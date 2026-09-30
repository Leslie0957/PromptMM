"""Frozen fixed-model frontier: metadata preflight by default; CPU fixtures opt-in.

Real execution requires a separately committed human-authorized declaration.
Never imports the original diagnostic runner or its probe-capable Roles class.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import copy
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from quality_frontier_core import (ARMS, SelectRoles, array_hash, decision, evaluate_grid, grid,
                                   history_queries, label_sets, make_cache, paired_bootstrap,
                                   normalize, require, transformed_content)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "docs/research/innovation2/QUALITY_CONSTRAINED_FRONTIER_PROFILE_V1.json"
PROFILE_SHA = "a665ed386d58c47067fc33d04304f06752d814b07b0eb8077c524367dfd986cd"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def local_path(value):
    require(isinstance(value, str) and "\\" not in value and ":" not in value, "invalid relative path")
    path = (ROOT / value).resolve()
    require(not Path(value).is_absolute() and ".." not in Path(value).parts and path.is_relative_to(ROOT), "path escapes repo")
    return path


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def read_profile():
    # Compare canonical Git LF bytes as well as the frozen semantic contract.
    raw = PROFILE.read_bytes().replace(b"\r\n", b"\n")
    require(hashlib.sha256(raw).hexdigest() == PROFILE_SHA, "frozen profile changed")
    profile = json.loads(raw)
    require(len(grid(profile)) == 63 and profile["models"]["arms"] == list(ARMS), "grid/arms changed")
    return profile


def metadata_preflight(profile):
    rows = []
    for asset in profile["anchor"]["assets"] + profile["source"]["allowed_original_payloads"]:
        path = local_path(asset["path"])
        rows.append({"path": asset["path"], "exists": path.is_file(),
                     "byte_size_matches": path.is_file() and path.stat().st_size == asset["bytes"]})
    return {"status": "metadata_only_preflight", "runtime_exists": True, "formal_authorization": False,
            "real_payload_reads": 0, "real_payload_hash_reads": 0, "ranking_states": 0,
            "profile_sha256": PROFILE_SHA, "assets": rows,
            "formal_output_exists": local_path(profile["artifacts"]["root"]).exists(),
            "declaration_exists": local_path(profile["launch"]["declaration_path"]).is_file()}


def declaration_gate(profile, declaration_path):
    require(declaration_path == local_path(profile["launch"]["declaration_path"]), "unexpected declaration path")
    require(not git("status", "--porcelain", "--untracked-files=no"), "dirty tracked launch source")
    for relative in ("tools/run_quality_constrained_frontier.py", "tools/quality_frontier_core.py"):
        source = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
        require(source.replace(b"\r\n", b"\n") == local_path(relative).read_bytes().replace(b"\r\n", b"\n"), "uncommitted runtime")
    raw = declaration_path.read_bytes().replace(b"\r\n", b"\n")
    committed = subprocess.check_output(["git", "show", "HEAD:" + declaration_path.relative_to(ROOT).as_posix()], cwd=ROOT)
    require(raw == committed.replace(b"\r\n", b"\n"), "uncommitted declaration")
    d = json.loads(raw)
    validate_declaration(d, profile, git("branch", "--show-current"))
    require(not local_path(d["output_root"]).exists(), "output exists; no retry/overwrite")
    return d, git("rev-parse", "HEAD")


def validate_declaration(d, profile, branch):
    """Pure contract check so rejection paths can be tested without real assets."""
    require(d.get("human_authorized") is True and bool(d.get("authorization_evidence")), "human authority missing")
    require(d.get("protocol_id") == profile["protocol_id"] and d.get("profile_sha256") == PROFILE_SHA, "protocol identity mismatch")
    require(d.get("scope") == "one_fixed_model_frontier_development", "scope mismatch")
    require(d.get("anchor") == profile["anchor"], "anchor changed")
    require(d.get("output_root") == profile["artifacts"]["root"], "output changed")
    require(d.get("command") == profile["launch"]["prospective_command"], "command changed")
    require(d.get("information") == profile["information"], "information boundary changed")
    require(d.get("resources") == profile["resources"], "resources changed")
    require(d.get("source_rule") == "clean_committed_HEAD_containing_this_declaration", "source rule changed")
    require(d.get("branch") == branch, "launch branch mismatch")
    require(d.get("one_launch_no_retry_no_resume_no_next_stage") is True, "one-launch guard missing")
    require(d.get("originalTrain_partition_exception_acknowledged") is True, "partition contact undisclosed")
    require(d.get("record_update_targets") == profile["routing"], "routing missing/changed")


class Budget:
    """Cooperative time checks and sampled resources, not OS isolation."""
    def __init__(self, out, resources):
        self.out, self.resources = out, resources
        self.start = self.stage_start = time.monotonic()
        self.stage_name = "restore_assets"
        self.times = {}
        self.error = None
        self.peak_rss = 0
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.sample()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def sample(self):
        with self.lock:
            rss = psutil.Process().memory_info().rss
            size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
            free = shutil.disk_usage(self.out).free
            self.peak_rss = max(self.peak_rss, rss)
            row = {"seconds": time.monotonic() - self.start, "stage": self.stage_name,
                   "rss_bytes": rss, "output_bytes": size, "free_disk_bytes": free}
            with (self.out / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            require(rss <= self.resources["rss_limit_gib"] * 2**30, "RSS cap exceeded")
            require(size <= self.resources["output_limit_gib"] * 2**30, "output cap exceeded")
            require(free >= self.resources["free_disk_floor_gib"] * 2**30, "free disk floor exceeded")

    def watch(self):
        while not self.stop.wait(self.resources["telemetry_interval_seconds"]):
            try:
                self.sample()
                self.check()
            except Exception as error:
                self.error = error
                return

    def check(self):
        if self.error:
            raise self.error
        now = time.monotonic()
        require(now - self.start <= self.resources["total_wall_seconds"], "total time cap exceeded")
        require(now - self.stage_start <= self.resources["stage_limits_seconds"][self.stage_name], "stage time cap exceeded")

    def stage(self, name):
        self.check()
        self.times[self.stage_name] = time.monotonic() - self.stage_start
        self.stage_start, self.stage_name = time.monotonic(), name
        self.sample()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)
        self.times[self.stage_name] = time.monotonic() - self.stage_start
        self.sample()
        self.check()


class Access:
    def __init__(self, profile):
        self.assets = {a["path"]: a for a in profile["anchor"]["assets"] + profile["source"]["allowed_original_payloads"]}
        self.events, self.identities = [], {}

    def verified(self, relative):
        require(relative in self.assets, "payload outside allowlist")
        path = local_path(relative)
        if relative not in self.identities:
            anchor = self.assets[relative]
            require(path.stat().st_size == anchor["bytes"], "asset size changed: " + relative)
            actual = digest(path)
            require(actual == anchor["sha256"], "asset hash changed: " + relative)
            self.identities[relative] = dict(anchor)
            self.events.append({"operation": "verify_payload_hash", "path": relative})
        self.events.append({"operation": "load_allowed_payload", "path": relative})
        return path


def deny_forbidden_reads(profile):
    forbidden = {str(local_path(p)).casefold() for p in profile["source"]["forbidden_payloads"]}
    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            raw = os.fsdecode(args[0])
            resolved = str(Path(raw).resolve()).casefold()
            require(resolved not in forbidden and Path(raw).name.casefold() != "sealed_lock.json", "forbidden payload read")
    sys.addaudithook(audit)


def infer_checkpoint(path, x, dimensions):
    require(path.suffix in (".npz", ".pt"), "unsupported checkpoint")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as state:
            out = (x @ state["weight"] + state["bias"]).astype(np.float32)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(dimensions[0], dimensions[1]), torch.nn.ReLU(),
                                    torch.nn.Linear(dimensions[1], dimensions[2]))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        with torch.inference_mode():
            out = model(torch.from_numpy(x)).numpy().copy()
    require(out.shape == (len(x), dimensions[2]) and np.isfinite(out).all(), "invalid ingress vectors")
    return out


def real_inputs(profile, access, check):
    train_path = access.verified("data/sports/train_mat")
    with train_path.open("rb") as stream:
        matrix = pickle.load(stream)  # trusted hash-bound prior conversion asset only
    require(tuple(matrix.shape) == (35598, 18357), "Train shape changed")
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    require(matrix.nnz == profile["source"]["original_shapes"]["original_train_nnz"]
            and np.isfinite(matrix.data).all() and np.all(matrix.data > 0), "invalid Train")
    row, col = matrix.nonzero()
    pairs = np.column_stack((row, col)).astype(np.int64)
    roles = SelectRoles(pairs, matrix.shape[1], profile["restoration"]["split"])
    roles.verify_identity(profile["restoration"])
    del matrix, row, col, pairs
    check()
    oldroot = "exp/innovation2/sports_unseen_transfer_v1/"
    # Verify small identity/selection records; never use their probe outcomes.
    for name in ("teacher_selection.json", "student_selection.json", "split_manifest.json", "source_manifest.json"):
        access.verified(oldroot + name)
    with np.load(access.verified(oldroot + "T_CF_tables.npz"), allow_pickle=False) as state:
        require(np.array_equal(state["warm_ids"], roles.warm), "CF warm mapping changed")
        pu, qw = state["users"].astype(np.float32), state["items"].astype(np.float32)
    require(pu.shape == (len(roles.users), 64) and qw.shape == (len(roles.warm), 64), "CF table shape changed")
    with np.load(access.verified(oldroot + "transform.npz"), allow_pickle=False) as state:
        transform = {name: state[name].copy() for name in state.files}
    sources = []
    for name in ("image", "text"):
        source = np.load(access.verified("data/sports/" + name + "_feat.npy"), mmap_mode="r", allow_pickle=False)
        require(list(source.shape) == profile["source"]["original_shapes"][name], "feature shape changed")
        sources.append(source)
    xw = transformed_content(sources, transform, roles.ids("warm_fit"))
    xc = transformed_content(sources, transform, roles.ids("cold_select"))
    del sources, transform
    query = history_queries(xw, roles)
    require(array_hash(query) == profile["models"]["N_query_sha256"], "N query identity changed")
    dims = tuple(profile["models"][k] for k in ("input_dim", "hidden_dim", "output_dim"))
    require(xw.shape[1] == xc.shape[1] == dims[0], "content dimension changed")
    ingress, identities = {}, {}
    for arm in ARMS[:-1]:
        check()
        path = access.verified(oldroot + profile["models"]["fixed_selected_states"][arm]["checkpoint"])
        ingress[arm] = infer_checkpoint(path, xc, dims)
        identities[arm] = dict(profile["models"]["fixed_selected_states"][arm])
    ingress["N"] = normalize(xc).astype(np.float32)
    identities["N"] = dict(profile["models"]["fixed_selected_states"]["N"])
    return roles, pu, qw, query, xw, ingress, identities


def synthetic_inputs(profile, out, check):
    """Tiny generated payloads only; fixed random states, no optimization/PCA fit."""
    rng = np.random.default_rng(20260930)
    nitems, nusers = 160, 16
    pairs = [(u, i) for u in range(nusers) for i in rng.choice(nitems, 70, replace=False)]
    roles = SelectRoles(pairs, nitems, profile["restoration"]["split"])
    sources = [rng.normal(size=(nitems, 12)), rng.normal(size=(nitems, 10))]
    transform = {"mean__image": np.zeros(12), "mean__text": np.zeros(10),
                 "components__image": np.eye(12)[:4], "components__text": np.eye(10)[:4]}
    fixture = out / "fixture_inputs"
    fixture.mkdir()
    np.savez_compressed(fixture / "transform.npz", **transform)
    for name, values in zip(("image", "text"), sources):
        np.save(fixture / (name + ".npy"), values)
    with np.load(fixture / "transform.npz", allow_pickle=False) as state:
        loaded = {name: state[name].copy() for name in state.files}
    sources = [np.load(fixture / (name + ".npy"), mmap_mode="r", allow_pickle=False) for name in ("image", "text")]
    xw = transformed_content(sources, loaded, roles.ids("warm_fit"))
    xc = transformed_content(sources, loaded, roles.ids("cold_select"))
    pu = rng.normal(size=(len(roles.users), 6)).astype(np.float32)
    qw = rng.normal(size=(len(roles.warm), 6)).astype(np.float32)
    query = history_queries(xw, roles)
    ingress, identities = {}, {}
    torch.manual_seed(20260930)
    for arm in ARMS[:-1]:
        check()
        if arm == "R":
            path = fixture / (arm + ".npz")
            np.savez_compressed(path, weight=rng.normal(size=(8, 6)), bias=np.zeros(6))
        else:
            path = fixture / (arm + ".pt")
            model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 6))
            torch.save(model.state_dict(), path)
        ingress[arm] = infer_checkpoint(path, xc, (8, 16, 6))
        identities[arm] = {"fixture_checkpoint": path.relative_to(out).as_posix(), "sha256": digest(path), "trained": False}
    ingress["N"] = normalize(xc).astype(np.float32)
    identities["N"] = {"query_sha256": array_hash(query), "trained": False}
    return roles, pu, qw, query, xw, ingress, identities


def execute(profile, out, synthetic=False, inject_failure=False, launch=None):
    require(not out.exists(), "output exists; no overwrite/retry")
    out.mkdir(parents=True, exist_ok=False)
    access, budget, roles = Access(profile), None, None
    batch = {"status": "running", "mode": "non_formal_synthetic" if synthetic else "formal_development",
             "profile_sha256": PROFILE_SHA, "source_commit": launch, "automatic_retry": False,
             "test_access": 0, "old_validation_access": 0, "old_probe_evaluation": 0,
             "lock_payload_reads": 0, "training_updates": 0}
    write_json(out / "batch.json", batch)
    write_json(out / "resolved_profile.json", profile)
    try:
        torch.set_num_threads(4)
        budget = Budget(out, profile["resources"])
        inputs = synthetic_inputs(profile, out, budget.check) if synthetic else real_inputs(profile, access, budget.check)
        roles, pu, qw, query, xw, ingress, identities = inputs
        del inputs
        write_json(out / "restoration_manifest.json", roles.manifest)
        write_json(out / "model_manifest.json", {"states": identities, "teacher_refits": 0, "checkpoint_reselection": False,
                   "cold_vectors": {a: {"sha256": array_hash(q), "norm_mean": float(np.linalg.norm(q, axis=1).mean())} for a, q in ingress.items()}})
        if inject_failure:
            raise ValueError("intentional synthetic failure after allowed-role restoration")
        budget.stage("score_grid")
        labels = label_sets(roles)
        k, user_batch = profile["scoring"]["k"], profile["scoring"]["user_batch"]
        core = make_cache(pu, qw, roles.warm, k, user_batch, budget.check, mask=labels["fit"])
        natural = make_cache(query, normalize(xw).astype(np.float32), roles.warm, k, user_batch, budget.check, mask=labels["fit"])
        warm = {a: natural if a == "N" else core for a in ARMS}
        cold = {}
        cache_dir = out / "score_cache"
        cache_dir.mkdir()
        for name, values in (("CF_warm", core), ("N_warm", natural)):
            np.savez_compressed(cache_dir / (name + ".npz"), ids=values["ids"], scores=values["scores"])
        for arm in ARMS:
            cold[arm] = make_cache(query if arm == "N" else pu, ingress[arm], roles.cold, k, user_batch,
                                   budget.check, candidates=grid(profile), sigma=warm[arm]["sigma"])
            np.savez_compressed(cache_dir / (arm + "_cold.npz"), ids=cold[arm]["ids"], scores=cold[arm]["scores"])
        write_json(out / "score_cache_manifest.json", {"full_score_matrix_saved": False, "warm_base_passes": 2,
                   "cold_base_passes": 6, "core_warm_shared": True, "group_topK": k,
                   "cold_order_certificates": {a: cold[a]["order_certificate"] for a in ARMS},
                   "warm_sigma": {a: warm[a]["sigma"] for a in ARMS},
                   "cache_files": [{"path": p.relative_to(out).as_posix(), "sha256": digest(p)} for p in sorted(cache_dir.glob("*.npz"))]})
        state_map = {}
        def save_arrays(saved):
            arrays = {"users": roles.users}
            for j, (state, values) in enumerate(saved.items()):
                prefix = f"s{j:03d}"
                state_map[state] = prefix
                arrays.update({prefix + "__" + name: value for name, value in values.items()})
            np.savez_compressed(out / "per_user_selected.npz", **arrays)
        result, saved = evaluate_grid(profile, roles, warm, cold, budget.check, save_arrays)
        write_json(out / "grid_metrics.json", result["grid"])
        write_json(out / "frontier.json", result["frontier"])
        write_json(out / "budget_selection.json", {"selections": result["selection"], "state_to_prefix": state_map,
                   "reference_warm": result["reference_warm"], "reference_overall": result["reference_overall"],
                   "references": result["references"], "per_user_states": len(state_map)})
        budget.stage("bootstrap")
        intervals = paired_bootstrap(result, saved, profile, budget.check)
        interpretation = decision(result, roles, profile)
        budget.stage("closeout")
        write_json(out / "report.json", {"scope": "synthetic_no_Sports_conclusion" if synthetic else profile["scientific_scope"],
                   "decision": interpretation, "auxiliary": result["auxiliary"], "intervals": intervals,
                   "ranking_states_evaluated": result["ranking_states_evaluated"],
                   "ranking_states_skipped": result["ranking_states_skipped"], "ranking_passes_repeated": 0,
                   "stage_timing": budget.times, "bootstrap_replicates": profile["evaluation"]["bootstrap_replicates"]})
        write_json(out / "source_manifest.json", {"source_commit": launch, "profile_sha256": PROFILE_SHA,
                   "source_sha256": {p.name: digest(p) for p in (Path(__file__), ROOT / "tools/quality_frontier_core.py")},
                   "assets": list(access.identities.values()), "synthetic_delta": synthetic,
                   "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                                   "psutil": psutil.__version__, "device": "cpu", "threads": torch.get_num_threads()}})
        write_json(out / "access_audit.json", access_record(access, roles, synthetic))
        missing = [p for p in profile["artifacts"]["required"] if not (out / p).is_file()]
        require(not missing, "missing artifacts: " + str(missing))
        require(len(result["grid"]) == 378 and len(result["references"]) == 2, "incomplete state inventory")
        require(len(intervals) == 36, "incomplete comparison inventory")
        budget.close()
        artifacts = [{"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
                     for p in sorted(out.rglob("*")) if p.is_file() and p.name != "batch.json"]
        budget.check()
        batch.update(status="completed", stage_timing=budget.times, wall_seconds=time.monotonic() - budget.start,
                     peak_rss_bytes=budget.peak_rss, scientific_decision=interpretation["status"],
                     ranking_states_evaluated=result["ranking_states_evaluated"], ranking_states_skipped=result["ranking_states_skipped"],
                     artifacts=artifacts)
        write_json(out / "batch.json", batch)
    except Exception as error:
        if budget:
            budget.stop.set()
            budget.thread.join(timeout=6)
        batch.update(status="failed", error_type=type(error).__name__, error=str(error), recovery_authority_required=True)
        write_json(out / "batch.json", batch)
        raise
    finally:
        if batch["status"] != "completed":
            write_json(out / "access_audit.json", access_record(access, roles, synthetic))
    return batch


def access_record(access, roles, synthetic):
    return {"application_guard_not_OS_trace": True, "events": access.events,
            "role_events": roles.events if roles else [],
            "real_payload_load_events": sum(e["operation"] == "load_allowed_payload" for e in access.events),
            "originalTrain_partition_contact": not synthetic and bool(roles), "probe_labels_exported": False,
            "test_access": 0, "old_validation_access": 0, "old_probe_evaluation": 0, "lock_payload_reads": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--synthetic", action="store_true")
    parser.add_argument("--declaration")
    parser.add_argument("--output", help="synthetic namespace only")
    parser.add_argument("--inject-failure", action="store_true", help="synthetic failure-preservation fixture")
    args = parser.parse_args()
    profile = read_profile()
    deny_forbidden_reads(profile)
    if args.synthetic:
        require(args.output is not None and args.declaration is None, "synthetic output required; no declaration")
        out = local_path(args.output)
        require(out.parent == ROOT / "exp/innovation2" and out.name.startswith("synthetic_quality_frontier_"), "isolated synthetic namespace required")
        p = copy.deepcopy(profile)
        p["evaluation"]["bootstrap_replicates"] = 40
        p["scoring"]["k"], p["scoring"]["user_batch"] = 5, 8
        p["resources"].update(total_wall_seconds=120, output_limit_gib=0.5, free_disk_floor_gib=2)
        p["resources"]["stage_limits_seconds"] = {s: 120 for s in p["resources"]["stage_limits_seconds"]}
        p["synthetic_delta"] = {"users": 16, "items": 160, "content_dim": 8, "hidden_dim": 16,
                                 "output_dim": 6, "k": 5, "bootstrap_replicates": 40, "trained": False,
                                 "all_380_states_retained": True, "real_payloads": 0}
        print(json.dumps(execute(p, out, synthetic=True, inject_failure=args.inject_failure), ensure_ascii=False))
    elif args.execute:
        require(args.output is None and not args.inject_failure and args.declaration is not None, "formal interface mismatch")
        _, launch = declaration_gate(profile, local_path(args.declaration))
        print(json.dumps(execute(profile, local_path(profile["artifacts"]["root"]), launch=launch), ensure_ascii=False))
    else:
        require(args.declaration is None and args.output is None and not args.inject_failure, "preflight takes no payload arguments")
        print(json.dumps(metadata_preflight(profile), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
