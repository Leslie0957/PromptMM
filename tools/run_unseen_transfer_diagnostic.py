"""Sports unseen-transfer runner. Default is metadata-only preflight.

Real payloads require --execute plus a committed, explicit cohort declaration.
Synthetic checks force CPU, small fixtures, isolated output, and non-formal status.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import threading
import time
import traceback

for _threads_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_threads_env] = "4"

import numpy as np
import psutil
import scipy
from scipy import sparse
import sklearn
import torch

from validate_unseen_transfer_protocol import PROFILE, ROOT, check_payload_permission, local_path, require, validate, verify_metadata
from unseen_transfer_core import (Content, Roles, Tape, Teacher, Undetermined, array_hash, content_reference,
                                 decide, evaluate, evaluate_arrays, graph, kd_loss, margin, mlp,
                                 paired_intervals, ridge)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def artifact_hashes(out, sealed_hash):
    return {path.relative_to(out).as_posix(): (sealed_hash if path == out / "sealed_lock.json" else digest(path))
            for path in sorted(out.rglob("*")) if path.is_file() and path.name != "batch.json"}


def env_identity():
    return {"python": os.sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__, "psutil": psutil.__version__}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def declaration_gate(path, profile):
    """Check concrete source/declaration identity; not a substitute for human authority."""
    path = path.resolve()
    require(path.is_relative_to(ROOT), "declaration outside repository")
    rel = path.relative_to(ROOT).as_posix()
    require(not git("status", "--porcelain", "--untracked-files=no"), "tracked launch source is dirty")
    tracked = git("ls-files", "--", rel)
    require(tracked == rel, "declaration must be committed")
    head_bytes = subprocess.check_output(["git", "show", "HEAD:" + rel], cwd=ROOT)
    require(head_bytes.replace(b"\r\n", b"\n") == path.read_bytes().replace(b"\r\n", b"\n"), "declaration differs from HEAD")
    d = json.loads(path.read_text(encoding="utf-8"))
    require(d["protocol_id"] == profile["protocol_id"] and d["scope"] == "one_serial_cohort_assets_teachers_students_report",
            "wrong declaration scope")
    require(d["explicit_user_authorization"] is True and bool(d["authorization_evidence"]), "missing explicit authorization evidence")
    require(d["launch_source_rule"] == "committed_HEAD_containing_this_declaration", "unsupported source rule")
    require(d["profile_sha256"] == digest(PROFILE), "profile identity differs from declaration")
    require(d["output"] == profile["artifacts"]["root"] and d["test_access"] == d["old_validation_access"] == d["lock_confirmation_access"] == 0,
            "output/access policy differs")
    require(d["one_launch"] is True and d["retry"] is False and d["next_stage"] is False, "execution expansion forbidden")
    expected = "& 'D:\\miniconda\\envs\\run_5060\\python.exe' -B tools/run_unseen_transfer_diagnostic.py --execute --declaration " + rel
    require(d["command"] == expected, "declared command differs from fixed interface")
    require(d["routing"] == profile["routing"], "record targets differ from profile")
    return d, git("rev-parse", "HEAD")


class Budget:
    """Sampled watchdog plus cooperative checkpoints; not OS hard isolation."""
    def __init__(self, out, p, device, synthetic):
        self.out, self.rules, self.device, self.synthetic = out, p["resources"], device, synthetic
        self.start = self.stage_start = time.monotonic()
        self.stage, self.violation, self.samples = "assets", None, []
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self.watch, daemon=True)
        self.worker.start()

    def switch(self, stage):
        self.check()
        self.stage, self.stage_start = stage, time.monotonic()

    def sample(self):
        with self.lock:
            now = time.monotonic()
            row = {"elapsed_seconds": now - self.start, "stage": self.stage, "rss_bytes": psutil.Process().memory_info().rss,
                   "free_disk_bytes": shutil.disk_usage(self.out).free,
                   "output_bytes": sum(path.stat().st_size for path in self.out.rglob("*") if path.is_file()),
                   "cuda_allocated_bytes": torch.cuda.memory_allocated() if self.device == "cuda" else 0,
                   "os_gpu_used_mib": None}
            if self.device == "cuda":
                try:
                    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    row["os_gpu_used_mib"] = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        timeout=2, creationflags=flags).decode().strip()
                except (OSError, subprocess.SubprocessError) as exc:
                    row["os_gpu_query_error"] = type(exc).__name__
            wall_limit = 120 if self.synthetic else self.rules["total_wall_time_limit_hours"] * 3600
            limits = [(row["elapsed_seconds"] > wall_limit, "total wall time"),
                      (now - self.stage_start > self.rules["stage_timeout_hours"][self.stage] * 3600, "stage wall time"),
                      (row["rss_bytes"] > self.rules["process_rss_limit_gib"] * 2**30, "RSS"),
                      (row["free_disk_bytes"] < self.rules["free_disk_floor_gib"] * 2**30, "free disk"),
                      (row["output_bytes"] > self.rules["output_limit_gib"] * 2**30, "output size"),
                      (row["cuda_allocated_bytes"] > self.rules["cuda_allocated_limit_gib"] * 2**30, "CUDA allocation")]
            for triggered, name in limits:
                if triggered:
                    self.violation = self.violation or name
            self.samples.append(row)
            with (self.out / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")

    def watch(self):
        try:
            while not self.closed.is_set():
                self.sample()
                self.closed.wait(self.rules["telemetry_interval_seconds"])
        except Exception as exc:
            self.violation = "telemetry failure: " + repr(exc)

    def check(self):
        require(self.violation is None, "resource stop: " + str(self.violation))
        if self.device == "cuda":
            require(torch.cuda.memory_allocated() <= self.rules["cuda_allocated_limit_gib"] * 2**30, "CUDA allocation limit")
        require(time.monotonic() - self.start <= (120 if self.synthetic else self.rules["total_wall_time_limit_hours"] * 3600), "total timeout")
        require(time.monotonic() - self.stage_start <= self.rules["stage_timeout_hours"][self.stage] * 3600, "stage timeout")

    def close(self):
        self.closed.set()
        self.worker.join(timeout=4)
        self.sample()


def synthetic_profile(original):
    p = copy.deepcopy(original)
    p["teachers"].update(dim=8, max_epochs=2, evaluate_every_epochs=1, batch_size=128)
    p["features"].update(image_pca_components=4, text_pca_components=4, input_dim=8)
    p["students"]["mlp"].update(input=8, hidden=16, output=8)
    p["students"].update(max_epochs=2, evaluate_every_epochs=1, batch_size=128)
    p["students"]["kd"]["scale_tape_triplets"] = 256
    p["evaluation"].update(k=5, user_batch=8, bootstrap_replicates=40)
    return p


def load_inputs(p, synthetic, budget):
    if synthetic:
        pairs = np.asarray([(u, (u * 10 + j) % 160) for u in range(16) for j in range(64)], dtype=np.int64)
        rng = np.random.default_rng(73)
        return pairs, rng.normal(size=(160, 8)), rng.normal(size=(160, 8)), {"kind": "synthetic", "payload_reads": []}
    assets = []
    for asset in p["source"]["allowed_payloads"]:
        budget.check()
        path = check_payload_permission(asset["path"])
        require(path.stat().st_size == asset["bytes"] and digest(path) == asset["sha256"], "source identity mismatch before deserialization")
        assets.append(dict(asset))
    # Exact verified local Train pickle only; never pass external/unverified pickle here.
    with check_payload_permission("data/sports/train_mat").open("rb") as stream:
        matrix = pickle.load(stream)
    shapes = p["source"]["metadata_shapes"]
    require(sparse.issparse(matrix) and matrix.shape == (shapes["users"], shapes["items"]), "Train matrix shape/type mismatch")
    matrix = matrix.tocsr(copy=True)
    require(np.isfinite(matrix.data).all() and np.all(matrix.data > 0), "invalid Train positive weight")
    matrix.sum_duplicates()
    require(matrix.nnz == shapes["original_train_nnz"], "Train nnz mismatch")
    row, col = matrix.nonzero()
    pairs = np.column_stack((row, col))
    del matrix
    image = np.load(check_payload_permission("data/sports/image_feat.npy"), mmap_mode="r", allow_pickle=False)
    text = np.load(check_payload_permission("data/sports/text_feat.npy"), mmap_mode="r", allow_pickle=False)
    require(list(image.shape) == shapes["image"] and list(text.shape) == shapes["text"], "feature payload shape mismatch")
    return pairs, image, text, {"kind": "originalTrain_and_preextracted_features", "payload_reads": assets,
                              "feature_loading": "mmap containers; numerical rows role-gated; lock rows not processed"}


def snapshot(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def gradients(model):
    total = 0.
    for parameter in model.parameters():
        if parameter.grad is not None:
            require(torch.isfinite(parameter.grad).all().item(), "nonfinite gradient")
            total += float(parameter.grad.detach().square().sum().item())
    return total ** .5


def selection_key(metrics, epoch, lr=0, strength=0):
    overall, cold = metrics["overall"]["recall20"], metrics["cold"]["recall20"]
    if overall is None:
        raise Undetermined("no selection positives")
    return overall, cold if cold is not None else -1., -epoch, -lr, -strength


def make_score(p_user, warm_q, cold_q, roles, cold_ids, scale=1., content_mode=False):
    ids = np.concatenate((roles.warm, cold_ids))
    order = np.argsort(ids)
    all_ids = ids[order]
    q = np.concatenate((warm_q, cold_q))[order].astype(np.float32)
    if content_mode:
        from unseen_transfer_core import normalize
        q = normalize(q).astype(np.float32)
    q[np.isin(all_ids, cold_ids)] *= scale
    pu = np.asarray(p_user, dtype=np.float32)
    def score(users, requested):
        indices = np.searchsorted(all_ids, requested)
        require(np.all(all_ids[indices] == requested), "requested unknown candidate")
        return pu[users] @ q[indices].T
    return score


def train_teachers(roles, content, tape, p, device, out, budget):
    budget.switch("teachers_total")
    x = torch.tensor(content.warm, device=device)
    adj = graph(roles, device)
    write_json(out / "graph_manifest.json", {"fit_role": "warm_fit", "indices_sha256": array_hash(adj.indices().cpu().numpy()),
                "normalized_values_sha256": array_hash(adj.values().cpu().numpy()), "shape": list(adj.shape), "self_loops": False})
    selected, tables, fit_records = {}, {}, []
    for name in p["teachers"]["names"]:
        best_key, best_state = None, None
        for lr in p["teachers"]["learning_rates"]:
            roles.set_phase("fit")
            model = Teacher(len(roles.users), len(roles.warm), x, adj, p, name == "T_MM").to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p["teachers"]["weight_decay"])
            curve, losses, local_key, local_state, picked = [], [], None, None, None
            fit_dir = out / "teachers" / (name + "_lr" + str(lr))
            fit_dir.mkdir(parents=True)
            for epoch in range(1, p["teachers"]["max_epochs"] + 1):
                triplets = tape.epoch(epoch)
                total, norms, batches = 0., [], 0
                for start in range(0, len(triplets), p["teachers"]["batch_size"]):
                    budget.check()
                    batch = torch.tensor(triplets[start:start + p["teachers"]["batch_size"]], device=device)
                    optimizer.zero_grad(set_to_none=True)
                    pu, qi = model()
                    loss = torch.nn.functional.softplus(-margin(pu, qi, batch)).mean()
                    require(torch.isfinite(loss).item(), "nonfinite teacher objective")
                    loss.backward()
                    norms.append(gradients(model))
                    optimizer.step()
                    total, batches = total + float(loss.item()) * len(batch), batches + len(batch)
                losses.append({"epoch": epoch, "loss": total / batches, "gradient_norm_max": max(norms), "gradient_norm_mean": float(np.mean(norms))})
                if epoch % p["teachers"]["evaluate_every_epochs"] == 0:
                    roles.set_phase("selection")
                    with torch.no_grad():
                        pu, qi = (v.detach().cpu().numpy() for v in model())
                    score = make_score(pu, qi, np.empty((0, qi.shape[1]), np.float32), roles, np.empty(0, np.int64))
                    metrics, _ = evaluate_arrays(score, roles.users, roles.warm, np.empty(0, np.int64), roles.labels("warm_fit"),
                                                 roles.labels("warm_select"), np.empty((0, 2), np.int64), p["evaluation"]["k"],
                                                 p["evaluation"]["user_batch"], budget.check, cold_only=False)
                    value = metrics["warm"]["recall20"]
                    if value is None:
                        raise Undetermined("teacher warm selection empty")
                    key = value, -epoch, -lr
                    curve.append({"epoch": epoch, "warm_recall": value})
                    if local_key is None or key > local_key:
                        local_key, local_state, picked = key, snapshot(model), {"epoch": epoch, "metrics": metrics}
                        torch.save(local_state, fit_dir / "selected.pt")
                    write_json(fit_dir / "curve.json", {"selection": curve, "optimization": losses})
                    roles.set_phase("fit")
            write_json(fit_dir / "curve.json", {"selection": curve, "optimization": losses})
            torch.save(local_state, fit_dir / "selected.pt")
            torch.save(snapshot(model), fit_dir / "final.pt")
            record = {"name": name, "lr": lr, "selected_epoch": picked["epoch"], "curve": curve,
                      "checkpoint": (fit_dir / "selected.pt").relative_to(out).as_posix(), "checkpoint_sha256": digest(fit_dir / "selected.pt"),
                      "selected_metrics": picked["metrics"]}
            fit_records.append(record)
            if best_key is None or local_key > best_key:
                best_key, best_state, selected[name] = local_key, local_state, record
            del model, optimizer, local_state
        chosen = Teacher(len(roles.users), len(roles.warm), x, adj, p, name == "T_MM").to(device)
        chosen.load_state_dict(best_state)
        with torch.no_grad():
            tables[name] = tuple(v.cpu().numpy().copy() for v in chosen())
        np.savez_compressed(out / (name + "_tables.npz"), users=tables[name][0], items=tables[name][1], warm_ids=roles.warm)
        del chosen, best_state
    write_json(out / "teacher_selection.json", {"selected": selected, "all_fits": fit_records})
    return selected, tables, fit_records


def train_students(roles, content, tape, p, device, out, budget, tables):
    budget.switch("students_total")
    roles.set_phase("selection")
    cold_ids, cold_x = roles.ids("cold_select"), content.rows("cold_select")
    roles.set_phase("fit")
    pu, qcf = tables["T_CF"]
    target_energy = float(np.square(qcf.astype(np.float64)).mean())
    if target_energy < 1e-12:
        raise Undetermined("degenerate warm CF vector energy")
    scales = {}
    scale_tape = torch.tensor(tape.scale(p["students"]["kd"]["scale_tape_triplets"]))
    for name, (pt, qt) in tables.items():
        values = margin(torch.from_numpy(pt), torch.from_numpy(qt), scale_tape).numpy().astype(np.float64)
        scales[name] = float(np.std(values, ddof=0))
        if scales[name] < p["students"]["kd"]["scale_floor"]:
            raise Undetermined("degenerate teacher margin scale: " + name)
    write_json(out / "target_manifest.json", {"scales": scales, "warm_vector_energy": target_energy,
                "scale_tape_sha256": array_hash(scale_tape.numpy()), "fit_tape_pairs_sha256": array_hash(tape.pairs),
                "excluded_no_negative_users": tape.excluded_users})
    x = torch.tensor(content.warm, device=device)
    p_tensor, q_tensor = torch.tensor(pu, device=device), torch.tensor(qcf, device=device)
    t_tables = {name: (torch.tensor(pt, device=device), torch.tensor(qt, device=device)) for name, (pt, qt) in tables.items()}
    selected, functions, all_fits = {}, {}, []
    # Closures are frozen learned content maps; cold_probe features enter only after seal.
    for arm in p["students"]["arms"]:
        best_key, best_function, best_record = None, None, None
        if arm == "R":
            for strength in p["students"]["ridge"]["lambdas"]:
                budget.check()
                weights, bias = ridge(content.warm, qcf, strength)
                fn = lambda value, w=weights.copy(), b=bias.copy(): (value @ w + b).astype(np.float32)
                roles.set_phase("selection")
                metrics, _ = evaluate(make_score(pu, qcf, fn(cold_x), roles, cold_ids), roles, "selection", p, budget.check)
                key = selection_key(metrics, 0, strength=strength)
                fit_dir = out / "students" / (arm + "_lambda" + str(strength))
                fit_dir.mkdir(parents=True)
                np.savez_compressed(fit_dir / "selected.npz", weight=weights, bias=bias)
                record = {"arm": arm, "lambda": strength, "epoch": 0, "metrics": metrics, "curve": [],
                          "checkpoint": (fit_dir / "selected.npz").relative_to(out).as_posix(), "checkpoint_sha256": digest(fit_dir / "selected.npz")}
                all_fits.append(record)
                if best_key is None or key > best_key:
                    best_key, best_function, best_record = key, fn, record
        elif arm == "N":
            query = content_reference(content.warm, tape.history)
            fn = lambda value: value.astype(np.float32)
            roles.set_phase("selection")
            metrics, _ = evaluate(make_score(query, content.warm, cold_x, roles, cold_ids, content_mode=True), roles, "selection", p, budget.check)
            best_function, best_record = fn, {"arm": arm, "epoch": 0, "metrics": metrics, "curve": [],
                                                   "query_sha256": array_hash(query)}
            all_fits.append(best_record)
        else:
            strengths = p["students"]["kd"]["weights"] if arm.startswith("K_") else [0.]
            for lr in p["students"]["learning_rates"]:
                for strength in strengths:
                    roles.set_phase("fit")
                    model = mlp(p).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p["students"]["weight_decay"])
                    curve, losses, local_key, local_state, picked = [], [], None, None, None
                    fit_dir = out / "students" / (arm + "_lr" + str(lr) + "_beta" + str(strength))
                    fit_dir.mkdir(parents=True)
                    for epoch in range(1, p["students"]["max_epochs"] + 1):
                        total, examples, norms = 0., 0, []
                        data = (np.random.default_rng(np.random.SeedSequence([p["seeds"]["student"], epoch])).permutation(len(x))
                                if arm == "V" else tape.epoch(epoch))
                        for start in range(0, len(data), p["students"]["batch_size"]):
                            budget.check()
                            batch = torch.tensor(data[start:start + p["students"]["batch_size"]], device=device)
                            optimizer.zero_grad(set_to_none=True)
                            if arm == "V":
                                loss = (model(x[batch]) - q_tensor[batch]).square().mean() / target_energy
                            else:
                                u, i, j = batch.unbind(1)
                                ms = (p_tensor[u] * (model(x[i]) - model(x[j]))).sum(1)
                                if arm == "D":
                                    loss = torch.nn.functional.softplus(-ms).mean()
                                else:
                                    name = "T_CF" if arm == "K_CF" else "T_MM"
                                    mt = margin(*t_tables[name], batch)
                                    loss = kd_loss(ms, mt, scales["T_CF"], scales[name], strength)
                            require(torch.isfinite(loss).item(), "nonfinite student objective")
                            loss.backward()
                            norms.append(gradients(model))
                            optimizer.step()
                            total += float(loss.item()) * len(batch)
                            examples += len(batch)
                        losses.append({"epoch": epoch, "loss": total / examples, "gradient_norm_max": max(norms), "gradient_norm_mean": float(np.mean(norms))})
                        if epoch % p["students"]["evaluate_every_epochs"] == 0:
                            roles.set_phase("selection")
                            with torch.no_grad():
                                qcold = model(torch.tensor(cold_x, device=device)).cpu().numpy()
                            metrics, _ = evaluate(make_score(pu, qcf, qcold, roles, cold_ids), roles, "selection", p, budget.check)
                            key = selection_key(metrics, epoch, lr, strength)
                            curve.append({"epoch": epoch, "overall_recall": metrics["overall"]["recall20"], "cold_recall": metrics["cold"]["recall20"]})
                            if local_key is None or key > local_key:
                                local_key, local_state, picked = key, snapshot(model), {"epoch": epoch, "metrics": metrics}
                                torch.save(local_state, fit_dir / "selected.pt")
                            write_json(fit_dir / "curve.json", {"selection": curve, "optimization": losses})
                            roles.set_phase("fit")
                    torch.save(local_state, fit_dir / "selected.pt")
                    torch.save(snapshot(model), fit_dir / "final.pt")
                    write_json(fit_dir / "curve.json", {"selection": curve, "optimization": losses})
                    record = {"arm": arm, "lr": lr, "beta": strength, "epoch": picked["epoch"], "metrics": picked["metrics"], "curve": curve,
                              "checkpoint": (fit_dir / "selected.pt").relative_to(out).as_posix(), "checkpoint_sha256": digest(fit_dir / "selected.pt")}
                    all_fits.append(record)
                    if best_key is None or local_key > best_key:
                        chosen = mlp(p)
                        chosen.load_state_dict(local_state)
                        chosen.eval()
                        def fn(value, net=chosen):
                            with torch.no_grad():
                                return net(torch.tensor(value, dtype=torch.float32)).numpy()
                        best_key, best_function, best_record = local_key, fn, record
                    del model, optimizer, local_state
        selected[arm], functions[arm] = best_record, best_function
    # Raw choices fixed first. Calibration does not choose another fit/checkpoint.
    roles.set_phase("selection")
    query = content_reference(content.warm, tape.history)
    for arm in p["students"]["arms"]:
        best_key, calibration, calibration_rows = None, None, []
        qcold = functions[arm](cold_x)
        selected[arm]["cold_vector_norm_mean"] = float(np.linalg.norm(qcold, axis=1).mean())
        for scale in p["calibration"]["cold_score_scales"]:
            metrics, _ = evaluate(make_score(query if arm == "N" else pu, content.warm if arm == "N" else qcf,
                                             qcold, roles, cold_ids, scale, content_mode=arm == "N"), roles, "selection", p, budget.check)
            prefer = {1: 2, .5: 1, 2: 0}[scale]
            key = selection_key(metrics, 0)[:2] + (prefer,)
            calibration_rows.append({"scale": scale, "metrics": metrics})
            if best_key is None or key > best_key:
                best_key, calibration = key, scale
        selected[arm]["calibration_scale"] = calibration
        selected[arm]["calibration_choices"] = calibration_rows
    write_json(out / "student_selection.json", {"selected": selected, "all_fits": all_fits,
                 "raw_choices_before_calibration": True, "search_count": {a: sum(row["arm"] == a for row in all_fits) for a in p["students"]["arms"]}})
    return selected, functions, all_fits


def final_report(roles, content, tape, p, out, budget, tables, teachers, students, functions):
    budget.switch("report_total")
    seal = digest(out / "teacher_selection.json") + ":" + digest(out / "student_selection.json")
    roles.seal_selection(seal)
    summaries, arrays, all_arrays = {}, {}, {}
    for name, (pu, qi) in tables.items():
        score = make_score(pu, qi, np.empty((0, qi.shape[1]), np.float32), roles, np.empty(0, np.int64))
        summaries[name], values = evaluate_arrays(score, roles.users, roles.warm, np.empty(0, np.int64), roles.labels("warm_fit"),
                                                 roles.labels("warm_probe"), np.empty((0, 2), np.int64), p["evaluation"]["k"],
                                                 p["evaluation"]["user_batch"], budget.check, cold_only=False)
        all_arrays.update({name + "__" + k: v for k, v in values.items()})
    cold_ids, cold_x = roles.ids("cold_probe"), content.rows("cold_probe")
    pu, qcf = tables["T_CF"]
    query = content_reference(content.warm, tape.history)
    ingress = {}
    for arm in p["students"]["arms"]:
        start = time.monotonic()
        qcold = functions[arm](cold_x)
        ingress[arm] = {"cold_vector_generation_seconds": time.monotonic() - start, "items": len(qcold),
                        "cold_vector_sha256": array_hash(qcold), "cold_vector_norm_mean": float(np.linalg.norm(qcold, axis=1).mean())}
        for variant, scale in (("raw", 1.), ("calibrated", students[arm]["calibration_scale"])):
            key = arm + "/" + variant
            start = time.monotonic()
            summaries[key], arrays[key] = evaluate(make_score(query if arm == "N" else pu, content.warm if arm == "N" else qcf,
                                                              qcold, roles, cold_ids, scale, content_mode=arm == "N"), roles, "report", p, budget.check)
            summaries[key]["mixed_report_seconds"] = time.monotonic() - start
            all_arrays.update({key.replace("/", "__") + "__" + k: v for k, v in arrays[key].items()})
    intervals = paired_intervals(arrays, p, budget.check)
    decision = decide(summaries, intervals, roles, teachers, students, p)
    np.savez_compressed(out / "per_user_probe.npz", **all_arrays)
    calibration_observations = {a: {"raw_cold_only_gap_vs_K_MM": summaries["K_MM/raw"]["cold_only_recall"] - summaries[a + "/raw"]["cold_only_recall"]
                                          if summaries["K_MM/raw"]["cold_only_recall"] is not None and summaries[a + "/raw"]["cold_only_recall"] is not None else None,
                                     "raw_mixed_cold_gap": intervals["raw"]["cold"][a]["difference"],
                                     "calibrated_mixed_cold_gap": intervals["calibrated"]["cold"][a]["difference"]}
                                for a in p["students"]["arms"] if a != "K_MM"}
    return {"metrics": summaries, "paired_intervals": intervals, "decision": decision, "ingress_cost": ingress,
            "calibration_observations": calibration_observations,
            "selected_before_report_sha256": seal, "test_access": 0, "old_validation_access": 0, "lock_confirmation_access": 0,
            "teacher_probe_passes": 2, "student_probe_passes": 12, "one_seed_not_stability": True}


def run_cohort(original, out, synthetic, declaration=None, commit=None):
    require(not out.exists(), "output directory already exists; no overwrite/retry")
    if not synthetic:
        require(torch.cuda.is_available(), "declared CUDA unavailable")
    out.mkdir(parents=True, exist_ok=False)
    p = synthetic_profile(original) if synthetic else copy.deepcopy(original)
    device = "cpu" if synthetic else "cuda"
    torch.set_num_threads(p["resources"]["threads"])
    torch.use_deterministic_algorithms(True)
    # CUDA sparse reductions may require deterministic support on this environment.
    budget = Budget(out, p, device, synthetic)
    roles, report, source, sealed_hash = None, None, None, None
    start = time.monotonic()
    costs, status = {}, "running"
    source_role = "base_reference_with_uncommitted_preparation_runtime_hashes" if synthetic else "exact_committed_launch_HEAD"
    write_json(out / "resolved_profile.json", p)
    write_json(out / "batch.json", {"status": status, "synthetic": synthetic, "source_commit": commit or git("rev-parse", "HEAD"),
                                    "source_commit_role": source_role, "no_retry": True, "no_next_stage": True, "declaration": declaration, "environment": env_identity()})
    try:
        pairs, image, text, source = load_inputs(p, synthetic, budget)
        write_json(out / "source_manifest.json", {**source, "environment": env_identity(), "profile_sha256": digest(PROFILE),
                                                   "source_commit": commit or git("rev-parse", "HEAD"),
                                                   "source_commit_role": source_role,
                                                   "runtime_hashes": {n: digest(ROOT / "tools" / n) for n in
                                                        ("unseen_transfer_core.py", "run_unseen_transfer_diagnostic.py", "validate_unseen_transfer_protocol.py")}})
        def seal_lock(value):
            nonlocal sealed_hash
            sealed_hash = write_json(out / "sealed_lock.json", value)
        roles = Roles(pairs, len(image), p, seal_lock)
        del pairs
        write_json(out / "split_manifest.json", roles.manifest)
        content = Content(image, text, roles, p, budget.check)
        np.savez_compressed(out / "transform.npz", **content.transform_state())
        write_json(out / "transform_manifest.json", content.stats)
        tape = Tape(roles, p["seeds"]["triplet"])
        costs["assets_seconds"] = time.monotonic() - start
        step = time.monotonic()
        teachers, tables, teacher_fits = train_teachers(roles, content, tape, p, device, out, budget)
        costs["teachers_seconds"] = time.monotonic() - step
        step = time.monotonic()
        students, functions, student_fits = train_students(roles, content, tape, p, device, out, budget, tables)
        costs["students_targets_selection_seconds"] = time.monotonic() - step
        step = time.monotonic()
        report = final_report(roles, content, tape, p, out, budget, tables, teachers, students, functions)
        write_json(out / "transform_manifest.json", {**content.stats, "transform_npz_sha256": digest(out / "transform.npz")})
        costs["report_seconds"] = time.monotonic() - step
        report["fit_counts"] = {"teachers": len(teacher_fits), "students_including_ridge_N": len(student_fits), "neural_students": 12}
        status = "completed"
    except Undetermined as exc:
        status = "completed_undetermined_early_stop"
        report = {"decision": {"label": "undetermined", "reasons": [str(exc)], "innovation_certified": False},
                  "unexecuted_steps": "remainder of cohort not run; no synthetic or missing metrics fabricated"}
    except Exception as exc:
        status = "failed"
        report = {"error": repr(exc), "traceback": traceback.format_exc(), "partial_artifacts_preserved": True, "retry": False}
    finally:
        budget.close()
        if budget.violation:
            status = "failed"
            report = {**(report or {}), "resource_failure": budget.violation, "partial_artifacts_preserved": True}
        costs["total_seconds"] = time.monotonic() - start
        write_json(out / "access_audit.json", {"events": roles.events if roles else [], "source_payloads": (source or {}).get("payload_reads", []),
                 "old_val_payload_reads": 0, "test_payload_reads": 0, "sealed_lock_reads": 0,
                 "partition_lock_exception": "builder seal only", "scope": "application-level guards/events, not OS-wide file tracing"})
        write_json(out / "report.json", {**(report or {}), "status": status, "synthetic": synthetic, "costs": costs,
                                         "scientific_interpretation": "fixture only; no Sports result" if synthetic else "one-seed exploratory diagnostic"})
        write_json(out / "batch.json", {"status": status, "synthetic": synthetic, "source_commit": commit or git("rev-parse", "HEAD"),
                   "source_commit_role": source_role, "no_retry": True, "no_next_stage": True, "declaration": declaration, "environment": env_identity(),
                   "report_sha256": digest(out / "report.json"),
                   "files": artifact_hashes(out, sealed_hash)})
    return status, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--synthetic-check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=str)
    parser.add_argument("--declaration", type=str)
    args = parser.parse_args()
    p = json.loads(PROFILE.read_text(encoding="utf-8"))
    validate(p)
    if args.synthetic_check:
        require(args.output is not None and args.declaration is None, "synthetic mode requires isolated --output only")
        require(args.output.startswith("exp/innovation2/synthetic_unseen_") and ".." not in Path(args.output).parts,
                "synthetic output must use its own ignored family")
        status, report = run_cohort(p, local_path(args.output), True)
        print(json.dumps({"status": status, "mode": "non_formal_CPU_fixture", "output": args.output,
                          "fit_counts": report.get("fit_counts"), "Sports_result": False}, ensure_ascii=False))
        raise SystemExit(0 if status == "completed" else 1)
    if args.execute:
        require(args.declaration is not None and args.output is None, "real execution requires fixed output and declaration")
        declaration, commit = declaration_gate(local_path(args.declaration), p)
        require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "set CUBLAS_WORKSPACE_CONFIG=:4096:8 before real launch")
        verify_metadata(p)
        status, _ = run_cohort(p, local_path(p["artifacts"]["root"]), False, declaration, commit)
        print(json.dumps({"status": status, "output": p["artifacts"]["root"]}))
        raise SystemExit(0 if status.startswith("completed") else 1)
    require(args.output is None and args.declaration is None, "preflight does not write artifacts")
    print(json.dumps({"status": "preflight_passed", "source_metadata": verify_metadata(p), "environment": env_identity(),
                      "payload_access": 0, "training_or_evaluation": False, "launch_authorized": False,
                      "formal_command_interface": "--execute --declaration <committed-authorized-declaration>"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
