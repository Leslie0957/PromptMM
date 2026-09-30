"""Check the frozen diagnostic specification, never load data/model payloads.

Synthetic partition checks validate role isolation; they are not a runtime audit.
Only the optional conversion-manifest JSON and allowed file stat metadata are read.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "docs/research/innovation2/UNSEEN_ITEM_TRANSFER_DIAGNOSTIC_PROFILE_V1.json"
ALLOWED = {"data/sports/train_mat", "data/sports/image_feat.npy", "data/sports/text_feat.npy"}
ROLES = ("warm", "cold_select", "cold_probe", "cold_lock")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(value):
    require(isinstance(value, str) and "\\" not in value, "use a repository-relative POSIX path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and ":" not in value,
            "path escapes repository")
    require((ROOT / path).resolve().is_relative_to(ROOT), "resolved path escapes repository")
    return ROOT / path


def check_payload_permission(path):
    require(path in ALLOWED, "payload is outside Train-only allowlist")
    return local_path(path)


def validate(p):
    require(p["schema_version"] == 1 and p["protocol_id"] == "sports_unseen_transfer_v1", "unknown protocol")
    require(p["status"] == "protocol_frozen_runtime_not_implemented"
            and p["authority"] == "preparation_only", "this checker does not authorize execution")
    require(all(p["launch"][key] is False for key in ("ready", "runtime_exists", "formal_authorization")),
            "preparation cannot claim launch readiness")
    source = p["source"]
    paths = [a["path"] for a in source["allowed_payloads"]]
    require(len(paths) == 3 and set(paths) == ALLOWED, "allowed source set changed")
    for asset in source["allowed_payloads"]:
        check_payload_permission(asset["path"])
        require(asset["bytes"] > 0 and len(asset["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in asset["sha256"]), "invalid source anchor")
    require(source["manifest"] == "data/sports/conversion_manifest.json", "unexpected metadata source")
    require(source["reuse_original_teachers"] is False and source["reuse_full_catalog_transforms"] is False,
            "full-catalog assets are unsafe here")
    require(set(source["forbidden_payloads"]) >= {"data/sports/test_mat", "data/sports/val_mat",
            "data/_incoming/mmrec_sports/sports.inter"}, "missing forbidden source")
    info = p["information"]
    require(all(info[k] == 0 for k in ("test_access", "old_validation_access", "lock_confirmation_access")),
            "old evaluation/lock access is forbidden")
    require(info["fit_roles"] == ["warm_fit"], "cold or holdout labels in gradients")
    require(set(info["selection_roles"]) == {"warm_select", "cold_select"}
            and set(info["report_roles"]) == {"warm_probe", "cold_probe"}, "invalid role routing")
    groups = [set(info[k]) for k in ("fit_roles", "selection_roles", "report_roles")]
    require(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]), "overlapping roles")
    split = p["split"]
    fractions = split["fractions"]
    require(set(fractions) == set(ROLES) and all(0 < x < 1 for x in fractions.values())
            and math.isclose(sum(fractions.values()), 1, abs_tol=1e-12), "invalid split fractions")
    require(split["min_warm_degree"] - split["warm_holdouts"] >= split["min_warm_fit_degree"] >= 1,
            "insufficient warm-fit degree")
    require(split["warm_holdouts"] == 2 and split["resample"] is False
            and split["filter_users_by_cold_labels"] is False
            and split["drop_items_without_warm_fit"] is True, "invalid split/filter policy")
    require(p["seeds"]["split"] == 20260930
            and split["key_format"] == "sports_unseen_transfer_v1|20260930|item|{id}"
            and split["warm_pair_key"] == "sports_unseen_transfer_v1|20260930|warm_pair|{user}|{item}",
            "partition key and seed disagree")
    f, t, s, e = (p[k] for k in ("features", "teachers", "students", "evaluation"))
    require(f["fit_items"] == "warm_active_only" and f["input_dim"] ==
            f["image_pca_components"] + f["text_pca_components"], "unsafe/mismatched feature transform")
    require(t["names"] == ["T_CF", "T_MM"] and t["fit_scope"] == "warm_fit_only"
            and t["no_cold_teacher_oracle"] and t["shared_initial_id_tables"] and t["shared_triplets"],
            "teacher isolation/pairing changed")
    require(s["arms"] == ["R", "V", "D", "K_CF", "K_MM", "N"]
            and s["mlp"]["input"] == f["input_dim"] and s["mlp"]["output"] == t["dim"], "invalid controls")
    require(s["no_cold_labels_in_gradients"] and s["kd"]["teacher_targets_warm_only"], "unsafe targets")
    require(p["anchor"]["refit_after_selection"] is False and p["anchor"]["warm_scores_fixed"], "anchor refit")
    fits = len(s["learning_rates"]) * (2 + 2 * len(s["kd"]["weights"]))
    require(fits == s["max_neural_fits"] == 12, "unbounded/mismatched fit budget")
    require(s["max_epochs"] % s["evaluate_every_epochs"] == 0, "selection count is ambiguous")
    c = p["calibration"]
    require(c["selection_total_neural_checkpoints_max"] == fits * s["max_epochs"] // s["evaluate_every_epochs"],
            "checkpoint selection budget disagrees")
    require(c["cold_score_scales"] == [0.5, 1, 2] and c["raw_scale"] == 1
            and c["selection_calibration_evaluations_max"] == len(s["arms"]) * len(c["cold_score_scales"]),
            "calibration budget changed")
    require(c["conditional_calibrated_copy_of_raw_choice"] and c["selection_only"]
            and c["no_refit"] and c["warm_scores_unchanged"] and c["apply_to_all_arms"], "unsafe calibration")
    require(e["selection_catalog"] == ["warm_active", "cold_select"]
            and e["probe_catalog"] == ["warm_active", "cold_probe"]
            and e["mask"] == "warm_fit_positives_only" and e["report_after_all_choices_frozen"]
            and e["probe_passes_per_arm_per_variant"] == 1, "report leakage or repeated probe")
    r = p["resources"]
    require(r["serial"] and r["gpu_count"] == 1 and all(r[k] is False for k in
            ("automatic_retry", "automatic_next_seed", "automatic_next_stage")), "automatic expansion is forbidden")
    require(sum(r["stage_timeout_hours"].values()) <= r["total_wall_time_limit_hours"] == 48
            and r["cuda_allocated_limit_gib"] <= 6, "resource budget disagrees")
    require(p["artifacts"]["overwrite"] is False, "artifact overwrite is forbidden")
    local_path(p["artifacts"]["root"])
    rows = ["T_CF", "T_MM"] + [a + "/" + v for a in s["arms"] for v in e["variants"]]
    require(p["routing"]["matrix_rows"] == rows and len(rows) == 14, "outcome routing misses arms")


def partition_items(ids, p):
    ids = sorted(set(ids))
    require(all(isinstance(i, int) and i >= 0 for i in ids), "invalid item ID")
    key = p["split"]["key_format"]
    ids.sort(key=lambda i: (hashlib.sha256(key.format(id=i).encode()).hexdigest(), i))
    result, start = {}, 0
    for role in ROLES:
        end = (start + math.floor(len(ids) * p["split"]["fractions"][role])
               if role != "cold_lock" else len(ids))
        result[role] = ids[start:end]
        start = end
    return result


def warm_roles(pairs, warm_items, p):
    """Synthetic role reference, not a real source loader or split builder."""
    by_user = {}
    for u, i in set(pairs):
        if i in warm_items:
            by_user.setdefault(u, []).append(i)
    fit, select, probe = set(), set(), set()
    key = p["split"]["warm_pair_key"]
    for u, items in by_user.items():
        if len(items) < p["split"]["min_warm_degree"]:
            continue
        items.sort(key=lambda i: (hashlib.sha256(key.format(user=u, item=i).encode()).hexdigest(), i))
        select.add((u, items[0]))
        probe.add((u, items[1]))
        fit.update((u, i) for i in items[2:])
    active = {i for _, i in fit}
    return fit, {(u, i) for u, i in select if i in active}, {(u, i) for u, i in probe if i in active}


def self_check(p):
    checks = 0
    for n in (0, 1, 19, 20, 103):
        part = partition_items(range(n), p)
        require(part == partition_items(list(reversed(range(n))) + list(range(n)), p), "partition order dependence")
        flat = [i for values in part.values() for i in values]
        require(len(flat) == len(set(flat)) == n and set(flat) == set(range(n)), "partition loses/leaks IDs")
        require(len(part["cold_lock"]) == n - sum(math.floor(n * p["split"]["fractions"][k])
                for k in ROLES[:-1]), "lock remainder mismatch")
        checks += 3
    pairs = [(u, i) for u in (0, 1, 2) for i in range(10)] + [(3, i) for i in range(4)] + [(0, 99)]
    fit, select, probe = warm_roles(pairs, set(range(10)), p)
    require(not (fit & select or fit & probe or select & probe), "warm labels overlap")
    require(all(u != 3 and i != 99 for u, i in fit | select | probe), "cold/ineligible labels enter fit")
    require(all(sum(v == u for v, _ in fit) >= 3 for u in (0, 1, 2)), "eligible user lost fit support")
    require((fit, select, probe) == warm_roles(pairs + [(3, 99)], set(range(10)), p), "cold labels filter users")
    checks += 4
    mutations = [
        ("source", "reuse_original_teachers", True),
        ("source", "reuse_full_catalog_transforms", True),
        ("information", "fit_roles", ["warm_fit", "cold_select"]),
        ("information", "report_roles", ["warm_select", "cold_probe"]),
        ("launch", "ready", True),
        ("calibration", "selection_only", False),
        ("students", "max_neural_fits", 120),
        ("resources", "automatic_retry", True),
    ]
    for section, field, value in mutations:
        modified = copy.deepcopy(p)
        modified[section][field] = value
        try:
            validate(modified)
        except ValueError:
            checks += 1
        else:
            raise ValueError("unsafe mutation accepted: " + section + "." + field)
    for path in ("data/sports/test_mat", "data/sports/val_mat", "data/_incoming/mmrec_sports/sports.inter",
                 "exp/innovation2/sports_unseen_transfer_v1/sealed_lock.json", "../data/sports/train_mat"):
        try:
            check_payload_permission(path)
        except ValueError:
            checks += 1
        else:
            raise ValueError("unsafe source accepted: " + path)
    return checks


def verify_metadata(p):
    manifest = json.loads(local_path(p["source"]["manifest"]).read_text(encoding="utf-8"))
    for asset in p["source"]["allowed_payloads"]:
        recorded = manifest["output"]["artifacts"][Path(asset["path"]).name]
        require(all(recorded[k] == asset[k] for k in ("bytes", "sha256")), "source metadata anchor changed")
        require(check_payload_permission(asset["path"]).stat().st_size == asset["bytes"], "source byte size changed")
    data, shapes = manifest["dataset"], p["source"]["metadata_shapes"]
    require(data["users"] == shapes["users"] and data["items"] == shapes["items"]
            and data["split_interactions"]["train"] == shapes["original_train_nnz"], "Train dimensions changed")
    require(data["image_features"]["shape"] == shapes["image"]
            and data["text_features"]["shape"] == shapes["text"], "feature dimensions changed")
    return "manifest anchor and stat sizes match; payload hashes NOT recomputed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--verify-source-metadata", action="store_true")
    args = parser.parse_args()
    p = json.loads(args.profile.read_text(encoding="utf-8"))
    validate(p)
    print(json.dumps({"status": "passed", "protocol": p["protocol_id"],
                      "synthetic_checks": self_check(p) if args.self_check else 0,
                      "source_metadata": verify_metadata(p) if args.verify_source_metadata else "not checked",
                      "runtime_validation": "not implemented", "payload_access": 0,
                      "training_or_evaluation": False, "launch_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
