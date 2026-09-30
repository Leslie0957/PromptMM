"""Selection-only frontier mathematics. No dataset, checkpoint or project imports."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict

import numpy as np

ARMS = ("R", "V", "D", "K_CF", "K_MM", "N")
GROUPS = ("overall", "warm", "cold")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def pair_array(pairs):
    return np.asarray(sorted(set(map(tuple, pairs))), dtype=np.int64).reshape(-1, 2)


class SelectRoles:
    """Discard reserved warm rank1; never construct probe/lock label arrays.

    The partition-only caller necessarily contacts the originalTrain container.
    This class is an application guard, not OS memory isolation.
    """
    allowed = ("warm_fit", "warm_select", "cold_select")

    def __init__(self, pairs, item_count, split):
        pairs = pair_array(pairs)
        require(len(pairs) > 0 and np.all(pairs >= 0) and np.all(pairs[:, 1] < item_count), "invalid pairs")
        key = split["key_format"]
        ordered = sorted(set(map(int, pairs[:, 1])), key=lambda i: (hashlib.sha256(key.format(id=i).encode()).hexdigest(), i))
        w = math.floor(len(ordered) * split["fractions"]["warm"])
        c = math.floor(len(ordered) * split["fractions"]["cold_select"])
        warm_set, cold_set = set(ordered[:w]), set(ordered[w:w + c])
        del ordered
        history = defaultdict(list)
        for u, i in pairs:
            if int(i) in warm_set:
                history[int(u)].append(int(i))
        fit, select = [], []
        for u, items in history.items():
            if len(items) < split["min_warm_degree"]:
                continue
            ranked = sorted(items, key=lambda i: (hashlib.sha256(split["warm_pair_key"].format(user=u, item=i).encode()).hexdigest(), i))
            select.append((u, ranked[0]))
            fit.extend((u, i) for i in ranked[2:])
        require(bool(fit), "no warm_fit users")
        self.users = np.asarray(sorted({u for u, _ in fit}), dtype=np.int64)
        self.warm = np.asarray(sorted({i for _, i in fit}), dtype=np.int64)
        self.cold = np.asarray(sorted(cold_set), dtype=np.int64)
        require(len(self.cold) > 0, "no cold_select candidates")
        active = set(map(int, self.warm))
        eligible = set(map(int, self.users))
        self._labels = {
            "warm_fit": pair_array(fit),
            "warm_select": pair_array((u, i) for u, i in select if i in active),
            "cold_select": pair_array((int(u), int(i)) for u, i in pairs if int(u) in eligible and int(i) in cold_set),
        }
        self.events = []
        self.manifest = {"users": len(self.users), "eligible_users_sha256": array_hash(self.users),
                         "warm_active_items": len(self.warm), "warm_active_sha256": array_hash(self.warm),
                         "cold_select_candidates": len(self.cold), "cold_select_ids_sha256": array_hash(self.cold),
                         "roles": {}, "reserved_rank1_retained": False, "probe_lock_labels_constructed": False,
                         "originalTrain_partition_contact": True, "source_discarded_before_scoring": True}
        for name, values in self._labels.items():
            self.manifest["roles"][name] = {"pairs": len(values), "sha256": array_hash(values),
                                          "users": len(set(values[:, 0])), "items": len(set(values[:, 1]))}

    def labels(self, role):
        require(role in self.allowed, "role unavailable: " + role)
        self.events.append({"operation": "labels", "role": role})
        return self._labels[role].copy()

    def ids(self, role):
        require(role in ("warm_fit", "cold_select"), "feature role unavailable: " + role)
        self.events.append({"operation": "feature_rows", "role": role})
        return (self.warm if role == "warm_fit" else self.cold).copy()

    def verify_identity(self, expected):
        for key in ("eligible_users_sha256", "warm_active_sha256", "warm_active_items", "cold_select_candidates"):
            require(self.manifest[key] == expected[key], "restoration identity mismatch: " + key)
        require(len(self.users) == expected["expected_users"], "user count mismatch")
        require(self.manifest["roles"] == expected["expected_role_metadata"], "role identity mismatch")


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def transformed_content(sources, state, ids):
    parts = []
    for name, source in zip(("image", "text"), sources):
        values = np.asarray(source[ids], dtype=np.float64)
        require(np.isfinite(values).all(), "nonfinite source content")
        # Preserve the pinned sklearn PCA.transform projection/centering order.
        out = values @ state["components__" + name].T
        out -= state["mean__" + name].reshape(1, -1) @ state["components__" + name].T
        require(np.isfinite(out).all(), "nonfinite PCA output")
        parts.append(normalize(out).astype(np.float32))
    return np.concatenate(parts, axis=1)


def history_queries(content, roles):
    umap = {int(u): j for j, u in enumerate(roles.users)}
    imap = {int(i): j for j, i in enumerate(roles.warm)}
    history = [set() for _ in roles.users]
    for u, i in roles.labels("warm_fit"):
        history[umap[int(u)]].add(imap[int(i)])
    return normalize(np.stack([content[sorted(h)].mean(0) for h in history])).astype(np.float32)


def grid(profile):
    return [(2 ** (j / 2), b) for j in profile["grid"]["scale_exponents_half_steps"]
            for b in profile["grid"]["offset_sigma_units"]]


def state_id(arm, a, b):
    return f"{arm}__a{a:.17g}__b{b:.17g}"


def top(scores, ids, k):
    """Exact float64 ranking, unlike v1's float32 post-calibration cast."""
    scores = np.asarray(scores, dtype=np.float64)
    require(not np.isnan(scores).any() and not np.isposinf(scores).any(), "invalid ranked scores")
    valid = np.flatnonzero(np.isfinite(scores))
    order = valid[np.lexsort((ids[valid], -scores[valid]))[:k]]
    return ids[order], scores[order]


def certify_cold(scores, candidates, sigma):
    """Check all adjacent distinct base values, including the topK boundary.

    Positive affine transformations are mathematically monotone but a large
    offset can collapse distinct floats. In that case original-ID ties could
    change the retained topK; rejecting is safer than an approximate ranking.
    """
    require(np.isfinite(scores).all(), "nonfinite cold scores")
    ordered = np.sort(scores, axis=1).astype(np.float64)
    distinct = ordered[:, 1:] > ordered[:, :-1]
    for a, b in candidates:
        require(a > 0, "nonpositive scale")
        changed = a * ordered + b * sigma
        require(np.isfinite(changed).all(), "nonfinite affine score")
        require(np.all((changed[:, 1:] > changed[:, :-1])[distinct]), "affine cold order collapse")


def make_cache(queries, vectors, ids, k, batch, check, mask=None, candidates=None, sigma=None):
    """One base pass; save only group topK and label-independent moments."""
    require(queries.dtype == vectors.dtype == np.float32, "base inputs must be float32")
    require(np.isfinite(queries).all() and np.isfinite(vectors).all(), "nonfinite scoring tables")
    require(queries.shape[1] == vectors.shape[1] and len(ids) == len(vectors), "score shape mismatch")
    id_cache = np.full((len(queries), k), -1, dtype=np.int64)
    score_cache = np.full((len(queries), k), -np.inf, dtype=np.float64)
    sums, squares, count = 0., 0., 0
    for start in range(0, len(queries), batch):
        check()
        end = min(start + batch, len(queries))
        scores = np.asarray(queries[start:end] @ vectors.T, dtype=np.float32)
        require(np.isfinite(scores).all(), "nonfinite matmul")
        values = scores.astype(np.float64)
        sums += float(values.sum())
        squares += float(np.square(values).sum())
        count += values.size
        if candidates is not None:
            certify_cold(scores, candidates, sigma)
        for local, u in enumerate(range(start, end)):
            row = values[local]
            if mask is not None:
                row[np.isin(ids, list(mask[u]))] = -np.inf
            ranked, ranked_scores = top(row, ids, k)
            id_cache[u, :len(ranked)], score_cache[u, :len(ranked)] = ranked, ranked_scores
    require(count > 0, "empty score population")
    variance = max(0., squares / count - (sums / count) ** 2)
    return {"ids": id_cache, "scores": score_cache, "sigma": math.sqrt(variance),
            "population_count": count, "pre_mask_mean": sums / count,
            "base_passes": 1, "order_certificate": candidates is not None}


def merge_cache(warm, cold, a, b, sigma, k, check):
    result = np.full_like(warm["ids"], -1)
    for u in range(len(result)):
        if u % 128 == 0:
            check()
        wmask = warm["ids"][u] >= 0
        ids = warm["ids"][u, wmask]
        scores = warm["scores"][u, wmask]
        if cold is not None:
            cmask = cold["ids"][u] >= 0
            ids = np.concatenate((ids, cold["ids"][u, cmask]))
            scores = np.concatenate((scores, a * cold["scores"][u, cmask] + b * sigma))
        ranked, _ = top(scores, ids, k)
        result[u, :len(ranked)] = ranked
    return result


def label_sets(roles):
    umap = {int(u): j for j, u in enumerate(roles.users)}
    labels = {name: [set() for _ in roles.users] for name in ("fit", "warm", "cold")}
    for name, role in (("fit", "warm_fit"), ("warm", "warm_select"), ("cold", "cold_select")):
        for u, i in roles.labels(role):
            labels[name][umap[int(u)]].add(int(i))
    return labels


def metrics(ranks, labels, cold_ids, check):
    values = {"top_ids": ranks}
    k = ranks.shape[1]
    for group in GROUPS:
        for field in ("recall", "ndcg", "hits", "denominator"):
            values[group + "_" + field] = np.zeros(len(ranks), dtype=np.float64)
    exposure = Counter()
    slots, valid_slots = 0, 0
    cold_set = set(map(int, cold_ids))
    for u, rank in enumerate(ranks):
        if u % 128 == 0:
            check()
        rank = rank[rank >= 0]
        require(len(set(rank)) == len(rank) and not set(rank) & labels["fit"][u], "invalid/masked ranking")
        require(set(rank) <= cold_set | labels["catalog_warm"], "candidate outside catalog")
        cold_rank = [int(i) for i in rank if int(i) in cold_set]
        exposure.update(cold_rank)
        slots += len(cold_rank)
        valid_slots += len(rank)
        for group in GROUPS:
            target = labels["warm"][u] | labels["cold"][u] if group == "overall" else labels[group][u]
            hits = np.asarray([int(i) in target for i in rank], dtype=np.float64)
            n = len(target)
            ideal = float((1 / np.log2(np.arange(min(n, k)) + 2)).sum())
            values[group + "_denominator"][u] = n
            values[group + "_hits"][u] = hits.sum()
            values[group + "_recall"][u] = hits.sum() / n if n else 0.
            values[group + "_ndcg"][u] = float((hits / np.log2(np.arange(len(rank)) + 2)).sum()) / ideal if ideal else 0.
    summary = {}
    for group in GROUPS:
        valid = values[group + "_denominator"] > 0
        summary[group] = {"recall20": float(values[group + "_recall"][valid].mean()) if valid.any() else None,
                          "ndcg20": float(values[group + "_ndcg"][valid].mean()) if valid.any() else None,
                          "users": int(valid.sum()), "positives": int(values[group + "_denominator"].sum()),
                          "hits": int(values[group + "_hits"].sum())}
    summary["cold_slot_share"] = slots / valid_slots if valid_slots else 0.
    summary["cold_exposure_concentration"] = sum((v / slots) ** 2 for v in exposure.values()) if slots else 0.
    return summary, values


def selection_key(row):
    return (-row["metrics"]["cold"]["recall20"], -row["metrics"]["overall"]["recall20"],
            -row["metrics"]["warm"]["recall20"], abs(math.log2(row["a"])), abs(row["b"]), row["a"], row["b"])


def feasible(row, ref_warm, ref_overall, epsilon):
    return (ref_warm is not None and ref_overall is not None
            and all(row["metrics"][g]["recall20"] is not None for g in GROUPS)
            and row["metrics"]["warm"]["recall20"] >= ref_warm - epsilon
            and row["metrics"]["overall"]["recall20"] >= ref_overall - epsilon)


def pareto(rows):
    unique = {}
    for row in sorted(rows, key=lambda r: (abs(math.log2(r["a"])), abs(r["b"]), r["a"], r["b"])):
        point = tuple(row["metrics"][g]["recall20"] for g in GROUPS)
        if None in point:
            continue
        if point not in unique:
            unique[point] = {"state": row["state"], "point_overall_warm_cold": list(point), "equivalent_states": []}
        unique[point]["equivalent_states"].append(row["state"])
    return [row for point, row in unique.items() if not any(
        all(y >= x for x, y in zip(point, other)) and any(y > x for x, y in zip(point, other))
        for other in unique)]


def evaluate_grid(profile, roles, warm_caches, cold_caches, check, save_arrays):
    labels = label_sets(roles)
    labels["catalog_warm"] = set(map(int, roles.warm))
    k = profile["scoring"]["k"]
    references, saved, evaluated = {}, {}, {}
    def evaluate(name, warm, cold=None, a=1., b=0.):
        rank = merge_cache(warm, cold, a, b, warm["sigma"], k, check)
        summary, arrays = metrics(rank, labels, roles.cold, check)
        evaluated[name] = summary
        return summary, arrays
    for name, arm in (("CF_W_ONLY", "D"), ("N_W_ONLY", "N")):
        summary, arrays = evaluate(name, warm_caches[arm])
        references[name] = {"metrics": summary, "cold_excluded": True, "search_candidate": False}
        saved[name] = arrays
    raw_id = state_id("D", 1., 0.)
    if warm_caches["D"]["sigma"] >= profile["scoring"]["sigma_floor"]:
        d_raw, d_arrays = evaluate(raw_id, warm_caches["D"], cold_caches["D"])
        saved[raw_id] = d_arrays
    else:
        d_raw = {g: {"recall20": None} for g in GROUPS}
        d_arrays = None
    ref_warm, ref_overall = references["CF_W_ONLY"]["metrics"]["warm"]["recall20"], d_raw["overall"]["recall20"]
    all_rows, selections, frontiers, auxiliary = [], {}, {}, {}
    candidates = grid(profile)
    budgets = profile["quality"]["budgets"]
    for arm in ARMS:
        winners = {e: None for e in budgets}
        winner_arrays = {}
        counts = {e: 0 for e in budgets}
        rows = []
        warm, cold = warm_caches[arm], cold_caches[arm]
        sigma_missing = warm["sigma"] < profile["scoring"]["sigma_floor"]
        cold_summary, _ = metrics(cold["ids"], labels, roles.cold, check)
        auxiliary[arm] = {"cold_only_recall": cold_summary["cold"]["recall20"], "sigma": warm["sigma"],
                          "sigma_below_floor": sigma_missing, "order_certificate": cold["order_certificate"]}
        for a, b in candidates:
            name = state_id(arm, a, b)
            if sigma_missing:
                row = {"state": name, "arm": arm, "a": a, "b": b, "status": "skipped_sigma_below_floor"}
                rows.append(row)
                continue
            if name == raw_id:
                summary, arrays = d_raw, d_arrays
            else:
                summary, arrays = evaluate(name, warm, cold, a, b)
            row = {"state": name, "arm": arm, "a": a, "b": b, "status": "evaluated", "metrics": summary}
            rows.append(row)
            if a == 1. and b == 0.:
                saved[name] = arrays
            for e in budgets:
                if feasible(row, ref_warm, ref_overall, e):
                    counts[e] += 1
                    if winners[e] is None or selection_key(row) < selection_key(winners[e]):
                        winners[e] = row
                        winner_arrays[e] = arrays
        arm_selections = {}
        previous = None
        previous_count = 0
        for e in sorted(budgets):
            chosen = winners[e]
            require(counts[e] >= previous_count, "feasible count not nested")
            previous_count = counts[e]
            if chosen:
                if previous is not None:
                    require(chosen["metrics"]["cold"]["recall20"] >= previous, "selected cold quality not nested")
                previous = chosen["metrics"]["cold"]["recall20"]
                saved[chosen["state"]] = winner_arrays[e]
            arm_selections[str(e)] = {"status": "selected" if chosen else "no_feasible_grid_candidate",
                                      "state": chosen["state"] if chosen else None,
                                      "metrics": chosen["metrics"] if chosen else None,
                                      "feasible_count": counts[e], "feasible_fraction": counts[e] / len(candidates)}
        selections[arm] = arm_selections
        all_rows.extend(rows)
        frontiers[arm] = pareto([r for r in rows if r["status"] == "evaluated"])
    require(len(saved) <= 32, "too many per-user states")
    save_arrays(saved)
    return {"grid": all_rows, "selection": selections, "frontier": frontiers, "references": references,
            "reference_warm": ref_warm, "reference_overall": ref_overall, "auxiliary": auxiliary,
            "ranking_states_evaluated": len(evaluated), "ranking_states_skipped": 380 - len(evaluated)}, saved


def paired_bootstrap(result, saved, profile, check):
    summaries = []
    rng = np.random.default_rng(profile["evaluation"]["bootstrap_seed"])
    for group in GROUPS:
        check()
        comparisons = []
        population = None
        for e in profile["quality"]["budgets"]:
            for left, right in profile["evaluation"]["selected_pair_differences"]:
                ls, rs = result["selection"][left][str(e)], result["selection"][right][str(e)]
                if ls["state"] is None or rs["state"] is None:
                    summaries.append({"group": group, "epsilon": e, "left": left, "right": right,
                                      "status": "skipped_missing_feasible_candidate"})
                    continue
                lv, rv = saved[ls["state"]], saved[rs["state"]]
                mask = lv[group + "_denominator"] > 0
                require(np.array_equal(lv[group + "_denominator"], rv[group + "_denominator"]), "unpaired labels")
                if population is not None:
                    require(np.array_equal(mask, population), "different bootstrap population")
                population = mask
                delta = (lv[group + "_recall"] - rv[group + "_recall"])[mask]
                require(len(delta) > 0, "empty bootstrap population")
                comparisons.append((e, left, right, delta))
        if not comparisons:
            continue
        matrix = np.stack([c[3] for c in comparisons], axis=1)
        samples = np.empty((profile["evaluation"]["bootstrap_replicates"], len(comparisons)))
        for j in range(len(samples)):
            if j % 20 == 0:
                check()
            samples[j] = matrix[rng.integers(len(matrix), size=len(matrix))].mean(axis=0)
        intervals = np.quantile(samples, profile["evaluation"]["bootstrap_quantiles"], axis=0, method="linear")
        for j, (e, left, right, delta) in enumerate(comparisons):
            summaries.append({"group": group, "epsilon": e, "left": left, "right": right, "status": "completed",
                              "difference": float(delta.mean()), "interval95": intervals[:, j].tolist(),
                              "users": len(delta), "interpretation": "descriptive_post_selection_reused_select"})
    return summaries


def decision(result, roles, profile):
    limits = profile["evaluation"]["minimum_samples"]
    metadata = roles.manifest["roles"]
    sufficient = (metadata["warm_select"]["users"] >= limits["warm_select_users"]
                  and metadata["cold_select"]["users"] >= limits["cold_select_users"]
                  and metadata["cold_select"]["pairs"] >= limits["cold_select_interactions"]
                  and metadata["cold_select"]["items"] >= limits["positive_cold_select_items"])
    e = str(profile["quality"]["primary_budget"])
    selected = {arm: result["selection"][arm][e] for arm in ARMS}
    core = [s for arm, s in selected.items() if arm != "N" and s["state"] is not None]
    missing = (result["reference_warm"] is None or result["reference_overall"] is None
               or any(result["auxiliary"][a]["sigma_below_floor"] for a in ARMS))
    if not sufficient or missing:
        status = "development_undetermined"
    elif not core:
        status = "development_no_feasible_candidate"
    elif max(s["metrics"]["cold"]["recall20"] for s in core) < profile["decision"]["effect_delta"]:
        status = "development_no_useful_tradeoff_in_grid"
    elif any(selected[a]["state"] is None for a in ("D", "K_CF", "K_MM")):
        status = "development_incomplete_control_comparison"
    else:
        simple = max(selected[a]["metrics"]["cold"]["recall20"] for a in ("R", "V", "D") if selected[a]["state"])
        kd = max(selected[a]["metrics"]["cold"]["recall20"] for a in ("K_CF", "K_MM"))
        status = ("simple_control_sufficient_in_development" if kd - simple <= profile["decision"]["effect_delta"]
                  else "KD_difference_remains_in_development")
    mm_delta = None
    if selected["K_MM"]["state"] and selected["K_CF"]["state"]:
        mm_delta = {g: selected["K_MM"]["metrics"][g]["recall20"] - selected["K_CF"]["metrics"][g]["recall20"] for g in GROUPS}
    return {"status": status, "sample_sufficient": sufficient, "quality_missing": missing,
            "K_MM_minus_K_CF_primary": mm_delta, "innovation_certified": False, "changes_v1_decision": False,
            "automatic_next_stage": False}
