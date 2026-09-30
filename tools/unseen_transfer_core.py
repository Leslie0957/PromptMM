"""Independent mathematical/data-role primitives; no legacy project imports."""
from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np
import torch
from sklearn.decomposition import PCA

from validate_unseen_transfer_protocol import partition_items, require, warm_roles


class Undetermined(RuntimeError):
    """Scientifically missing quality/scale/sample information, not execution failure."""


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def pairs_array(pairs):
    return np.asarray(sorted(set(map(tuple, pairs))), dtype=np.int64).reshape(-1, 2)


class Roles:
    """Application access guard, not an OS isolation boundary.

    The partition builder alone sees originalTrain/lock. No lock data survives
    in this object; probe methods deny access until selection is sealed.
    """
    def __init__(self, pairs, item_count, profile, seal_lock):
        pairs = pairs_array(pairs)
        require(len(pairs) > 0 and np.all(pairs >= 0) and np.all(pairs[:, 1] < item_count), "invalid positive pairs")
        part = partition_items(pairs[:, 1].tolist(), profile)
        fit, ws, wp = warm_roles(list(map(tuple, pairs.tolist())), set(part["warm"]), profile)
        require(bool(fit), "no eligible warm_fit pairs")
        self.users = np.asarray(sorted({u for u, _ in fit}), dtype=np.int64)
        self.warm = np.asarray(sorted({i for _, i in fit}), dtype=np.int64)
        user_set = set(self.users.tolist())
        label_roles = {"warm_fit": fit, "warm_select": ws, "warm_probe": wp}
        for name in ("cold_select", "cold_probe", "cold_lock"):
            ids = set(part[name])
            label_roles[name] = {(int(u), int(i)) for u, i in pairs if u in user_set and i in ids}
        lock = {"ids": part["cold_lock"], "pairs": sorted(label_roles.pop("cold_lock")),
                "policy": "partition-only seal; never read by diagnostic"}
        seal_lock(lock)
        self._labels = {name: pairs_array(value) for name, value in label_roles.items()}
        self._ids = {name: np.asarray(sorted(part[name]), dtype=np.int64) for name in ("cold_select", "cold_probe")}
        self.phase = "fit"
        self.sealed = False
        self.events = [{"kind": "partition_original_train", "lock_exception": "fixed partition/seal only"}]
        self.manifest = {"users": len(self.users), "warm_active_items": len(self.warm),
                         "excluded_warm_zero_fit_items": len(part["warm"]) - len(self.warm),
                         "eligible_users_sha256": array_hash(self.users), "warm_active_sha256": array_hash(self.warm),
                         "item_count": item_count, "roles": {}, "locked_ids_count": len(lock["ids"]),
                         "locked_pairs_count": len(lock["pairs"]), "locked_ids_sha256": array_hash(np.asarray(lock["ids"], dtype=np.int64))}
        for name, labels in self._labels.items():
            self.manifest["roles"][name] = {"pairs": len(labels), "sha256": array_hash(labels),
                                          "users": len(set(labels[:, 0])), "items": len(set(labels[:, 1]))}
        self.manifest["cold_id_counts"] = {name: len(ids) for name, ids in self._ids.items()}

    def set_phase(self, phase):
        require(phase in ("fit", "selection", "report"), "invalid phase")
        require(not self.sealed or phase == "report", "cannot return to optimization after seal")
        require(phase != "report" or self.sealed, "report before selections sealed")
        self.phase = phase

    def seal_selection(self, identity):
        require(not self.sealed and bool(identity), "invalid/repeated selection seal")
        self.sealed = True
        self.events.append({"kind": "selection_sealed", "identity": identity})
        self.set_phase("report")

    def _permit(self, role):
        require(role != "cold_lock", "locked role is never accessible")
        if role == "warm_fit":
            return
        expected = "selection" if role.endswith("select") else "report"
        require(self.phase == expected and (expected != "report" or self.sealed), "role/phase mismatch: " + role)

    def labels(self, role):
        self._permit(role)
        self.events.append({"kind": "labels", "role": role, "phase": self.phase})
        return self._labels[role].copy()

    def ids(self, role):
        if role == "warm_fit":
            return self.warm.copy()
        self._permit(role)
        self.events.append({"kind": "feature_rows", "role": role, "phase": self.phase})
        return self._ids[role].copy()

    def fit_features(self, role):
        require(role == "warm_fit", "feature transformer can fit warm rows only")
        return self.ids(role)


def normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


class Content:
    def __init__(self, image, text, roles, profile, check):
        self.sources, self.roles, self.check = (image, text), roles, check
        self.models, self.cache, self.stats = [], {}, {}
        warm_ids = roles.fit_features("warm_fit")
        for name, values, dim in zip(("image", "text"), self.sources,
                                     (profile["features"]["image_pca_components"], profile["features"]["text_pca_components"])):
            check()
            require(values.ndim == 2 and len(values) == roles.manifest["item_count"], "feature shape mismatch")
            if min(len(warm_ids), values.shape[1]) < dim:
                raise Undetermined("warm fit dimension insufficient for frozen PCA")
            data = np.asarray(values[warm_ids], dtype=np.float64)
            require(np.isfinite(data).all(), "nonfinite warm input")
            model = PCA(n_components=dim, svd_solver="full", whiten=False).fit(data)
            self.models.append(model)
            self.stats[name] = {"fit_items_sha256": array_hash(warm_ids), "mean_sha256": array_hash(model.mean_),
                                "components_sha256": array_hash(model.components_)}
            check()
        self.warm = self.rows("warm_fit")

    def rows(self, role):
        ids = self.roles.ids(role)  # guard also applies when cached
        if role not in self.cache:
            parts = []
            for name, source, model in zip(("image", "text"), self.sources, self.models):
                data = np.asarray(source[ids], dtype=np.float64)
                require(np.isfinite(data).all(), "nonfinite content: " + role)
                out = model.transform(data)
                require(np.isfinite(out).all(), "nonfinite transformed content")
                self.stats[name][role + "_zero_rows"] = int(np.count_nonzero(np.linalg.norm(out, axis=1) == 0))
                parts.append(normalize(out).astype(np.float32))
            self.cache[role] = np.concatenate(parts, axis=1)
        return self.cache[role]

    def transform_state(self):
        return {key + "_" + name: getattr(model, key) for name, model in zip(("image", "text"), self.models)
                for key in ("mean_", "components_")}


class Tape:
    def __init__(self, roles, seed):
        pairs = roles.labels("warm_fit")
        umap, imap = {int(u): j for j, u in enumerate(roles.users)}, {int(i): j for j, i in enumerate(roles.warm)}
        self.history = [set() for _ in roles.users]
        for u, i in pairs:
            self.history[umap[int(u)]].add(imap[int(i)])
        valid = [(umap[int(u)], imap[int(i)]) for u, i in pairs if len(self.history[umap[int(u)]]) < len(roles.warm)]
        self.excluded_users = sum(len(h) == len(roles.warm) for h in self.history)
        require(bool(valid), "no legal negative triplets")
        self.pairs, self.n_items, self.seed = pairs_array(valid), len(roles.warm), seed

    def _negative(self, users, rng):
        result = rng.integers(self.n_items, size=len(users))
        invalid = np.asarray([j in self.history[int(u)] for u, j in zip(users, result)])
        while invalid.any():
            result[invalid] = rng.integers(self.n_items, size=int(invalid.sum()))
            invalid = np.asarray([j in self.history[int(u)] for u, j in zip(users, result)])
        return result

    def epoch(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 1]))
        pairs = self.pairs[rng.permutation(len(self.pairs))]
        return np.column_stack((pairs, self._negative(pairs[:, 0], rng)))

    def scale(self, count):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 0, 2]))
        pairs = self.pairs[rng.integers(len(self.pairs), size=count)]
        return np.column_stack((pairs, self._negative(pairs[:, 0], rng)))


def graph(roles, device):
    pairs = roles.labels("warm_fit")
    u = np.searchsorted(roles.users, pairs[:, 0])
    i = np.searchsorted(roles.warm, pairs[:, 1]) + len(roles.users)
    row, col = np.concatenate((u, i)), np.concatenate((i, u))
    degree = np.bincount(row, minlength=len(roles.users) + len(roles.warm))
    val = (degree[row] * degree[col]).astype(np.float64) ** -0.5
    return torch.sparse_coo_tensor(torch.tensor(np.stack((row, col))), torch.tensor(val, dtype=torch.float32),
                                   (len(degree), len(degree)), device=device, check_invariants=True).coalesce()


class Teacher(torch.nn.Module):
    def __init__(self, users, items, x, adj, p, multimodal):
        super().__init__()
        dim, seed = p["teachers"]["dim"], p["seeds"]["teacher"]
        self.p, self.q = torch.nn.Parameter(torch.empty(users, dim)), torch.nn.Parameter(torch.empty(items, dim))
        gen = torch.Generator().manual_seed(seed)
        torch.nn.init.xavier_normal_(self.p, generator=gen)
        torch.nn.init.xavier_normal_(self.q, generator=gen)
        self.projections = torch.nn.ModuleList()
        if multimodal:
            widths = [p["features"]["image_pca_components"], p["features"]["text_pca_components"]]
            gen = torch.Generator().manual_seed(seed + 1)
            for width in widths:
                layer = torch.nn.Linear(width, dim, bias=False)
                torch.nn.init.xavier_normal_(layer.weight, generator=gen)
                self.projections.append(layer)
        self.register_buffer("content", x)
        self.register_buffer("adj", adj)
        self.layers, self.users = p["teachers"]["graph_layers"], users

    def forward(self):
        q = self.q
        if self.projections:
            split = self.projections[0].in_features
            q = q + (self.projections[0](self.content[:, :split]) + self.projections[1](self.content[:, split:])) / math.sqrt(2)
        h = torch.cat((self.p, q))
        states = [h]
        for _ in range(self.layers):
            h = torch.sparse.mm(self.adj, h)
            states.append(h)
        h = torch.stack(states).mean(0)
        return h[:self.users], h[self.users:]


def mlp(p):
    m = p["students"]["mlp"]
    net = torch.nn.Sequential(torch.nn.Linear(m["input"], m["hidden"]), torch.nn.ReLU(),
                              torch.nn.Linear(m["hidden"], m["output"]))
    gen = torch.Generator().manual_seed(p["seeds"]["student"])
    for layer in net:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight, generator=gen)
            torch.nn.init.zeros_(layer.bias)
    return net


def margin(p, q, triplets):
    u, i, j = triplets.unbind(1)
    return (p[u] * (q[i] - q[j])).sum(1)


def kd_loss(student_margin, teacher_margin, sigma_cf, sigma_t, beta):
    return torch.nn.functional.softplus(-student_margin).mean() + beta * ((student_margin / sigma_cf - teacher_margin / sigma_t) ** 2).mean()


def ridge(x, q, strength):
    x, q = np.asarray(x, dtype=np.float64), np.asarray(q, dtype=np.float64)
    mx, mq = x.mean(0), q.mean(0)
    xc, qc = x - mx, q - mq
    # loss mean over N*D coordinates, intercept unpenalized
    w = np.linalg.solve(xc.T @ xc + len(x) * q.shape[1] * strength * np.eye(x.shape[1]), xc.T @ qc)
    return w, mq - mx @ w


def content_reference(x, history):
    query = np.stack([x[sorted(h)].mean(0) for h in history])
    return normalize(query).astype(np.float32)


def top_ids(scores, ids, k):
    scores = np.asarray(scores, dtype=np.float32)
    require(not np.isnan(scores).any() and not np.isposinf(scores).any(), "invalid ranked score")
    valid = np.flatnonzero(np.isfinite(scores))
    if len(valid) > k:
        threshold = np.partition(scores[valid], -k)[-k]
        valid = valid[scores[valid] >= threshold]
    order = np.lexsort((ids[valid], -scores[valid]))[:k]
    return ids[valid[order]]


def metric_summary(values):
    result = {}
    for group in ("overall", "warm", "cold"):
        mask = values[group + "_denominator"] > 0
        result[group] = {"recall20": float(values[group + "_recall"][mask].mean()) if mask.any() else None,
                         "ndcg20": float(values[group + "_ndcg"][mask].mean()) if mask.any() else None,
                         "users": int(mask.sum()), "positives": int(values[group + "_denominator"].sum()),
                         "hits": int(values[group + "_hits"].sum())}
    return result


def evaluate(score_fn, roles, phase, profile, check, *, cold_only=True):
    """One exact shared ranking; group denominators never define separate rankings."""
    roles.set_phase(phase)
    suffix = "select" if phase == "selection" else "probe"
    warm_labels = roles.labels("warm_" + suffix)
    cold_labels = roles.labels("cold_" + suffix)
    cold_ids = roles.ids("cold_" + suffix)
    return evaluate_arrays(score_fn, roles.users, roles.warm, cold_ids, roles.labels("warm_fit"),
                           warm_labels, cold_labels, profile["evaluation"]["k"],
                           profile["evaluation"]["user_batch"], check, cold_only=cold_only)


def evaluate_arrays(score_fn, users, warm, cold, fit, warm_labels, cold_labels, k, batch, check, cold_only=True):
    ids = np.sort(np.concatenate((warm, cold)))
    is_cold = np.isin(ids, cold)
    user_index = {int(u): j for j, u in enumerate(users)}
    labels = {"warm": [set() for _ in users], "cold": [set() for _ in users], "fit": [set() for _ in users]}
    for name, pairs in (("warm", warm_labels), ("cold", cold_labels), ("fit", fit)):
        for u, i in pairs:
            labels[name][user_index[int(u)]].add(int(i))
    values = {"users": users.copy(), "top_ids": np.full((len(users), k), -1, dtype=np.int64)}
    for group in ("overall", "warm", "cold"):
        for field in ("recall", "ndcg", "hits", "denominator"):
            values[group + "_" + field] = np.zeros(len(users), dtype=np.float64)
    values["cold_only_recall"] = np.zeros(len(users), dtype=np.float64)
    exposure, score_sum, score_square, score_count, score_min, score_max = Counter(), 0., 0., 0, math.inf, -math.inf
    for start in range(0, len(users), batch):
        check()
        end = min(start + batch, len(users))
        scores = np.asarray(score_fn(np.arange(start, end), ids), dtype=np.float32)
        require(scores.shape == (end - start, len(ids)) and np.isfinite(scores).all(), "invalid scoring output")
        score_sum += float(scores.astype(np.float64).sum())
        score_square += float(np.square(scores.astype(np.float64)).sum())
        score_count += scores.size
        score_min, score_max = min(score_min, float(scores.min())), max(score_max, float(scores.max()))
        for local, uidx in enumerate(range(start, end)):
            row = scores[local].copy()
            row[np.isin(ids, list(labels["fit"][uidx]))] = -np.inf
            rank = top_ids(row, ids, k)
            require(not set(rank) & labels["fit"][uidx], "fit item appeared in recommendation")
            values["top_ids"][uidx, :len(rank)] = rank
            exposure.update(map(int, rank))
            if cold_only:
                cr = top_ids(row[is_cold], ids[is_cold], k)
                target = labels["cold"][uidx]
                values["cold_only_recall"][uidx] = len(set(cr) & target) / len(target) if target else 0
            for group in ("overall", "warm", "cold"):
                target = labels["warm"][uidx] | labels["cold"][uidx] if group == "overall" else labels[group][uidx]
                hits = np.asarray([int(i) in target for i in rank], dtype=np.float64)
                count = len(target)
                dcg = float((hits / np.log2(np.arange(len(rank)) + 2)).sum())
                ideal = float((1 / np.log2(np.arange(min(count, k)) + 2)).sum())
                values[group + "_denominator"][uidx] = count
                values[group + "_hits"][uidx] = hits.sum()
                values[group + "_recall"][uidx] = hits.sum() / count if count else 0
                values[group + "_ndcg"][uidx] = dcg / ideal if ideal else 0
    summary = metric_summary(values)
    summary["cold_only_recall"] = float(values["cold_only_recall"][values["cold_denominator"] > 0].mean()) if np.any(values["cold_denominator"] > 0) else None
    summary["score_distribution"] = {"mean": score_sum / score_count, "std": math.sqrt(max(0, score_square / score_count - (score_sum / score_count) ** 2)),
                                      "min": score_min, "max": score_max, "count": score_count}
    summary["exposure"] = {"item_ids": sorted(exposure), "counts": [exposure[i] for i in sorted(exposure)],
                            "top_item_share": max(exposure.values(), default=0) / max(1, sum(exposure.values()))}
    summary["candidate_sha256"] = array_hash(ids)
    return summary, values


def paired_intervals(results, profile, check):
    intervals = {}
    for variant in ("raw", "calibrated"):
        intervals[variant] = {}
        ref = results["K_MM/" + variant]
        for arm in profile["students"]["arms"]:
            other = results[arm + "/" + variant]
            require(np.array_equal(ref["users"], other["users"]) and all(np.array_equal(ref[g + "_denominator"], other[g + "_denominator"])
                    for g in ("warm", "cold", "overall")), "bootstrap populations/labels differ between arms")
        for group in ("cold", "warm", "overall"):
            mask = ref[group + "_denominator"] > 0
            n = int(mask.sum())
            rng = np.random.default_rng(profile["seeds"]["bootstrap"])
            controls = [a for a in profile["students"]["arms"] if a != "K_MM"]
            differences = np.stack([ref[group + "_recall"][mask] - results[a + "/" + variant][group + "_recall"][mask] for a in controls])
            samples = np.empty((len(controls), profile["evaluation"]["bootstrap_replicates"]))
            for rep in range(samples.shape[1]):
                if rep % 50 == 0:
                    check()
                if n:
                    indices = rng.integers(n, size=n)
                    samples[:, rep] = differences[:, indices].mean(1)
            intervals[variant][group] = {a: {"difference": float(differences[j].mean()) if n else None,
                                                  "lower": float(np.quantile(samples[j], .025)) if n else None,
                                                  "upper": float(np.quantile(samples[j], .975)) if n else None,
                                                  "users": n} for j, a in enumerate(controls)}
    return intervals


def plateau(curve, rule, key):
    tail = [row[key] for row in curve[-rule["last_evaluations"]:]]
    return len(tail) == rule["last_evaluations"] and all(x is not None for x in tail) and max(tail) - min(tail) <= rule["allowed_range"]


def decide(summary, intervals, roles, teachers, students, profile):
    d = profile["decision"]
    reasons = []
    base = summary["K_MM/raw"]
    if (roles.manifest["roles"]["cold_probe"]["items"] < d["min_cold_probe_items"]
            or base["cold"]["users"] < d["min_cold_probe_users"]
            or base["cold"]["positives"] < d["min_cold_probe_interactions"]
            or base["warm"]["users"] < d["min_warm_probe_users"]):
        reasons.append("insufficient frozen probe sample")
    tw = {name: summary[name]["warm"]["recall20"] for name in ("T_CF", "T_MM")}
    if any(v is None for v in tw.values()) or tw["T_MM"] - tw["T_CF"] < d["teacher_mm_recall_advantage"]:
        reasons.append("MM warm teacher advantage not established")
    if any(not plateau(row["curve"], d["teacher_plateau"], "warm_recall") for row in teachers.values()):
        reasons.append("teacher platform check missing")
    if any(not plateau(students[name]["curve"], d["student_plateau"], "overall_recall") for name in ("V", "D", "K_CF", "K_MM")):
        reasons.append("student platform check missing")
    if reasons:
        return {"label": "undetermined", "reasons": reasons, "innovation_certified": False}
    support = {}
    controls = [a for a in profile["students"]["arms"] if a != "K_MM"]
    for variant in ("raw", "calibrated"):
        cold, warm, overall = (intervals[variant][group] for group in ("cold", "warm", "overall"))
        support[variant] = all(cold[a]["difference"] >= d["cold_recall_advantage"] and cold[a]["lower"] > 0
                               and warm[a]["difference"] >= -d["warm_recall_tolerance"]
                               and overall[a]["difference"] >= -d["mixed_recall_tolerance"] for a in controls)
    if support["raw"] or support["calibrated"]:
        return {"label": "support_transfer" if support["raw"] else "calibration_signal",
                "passes_transfer_gate": True, "support_by_variant": support, "innovation_certified": False}
    stopped = []
    for variant in ("raw", "calibrated"):
        simple = max(("R", "V", "N"), key=lambda a: summary[a + "/" + variant]["cold"]["recall20"])
        small = all(intervals[variant]["cold"][a]["upper"] < d["cold_recall_advantage"] for a in ("D", "K_CF", simple))
        costly = any(intervals[variant][g][a]["upper"] < -d[t] for g, t in
                     (("warm", "warm_recall_tolerance"), ("overall", "mixed_recall_tolerance")) for a in controls)
        stopped.append(small or costly)
    return {"label": "screen_stop" if all(stopped) else "undetermined", "passes_transfer_gate": False,
            "reasons": ["bounded controls show inadequate gain/quality cost" if all(stopped) else "intervals or control tradeoffs remain ambiguous"],
            "innovation_certified": False}
