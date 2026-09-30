"""Independent small-fixture checks; never load Sports or its saved models."""
import copy
import json
import math
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import quality_frontier_core as c
import run_quality_constrained_frontier as r
from validate_unseen_transfer_protocol import partition_items, warm_roles


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = r.read_profile()

    def test_restoration_matches_reference_and_denies_other_roles(self):
        rng = np.random.default_rng(71)
        pairs = [(u, int(i)) for u in range(12) for i in rng.choice(160, 60, replace=False)]
        original = {"split": self.p["restoration"]["split"]}
        part = partition_items([i for _, i in pairs], original)
        pairs += [(999, i) for i in part["warm"][:3] + part["cold_select"]]
        roles = c.SelectRoles(pairs + pairs[:20], 160, original["split"])
        fit, select, reserved = warm_roles(pairs, set(part["warm"]), original)
        np.testing.assert_array_equal(roles.labels("warm_fit"), c.pair_array(fit))
        np.testing.assert_array_equal(roles.labels("warm_select"), c.pair_array(select))
        expected_cold = [(u, i) for u, i in pairs if u in roles.users and i in part["cold_select"]]
        np.testing.assert_array_equal(roles.labels("cold_select"), c.pair_array(expected_cold))
        self.assertNotIn(999, roles.users)
        self.assertEqual(set(roles._labels), set(c.SelectRoles.allowed))
        self.assertFalse(set(map(tuple, roles.labels("warm_fit"))) & reserved)
        self.assertFalse(set(map(tuple, roles.labels("warm_select"))) & reserved)
        reordered = c.SelectRoles(list(reversed(pairs)), 160, original["split"])
        self.assertEqual(roles.manifest, reordered.manifest)
        for forbidden in ("warm_probe", "cold_probe", "cold_lock", "test", "warm_select"):
            with self.assertRaises(ValueError):
                roles.ids(forbidden)
        for forbidden in ("warm_probe", "cold_probe", "cold_lock", "test"):
            with self.assertRaises(ValueError):
                roles.labels(forbidden)
        expected = copy.deepcopy(self.p["restoration"])
        expected.update(expected_users=len(roles.users), eligible_users_sha256=roles.manifest["eligible_users_sha256"],
                        warm_active_items=len(roles.warm), warm_active_sha256=roles.manifest["warm_active_sha256"],
                        cold_select_candidates=len(roles.cold), expected_role_metadata=roles.manifest["roles"])
        roles.verify_identity(expected)
        expected["eligible_users_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            roles.verify_identity(expected)

    def test_exact_group_merge_all_grid_points_with_masks_ties_and_boundary(self):
        rng = np.random.default_rng(12)
        users, k = 7, 20
        ids = rng.permutation(np.arange(63, dtype=np.int64) * 3 + 1)
        wid, cid = ids[:36], ids[36:]
        ws = np.round(rng.normal(size=(users, len(wid))), 1).astype(np.float32)
        cs = np.round(rng.normal(size=(users, len(cid))), 1).astype(np.float32)
        ws[:, :3] = 100  # high fit scores count in sigma but never in ranking
        cs[:, :23] = 0.125  # tied values straddle the cold top20 boundary
        masks = [set(map(int, wid[:3])) for _ in range(users)]
        q = np.eye(users, dtype=np.float32)
        warm = c.make_cache(q, ws.T.copy(), wid, k, 3, lambda: None, mask=masks)
        cold = c.make_cache(q, cs.T.copy(), cid, k, 2, lambda: None, candidates=c.grid(self.p), sigma=warm["sigma"])
        self.assertAlmostEqual(warm["sigma"], np.std(ws.astype(np.float64)), places=12)
        for a, b in c.grid(self.p):
            actual = c.merge_cache(warm, cold, a, b, warm["sigma"], k, lambda: None)
            for u in range(users):
                scores = np.concatenate((ws[u].astype(np.float64), a * cs[u].astype(np.float64) + b * warm["sigma"]))
                scores[np.isin(ids, list(masks[u]))] = -np.inf
                valid = np.flatnonzero(np.isfinite(scores))
                reference = ids[valid[np.lexsort((ids[valid], -scores[valid]))[:k]]]
                np.testing.assert_array_equal(actual[u], reference)
        another_batch = c.make_cache(q, ws.T.copy(), wid, k, 7, lambda: None, mask=masks)
        np.testing.assert_array_equal(warm["ids"], another_batch["ids"])

    def test_order_certificate_rejects_rounding_collapse_and_nonfinite(self):
        scores = np.array([[1., np.nextafter(np.float32(1.), np.float32(2.))]], dtype=np.float32)
        c.certify_cold(scores, c.grid(self.p), 1.)
        with self.assertRaisesRegex(ValueError, "collapse"):
            c.certify_cold(scores, [(0.25, 1)], 1e20)
        with self.assertRaises(ValueError):
            c.certify_cold(scores, [(0., 0.)], 1.)
        with self.assertRaises(ValueError):
            c.certify_cold(np.array([[np.nan]], np.float32), [(1., 0.)], 1.)
        with self.assertRaises(ValueError):
            c.top(np.array([np.inf]), np.array([3]), 1)

    def test_metrics_use_same_ranking_and_group_denominators(self):
        labels = {"fit": [{2}, {3}, set()], "warm": [{1}, {4}, set()],
                  "cold": [{9, 10}, set(), {10}], "catalog_warm": {1, 2, 3, 4}}
        ranks = np.array([[1, 9], [4, 9], [1, 9]])
        summary, values = c.metrics(ranks, labels, np.array([9, 10]), lambda: None)
        self.assertAlmostEqual(summary["overall"]["recall20"], (2 / 3 + 1 + 0) / 3)
        self.assertEqual(summary["warm"]["recall20"], 1.)
        self.assertEqual(summary["cold"]["recall20"], .25)
        self.assertEqual(summary["cold"]["users"], 2)
        self.assertAlmostEqual(values["cold_ndcg"][0], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertEqual(summary["cold_slot_share"], .5)
        self.assertEqual(summary["cold_exposure_concentration"], 1.)
        with self.assertRaises(ValueError):
            c.metrics(np.array([[2, 9], [4, 9], [1, 9]]), labels, np.array([9, 10]), lambda: None)

    def row(self, a, b, overall, warm, cold, state=None):
        return {"state": state or c.state_id("D", a, b), "a": a, "b": b,
                "metrics": {g: {"recall20": v} for g, v in zip(c.GROUPS, (overall, warm, cold))}}

    def test_quality_constraints_tie_priority_and_nondominated_duplicates(self):
        rows = [self.row(1., 0., .5, .49, .8), self.row(2., 0., .5, .5, .2),
                self.row(1., -.5, .5, .5, .2), self.row(1., .5, .5, .5, .2), self.row(1., 0., .5, .5, .2)]
        feasible = [x for x in rows if c.feasible(x, .5, .5, .001)]
        self.assertEqual(min(feasible, key=c.selection_key)["state"], c.state_id("D", 1., 0.))
        feasible = [x for x in rows[:-1] if c.feasible(x, .5, .5, .001)]
        self.assertEqual(min(feasible, key=c.selection_key)["b"], -.5)
        self.assertFalse(c.feasible(rows[1], None, .5, .001))
        self.assertFalse(c.feasible(rows[1], .6, .5, .001))
        frontier = c.pareto(rows)
        self.assertEqual(len(frontier), 2)
        self.assertEqual(max(len(x["equivalent_states"]) for x in frontier), 4)
        for e in (0., .0005, .001, .002, .02):
            fs = [x for x in rows if c.feasible(x, .5, .5, e)]
            if fs:
                self.assertGreaterEqual(min(fs, key=c.selection_key)["metrics"]["cold"]["recall20"], .2)

    def test_frozen_pca_arithmetic_without_fit(self):
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(8)
        sources = [rng.normal(size=(11, 9)), rng.normal(size=(11, 5))]
        state, expected = {}, []
        ids = np.array([0, 2, 7])
        for name, values in zip(("image", "text"), sources):
            model = PCA(n_components=3, whiten=False)
            model.mean_ = rng.normal(size=values.shape[1])
            model.components_ = rng.normal(size=(3, values.shape[1]))
            state["mean__" + name], state["components__" + name] = model.mean_, model.components_
            expected.append(c.normalize(model._transform(values[ids], np)).astype(np.float32))
        actual = c.transformed_content(sources, state, ids)
        np.testing.assert_array_equal(actual, np.concatenate(expected, axis=1))
        self.assertTrue(np.isfinite(c.normalize(np.zeros((3, 5), np.float32))).all())

    def test_bootstrap_pairs_share_group_resamples(self):
        p = copy.deepcopy(self.p)
        p["evaluation"]["bootstrap_replicates"] = 40
        result = {"selection": {a: {str(e): {"state": a} for e in p["quality"]["budgets"]} for a in c.ARMS}}
        saved = {}
        for j, arm in enumerate(c.ARMS):
            saved[arm] = {}
            for group in c.GROUPS:
                saved[arm][group + "_denominator"] = np.array([1, 1, 0, 1])
                saved[arm][group + "_recall"] = np.array([.05, .1, 0., .15]) * j
        intervals = c.paired_bootstrap(result, saved, p, lambda: None)
        self.assertEqual(len(intervals), 36)
        rng = np.random.default_rng(p["evaluation"]["bootstrap_seed"])
        delta = np.array([.05, .1, .15])  # K_CF(index3)-D(index2)
        for group in c.GROUPS:
            means = np.array([delta[rng.integers(3, size=3)].mean() for _ in range(40)])
            expected = np.quantile(means, [.025, .975], method="linear")
            for item in [x for x in intervals if x["group"] == group and x["left"] == "K_CF" and x["right"] == "D"]:
                np.testing.assert_allclose(item["interval95"], expected, atol=1e-15)
        result["selection"]["K_MM"]["0.001"]["state"] = None
        self.assertEqual(sum(x["status"].startswith("skipped") for x in c.paired_bootstrap(result, saved, p, lambda: None)), 6)

    def test_payload_allowlist_paths_and_metadata_only_preflight(self):
        access = r.Access(self.p)
        for path in self.p["source"]["forbidden_payloads"] + ["../data/sports/train_mat"]:
            with self.assertRaises(ValueError):
                access.verified(path)
        for path in ("../escape", "C:/outside", "/absolute", "data\\sports\\train_mat"):
            with self.assertRaises(ValueError):
                r.local_path(path)
        with patch.object(r, "digest", side_effect=AssertionError("no payload hash in preflight")), \
             patch.object(np, "load", side_effect=AssertionError("no payload loading")), \
             patch.object(Path, "read_bytes", side_effect=AssertionError("no payload bytes")):
            record = r.metadata_preflight(self.p)
        self.assertEqual(record["real_payload_reads"], 0)
        self.assertFalse(record["formal_authorization"])
        self.assertEqual(len(record["assets"]), 14)

    def test_formal_declaration_rejects_authority_and_protocol_changes(self):
        p = self.p
        declaration = {"human_authorized": True, "authorization_evidence": "fixture authority, never executable",
                       "protocol_id": p["protocol_id"], "profile_sha256": r.PROFILE_SHA,
                       "scope": "one_fixed_model_frontier_development", "anchor": p["anchor"],
                       "output_root": p["artifacts"]["root"], "command": p["launch"]["prospective_command"],
                       "information": p["information"], "resources": p["resources"],
                       "source_rule": "clean_committed_HEAD_containing_this_declaration", "branch": "fixture",
                       "one_launch_no_retry_no_resume_no_next_stage": True,
                       "originalTrain_partition_exception_acknowledged": True, "record_update_targets": p["routing"]}
        r.validate_declaration(declaration, p, "fixture")
        for key, value in (("human_authorized", False), ("authorization_evidence", ""), ("profile_sha256", "0" * 64),
                           ("scope", "expanded"), ("output_root", "elsewhere"), ("command", "changed"),
                           ("information", {}), ("resources", {}), ("branch", "other"),
                           ("one_launch_no_retry_no_resume_no_next_stage", False),
                           ("originalTrain_partition_exception_acknowledged", False), ("record_update_targets", {})):
            changed = copy.deepcopy(declaration)
            changed[key] = value
            with self.assertRaises(ValueError, msg=key):
                r.validate_declaration(changed, p, "fixture")
        with patch.object(r, "git", return_value="dirty"), patch.object(Path, "read_bytes", side_effect=AssertionError("dirty gate before reading declaration")):
            with self.assertRaisesRegex(ValueError, "dirty"):
                r.declaration_gate(p, r.local_path(p["launch"]["declaration_path"]))

    def test_scientific_decision_precedence_including_no_feasible(self):
        e = str(self.p["quality"]["primary_budget"])
        roles = SimpleNamespace(manifest={"roles": {"warm_select": {"users": 1500},
                   "cold_select": {"users": 1500, "pairs": 3000, "items": 300}}})
        result = {"selection": {a: {e: {"state": a, "metrics": self.row(1., 0., .5, .5, .05)["metrics"]}} for a in c.ARMS},
                  "reference_warm": .5, "reference_overall": .5,
                  "auxiliary": {a: {"sigma_below_floor": False} for a in c.ARMS}}
        self.assertEqual(c.decision(result, roles, self.p)["status"], "simple_control_sufficient_in_development")
        result["selection"]["K_MM"][e]["metrics"]["cold"]["recall20"] = .06
        self.assertEqual(c.decision(result, roles, self.p)["status"], "KD_difference_remains_in_development")
        result["selection"]["D"][e]["state"] = None
        self.assertEqual(c.decision(result, roles, self.p)["status"], "development_incomplete_control_comparison")
        for arm in c.ARMS[:-1]:
            result["selection"][arm][e]["metrics"]["cold"]["recall20"] = 0.
        self.assertEqual(c.decision(result, roles, self.p)["status"], "development_no_useful_tradeoff_in_grid")
        for arm in c.ARMS[:-1]:
            result["selection"][arm][e]["state"] = None
        self.assertEqual(c.decision(result, roles, self.p)["status"], "development_no_feasible_candidate")
        result["reference_warm"] = None
        self.assertEqual(c.decision(result, roles, self.p)["status"], "development_undetermined")
        result["reference_warm"] = .5
        roles.manifest["roles"]["cold_select"]["items"] = 3
        self.assertEqual(c.decision(result, roles, self.p)["status"], "development_undetermined")
        self.assertFalse(c.decision(result, roles, self.p)["innovation_certified"])

    def test_zero_sigma_grid_is_explicitly_skipped_without_fabricated_winner(self):
        roles = c.SelectRoles([(u, i) for u in range(3) for i in range(40)], 40, self.p["restoration"]["split"])
        labels = c.label_sets(roles)
        q = np.ones((3, 1), np.float32)
        warm = c.make_cache(q, np.ones((len(roles.warm), 1), np.float32), roles.warm, 20, 3, lambda: None, mask=labels["fit"])
        cold = c.make_cache(q, np.ones((len(roles.cold), 1), np.float32), roles.cold, 20, 3, lambda: None,
                            candidates=c.grid(self.p), sigma=warm["sigma"])
        result, saved = c.evaluate_grid(self.p, roles, {a: warm for a in c.ARMS}, {a: cold for a in c.ARMS}, lambda: None, lambda _: None)
        self.assertEqual(result["ranking_states_evaluated"], 2)
        self.assertEqual(result["ranking_states_skipped"], 378)
        self.assertTrue(all(row["status"] == "skipped_sigma_below_floor" for row in result["grid"]))
        self.assertTrue(all(x["state"] is None for row in result["selection"].values() for x in row.values()))
        self.assertEqual(set(saved), {"CF_W_ONLY", "N_W_ONLY"})

    def test_overwrite_and_resource_limits_stop_before_work(self):
        with patch.object(r, "write_json", side_effect=AssertionError("must not write")), \
             patch.object(r, "Budget", side_effect=AssertionError("must not launch")):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                r.execute(self.p, r.ROOT, synthetic=True)
        budget = r.Budget.__new__(r.Budget)
        budget.start = budget.stage_start = 0.
        budget.stage_name = "restore_assets"
        budget.resources = {"total_wall_seconds": 2, "stage_limits_seconds": {"restore_assets": 1}}
        budget.error = None
        with patch.object(r.time, "monotonic", return_value=.5):
            budget.check()
        with patch.object(r.time, "monotonic", return_value=1.1):
            with self.assertRaisesRegex(ValueError, "stage time"):
                budget.check()
        with patch.object(r.time, "monotonic", return_value=2.1):
            with self.assertRaisesRegex(ValueError, "total time"):
                budget.check()
        budget.error = ValueError("sampled resource failure")
        with self.assertRaisesRegex(ValueError, "sampled resource"):
            budget.check()


if __name__ == "__main__":
    unittest.main()
