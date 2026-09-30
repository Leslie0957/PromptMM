"""Numerical and role-boundary checks use artificial arrays only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np
import torch

from validate_unseen_transfer_protocol import PROFILE, check_payload_permission
from unseen_transfer_core import (Content, Roles, Tape, Teacher, array_hash, decide, evaluate_arrays,
                                 graph, kd_loss, margin, mlp, paired_intervals, ridge, top_ids)
from run_unseen_transfer_diagnostic import artifact_hashes, declaration_gate, run_cohort, selection_key, synthetic_profile, write_json


def fixture():
    p = synthetic_profile(json.loads(PROFILE.read_text(encoding="utf-8")))
    pairs = [(u, (u * 10 + j) % 160) for u in range(16) for j in range(64)]
    lock = []
    roles = Roles(pairs, 160, p, lock.append)
    return p, roles, lock[0]


class RuntimeChecks(unittest.TestCase):
    def test_probe_lock_and_seal_guards(self):
        p, roles, lock = fixture()
        for name in ("cold_probe", "warm_probe", "cold_lock", "cold_select"):
            with self.assertRaises(ValueError):
                roles.labels(name)
        for name in ("cold_select", "cold_probe"):
            with self.assertRaises(ValueError):
                roles.fit_features(name)
        roles.set_phase("selection")
        self.assertGreater(len(roles.labels("cold_select")), 0)
        with self.assertRaises(ValueError):
            roles.set_phase("report")
        roles.seal_selection("fixed_identity")
        self.assertGreater(len(roles.labels("cold_probe")), 0)
        with self.assertRaises(ValueError):
            roles.set_phase("fit")
        self.assertNotIn("cold_lock", roles._labels)
        self.assertTrue(set(lock["ids"]).isdisjoint(roles.warm.tolist()))

    def test_warm_transform_immune_to_cold_values(self):
        p, roles, lock = fixture()
        rng = np.random.default_rng(77)
        image, text = rng.normal(size=(160, 8)), rng.normal(size=(160, 8))
        a = Content(image, text, roles, p, lambda: None)
        modified_i, modified_t = image.copy(), text.copy()
        cold = np.setdiff1d(np.arange(160), roles.warm)
        modified_i[cold], modified_t[cold] = 1e12, -1e12
        b = Content(modified_i, modified_t, roles, p, lambda: None)
        np.testing.assert_array_equal(a.warm, b.warm)
        for ma, mb in zip(a.models, b.models):
            np.testing.assert_array_equal(ma.components_, mb.components_)
        with self.assertRaises(ValueError):
            a.rows("cold_probe")

    def test_tapes_replay_legal_negatives(self):
        p, roles, _ = fixture()
        tape = Tape(roles, 2022)
        np.testing.assert_array_equal(tape.epoch(1), Tape(roles, 2022).epoch(1))
        self.assertFalse(np.array_equal(tape.epoch(1), tape.epoch(2)))
        for data in (tape.epoch(1), tape.scale(1000)):
            for u, i, j in data:
                self.assertIn(int(i), tape.history[int(u)])
                self.assertNotIn(int(j), tape.history[int(u)])

    def test_teacher_pair_initialization_and_graph(self):
        p, roles, _ = fixture()
        x = torch.zeros((len(roles.warm), 8))
        adj = graph(roles, "cpu")
        cf = Teacher(len(roles.users), len(roles.warm), x, adj, p, False)
        mm = Teacher(len(roles.users), len(roles.warm), x, adj, p, True)
        torch.testing.assert_close(cf.p, mm.p, rtol=0, atol=0)
        torch.testing.assert_close(cf.q, mm.q, rtol=0, atol=0)
        torch.testing.assert_close(cf()[0], mm()[0])
        dense = adj.to_dense()
        torch.testing.assert_close(dense, dense.T)
        self.assertTrue(torch.count_nonzero(dense.diag()) == 0)
        h0 = torch.cat((cf.p, cf.q))
        expected = (h0 + dense @ h0 + dense @ dense @ h0) / 3
        torch.testing.assert_close(torch.cat(cf()), expected)

    def test_mlp_initialization_replays(self):
        p, _, _ = fixture()
        a, b = mlp(p), mlp(p)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertTrue(torch.count_nonzero(a[0].bias) == 0)

    def test_score_kd_is_coordinate_rotation_invariant(self):
        rng = np.random.default_rng(90)
        pt, qt = torch.tensor(rng.normal(size=(3, 4)), dtype=torch.float64), torch.tensor(rng.normal(size=(6, 4)), dtype=torch.float64)
        rotation = torch.tensor(np.linalg.qr(rng.normal(size=(4, 4)))[0])
        tape = torch.tensor([[0, 0, 1], [1, 2, 3], [2, 4, 5]])
        target = margin(pt, qt, tape)
        rotated = margin(pt @ rotation, qt @ rotation, tape)
        torch.testing.assert_close(target, rotated)
        student = torch.tensor([.2, -.4, .9], dtype=torch.float64, requires_grad=True)
        loss = kd_loss(student, target, .7, 1.2, .1)
        expected = torch.nn.functional.softplus(-student).mean() + .1 * ((student / .7 - target / 1.2) ** 2).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(loss, kd_loss(student, rotated, .7, 1.2, .1))
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_ridge_stationarity_and_unpenalized_intercept(self):
        rng = np.random.default_rng(2)
        x, q = rng.normal(size=(20, 3)), rng.normal(size=(20, 2)) + 7
        w, b = ridge(x, q, .1)
        error = x @ w + b - q
        np.testing.assert_allclose(2 * x.T @ error / q.size + .2 * w, 0, atol=1e-12)
        np.testing.assert_allclose(error.mean(0), 0, atol=1e-12)
        w2, b2 = ridge(x, q + 100, .1)
        np.testing.assert_allclose(w, w2, atol=1e-12)
        np.testing.assert_allclose(b2 - b, 100, atol=1e-12)

    def test_exact_boundary_ties_match_full_lexsort(self):
        rng = np.random.default_rng(31)
        for n in (1, 20, 103):
            ids = rng.permutation(n).astype(np.int64)
            scores = rng.integers(-2, 3, size=n).astype(np.float32)
            scores[::5] = -np.inf
            for k in (1, 3, 20):
                eligible = np.flatnonzero(np.isfinite(scores))
                expected = ids[eligible[np.lexsort((ids[eligible], -scores[eligible]))[:k]]]
                np.testing.assert_array_equal(top_ids(scores, ids, k), expected)

    def test_union_denominator_group_ndcg_and_mask(self):
        users, warm, cold = np.array([0, 1]), np.array([10, 20, 30]), np.array([40, 50])
        scores = np.array([[1, 1, 1, 1, 1], [5, 100, 1, 4, 0]], dtype=np.float32)
        summary, values = evaluate_arrays(lambda u, ids: scores[u], users, warm, cold,
            np.array([[0, 10], [1, 20]]), np.array([[0, 20]]), np.array([[0, 40], [0, 50], [1, 40]]), 2, 2, lambda: None)
        np.testing.assert_array_equal(values["top_ids"], [[20, 30], [10, 40]])
        self.assertAlmostEqual(summary["overall"]["recall20"], 2/3)
        self.assertAlmostEqual(summary["warm"]["recall20"], 1)
        self.assertAlmostEqual(summary["cold"]["recall20"], .5)
        self.assertAlmostEqual(summary["cold"]["ndcg20"], .5 / np.log2(3))
        self.assertNotAlmostEqual(summary["overall"]["recall20"], .75)

    def test_selection_uses_quality_then_earliest_and_smallest(self):
        m = {"overall": {"recall20": .2}, "cold": {"recall20": .1}}
        self.assertGreater(selection_key(m, 10, .001, .1), selection_key(m, 20, .001, .1))
        self.assertGreater(selection_key(m, 10, .001, .1), selection_key(m, 10, .003, .1))

    def test_old_payload_and_uncommitted_launch_rejected(self):
        for name in ("data/sports/test_mat", "data/sports/val_mat", "data/_incoming/mmrec_sports/sports.inter"):
            with self.assertRaises(ValueError):
                check_payload_permission(name)
        with self.assertRaises(ValueError):
            declaration_gate(Path(__file__).resolve(), json.loads(PROFILE.read_text(encoding="utf-8")))

    def test_no_overwrite_guard(self):
        p = json.loads(PROFILE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_cohort(p, Path(directory), True)

    def test_artifact_index_never_reopens_sealed_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            lock_hash = write_json(out / "sealed_lock.json", {"ids": [1, 2], "pairs": [[0, 1]]})
            write_json(out / "report.json", {"status": "completed"})
            original = Path.open
            def guarded(path, *args, **kwargs):
                if path == out / "sealed_lock.json":
                    raise AssertionError("sealed lock reopened")
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, "open", guarded):
                records = artifact_hashes(out, lock_hash)
            self.assertEqual(records["sealed_lock.json"], lock_hash)
            self.assertEqual(len(records["report.json"]), 64)

    def test_failure_preserves_artifacts_without_retry(self):
        p = json.loads(PROFILE.read_text(encoding="utf-8"))
        out = PROFILE.parents[3] / "exp/innovation2" / ("synthetic_unseen_failure_gate_" + uuid.uuid4().hex[:10])
        with mock.patch("run_unseen_transfer_diagnostic.load_inputs", side_effect=RuntimeError("injected synthetic-only failure")) as loader:
            status, report = run_cohort(p, out, True)
        self.assertEqual(status, "failed")
        self.assertEqual(loader.call_count, 1)
        self.assertTrue((out / "batch.json").exists())
        self.assertTrue((out / "report.json").exists())
        self.assertTrue(report["partial_artifacts_preserved"])
        print("preserved synthetic failure fixture: " + str(out))

    def test_bootstrap_rejects_misaligned_users(self):
        p = json.loads(PROFILE.read_text(encoding="utf-8"))
        p["evaluation"]["bootstrap_replicates"] = 20
        results = {}
        for arm in p["students"]["arms"]:
            for variant in ("raw", "calibrated"):
                results[arm + "/" + variant] = {"users": np.array([0, 1])}
                for group in ("overall", "warm", "cold"):
                    results[arm + "/" + variant][group + "_denominator"] = np.ones(2)
                    results[arm + "/" + variant][group + "_recall"] = np.array([.2, .4]) + (0.1 if arm == "K_MM" else 0)
        intervals = paired_intervals(results, p, lambda: None)
        self.assertAlmostEqual(intervals["raw"]["cold"]["D"]["lower"], .1)
        results["D/raw"]["users"] = np.array([1, 0])
        with self.assertRaises(ValueError):
            paired_intervals(results, p, lambda: None)

    def test_decision_support_stop_and_missing_quality(self):
        p = json.loads(PROFILE.read_text(encoding="utf-8"))
        _, roles, _ = fixture()
        roles.manifest["roles"]["cold_probe"]["items"] = 300
        base = {"cold": {"users": 2000, "positives": 4000, "recall20": .1}, "warm": {"users": 2000, "recall20": .2}}
        summary = {a + "/" + v: copy.deepcopy(base) for a in p["students"]["arms"] for v in ("raw", "calibrated")}
        summary.update(T_CF={"warm": {"recall20": .2}}, T_MM={"warm": {"recall20": .21}})
        intervals = {v: {g: {a: {"difference": .01, "lower": .005, "upper": .015} for a in ("R", "V", "D", "K_CF", "N")}
                            for g in ("cold", "warm", "overall")} for v in ("raw", "calibrated")}
        teachers = {t: {"curve": [{"warm_recall": .2}] * 5} for t in ("T_CF", "T_MM")}
        students = {a: {"curve": [{"overall_recall": .2}] * 5} for a in ("V", "D", "K_CF", "K_MM")}
        self.assertEqual(decide(summary, intervals, roles, teachers, students, p)["label"], "support_transfer")
        for v in intervals.values():
            for row in v["cold"].values():
                row.update(difference=0, lower=-.0005, upper=.0005)
        self.assertEqual(decide(summary, intervals, roles, teachers, students, p)["label"], "screen_stop")
        summary["T_MM"]["warm"]["recall20"] = .2
        self.assertEqual(decide(summary, intervals, roles, teachers, students, p)["label"], "undetermined")


if __name__ == "__main__":
    unittest.main()
