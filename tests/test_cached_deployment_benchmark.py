"""Synthetic CPU checks only; no Sports assets, CUDA workload or split loading."""
import ast
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tools'))
import cached_deployment_benchmark as b
import run_sports_cached_deployment as runner


class DeploymentTests(unittest.TestCase):
    def test_score_exclusion_matches_exhaustive_sort_and_is_read_only(self):
        users = b.torch.tensor([[1., 0.], [0., 1.]])
        items = b.torch.tensor([[9., 1.], [8., 2.], [7., 3.], [6., 4.]])
        before = b.tensor_digest((users, items))
        request = (b.torch.tensor([0, 1]), b.torch.tensor([0, 1]), b.torch.tensor([0, 3]))
        values, ids = b.score((users, items), request, k=2)
        self.assertEqual(ids.tolist(), [[1, 2], [2, 1]])
        self.assertEqual(values.tolist(), [[8., 7.], [3., 2.]])
        self.assertEqual(before, b.tensor_digest((users, items)))
        tied = b.torch.ones((4, 2))
        values, ids = b.score((users, tied), request, k=2)
        self.assertTrue(b.torch.isfinite(values).all())
        self.assertNotIn(0, ids[0].tolist())
        self.assertNotIn(3, ids[1].tolist())

    def test_requests_use_train_only_and_cycle_identically(self):
        train = b.sp.csr_matrix(([1., 1., 1.], ([0, 1, 2], [0, 1, 2])), shape=(4, 30))
        order = b.request_order(train, largest_batch=2)
        self.assertEqual(set(order), {0, 1, 2})
        self.assertTrue(b.np.array_equal(order, b.request_order(train, largest_batch=2)))
        requests = b.make_requests(train, order, 2, count=4)
        self.assertEqual(requests[0][0].tolist(), requests[3][0].tolist())
        for ids, rows, columns in requests:
            for row, col in zip(rows, columns):
                self.assertEqual(train[int(ids[row]), int(col)], 1.)
        with self.assertRaises(ValueError):
            b.request_order(train, largest_batch=4)

    def test_storage_counts_aliases_once_and_sparse_components(self):
        x = b.torch.arange(12, dtype=b.torch.float32).reshape(3, 4)
        inventory = b.storage_inventory([('x', x), ('view', x[:, :2]), ('copy', x.clone())])
        self.assertEqual(inventory['unique_bytes'], 96)
        graph = b.sparse_tensor(b.sp.eye(3, dtype=b.np.float32), 'cpu')
        inventory = b.storage_inventory([('g', graph), ('alias', graph)])
        self.assertEqual(inventory['unique_bytes'], 3*2*8 + 3*4)

    def test_graph_adapter_matches_original_trainer_method(self):
        tree = ast.parse((ROOT / 'codes/main_mmlight.py').read_text(encoding='utf-8'))
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Trainer')
        method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'csr_norm')
        namespace = dict(np=b.np, sp=b.sp)
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<original normalization>', 'exec'), namespace)
        matrix = b.sp.csr_matrix(b.np.array([[1., 2., 0.], [0., 0., 0.]], dtype='float32'))
        for m in (matrix, matrix.T):
            original = namespace['csr_norm'](None, m, mean_flag=True)
            self.assertTrue(b.np.array_equal(original.toarray(), b.graph_normalize(m).toarray()))

    def test_schedule_complete_and_balanced(self):
        schedule = runner.schedule()
        self.assertEqual(len(schedule), 36)
        self.assertEqual(len({runner.condition_name(c) for c in schedule}), 36)
        self.assertEqual(sum(c['phase'] == 'offline' for c in schedule), 9)
        for r in range(3):
            group = [c for c in schedule if c['round'] == r]
            self.assertEqual([c['arm'] for c in group[:3]], list(('TFB', 'FBT', 'BTF')[r]))
        for arm in 'TFB':
            for size in b.BATCHES:
                self.assertEqual(sum(c['phase']=='online' and c['arm']==arm and c['batch_size']==size for c in schedule), 3)

    def test_reference_gate_and_nonfinite_timing(self):
        pair = (b.torch.zeros((2, 4)), b.torch.zeros((3, 4)))
        changed = (pair[0] + 0.01, pair[1])
        self.assertFalse(all(x['allclose'] for x in b.compare_reference(changed, pair)))
        for data in ([], [0.], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                b.summary(data)

    def test_io_guard_in_isolated_synthetic_process(self):
        with tempfile.TemporaryDirectory() as temp:
            program = '''
import os, platform, subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from cached_deployment_benchmark import install_io_guard
root=Path(sys.argv[2]); install_io_guard(root)
(root/'allowed.json').write_text('{}')
handle=os.open(os.devnull, os.O_RDWR)
os.close(handle)
subprocess.run([sys.executable, '-B', '-c', 'print("null-device regression")'],
               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL, check=True)
assert platform.platform()
for path in (root/'test_mat', root.parent/'forbidden-benchmark-write.tmp'):
    try: path.write_text('blocked')
    except RuntimeError: pass
    else: raise AssertionError('guard failed')
'''
            subprocess.run([sys.executable, '-B', '-c', program, str(ROOT/'codes'), temp], check=True)

    def test_original_teacher_adapter_strict_restore_with_synthetic_assets(self):
        # Exercise the complete adapter with tiny synthetic graph/modalities and
        # CPU substitutions for constructors' hard-coded .cuda(). No real assets.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = dict(embed_size=64, weight_size='[64,64]', mess_dropout='[.1,.1]',
                        layers=1, sparse=1, feat_soft_token_rate=1., soft_token_rate=.005,
                        model_cat_rate=.55, drop_rate=.2, prompt_dropout=0.,
                        hard_token_type='pca', hard_token_seed=2022, eval_protocol='val_test_once_v1',
                        data_path=str(root), dataset='synthetic', teacher_checkpoint='teacher.pt')
            train = b.sp.csr_matrix(b.np.array([[1]*15+[0]*15, [0]*15+[1]*15, [1,0]*15], dtype='float32'))
            train_path = root/'train_mat'
            with train_path.open('wb') as f:
                pickle.dump(train, f)
            rng = b.np.random.default_rng(3)
            records, features, caches = {}, {}, {}
            for modality, dim in [('image', 8), ('text', 6)]:
                features[modality] = rng.normal(size=(30,dim)).astype('float32')
                caches[modality] = rng.normal(size=(30,64)).astype('float32')
                npy, cache = root/(modality+'.npy'), root/(modality+'.pkl')
                b.np.save(npy, features[modality])
                with cache.open('wb') as f:
                    pickle.dump(dict(metadata=dict(cache_identity=modality), values=caches[modality]), f)
                records[modality] = dict(cache_identity=modality, cache_hit=True,
                    source_feature=dict(path=str(npy), sha256=b.sha256(npy)),
                    cache_file=dict(path=str(cache), sha256=b.sha256(cache)))
            patches = [mock.patch.object(b.torch.Tensor, 'cuda', lambda self,*a,**kw:self),
                       mock.patch.object(b.torch.nn.Module, 'cuda', lambda self,*a,**kw:self),
                       mock.patch.object(b.torch.cuda, 'synchronize', lambda:None)]
            for patch in patches:
                patch.start(); self.addCleanup(patch.stop)
            models = b.load_original_models(args)
            self.addCleanup(lambda: sys.modules.pop('Models_mmlight', None))
            self.addCleanup(lambda: sys.modules.pop('utility.parser', None))
            models.load_or_create_hard_token_cache = lambda *a: (caches[a[3]], records[a[3]])
            ui = b.sparse_tensor(b.graph_normalize(train), 'cpu')
            iu = b.sparse_tensor(b.graph_normalize(train.T), 'cpu')
            teacher = models.Teacher_Model(3,30,64,[64,64],[.1,.1],features['image'],features['text']).eval()
            prompt = models.PromptLearner(features['image'],features['text'],ui).eval()
            with b.torch.inference_mode():
                expected = tuple(t.clone() for t in teacher(ui,iu,prompt)[:2])
            config = {k: args[k] for k in ('embed_size','layers','sparse','feat_soft_token_rate',
                      'soft_token_rate','model_cat_rate','hard_token_type','hard_token_seed')}
            config['weight_size'] = [64,64]
            b.torch.save(dict(teacher_model=teacher.state_dict(),prompt_module=prompt.state_dict(),
                             teacher_inference_config=config), root/'teacher.pt')
            m = dict(resolved_arguments=args, active_teacher_hard_token_cache=records,
                     teacher_checkpoint_metadata=dict(teacher_inference_config=config))
            del sys.modules['Models_mmlight']; del sys.modules['utility.parser']
            real_root = b.ROOT
            # Keep module source anchored at the real repository while fixture
            # checkpoint path is absolute; no synthetic source override needed.
            args['teacher_checkpoint'] = str(root/'teacher.pt')
            with mock.patch.object(b,'manifest',return_value=(m,None)), \
                 mock.patch.object(b,'TEACHER_SHA',b.sha256(root/'teacher.pt')), \
                 mock.patch.object(b,'TRAIN',train_path), \
                 mock.patch.object(b,'TRAIN_SHA',b.sha256(train_path)):
                restored, restored_prompt, a, c, phases = b.load_teacher(device='cpu')
            before = b.tensor_digest(list(restored.state_dict().values()) + list(restored_prompt.state_dict().values()))
            with b.torch.inference_mode():
                actual = restored(a,c,restored_prompt)[:2]
            self.assertTrue(all(b.torch.equal(x,y) for x,y in zip(expected,actual)))
            self.assertFalse(restored.training or restored_prompt.training)
            self.assertEqual(before,b.tensor_digest(list(restored.state_dict().values()) + list(restored_prompt.state_dict().values())))
            self.assertIn('construction_modal_transfer_restore_seconds', phases)
            self.assertEqual(b.ROOT,real_root)

    def test_serial_failure_preserves_attempt_and_never_advances(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'exclusive_attempt'
            calls = []
            def child(command, **kwargs):
                calls.append(command)
                if '--preflight-worker' in command:
                    b.save_json(output/'preflight.json', {'status':'passed'})
                else:
                    raise subprocess.CalledProcessError(7, command)
            with mock.patch.object(b, 'OUT', output), \
                 mock.patch.object(b, 'PYTHON', Path(sys.executable)), \
                 mock.patch.object(b, 'install_io_guard', lambda _:None), \
                 mock.patch.object(runner, 'source_state', return_value='synthetic-head'), \
                 mock.patch.object(runner, 'telemetry', lambda *args:None), \
                 mock.patch.object(runner.subprocess, 'run', side_effect=child), \
                 mock.patch.object(runner.subprocess, 'check_output', return_value='synthetic-branch'):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run(True)
                result = __import__('json').loads((output/'batch.json').read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['completed'], [])
                self.assertEqual(len(calls), 2)
                with self.assertRaises(FileExistsError):
                    runner.run(True)
                self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
