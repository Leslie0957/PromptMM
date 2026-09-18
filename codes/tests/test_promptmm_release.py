"""CPU synthetic parity against pinned, exact upstream source extracts."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'codes'))
from promptmm_release import (ReleaseStudent, ResourceConfig, release_graphs, sparse_tensor,
                              release_objective, release_candidates)
from promptmm_release_resource import (parse_args, claim_run_directory, load_training_inputs,
                                       load_teacher_class, load_dgl)

REFERENCE = json.loads((Path(__file__).parent / 'fixtures/promptmm_release_reference.json').read_text())


def upstream(config):
    namespace = dict(torch=torch, nn=nn, F=F, np=np, sp=sp,
                     args=SimpleNamespace(student_tau=1., kd_loss_rate=config.pair_rate,
                                          kd_loss_list_rate=config.list_rate,
                                          kd_loss_feat_rate=config.feature_rate,
                                          alpha_l=config.sce_power, tip_rate_feat=0))
    exec(REFERENCE['student'], namespace)
    methods = '\n\n'.join(textwrap.indent(REFERENCE[k], '    ') for k in
                           ['bpr_loss', 'bpr_loss_for_KD', 'sce_criterion', 'distillation', 'csr_norm'])
    exec('class UpstreamHarness:\n' + methods, namespace)
    body = REFERENCE['objective_statements']
    prefix = ('def objective(self, student, teacher, users, pos_items, neg_items, item_index):\n'
              '    u_embed, i_embed = student\n'
              '    self.u_final_embed, self.i_final_embed, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds = teacher\n')
    suffix = ('\n    return dict(total=student_batch_loss, bpr=batch_mf_loss, reg=batch_emb_loss, '
              'pair=kd_loss, list_image=kd_loss_list_image, list_text=kd_loss_list_text, feature=kd_loss_feat)\n')
    exec(prefix + textwrap.indent(body, '    ') + suffix, namespace)
    obj = namespace['UpstreamHarness']()
    obj.decay, obj.batch_size = config.decay, config.batch_size
    return namespace['Student_LightGCN'], obj, namespace['objective']


class ReleaseParity(unittest.TestCase):
    def test_forward_losses_gradients_and_alias_updates(self):
        self._parity(torch.float64)

    def test_float32_forward_losses_gradients_and_alias_updates(self):
        self._parity(torch.float32)

    def _parity(self, dtype):
        cfg = ResourceConfig(batch_size=2)
        rtol, atol = (1e-12, 1e-10) if dtype == torch.float64 else (2e-5, 2e-5)
        ref_type, ref_loss, objective = upstream(cfg)
        torch.manual_seed(31)
        teacher = tuple(torch.randn(n, 4, dtype=dtype) * .2 for n in [3,5,5,5,3,3])
        train = sp.csr_matrix(np.array([[1,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]], dtype=np.float64))
        adj = sparse_tensor(release_graphs(train)[2], 'cpu').to(dtype)
        ref = ref_type(3,5,4,1,[]).to(dtype)
        new = ReleaseStudent(3,5,4,1).to(dtype)
        for model in [ref,new]:
            model.init_user_item_embed(teacher[0].clone(),teacher[1].clone())
        users, pos, neg = [0,2], [0,4], [3,2]
        candidates = torch.tensor([[0,2,3],[4,1,2]])
        opts = [torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay, foreach=False) for m in [ref,new]]
        for _ in range(3):
            ro, no = ref(adj),new(adj)
            for a,b in zip(ro,no): torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
            rl = objective(ref_loss,ro,teacher,users,pos,neg,candidates)
            nl = release_objective(no,teacher,users,pos,neg,candidates,cfg)
            for key in rl: torch.testing.assert_close(rl[key],nl[key],rtol=rtol,atol=atol)
            for opt in opts: opt.zero_grad(set_to_none=True)
            rl['total'].backward(); nl['total'].backward()
            for a,b in zip(ref.parameters(),new.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=rtol,atol=atol)
            for opt in opts: opt.step()
            for a,b in zip(ref.parameters(),new.parameters()):
                torch.testing.assert_close(a,b,rtol=rtol,atol=min(atol,1e-7))
            for rs, ns in zip(opts[0].state.values(),opts[1].state.values()):
                for key in rs:torch.testing.assert_close(rs[key],ns[key],rtol=rtol,atol=atol)
            new.assert_aliases()
        self.assertNotEqual(new.user_id_embedding.weight.data_ptr(), teacher[0].data_ptr())

    def test_release_graph_order_and_normalization(self):
        _, harness, _ = upstream(ResourceConfig())
        train = sp.csr_matrix(np.array([[1.,0,1],[0,1,0]],dtype=np.float32))
        ui,iu,adj = release_graphs(train)
        ru=harness.csr_norm(train,True);ri=harness.csr_norm(train.T,True)
        expected=sp.vstack([sp.hstack([ru,sp.csr_matrix((2,2))]),
                            sp.hstack([sp.csr_matrix((3,3)),ri])])
        np.testing.assert_array_equal(adj.toarray(),expected.toarray())
        conventional=sp.bmat([[sp.csr_matrix((2,2)),ru],[ri,sp.csr_matrix((3,3))]])
        self.assertFalse(np.array_equal(adj.toarray(),conventional.toarray()))

    def test_no_grad_teacher_prompt_contract(self):
        teacher=nn.Linear(4,4);teacher.requires_grad_(False)
        prompt=nn.Linear(4,4)
        with torch.no_grad(): targets=teacher(prompt(torch.ones(5,4)))
        model=ReleaseStudent(3,5,4,1)
        model.init_user_item_embed(targets[:3].clone(),targets.clone())
        before=[p.clone() for p in prompt.parameters()]
        opt=torch.optim.AdamW([{'params':model.parameters()},{'params':prompt.parameters()}],foreach=False)
        out=model(torch.eye(8).to_sparse())
        cfg=ResourceConfig(batch_size=2)
        losses=release_objective(out,(targets[:3],targets,targets,targets,targets[:3],targets[:3]),
                                 [0,1],[0,1],[2,3],torch.tensor([[0,2],[1,3]]),cfg)
        losses['total'].backward();opt.step()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in prompt.parameters()))
        for a,b in zip(before,prompt.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_cpu_dgl_sampling_matches_release_call(self):
        dgl=load_dgl()
        train=sp.csr_matrix(np.array([[1,0,1,0,0],[0,1,0,1,0],[1,0,0,0,1]],dtype=np.float32))
        users,pos=[0,2],[0,4]
        dgl.seed(2022)
        row,col=train[users].nonzero();g=dgl.graph((row,col))
        nr,nc=dgl.sampling.global_uniform_negative_sampling(g,4,replace=True)
        expected=torch.cat((torch.tensor(pos).unsqueeze(1),nc.reshape(2,2)),dim=1)
        # DGL CPU sampling can be nondeterministic across repeated seeded calls.
        # Compare identical sampled pairs, and separately verify the real call above works.
        with patch.object(dgl.sampling,'global_uniform_negative_sampling',return_value=(nr,nc)) as sample:
            actual=release_candidates(train,users,pos,2,'cpu',dgl)
            received=sample.call_args.args[0]
            self.assertEqual(received.num_nodes(),g.num_nodes())
            for a,b in zip(received.edges(),g.edges()):torch.testing.assert_close(a,b)
            self.assertEqual(sample.call_args.args[1],4)
            self.assertTrue(sample.call_args.kwargs['replace'])
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)


class ResourceGuards(unittest.TestCase):
    def test_cap_and_no_extra_training_flags(self):
        self.assertEqual(parse_args(['--promptmm_release_resource_check']).steps,3)
        for extra in [['--steps','4'],['--steps','0'],['--dataset','baby'],['--run_final_test','true'],['--epoch','300']]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--promptmm_release_resource_check']+extra)

    def test_one_launch_guard(self):
        with tempfile.TemporaryDirectory() as d:
            claim_run_directory(d)
            with self.assertRaises(FileExistsError):claim_run_directory(d)

    def test_training_loader_does_not_open_heldout(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with (p/'train_mat').open('wb') as f:pickle.dump(sp.eye(3,format='csr'),f)
            np.save(p/'image_feat.npy',np.ones((3,4)));np.save(p/'text_feat.npy',np.ones((3,2)))
            # Held-out files do not exist; loader must still succeed.
            train,image,text=load_training_inputs(p)
            self.assertEqual(train.shape,(3,3));self.assertEqual(image.shape,(3,4));self.assertEqual(text.shape,(3,2))

    def test_early_dispatch_and_describe_have_no_legacy_imports(self):
        code = """import sys,runpy
sys.path.insert(0,'codes')
sys.argv=['codes/main_mmlight.py','--promptmm_release_resource_check','--describe']
try: runpy.run_path('codes/main_mmlight.py',run_name='__main__')
except SystemExit as e: assert e.code==0
assert 'utility.batch_test' not in sys.modules
assert 'utility.load_data' not in sys.modules
assert 'utility.parser' not in sys.modules
assert 'dgl' not in sys.modules
"""
        r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout)['optimizer_steps_limit'],3)

    def test_teacher_import_does_not_parse_legacy_cli(self):
        old=sys.modules.get('utility.parser')
        cls=load_teacher_class(dict(embed_size=64,drop_rate=.2))
        self.assertEqual(cls.__name__,'Teacher_Model')
        self.assertIs(sys.modules.get('utility.parser'),old)


if __name__=='__main__':unittest.main()
