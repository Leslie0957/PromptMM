"""No real training: stub subprocess tests for serial completion gates."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_promptmm_validation300 as batch


class BatchContracts(unittest.TestCase):
    def test_serial_success_and_stop_on_failure_or_incomplete_success(self):
        for mode in ('success','crash','incomplete','old120'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root=Path(d); calls=[]
                def launch(cmd,**kw):
                    seed=int(cmd[cmd.index('--seed')+1]);calls.append(seed)
                    self.assertEqual(cmd[cmd.index('--epochs')+1],'300')
                    if mode=='crash' and seed==2023:return SimpleNamespace(returncode=1)
                    run=root/'exp/promptmm_release'/batch.identity(seed);run.mkdir(parents=True)
                    (run/'best.pt').write_bytes(b'synthetic checkpoint')
                    curve=[dict(epoch=e,losses_mean={'total':-1.},validation={'recall':[0,e/1000.]}) for e in range(1,301)]
                    r=dict(status='completed',run_id=batch.identity(seed),epochs=300,optimizer_steps_completed=64200,
                        validation_evaluations=301,test_evaluations=0,teacher_test_evaluations=0,test_split_loaded=False,
                        launch_commit='head',launch_dirty=False,early_stopping=False,teacher_unchanged=True,prompt_unchanged=True,
                        alias_preserved=True,student_updated=True,best_checkpoint_roundtrip=True,completion_source_unchanged=True,
                        finite_updates_checked=True,paper_ready_eligible=False,config={'seed':seed,'learning_rate':2e-5},
                        curve=curve,best_epoch=300,best_validation=curve[-1]['validation'],input_sha256={'teacher':'fixed'},
                        best_checkpoint_sha256=batch.digest(run/'best.pt'))
                    if mode=='incomplete' and seed==2023:r['validation_evaluations']=300
                    if mode=='old120' and seed==2023:
                        r.update(epochs=120,optimizer_steps_completed=25680,validation_evaluations=121)
                    (run/'report.json').write_text(json.dumps(r))
                    return SimpleNamespace(returncode=0)
                with patch.object(batch.subprocess,'check_output',side_effect=lambda args,**kw:'head' if args[1]=='rev-parse' else ''),patch.object(batch.subprocess,'run',side_effect=launch),contextlib.redirect_stdout(io.StringIO()):
                    if mode=='success':self.assertEqual(batch.run_batch(root,'fake-python'),0)
                    else:
                        with self.assertRaises(RuntimeError):batch.run_batch(root,'fake-python')
                    self.assertEqual(calls,[2022,2023,2024] if mode=='success' else [2022,2023])
                    summary=json.loads((root/'exp/promptmm_release'/batch.BATCH_ID/'batch.json').read_text())
                    self.assertEqual(summary['status'],'completed' if mode=='success' else 'failed')
                    self.assertEqual(len(summary['completed']),3 if mode=='success' else 1)
                    with self.assertRaises(FileExistsError):batch.run_batch(root,'fake-python')


if __name__=='__main__':unittest.main()
