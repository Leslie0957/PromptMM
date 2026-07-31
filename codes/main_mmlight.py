from datetime import datetime
import math
import os
import random
import sys
import types
from time import time
from tqdm import tqdm
# ===== DGL Compatibility =====
if os.name == 'nt':
    dgl_distributed_stub = types.ModuleType('dgl.distributed')
    dgl_distributed_stub.__file__ = __file__

    class _UnavailableDistGraph(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError('dgl.distributed is unavailable in this Windows GraphBolt environment.')

    def _unavailable_distributed_api(*args, **kwargs):
        raise RuntimeError('dgl.distributed is unavailable in this Windows GraphBolt environment.')

    dgl_distributed_stub.DistGraph = _UnavailableDistGraph
    dgl_distributed_stub.DistGraphServer = _UnavailableDistGraph
    dgl_distributed_stub.DistDataLoader = _UnavailableDistGraph
    dgl_distributed_stub.DistNodeDataLoader = _UnavailableDistGraph
    dgl_distributed_stub.DistEdgeDataLoader = _UnavailableDistGraph
    dgl_distributed_stub.edge_split = _unavailable_distributed_api
    dgl_distributed_stub.node_split = _unavailable_distributed_api

    def _distributed_getattr(name):
        if name.startswith('__'):
            raise AttributeError(name)
        if name.startswith('Dist'):
            return _UnavailableDistGraph
        return _unavailable_distributed_api

    dgl_distributed_stub.__getattr__ = _distributed_getattr
    sys.modules.setdefault('dgl.distributed', dgl_distributed_stub)
    dgl_graphbolt_stub = types.ModuleType('dgl.graphbolt')
    dgl_graphbolt_stub.__file__ = __file__
    sys.modules.setdefault('dgl.graphbolt', dgl_graphbolt_stub)

import dgl
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
# import  visdom

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse
from torch import autograd

# ===== CPU Fallback =====
if not torch.cuda.is_available():
    def _tensor_cuda_fallback(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to('cpu')

    def _module_cuda_fallback(self, device=None):
        return self.to('cpu')

    torch.Tensor.cuda = _tensor_cuda_fallback
    torch.nn.Module.cuda = _module_cuda_fallback

import copy

from utility.parser import args, select_dataset
from utility.dataset_profiles import (
    BABY_TEACHER_PROFILE_NAME,
    BABY_TEACHER_PROFILE_SCOPE,
    BABY_TEACHER_PROFILE_SOURCE,
    baby_student_paper_ready_blockers,
)
from utility.experiment_protocol import (
    LEGACY_PROTOCOL,
    PAPER_READY_PROTOCOL,
    candidate_exclusion_policy,
    dataset_identity_from_preflight,
    early_stopping_non_improvement_limit,
    file_fingerprint,
    namespace_to_dict,
    publish_file_atomically,
    protocol_uses_validation,
    resolve_primary_k_index,
    restore_checkpoint_then_evaluate,
    result_to_dict,
    run_dataset_preflight,
    teacher_inference_config_from_namespace,
    training_batch_count,
    validate_final_test_policy,
    validate_teacher_checkpoint_metadata,
    write_json,
)
# select_dataset()

TD_DISTILL_ARG_DEFAULTS = {
    'td_distill_alpha': 0.1,
    'td_item_image_rate': 1.0,
    'td_item_text_rate': 1.0,
    'td_user_image_rate': 1.0,
    'td_user_text_rate': 1.0,
    'td_init_from_teacher': True,
}
for _arg_name, _arg_default in TD_DISTILL_ARG_DEFAULTS.items():
    if not hasattr(args, _arg_name):
        setattr(args, _arg_name, _arg_default)
if isinstance(args.td_init_from_teacher, str):
    args.td_init_from_teacher = args.td_init_from_teacher.strip().lower() in ('true', '1', 'yes', 'y', 't')

if getattr(args, 'dataset_preflight', True):
    DATASET_PREFLIGHT_REPORT = run_dataset_preflight(
        args.data_path,
        args.dataset,
        getattr(args, 'duplicate_modalities_policy', 'warn'),
    )
else:
    DATASET_PREFLIGHT_REPORT = {
        'dataset': args.dataset,
        'status': 'disabled',
        'warnings': ['Dataset preflight was explicitly disabled.'],
    }

# from utility.parser import parse_args
from Models_mmlight import Teacher_Model, Student_LightGCN, Student_GCN, Student_MLP, PromptLearner, Student_MMLight
from utility.batch_test import *
from utility.logging import Logger
from utility.norm import build_sim, build_knn_normalized_graph
from torch.utils.tensorboard import SummaryWriter
# ===== TD-Distill =====
from td_distill_model import (
    TDDistillModel,
    bpr_loss as td_bpr_loss,
    directional_distillation_loss,
    export_infer_state_dict,
)
from td_distill_model_no_projection import (
    TDDistillNoProjectionModel,
    export_infer_state_dict as export_infer_state_dict_no_projection,
)
from efficiency_benchmark import (
    maybe_load_student_checkpoint_for_efficiency,
    run_efficiency_benchmark,
)

import setproctitle
setproctitle.setproctitle('EXP@weiw')

# args = parse_args()
# from utility.parser import args, select_dataset


class Trainer(object):
    def __init__(self, data_config):
       
        self.task_name = "%s_%s_%s_pid%d" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'), args.dataset, args.cf_model, os.getpid())
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.run_name = self.logger.filename
        self.eval_protocol = getattr(args, 'eval_protocol', PAPER_READY_PROTOCOL)
        self.uses_validation_selection = protocol_uses_validation(self.eval_protocol)
        self.selection_split = 'validation' if self.uses_validation_selection else 'legacy_test'
        self.candidate_exclusion_policy = candidate_exclusion_policy(self.eval_protocol)
        self.primary_k_index, self.ks = resolve_primary_k_index(args.Ks)
        self.primary_k = self.ks[self.primary_k_index]
        self.early_stopping_limit = early_stopping_non_improvement_limit(
            self.eval_protocol, args.early_stopping_patience
        )
        self.smoke_mode = getattr(args, 'smoke_train_batches', 0) > 0
        self.train_batch_count = training_batch_count(
            data_generator.n_train,
            args.batch_size,
            getattr(args, 'smoke_train_batches', 0),
        )
        self.run_final_test = validate_final_test_policy(
            getattr(args, 'smoke_train_batches', 0),
            getattr(args, 'run_final_test', True),
        )
        if self.eval_protocol == PAPER_READY_PROTOCOL and DATASET_PREFLIGHT_REPORT.get('status') == 'disabled':
            raise ValueError('val_test_once_v1 requires dataset preflight to remain enabled.')
        self.dataset_identity = (
            None
            if DATASET_PREFLIGHT_REPORT.get('status') == 'disabled'
            else dataset_identity_from_preflight(DATASET_PREFLIGHT_REPORT)
        )
        self.teacher_inference_config = teacher_inference_config_from_namespace(args)
        # ===== Local Artifact Paths =====
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(self.repo_root, 'Model', args.dataset)
        self.converge_dir = os.path.join(self.repo_root, 'exp', 'converge', args.dataset)
        self.efficiency_dir = os.path.join(self.repo_root, 'exp', 'efficiency', args.dataset)
        self.run_artifact_dir = os.path.join(self.repo_root, 'exp', 'runs', args.dataset)
        self.model_run_dir = os.path.join(self.model_dir, 'runs')
        self.student_dir = os.path.join(self.model_dir, 'students')
        self.td_distill_dir = os.path.join(self.model_dir, 'td_distill')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.converge_dir, exist_ok=True)
        os.makedirs(self.efficiency_dir, exist_ok=True)
        os.makedirs(self.run_artifact_dir, exist_ok=True)
        os.makedirs(self.model_run_dir, exist_ok=True)
        os.makedirs(self.student_dir, exist_ok=True)
        os.makedirs(self.td_distill_dir, exist_ok=True)
        explicit_teacher_path = getattr(args, 'teacher_checkpoint', '').strip()
        self.explicit_teacher_checkpoint = bool(explicit_teacher_path)
        if (
            self.explicit_teacher_checkpoint
            and args.if_train_teacher
            and not getattr(args, 'run_efficiency_benchmark', False)
        ):
            raise ValueError(
                '--teacher_checkpoint is a read-only reuse input. Use it together with '
                '--if_train_teacher false; training always preserves a run-specific archive.'
            )
        if getattr(args, 'teacher_only', False) and not args.if_train_teacher:
            raise ValueError('--teacher_only true requires --if_train_teacher true.')
        if explicit_teacher_path:
            self.teacher_model_path = (
                explicit_teacher_path
                if os.path.isabs(explicit_teacher_path)
                else os.path.join(self.repo_root, explicit_teacher_path)
            )
        else:
            teacher_alias_name = (
                'teacher_model_great.pt'
                if self.eval_protocol == LEGACY_PROTOCOL
                else 'teacher_model_%s.pt' % self.eval_protocol
            )
            self.teacher_model_path = os.path.join(self.model_dir, teacher_alias_name)
        legacy_teacher_alias = os.path.normcase(
            os.path.abspath(os.path.join(self.model_dir, 'teacher_model_great.pt'))
        )
        if (
            self.eval_protocol == PAPER_READY_PROTOCOL
            and os.path.normcase(os.path.abspath(self.teacher_model_path)) == legacy_teacher_alias
        ):
            raise ValueError(
                'Paper-ready runs cannot use or overwrite the historical legacy teacher alias: %s'
                % self.teacher_model_path
            )
        self.paper_ready_blockers = []
        if self.eval_protocol != PAPER_READY_PROTOCOL:
            self.paper_ready_blockers.append('legacy evaluation protocol selected')
        if DATASET_PREFLIGHT_REPORT.get('duplicate_modalities'):
            self.paper_ready_blockers.append('image and text feature arrays are identical')
        if getattr(args, 'dataset_config_profile', '') == 'unvalidated_amazon_default_fallback':
            self.paper_ready_blockers.append('dataset-specific defaults are not validated')
        if args.dataset == 'baby':
            baby_profile_metadata = (
                getattr(args, 'dataset_config_profile', None),
                getattr(args, 'dataset_config_scope', None),
                getattr(args, 'dataset_config_source', None),
            )
            expected_baby_profile_metadata = (
                BABY_TEACHER_PROFILE_NAME,
                BABY_TEACHER_PROFILE_SCOPE,
                BABY_TEACHER_PROFILE_SOURCE,
            )
            if baby_profile_metadata != expected_baby_profile_metadata:
                self.paper_ready_blockers.append(
                    'Baby reference profile metadata is missing or mismatched'
                )
            if getattr(args, 'dataset_config_overrides', {}):
                self.paper_ready_blockers.append(
                    'Baby reference profile has resolved overrides'
                )
            if getattr(args, 'teacher_only', False):
                if getattr(args, 'student_config_profile', 'none') != 'none':
                    self.paper_ready_blockers.append(
                        'Baby teacher-only run must not activate a student profile'
                    )
            else:
                self.paper_ready_blockers.extend(
                    baby_student_paper_ready_blockers(
                        getattr(args, 'student_config_profile', None),
                        getattr(args, 'student_config_scope', None),
                        getattr(args, 'student_config_source', None),
                        getattr(args, 'student_config_overrides', {}),
                    )
                )
        if getattr(args, 'smoke_train_batches', 0) > 0:
            self.paper_ready_blockers.append('training batches are capped for smoke')
        if not self.run_final_test:
            self.paper_ready_blockers.append('final test evaluation is disabled')
        if DATASET_PREFLIGHT_REPORT.get('status') == 'disabled':
            self.paper_ready_blockers.append('dataset preflight is disabled')
        self.paper_ready_eligible = not self.paper_ready_blockers
        self.teacher_alias_writable = (
            self.eval_protocol != LEGACY_PROTOCOL and self.paper_ready_eligible
        )
        self.allow_teacher_alias_overwrite = getattr(
            args, 'allow_teacher_alias_overwrite', False
        )
        if (
            args.if_train_teacher
            and not getattr(args, 'run_efficiency_benchmark', False)
            and self.teacher_alias_writable
            and os.path.exists(self.teacher_model_path)
            and not self.allow_teacher_alias_overwrite
        ):
            raise FileExistsError(
                'Refusing to overwrite existing teacher alias {}. Preserve it or pass '
                '--allow_teacher_alias_overwrite true only after explicit review.'.format(
                    self.teacher_model_path
                )
            )
        teacher_archive_label = self._safe_artifact_name(
            os.path.splitext(os.path.basename(self.teacher_model_path))[0]
        )
        self.teacher_model_archive_path = os.path.join(
            self.model_run_dir,
            '%s__%s.pt' % (teacher_archive_label, self.run_name),
        )
        self.student_checkpoint_path = os.path.join(
            self.student_dir,
            'student_full__%s__%s.pth' % (self.eval_protocol, self.run_name),
        )
        point_label = self._safe_artifact_name(args.point) if args.point else 'auto'
        self.converge_archive_path = os.path.join(self.converge_dir, '%s__%s.pkl' % (point_label, self.run_name))
        self.preflight_report_path = os.path.join(
            self.run_artifact_dir, 'dataset_preflight__%s.json' % self.run_name
        )
        self.run_manifest_path = os.path.join(
            self.run_artifact_dir, 'run_manifest__%s.json' % self.run_name
        )
        run_specific_paths = (
            self.teacher_model_archive_path,
            self.student_checkpoint_path,
            self.converge_archive_path,
            self.preflight_report_path,
            self.run_manifest_path,
            os.path.join(self.logger.path, self.run_name),
        )
        path_collisions = [path for path in run_specific_paths if os.path.exists(path)]
        if path_collisions:
            raise FileExistsError(
                'Refusing to overwrite existing run-specific artifacts: {}'.format(
                    path_collisions
                )
            )
        write_json(self.preflight_report_path, DATASET_PREFLIGHT_REPORT)
        self.run_manifest = {
            'status': 'initialized',
            'started_at': datetime.now().astimezone().isoformat(),
            'run_name': self.run_name,
            'dataset': args.dataset,
            'dataset_config_profile': getattr(args, 'dataset_config_profile', 'unknown'),
            'dataset_config_scope': getattr(args, 'dataset_config_scope', 'unknown'),
            'dataset_config_source': getattr(args, 'dataset_config_source', 'unknown'),
            'dataset_config_overrides': getattr(args, 'dataset_config_overrides', {}),
            'student_config_profile': getattr(args, 'student_config_profile', 'none'),
            'student_config_scope': getattr(args, 'student_config_scope', 'none'),
            'student_config_source': getattr(args, 'student_config_source', 'none'),
            'student_config_overrides': getattr(args, 'student_config_overrides', {}),
            'evaluation_protocol': self.eval_protocol,
            'selection_split': self.selection_split,
            'candidate_exclusion_policy': self.candidate_exclusion_policy,
            'primary_selection_metric': 'Recall@%d' % self.primary_k,
            'run_final_test': self.run_final_test,
            'paper_ready_eligible': self.paper_ready_eligible,
            'paper_ready_blockers': self.paper_ready_blockers,
            'resolved_arguments': namespace_to_dict(args),
            'data_config': data_config,
            'dataset_preflight': DATASET_PREFLIGHT_REPORT,
            'dataset_identity': self.dataset_identity,
            'teacher_inference_config': self.teacher_inference_config,
            'code_fingerprints': {
                'main_mmlight': file_fingerprint(os.path.join(self.repo_root, 'codes', 'main_mmlight.py')),
                'models_mmlight': file_fingerprint(os.path.join(self.repo_root, 'codes', 'Models_mmlight.py')),
                'parser': file_fingerprint(os.path.join(self.repo_root, 'codes', 'utility', 'parser.py')),
                'dataset_profiles': file_fingerprint(os.path.join(self.repo_root, 'codes', 'utility', 'dataset_profiles.py')),
                'batch_test': file_fingerprint(os.path.join(self.repo_root, 'codes', 'utility', 'batch_test.py')),
                'load_data': file_fingerprint(os.path.join(self.repo_root, 'codes', 'utility', 'load_data.py')),
                'experiment_protocol': file_fingerprint(os.path.join(self.repo_root, 'codes', 'utility', 'experiment_protocol.py')),
                'hard_token_cache': file_fingerprint(os.path.join(self.repo_root, 'codes', 'utility', 'hard_token_cache.py')),
                'td_distill_model': file_fingerprint(os.path.join(self.repo_root, 'codes', 'td_distill_model.py')),
                'td_distill_model_no_projection': file_fingerprint(os.path.join(self.repo_root, 'codes', 'td_distill_model_no_projection.py')),
                'efficiency_benchmark': file_fingerprint(os.path.join(self.repo_root, 'codes', 'efficiency_benchmark.py')),
            },
            'artifacts': {
                'teacher_alias': self.teacher_model_path,
                'teacher_alias_writable': self.teacher_alias_writable,
                'teacher_alias_allow_overwrite': self.allow_teacher_alias_overwrite,
                'teacher_alias_publication': 'after_successful_final_test',
                'teacher_run': self.teacher_model_archive_path,
                'student_run': self.student_checkpoint_path,
                'converge_run': self.converge_archive_path,
                'preflight_report': self.preflight_report_path,
                'run_manifest': self.run_manifest_path,
            },
        }
        write_json(self.run_manifest_path, self.run_manifest)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))
        self.logger.logging(
            'Evaluation protocol: %s; selection_split=%s; candidate_exclusion=%s; primary_metric=Recall@%d' % (
                self.eval_protocol,
                self.selection_split,
                self.candidate_exclusion_policy,
                self.primary_k,
            )
        )
        self.logger.logging(
            'Early stopping: stop after %d consecutive non-improvement epoch(s).' %
            self.early_stopping_limit
        )
        self.logger.logging("Artifacts: teacher_latest=%s, teacher_run=%s, student_run=%s, converge_run=%s, manifest=%s" % (
            self.teacher_model_path,
            self.teacher_model_archive_path,
            self.student_checkpoint_path,
            self.converge_archive_path,
            self.run_manifest_path,
        ))
        for preflight_warning in DATASET_PREFLIGHT_REPORT.get('warnings', []):
            self.logger.logging('DATASET PREFLIGHT WARNING: %s' % preflight_warning)
        self.logger.logging(
            'Paper-ready eligibility: %s%s' % (
                self.paper_ready_eligible,
                '' if self.paper_ready_eligible else '; blockers=' + '; '.join(self.paper_ready_blockers),
            )
        )
        if self.eval_protocol == LEGACY_PROTOCOL:
            self.logger.logging(
                'LEGACY ARTIFACT PRESERVATION: %s is read-only; this run will write only %s.' % (
                    self.teacher_model_path, self.teacher_model_archive_path
                )
            )
        elif not self.teacher_alias_writable:
            self.logger.logging(
                'TEACHER ALIAS WITHHELD: this run is not paper-ready eligible; '
                'the teacher will be preserved only at %s.' % self.teacher_model_archive_path
            )
        if getattr(args, 'dataset_config_profile', '') == 'unvalidated_amazon_default_fallback':
            self.logger.logging(
                'DATASET CONFIG WARNING: %s currently uses unvalidated Amazon fallback defaults; '
                'formal runs must pin and document dataset-specific settings.' % args.dataset
            )
        if getattr(args, 'smoke_train_batches', 0) > 0:
            self.logger.logging(
                'SMOKE TRAINING CAP: using %d batch(es) per epoch instead of the full %d; '
                'this run is not paper-ready.' % (
                    self.train_batch_count,
                    data_generator.n_train // args.batch_size + 1,
                )
            )

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.student_lr = args.student_lr
        self.emb_dim = args.embed_size
        self.student_emb_dim = args.student_embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.student_n_layers = args.student_n_layers
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        # ===== TD-Distill =====
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        # ===== Sparse Matrix Compatibility =====
        self.ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat','rb')).tocsr()
        self.ui_graph = self.ui_graph_raw.copy()

        # self.image_ui_graph_tmp = self.text_ui_graph_tmp = torch.tensor(self.ui_graph_raw.todense()).cuda()
        # self.image_iu_graph_tmp = self.text_iu_graph_tmp = torch.tensor(self.ui_graph_raw.T.todense()).cuda()

        self.image_ui_index = {'x':[], 'y':[]}
        self.text_ui_index = {'x':[], 'y':[]}

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]        
        self.iu_graph = self.ui_graph.T
  
        self.ui_graph_dgl = dgl.heterograph({('user','ui','item'):self.ui_graph.nonzero()})
        self.iu_graph_dgl = dgl.heterograph({('user','ui','item'):self.iu_graph.nonzero()})

        self.ui_graph = self.csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.csr_norm(self.iu_graph, mean_flag=True)
        self.adj = sp.vstack([sp.hstack([self.ui_graph, csr_matrix((self.n_users, self.n_users))]), sp.hstack([csr_matrix((self.n_items, self.n_items)), self.iu_graph])])

        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        self.adj = self.matrix_to_tensor(self.adj)  
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        self.teacher_model = Teacher_Model(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, self.image_feats, self.text_feats)      
        # self.student_model = Student_LightGCN(self.n_users, self.n_items, self.student_emb_dim, self.student_n_layers, self.mess_dropout, self.image_feats, self.text_feats)      
        self.teacher_model = self.teacher_model.cuda()
        # self.student_model = self.student_model.cuda()
        self.prompt_module = PromptLearner(self.image_feats, self.text_feats, self.ui_graph)
        self._update_run_manifest(
            hard_token_cache=getattr(self.prompt_module, 'hard_token_cache_records', {})
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCELoss()        

        self.opt_T = optim.AdamW([{'params':self.teacher_model.parameters()},
                                  {'params':self.prompt_module.parameters()}
                                  ], lr=self.lr, weight_decay=args.t_weight_decay)  
        # self.opt_S = optim.AdamW([{'params':self.student_model.parameters()},], lr=self.student_lr)  


        # self.scheduler_D = self.set_lr_scheduler()

    def _safe_artifact_name(self, name):
        safe_name = str(name)
        for invalid_char in ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ']:
            safe_name = safe_name.replace(invalid_char, '_')
        return safe_name.strip('_') or 'artifact'

    def _save_pickle(self, payload, file_path):
        with open(file_path, 'wb') as file_obj:
            pickle.dump(payload, file_obj)

    def _update_run_manifest(self, **updates):
        self.run_manifest.update(updates)
        write_json(self.run_manifest_path, self.run_manifest)

    def _selection_target(self):
        if self.uses_validation_selection:
            return list(data_generator.val_set.keys()), True, 'Validation'
        return list(data_generator.test_set.keys()), False, 'Legacy-Test'

    def _primary_recall(self, result):
        return float(result['recall'][self.primary_k_index])

    def _save_teacher_checkpoint(self, best_selection_recall, best_epoch):
        checkpoint = {
            'format_version': 5,
            'teacher_model': self.teacher_model.state_dict(),
            'prompt_module': self.prompt_module.state_dict(),
            'evaluation_protocol': self.eval_protocol,
            'selection_split': self.selection_split,
            'candidate_exclusion_policy': self.candidate_exclusion_policy,
            'primary_k': self.primary_k,
            'best_selection_recall': float(best_selection_recall),
            'best_epoch': int(best_epoch),
            'run_name': self.run_name,
            'dataset': args.dataset,
            'dataset_config_profile': getattr(args, 'dataset_config_profile', 'unknown'),
            'dataset_config_source': getattr(args, 'dataset_config_source', 'unknown'),
            'dataset_config_overrides': getattr(args, 'dataset_config_overrides', {}),
            'paper_ready_eligible': self.paper_ready_eligible,
            'paper_ready_blockers': self.paper_ready_blockers,
            'dataset_identity': self.dataset_identity,
            'teacher_inference_config': self.teacher_inference_config,
            'hard_token_cache': getattr(self.prompt_module, 'hard_token_cache_records', {}),
        }
        os.makedirs(os.path.dirname(self.teacher_model_path), exist_ok=True)
        temporary_path = '{}.tmp.{}'.format(
            self.teacher_model_archive_path, os.getpid()
        )
        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, self.teacher_model_archive_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _publish_teacher_alias(self):
        if not self.teacher_alias_writable:
            return None
        alias_fingerprint = publish_file_atomically(
            self.teacher_model_archive_path,
            self.teacher_model_path,
            allow_overwrite=self.allow_teacher_alias_overwrite,
        )
        self.logger.logging(
            'Teacher alias published after successful final test: %s' %
            self.teacher_model_path
        )
        return alias_fingerprint

    def _load_teacher_checkpoint(
        self, checkpoint_path, require_paper_ready_reuse=True
    ):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'teacher_model' in checkpoint:
            validate_teacher_checkpoint_metadata(
                checkpoint,
                self.eval_protocol,
                self.primary_k,
                args.dataset,
                checkpoint_path,
                self.dataset_identity,
                self.teacher_inference_config,
                self.paper_ready_eligible,
                self.paper_ready_blockers,
                self.candidate_exclusion_policy,
                require_paper_ready_checkpoint=require_paper_ready_reuse,
            )
            self.teacher_model.load_state_dict(checkpoint['teacher_model'])
            prompt_state = checkpoint.get('prompt_module')
            if prompt_state is None:
                raise ValueError(
                    'Teacher checkpoint at %s is missing prompt_module state. '
                    'Run once with --if_train_teacher true to refresh the checkpoint bundle.'
                    % checkpoint_path
                )
            if self.eval_protocol == PAPER_READY_PROTOCOL:
                self.prompt_module.load_state_dict(prompt_state, strict=True)
            else:
                incompatible = self.prompt_module.load_state_dict(prompt_state, strict=False)
                allowed_missing = {'item_hard_token', 'user_hard_token'}
                unexpected_missing = set(incompatible.missing_keys) - allowed_missing
                if unexpected_missing or incompatible.unexpected_keys:
                    raise ValueError(
                        'Legacy prompt checkpoint incompatibility at %s: missing=%s, unexpected=%s' % (
                            checkpoint_path,
                            sorted(unexpected_missing),
                            sorted(incompatible.unexpected_keys),
                        )
                    )
            checkpoint_hard_token_cache = checkpoint.get('hard_token_cache')
            if checkpoint_hard_token_cache:
                self.prompt_module.hard_token_cache_records = checkpoint_hard_token_cache
            metadata_keys = (
                'format_version',
                'evaluation_protocol',
                'selection_split',
                'candidate_exclusion_policy',
                'primary_k',
                'best_selection_recall',
                'best_epoch',
                'run_name',
                'dataset',
                'dataset_config_profile',
                'dataset_config_source',
                'dataset_config_overrides',
                'paper_ready_eligible',
                'paper_ready_blockers',
                'dataset_identity',
                'teacher_inference_config',
                'hard_token_cache',
            )
            self.active_teacher_checkpoint_metadata = {
                key: checkpoint.get(key) for key in metadata_keys if key in checkpoint
            }
            return checkpoint

        raise ValueError(
            'Legacy teacher checkpoint detected at %s. It only stores teacher_model weights, '
            'so teacher reuse is not a fully frozen reproduction. Run once with '
            '--if_train_teacher true to refresh the checkpoint bundle.'
            % checkpoint_path
        )

    def _save_student_checkpoint(self, best_selection_recall, best_epoch):
        checkpoint = {
            'format_version': 1,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.opt_S.state_dict(),
            'prompt_module': self.prompt_module.state_dict(),
            'student_model_type': args.student_model_type,
            'evaluation_protocol': self.eval_protocol,
            'selection_split': self.selection_split,
            'primary_k': self.primary_k,
            'best_selection_recall': float(best_selection_recall),
            'best_epoch': int(best_epoch),
            'run_name': self.run_name,
        }
        torch.save(checkpoint, self.student_checkpoint_path)

    def _load_student_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        prompt_state = checkpoint.get('prompt_module')
        if prompt_state is not None:
            self.prompt_module.load_state_dict(prompt_state)
        return checkpoint


    # def set_lr_scheduler(self):
    #     fac = lambda epoch: 0.96 ** (epoch / 50)

    #     scheduler_D = optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=fac)

    #     return scheduler_D

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data).to(torch.float32)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=self.device).coalesce()  #

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):  
        pred_i = torch.sum(torch.mul(u_pos,i_pos), dim=-1) 
        pred_j = torch.sum(torch.mul(u_neg,j_neg), dim=-1)  
        return pred_i, pred_j

    def sampleTrainBatch_dgl(self, batIds, pos_id=None, g=None, g_neg=None, sample_num=None, sample_num_neg=None):

        sub_g = dgl.sampling.sample_neighbors(g.cpu(), {'user':batIds}, sample_num, edge_dir='out', replace=True)
        row, col = sub_g.edges()
        row = row.reshape(len(batIds), sample_num)
        col = col.reshape(len(batIds), sample_num)

        if g_neg==None:
            return row, col
        else: 
            sub_g_neg = dgl.sampling.sample_neighbors(g_neg, {'user':batIds}, sample_num_neg, edge_dir='out', replace=True)
            row_neg, col_neg = sub_g_neg.edges()
            row_neg = row_neg.reshape(len(batIds), sample_num_neg)
            col_neg = col_neg.reshape(len(batIds), sample_num_neg)
            return row, col, col_neg 

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)



    def weighted_sum(self, anchor, nei, co):  

        ac = torch.multiply(anchor, co).sum(-1).sum(-1)  
        nc = torch.multiply(nei, co).sum(-1).sum(-1)  

        an = (anchor.permute(1, 0, 2)[0])
        ne = (nei.permute(1, 0, 2)[0])

        an_w = an*(ac.unsqueeze(-1).repeat(1, args.embed_size))
        ne_w = ne*(nc.unsqueeze(-1).repeat(1, args.embed_size))                                     
  
        res = (args.anchor_rate*an_w + (1-args.anchor_rate)*ne_w).reshape(-1, args.sample_num_ii, args.embed_size).sum(1)

        return res


    def sample_topk(self, u_sim, users, emb_type=None):
        topk_p, topk_id = torch.topk(u_sim, args.ad_topk*10, dim=-1)  
        topk_data = topk_p.reshape(-1).cpu()
        topk_col = topk_id.reshape(-1).cpu().int()
        topk_row = torch.tensor(np.array(users)).unsqueeze(1).repeat(1, args.ad_topk*args.ad_topk_multi_num).reshape(-1).int()  #
        topk_csr = csr_matrix((topk_data.detach().numpy(), (topk_row.detach().numpy(), topk_col.detach().numpy())), shape=(self.n_users, self.n_items))
        topk_g = dgl.heterograph({('user','ui','item'):topk_csr.nonzero()})
        _, topk_id = self.sampleTrainBatch_dgl(users, g=topk_g, sample_num=args.ad_topk, pos_id=None, g_neg=None, sample_num_neg=None)
        self.gene_fake[emb_type] = topk_id

        topk_id_u = torch.arange(len(users)).unsqueeze(1).repeat(1, args.ad_topk)
        topk_p = u_sim[topk_id_u, topk_id]
        return topk_p, topk_id

    def ssl_loss_calculation(self, ssl_image_logit, ssl_text_logit, ssl_common_logit):
        ssl_label_1_s2 = torch.ones(1, self.n_items).cuda()
        ssl_label_0_s2 = torch.zeros(1, self.n_items).cuda()
        ssl_label_s2 = torch.cat((ssl_label_1_s2, ssl_label_0_s2), 1)
        ssl_image_s2 = self.bce(ssl_image_logit, ssl_label_s2)
        ssl_text_s2 = self.bce(ssl_text_logit, ssl_label_s2)
        ssl_loss_s2 = ssl_image_s2 + ssl_text_s2

        ssl_label_1_c2 = torch.ones(1, self.n_items*2).cuda()
        ssl_label_0_c2 = torch.zeros(1, self.n_items*2).cuda()
        ssl_label_c2 = torch.cat((ssl_label_1_c2, ssl_label_0_c2), 1)
        ssl_result_c2 = self.bce(ssl_common_logit, ssl_label_c2)  
        ssl_loss_c2 = ssl_result_c2

        ssl_loss2 = args.ssl_s_rate*ssl_loss_s2 + args.ssl_c_rate*ssl_loss_c2 
        return ssl_loss2


    def sim(self, z1, z2):
        z1 = F.normalize(z1)  
        z2 = F.normalize(z2)
        # z1 = z1/((z1**2).sum(-1) + 1e-8)
        # z2 = z2/((z2**2).sum(-1) + 1e-8)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):

        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / args.tau)   #       

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))  
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))  

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()/ (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())+1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()


    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = 1./2*(g_item_image**2).sum() + 1./2*(g_item_text**2).sum() \
            + 1./2*(g_user_image**2).sum() + 1./2*(g_user_text**2).sum()        
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = args.feat_reg_decay * feat_reg
        return feat_emb_loss


    def fake_gene_loss_calculation(self, u_emb, i_emb, emb_type=None):
        if self.gene_u!=None:
            gene_real_loss = (-F.logsigmoid((u_emb[self.gene_u]*i_emb[self.gene_real]).sum(-1)+1e-8)).mean()
            gene_fake_loss = (1-(-F.logsigmoid((u_emb[self.gene_u]*i_emb[self.gene_fake[emb_type]]).sum(-1)+1e-8))).mean()

            gene_loss = gene_real_loss + gene_fake_loss
        else:
            gene_loss = 0

        return gene_loss

    def reward_loss_calculation(self, users, re_u, re_i, topk_id, topk_p):
        self.gene_u = torch.tensor(np.array(users)).unsqueeze(1).repeat(1, args.ad_topk)
        reward_u = re_u[self.gene_u]
        reward_i = re_i[topk_id]
        reward_value = (reward_u*reward_i).sum(-1)

        reward_loss = -(((topk_p*reward_value).sum(-1)).mean()+1e-8).log()
        
        return reward_loss




    def u_sim_calculation(self, users, user_final, item_final):
        topk_u = user_final[users]
        u_ui = torch.tensor(self.ui_graph_raw[users].todense()).cuda()

        num_batches = (self.n_items - 1) // args.batch_size + 1
        indices = torch.arange(0, self.n_items).cuda()
        u_sim_list = []

        for i_b in range(num_batches):
            index = indices[i_b * args.batch_size:(i_b + 1) * args.batch_size]
            sim = torch.mm(topk_u, item_final[index].T)
            sim_gt = torch.multiply(sim, (1-u_ui[:, index]))
            u_sim_list.append(sim_gt)
                
        u_sim = F.normalize(torch.cat(u_sim_list, dim=-1), p=2, dim=1)   
        return u_sim



    def loss_function(self, pred, drop_rate):
        # loss = F.cross_entropy(y, t, reduce = False)
        # loss_mul = loss * t
        ind_sorted = np.argsort(pred.cpu().data).cuda()
        loss_sorted = pred[ind_sorted]

        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))

        ind_update = ind_sorted[:num_remember]

        loss_update = pred[ind_update]

        return loss_update.mean()



    def mse_criterion(self, x, y, mask_nodes_dict=None, alpha=3):

        # res_list = []
        # for id, value in enumerate(x_dict):
        #     # x, y  = x_dict[value][mask_nodes_dict[value]], y_dict[value][mask_nodes_dict[value]]
        # x, y  = x_dict[value], y_dict[value]

        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)
        tmp_loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        tmp_loss = tmp_loss.mean()

        loss = F.mse_loss(x, y)
        # res_list.append(tmp_loss)
        # loss = sum(res_list)/len(res_list)
        return loss


    def sce_criterion(self, x, y, alpha=1, tip_rate=0):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1-(x*y).sum(dim=-1)).pow_(alpha)


        if tip_rate!=0:
            loss = self.loss_function(loss, tip_rate)   
            return loss

        loss = loss.mean() 

        # loss = loss.mean()

        return loss



    def test(self, users_to_test, is_val, is_teacher=True):
        self.teacher_model.eval()
        self.prompt_module.eval()
        with torch.no_grad():
            if is_teacher:
                u_embed, i_embed, *rest = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)
            else:
                # ===== TD-Distill =====
                if args.student_model_type in ('td_distill', 'td_distill_no_projection'):
                    self.td_distill_model.eval()
                    u_embed, i_embed = self.td_distill_model()
                else:
                    self.student_model.eval()
                    with torch.no_grad():
                            self.u_final_embed, self.i_final_embed, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds \
                            , G_user_emb, G_item_emb, prompt_user, prompt_item \
                            = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)

                    if args.student_model_type=='lightgcn':
                        u_embed, i_embed = self.student_model(self.adj, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds)
                    elif args.student_model_type=='gcn': 
                        u_embed, i_embed = self.student_model(self.u_final_embed, self.i_final_embed, self.ui_graph, self.iu_graph)
                    elif args.student_model_type=='mlp': 
                        u_embed, i_embed = self.student_model(self.u_final_embed, self.i_final_embed)  
                    elif args.student_model_type=='mm_light':
                        u_embed, i_embed = self.student_model(image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds)
                    

        result = test_torch(u_embed, i_embed, users_to_test, is_val)
        return result

    # ===== TD-Distill =====
        # ===== TD-Distill =====
    def save_td_distill_checkpoints(self, best_selection_recall, best_epoch):
        torch.save({
            'format_version': 2,
            'model_state_dict': self.td_distill_model.state_dict(),
            'optimizer_state_dict': self.opt_TD.state_dict(),
            'best_recall': best_selection_recall,
            'best_selection_recall': best_selection_recall,
            'best_epoch': best_epoch,
            'evaluation_protocol': self.eval_protocol,
            'selection_split': self.selection_split,
            'primary_k': self.primary_k,
            'run_name': self.run_name,
            'student_embedding_dim': self.td_distill_model.embedding_dim,
            'student_config_profile': getattr(args, 'student_config_profile', 'none'),
            'student_learning_rate': self.student_lr,
            'student_weight_decay': args.student_weight_decay,
            'item_teacher_semantic_dim': self.td_distill_model.item_teacher_dim,
            'user_teacher_semantic_dim': self.td_distill_model.user_teacher_dim,
            'td_distill_alpha': getattr(args, 'td_distill_alpha', 0.1),
            'td_component_rates': {
                'item_image': getattr(args, 'td_item_image_rate', 1.0),
                'item_text': getattr(args, 'td_item_text_rate', 1.0),
                'user_image': getattr(args, 'td_user_image_rate', 1.0),
                'user_text': getattr(args, 'td_user_text_rate', 1.0),
            },
            'td_init_from_teacher': getattr(args, 'td_init_from_teacher', True),
            'student_model_type': args.student_model_type,
        }, self.td_distill_full_path)

        if args.student_model_type == 'td_distill_no_projection':
            infer_state = export_infer_state_dict_no_projection(self.td_distill_model)
        else:
            infer_state = export_infer_state_dict(self.td_distill_model)

        torch.save(infer_state, self.td_distill_infer_path)

    def load_td_distill_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.td_distill_model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

    # ===== TD-Distill =====
    def train_td_distill(self):
        td_best_selection_recall = -1.
        td_best_epoch = None
        td_stopping_step = 0
        td_alpha = getattr(args, 'td_distill_alpha', 0.1)
        td_component_rates = {
            'item_image': max(0.0, float(getattr(args, 'td_item_image_rate', 1.0))),
            'item_text': max(0.0, float(getattr(args, 'td_item_text_rate', 1.0))),
            'user_image': max(0.0, float(getattr(args, 'td_user_image_rate', 1.0))),
            'user_text': max(0.0, float(getattr(args, 'td_user_text_rate', 1.0))),
        }
        td_direction_enabled = abs(td_alpha) > 0 and any(rate > 0 for rate in td_component_rates.values())
        n_batch = self.train_batch_count
        td_final_test_ret = None
        td_batch_loss_list, td_bpr_loss_list, td_distill_loss_list = [], [], []
        td_item_image_loss_list, td_item_text_loss_list = [], []
        td_user_image_loss_list, td_user_text_loss_list = [], []
        recall20_list, recall50_list, ndcg20_list, ndcg50_list = [], [], [], []
        selection_users, selection_is_val, selection_label = self._selection_target()

        with torch.no_grad():
            _, _, t_i_image_embed, t_i_text_embed, t_u_image_embed, t_u_text_embed \
            , _, _, _, _ \
            = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)
            td_teacher_semantics = {
                'item_image': t_i_image_embed.detach(),
                'item_text': t_i_text_embed.detach(),
                'user_image': t_u_image_embed.detach(),
                'user_text': t_u_text_embed.detach(),
            }

        self.logger.logging(
            'TD-Distill teacher image/text semantics for both items and users cached once from the frozen teacher checkpoint.'
        )
        self.logger.logging(
            'TD-Distill component rates: item_image=%.4f, item_text=%.4f, user_image=%.4f, user_text=%.4f' % (
                td_component_rates['item_image'],
                td_component_rates['item_text'],
                td_component_rates['user_image'],
                td_component_rates['user_text'],
            )
        )
        if not td_direction_enabled:
            self.logger.logging(
                'TD-Distill semantic loss disabled because td_distill_alpha=0.0 or all component rates are zero; this run reduces to BPR-only training under the current initialization setting.'
            )
        else:
            active_components = [name for name, rate in td_component_rates.items() if rate > 0]
            self.logger.logging(
                'TD-Distill active semantic heads: %s' % ', '.join(active_components)
            )

        for epoch in range(args.epoch):
            t1 = time()
            loss, td_bpr_total, td_distill_total = 0., 0., 0.
            td_component_totals = {
                'item_image': 0.,
                'item_text': 0.,
                'user_image': 0.,
                'user_text': 0.,
            }

            for idx in tqdm(range(n_batch)):
                self.teacher_model.eval()
                self.prompt_module.eval()
                self.td_distill_model.train()
                users, pos_items, neg_items = data_generator.sample()

                users = torch.as_tensor(users, dtype=torch.long, device=self.device)
                pos_items = torch.as_tensor(pos_items, dtype=torch.long, device=self.device)
                neg_items = torch.as_tensor(neg_items, dtype=torch.long, device=self.device)

                user_embeddings = self.td_distill_model.user_id_embedding(users)
                pos_item_embeddings = self.td_distill_model.item_id_embedding(pos_items)
                neg_item_embeddings = self.td_distill_model.item_id_embedding(neg_items)

                td_bpr = td_bpr_loss(user_embeddings, pos_item_embeddings, neg_item_embeddings)
                if td_direction_enabled:
                    projected_student_items = self.td_distill_model.project_items(pos_item_embeddings)
                    projected_student_users = self.td_distill_model.project_users(user_embeddings)
                    component_losses = {}

                    if td_component_rates['item_image'] > 0:
                        component_losses['item_image'] = directional_distillation_loss(
                            projected_student_items['image'],
                            td_teacher_semantics['item_image'][pos_items],
                        )
                    if td_component_rates['item_text'] > 0:
                        component_losses['item_text'] = directional_distillation_loss(
                            projected_student_items['text'],
                            td_teacher_semantics['item_text'][pos_items],
                        )
                    if td_component_rates['user_image'] > 0:
                        component_losses['user_image'] = directional_distillation_loss(
                            projected_student_users['image'],
                            td_teacher_semantics['user_image'][users],
                        )
                    if td_component_rates['user_text'] > 0:
                        component_losses['user_text'] = directional_distillation_loss(
                            projected_student_users['text'],
                            td_teacher_semantics['user_text'][users],
                        )

                    weighted_distill = torch.zeros((), dtype=td_bpr.dtype, device=self.device)
                    component_weight_sum = 0.0
                    for component_name, component_loss in component_losses.items():
                        component_rate = td_component_rates[component_name]
                        weighted_distill = weighted_distill + component_rate * component_loss
                        component_weight_sum += component_rate
                        td_component_totals[component_name] += component_loss.detach().item()

                    if component_weight_sum <= 0:
                        td_distill = torch.zeros((), dtype=td_bpr.dtype, device=self.device)
                    else:
                        td_distill = weighted_distill / component_weight_sum
                    td_batch_loss = td_bpr + td_alpha * td_distill
                else:
                    td_distill = torch.zeros((), dtype=td_bpr.dtype, device=self.device)
                    td_batch_loss = td_bpr

                self.opt_TD.zero_grad()
                td_batch_loss.backward(retain_graph=False)
                self.opt_TD.step()

                loss += td_batch_loss.detach().item()
                td_bpr_total += td_bpr.detach().item()
                td_distill_total += td_distill.detach().item()

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: TD-Distill loss is nan.')
                sys.exit()

            selection_ret = self.test(
                selection_users, is_val=selection_is_val, is_teacher=False
            )
            t2 = time()

            td_batch_loss_list.append(loss)
            td_bpr_loss_list.append(td_bpr_total)
            td_distill_loss_list.append(td_distill_total)
            td_item_image_loss_list.append(td_component_totals['item_image'])
            td_item_text_loss_list.append(td_component_totals['item_text'])
            td_user_image_loss_list.append(td_component_totals['user_image'])
            td_user_text_loss_list.append(td_component_totals['user_text'])
            recall20_list.append(selection_ret['recall'][self.primary_k_index])
            recall50_list.append(selection_ret['recall'][-1])
            ndcg20_list.append(selection_ret['ndcg'][self.primary_k_index])
            ndcg50_list.append(selection_ret['ndcg'][-1])

            td_results = {
                'evaluation_protocol': self.eval_protocol,
                'selection_split': self.selection_split,
                'primary_selection_k': self.primary_k,
                'td_batch_loss_List': td_batch_loss_list,
                'td_bpr_loss_List': td_bpr_loss_list,
                'td_distill_loss_List': td_distill_loss_list,
                'td_item_image_loss_List': td_item_image_loss_list,
                'td_item_text_loss_List': td_item_text_loss_list,
                'td_user_image_loss_List': td_user_image_loss_list,
                'td_user_text_loss_List': td_user_text_loss_list,
                'selection_recall20_List': recall20_list,
                'selection_recall50_List': recall50_list,
                'selection_ndcg20_List': ndcg20_list,
                'selection_ndcg50_List': ndcg50_list,
                # Compatibility aliases; selection_split records their meaning.
                'recall20_List': recall20_list,
                'recall50_List': recall50_list,
                'ndcg20_List': ndcg20_list,
                'ndcg50_List': ndcg50_list,
                'td_distill_alpha': td_alpha,
                'td_init_from_teacher': getattr(args, 'td_init_from_teacher', True),
                'td_component_rates': td_component_rates,
            }
            self._save_pickle(td_results, self.converge_archive_path)
            if args.point:
                self._save_pickle(td_results, os.path.join(self.converge_dir, args.point))

            if args.verbose > 0:
                perf_str = 'TD-Distill %s: Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f], components=[ii=%.5f, it=%.5f, ui=%.5f, ut=%.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (selection_label, epoch, t2 - t1, loss, td_bpr_total, td_distill_total,
                            td_component_totals['item_image'], td_component_totals['item_text'],
                            td_component_totals['user_image'], td_component_totals['user_text'],
                            selection_ret['recall'][0], selection_ret['recall'][1], selection_ret['recall'][2],
                            selection_ret['recall'][-1],
                            selection_ret['precision'][0], selection_ret['precision'][1], selection_ret['precision'][2], selection_ret['precision'][-1], selection_ret['hit_ratio'][0], selection_ret['hit_ratio'][1], selection_ret['hit_ratio'][2], selection_ret['hit_ratio'][-1],
                            selection_ret['ndcg'][0], selection_ret['ndcg'][1], selection_ret['ndcg'][2], selection_ret['ndcg'][-1])
                self.logger.logging(perf_str)

            selection_recall = self._primary_recall(selection_ret)
            if selection_recall > td_best_selection_recall:
                td_best_selection_recall = selection_recall
                td_best_epoch = epoch
                self.logger.logging(
                    'TD-Distill best %s checkpoint: epoch=%d, Recall@%d=%.5f' % (
                        selection_label, epoch, self.primary_k, td_best_selection_recall
                    )
                )
                td_stopping_step = 0
                self.save_td_distill_checkpoints(td_best_selection_recall, td_best_epoch)
            else:
                td_stopping_step += 1
                self.logger.logging(
                    '#####TD-Distill early stopping steps: %d/%d #####' % (
                        td_stopping_step, self.early_stopping_limit
                    )
                )
                if td_stopping_step >= self.early_stopping_limit:
                    self.logger.logging('#####TD-Distill early stop! #####')
                    break

        if td_best_epoch is None or not os.path.exists(self.td_distill_full_path):
            raise RuntimeError('TD-Distill did not produce a selectable checkpoint.')

        if not self.run_final_test:
            self.load_td_distill_checkpoint(self.td_distill_full_path)
            self.logger.logging(
                'TD-Distill final Test skipped after restoring the best %s '
                'checkpoint; test_mat was not evaluated.' % selection_label
            )
            td_results.update({
                'best_selection_epoch': td_best_epoch,
                'best_selection_recall': td_best_selection_recall,
                'td_full_checkpoint': self.td_distill_full_path,
                'td_infer_checkpoint': self.td_distill_infer_path,
            })
            self._save_pickle(td_results, self.converge_archive_path)
            if args.point:
                self._save_pickle(td_results, os.path.join(self.converge_dir, args.point))
            self._update_run_manifest(
                status='smoke_completed' if self.smoke_mode else 'validation_completed',
                completed_at=datetime.now().astimezone().isoformat(),
                model_stage='td_distill',
                best_selection_epoch=td_best_epoch,
                best_selection_recall=td_best_selection_recall,
                final_test_performed=False,
                td_full_checkpoint=self.td_distill_full_path,
                td_infer_checkpoint=self.td_distill_infer_path,
            )
            return

        users_to_test = list(data_generator.test_set.keys())
        _, td_final_test_ret = restore_checkpoint_then_evaluate(
            self.td_distill_full_path,
            self.load_td_distill_checkpoint,
            lambda: self.test(users_to_test, is_val=False, is_teacher=False),
        )
        self.logger.logging(
            'TD-Distill final Test after restoring best %s checkpoint: '
            'epoch=%d, best_%s_Recall@%d=%.5f, Test Recall@%d=%.5f, '
            'precision=[%.5f], ndcg=[%.5f]' % (
                selection_label,
                td_best_epoch,
                self.selection_split,
                self.primary_k,
                td_best_selection_recall,
                self.primary_k,
                td_final_test_ret['recall'][self.primary_k_index],
                td_final_test_ret['precision'][self.primary_k_index],
                td_final_test_ret['ndcg'][self.primary_k_index],
            )
        )
        self.logger.logging(str(td_final_test_ret))
        td_results.update({
            'best_selection_epoch': td_best_epoch,
            'best_selection_recall': td_best_selection_recall,
            'final_test_result': td_final_test_ret,
            'td_full_checkpoint': self.td_distill_full_path,
            'td_infer_checkpoint': self.td_distill_infer_path,
        })
        self._save_pickle(td_results, self.converge_archive_path)
        if args.point:
            self._save_pickle(td_results, os.path.join(self.converge_dir, args.point))
        self._update_run_manifest(
            status='completed',
            completed_at=datetime.now().astimezone().isoformat(),
            model_stage='td_distill',
            best_selection_epoch=td_best_epoch,
            best_selection_recall=td_best_selection_recall,
            final_test_performed=True,
            final_test_result=result_to_dict(td_final_test_ret),
            td_full_checkpoint=self.td_distill_full_path,
            td_infer_checkpoint=self.td_distill_infer_path,
        )


    def train(self):

        now_time = datetime.now()
        run_time = datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        line_var_loss, line_g_loss, line_d_loss, line_cl_loss, line_var_recall, line_var_precision, line_var_ndcg = [], [], [], [], [], [], []
        teacher_stopping_step = 0
        student_stopping_step = 0
        teacher_best_epoch = None
        teacher_best_selection_recall = -1.
        teacher_test_ret = None
        student_best_epoch = None
        student_test_ret = None
        should_stop = False
        cur_best_pre_0 = 0. 

        if getattr(args, 'run_efficiency_benchmark', False) and args.if_train_teacher:
            self.logger.logging('Efficiency benchmark mode enabled: overriding --if_train_teacher to false to avoid retraining teacher.')
            args.if_train_teacher = False

        if args.if_train_teacher: 
            # ----train_teacher-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            if args.point:
                print("########begin:T###################################")
                print(args.point)
                print("###########################################")
            n_batch = self.train_batch_count
            teacher_selection_users, teacher_selection_is_val, teacher_selection_label = self._selection_target()
            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
                contrastive_loss = 0.
                n_batch = self.train_batch_count
                f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
                sample_time = 0.
                build_item_graph = True

                self.gene_u, self.gene_real, self.gene_fake = None, None, {}
                self.topk_p_dict, self.topk_id_dict = {}, {}

                for idx in tqdm(range(n_batch)):
                    self.teacher_model.train()
                    self.prompt_module.train()
                    sample_t1 = time()
                    users, pos_items, neg_items = data_generator.sample()
                    sample_time += time() - sample_t1       


                    # self.prompt_module()
                    t_u_id_embed, t_i_id_embed, t_i_image_embed, t_i_text_embed, t_u_image_embed, t_u_text_embed \
                                    , G_user_emb, G_item_emb, prompt_user, prompt_item \
                            = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)


                    t_u_id_embed_pbr = t_u_id_embed[users]
                    t_i_id_embed_pbr_pos = t_i_id_embed[pos_items]
                    t_i_id_embed_pbr_neg = t_i_id_embed[neg_items]
                    t_mf_loss, t_emb_loss = self.bpr_loss(t_u_id_embed_pbr, t_i_id_embed_pbr_pos, t_i_id_embed_pbr_neg)
        
                    # prompt
                    u_id_embed_pbr_prompt = prompt_user[users]
                    i_id_embed_pbr_pos_prompt = prompt_item[pos_items]
                    i_id_embed_pbr_neg_prompt = prompt_item[neg_items]
                    mf_loss_prompt, emb_loss_prompt = self.bpr_loss(u_id_embed_pbr_prompt, i_id_embed_pbr_pos_prompt, i_id_embed_pbr_neg_prompt)
        

                    t_image_u_g_embeddings = t_u_image_embed[users]
                    t_image_pos_i_g_embeddings = t_i_image_embed[pos_items]
                    t_image_neg_i_g_embeddings = t_i_image_embed[neg_items]
                    t_image_batch_mf_loss, G_image_batch_emb_loss = self.bpr_loss(t_image_u_g_embeddings, t_image_pos_i_g_embeddings, t_image_neg_i_g_embeddings)

                    t_text_u_g_embeddings = t_u_text_embed[users]
                    t_text_pos_i_g_embeddings = t_i_text_embed[pos_items]
                    t_text_neg_i_g_embeddings = t_i_text_embed[neg_items]
                    t_text_batch_mf_loss, G_text_batch_emb_loss = self.bpr_loss(t_text_u_g_embeddings, t_text_pos_i_g_embeddings, t_text_neg_i_g_embeddings)


                    feat_emb_loss = self.feat_reg_loss_calculation(t_i_image_embed, t_i_text_embed, t_u_image_embed, t_u_text_embed)

                    # t_batch_loss = t_mf_loss + t_emb_loss + feat_emb_loss + args.t_prompt_rate1*mf_loss_prompt #+ args.t_prompt_rate2*emb_loss_prompt + args.t_feat_mf_rate*t_image_batch_mf_loss + args.t_feat_mf_rate*t_text_batch_mf_loss
                    t_batch_loss = t_mf_loss + t_emb_loss + feat_emb_loss + args.t_prompt_rate1*mf_loss_prompt + args.t_feat_mf_rate*t_image_batch_mf_loss + args.t_feat_mf_rate*t_text_batch_mf_loss
                    # t_batch_loss = t_mf_loss + t_emb_loss + feat_emb_loss + args.t_feat_mf_rate*t_image_batch_mf_loss + args.t_feat_mf_rate*t_text_batch_mf_loss



                    line_var_loss.append(t_batch_loss.detach().data)
                    # line_cl_loss.append(batch_contrastive_loss.detach().data)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    #+ ssl_loss2 #+ batch_contrastive_loss
                    self.opt_T.zero_grad()  
                    t_batch_loss.backward(retain_graph=False)
                    self.opt_T.step()

                    loss += t_batch_loss.detach().item()
                    mf_loss += t_emb_loss.detach().item()
                    # emb_loss += float(t_reg_loss)
                    # reg_loss += float(G_batch_reg_loss)
                    teacher_feat_dict = { 'item_image':t_i_image_embed.detach(),'item_text':t_i_text_embed.detach(),'user_image':t_u_image_embed.detach(),'user_text':t_u_text_embed.detach() }

        
                # del ua_embeddings, ia_embeddings, G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings
                del t_u_id_embed, t_i_id_embed, t_i_image_embed, t_i_text_embed, t_u_image_embed, t_u_text_embed \
                                    , G_user_emb, G_item_emb \
                                    , t_u_id_embed_pbr, t_i_id_embed_pbr_pos, t_i_id_embed_pbr_neg


                if math.isnan(loss) == True:
                    self.logger.logging('ERROR: loss is nan.')
                    sys.exit()

                t2 = time()
                selection_ret = self.test(
                    teacher_selection_users,
                    is_val=teacher_selection_is_val,
                    is_teacher=True,
                )
                training_time_list.append(t2 - t1)

                t3 = time()

                loss_loger.append(loss)
                rec_loger.append(selection_ret['recall'].data)
                pre_loger.append(selection_ret['precision'].data)
                ndcg_loger.append(selection_ret['ndcg'].data)
                hit_loger.append(selection_ret['hit_ratio'].data)

                line_var_recall.append(selection_ret['recall'][self.primary_k_index])
                line_var_precision.append(selection_ret['precision'][self.primary_k_index])
                line_var_ndcg.append(selection_ret['ndcg'][self.primary_k_index])

                tags = ["recall", "precision", "ndcg"]


                if args.verbose > 0:
                    perf_str = 'Teacher %s: Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                            'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                            (teacher_selection_label, epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, selection_ret['recall'][0], selection_ret['recall'][1], selection_ret['recall'][2],
                                selection_ret['recall'][-1],
                                selection_ret['precision'][0], selection_ret['precision'][1], selection_ret['precision'][2], selection_ret['precision'][-1], selection_ret['hit_ratio'][0], selection_ret['hit_ratio'][1], selection_ret['hit_ratio'][2], selection_ret['hit_ratio'][-1],
                                selection_ret['ndcg'][0], selection_ret['ndcg'][1], selection_ret['ndcg'][2], selection_ret['ndcg'][-1])
                    self.logger.logging(perf_str)

                teacher_selection_recall = self._primary_recall(selection_ret)
                if teacher_selection_recall > teacher_best_selection_recall:
                    teacher_best_selection_recall = teacher_selection_recall
                    teacher_best_epoch = epoch
                    self.logger.logging(
                        'Teacher best %s checkpoint: epoch=%d, Recall@%d=%.5f' % (
                            teacher_selection_label,
                            teacher_best_epoch,
                            self.primary_k,
                            teacher_best_selection_recall,
                        )
                    )
                    self._save_teacher_checkpoint(
                        teacher_best_selection_recall, teacher_best_epoch
                    )
                    teacher_stopping_step = 0
                    # torch.save(self.teacher_model.state_dict(), '/home/weiw/Code/MM/KDMM/Model/' + args.dataset + '/teacher_model_prompt.pt')
                    # torch.save(teacher_feat_dict, '/home/weiw/Code/MM/KDMM/Model/' + args.dataset + '/teacher_feat_dict_listwise_1000.pt')
                    # torch.save(self.teacher_model.state_dict(), '/home/weiw/Code/MM/KDMM/Model/' + args.dataset + '/teacher_model_listwise_1000.pt')
                else:
                    teacher_stopping_step += 1
                    self.logger.logging(
                        'Teacher early stopping step: %d/%d' % (
                            teacher_stopping_step, self.early_stopping_limit
                        )
                    )
                    if teacher_stopping_step >= self.early_stopping_limit:
                        self.logger.logging('Teacher early stop triggered.')
                        break

            if teacher_best_epoch is None or not os.path.exists(self.teacher_model_archive_path):
                raise RuntimeError('Teacher training did not produce a selectable checkpoint.')
            if self.run_final_test:
                users_to_test = list(data_generator.test_set.keys())
                _, teacher_test_ret = restore_checkpoint_then_evaluate(
                    self.teacher_model_archive_path,
                    self._load_teacher_checkpoint,
                    lambda: self.test(users_to_test, is_val=False, is_teacher=True),
                )
                self.logger.logging(
                    'Teacher final Test after restoring best %s checkpoint: '
                    'epoch=%d, best_%s_Recall@%d=%.5f, Test Recall@%d=%.5f, '
                    'precision=[%.5f], ndcg=[%.5f]' % (
                        teacher_selection_label,
                        teacher_best_epoch,
                        self.selection_split,
                        self.primary_k,
                        teacher_best_selection_recall,
                        self.primary_k,
                        teacher_test_ret['recall'][self.primary_k_index],
                        teacher_test_ret['precision'][self.primary_k_index],
                        teacher_test_ret['ndcg'][self.primary_k_index],
                    )
                )
                self.logger.logging(str(teacher_test_ret))
                self._update_run_manifest(
                    status='teacher_completed',
                    teacher_best_selection_epoch=teacher_best_epoch,
                    teacher_best_selection_recall=teacher_best_selection_recall,
                    teacher_final_test_performed=True,
                    teacher_final_test_result=result_to_dict(teacher_test_ret),
                    teacher_checkpoint=self.teacher_model_archive_path,
                    teacher_checkpoint_fingerprint=file_fingerprint(self.teacher_model_archive_path),
                    teacher_checkpoint_metadata=self.active_teacher_checkpoint_metadata,
                    active_teacher_hard_token_cache=getattr(
                        self.prompt_module, 'hard_token_cache_records', {}
                    ),
                )
                teacher_alias_fingerprint = self._publish_teacher_alias()
                self._update_run_manifest(
                    teacher_alias_published=teacher_alias_fingerprint is not None,
                    teacher_alias_fingerprint=teacher_alias_fingerprint,
                )
            else:
                self._load_teacher_checkpoint(
                    self.teacher_model_archive_path,
                    require_paper_ready_reuse=False,
                )
                self.logger.logging(
                    'Teacher final Test skipped after restoring the best %s '
                    'checkpoint; test_mat was not evaluated and no shared alias '
                    'was published.' % teacher_selection_label
                )
                self._update_run_manifest(
                    status='teacher_validation_completed',
                    teacher_best_selection_epoch=teacher_best_epoch,
                    teacher_best_selection_recall=teacher_best_selection_recall,
                    teacher_final_test_performed=False,
                    teacher_checkpoint=self.teacher_model_archive_path,
                    teacher_checkpoint_fingerprint=file_fingerprint(self.teacher_model_archive_path),
                    teacher_checkpoint_metadata=self.active_teacher_checkpoint_metadata,
                    teacher_alias_published=False,
                    active_teacher_hard_token_cache=getattr(
                        self.prompt_module, 'hard_token_cache_records', {}
                    ),
                )
            if args.point:
                print("######end:T#####################################")
                print(args.point)
                print("###########################################")
            if getattr(args, 'teacher_only', False):
                self._update_run_manifest(
                    status='smoke_completed' if self.smoke_mode else 'completed',
                    completed_at=datetime.now().astimezone().isoformat(),
                    model_stage='teacher_only',
                )
                self.logger.logging('Teacher-only run completed; student training skipped.')
                return
            # ----train_teacher-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        else:
            self.logger.logging('Teacher training skipped; will reuse checkpoint at %s' % self.teacher_model_path)



        # # ----train_student-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # print("########begin:S###################################")
        if args.point:
            print(args.point)
            print("###########################################")
        n_batch = self.train_batch_count
        student_best_selection_recall = -1.
        # self.teacher_feat_dict = torch.load('/home/weiw/Code/MM/KDMM/Model/' + args.dataset + '/teacher_feat_dict.pt')
        teacher_checkpoint_for_student = (
            self.teacher_model_archive_path if args.if_train_teacher else self.teacher_model_path
        )
        if not os.path.exists(teacher_checkpoint_for_student):
            raise FileNotFoundError('Teacher checkpoint not found at %s. Run once with --if_train_teacher true before reusing the teacher.' % teacher_checkpoint_for_student)
        self.logger.logging('Loading teacher checkpoint for student distillation from %s' % teacher_checkpoint_for_student)
        self._load_teacher_checkpoint(teacher_checkpoint_for_student)
        self.teacher_model.eval()
        self.prompt_module.eval()
        if teacher_test_ret is None and self.run_final_test:
            users_to_test = list(data_generator.test_set.keys())
            _, teacher_test_ret = restore_checkpoint_then_evaluate(
                teacher_checkpoint_for_student,
                self._load_teacher_checkpoint,
                lambda: self.test(users_to_test, is_val=False, is_teacher=True),
            )
            self.logger.logging("Teacher reuse summary: Recall@%d=%.5f, precision=[%.5f], ndcg=[%.5f]" % (
                self.primary_k,
                teacher_test_ret['recall'][self.primary_k_index],
                teacher_test_ret['precision'][self.primary_k_index],
                teacher_test_ret['ndcg'][self.primary_k_index],
            ))
            self.logger.logging(str(teacher_test_ret))
            self._update_run_manifest(
                status='teacher_reused',
                teacher_final_test_result=result_to_dict(teacher_test_ret),
                teacher_checkpoint=teacher_checkpoint_for_student,
                teacher_checkpoint_fingerprint=file_fingerprint(teacher_checkpoint_for_student),
                teacher_checkpoint_metadata=self.active_teacher_checkpoint_metadata,
                active_teacher_hard_token_cache=getattr(
                    self.prompt_module, 'hard_token_cache_records', {}
                ),
            )
        elif teacher_test_ret is None:
            self.logger.logging(
                'Teacher checkpoint reused without test evaluation; test_mat '
                'remains untouched for this run.'
            )
            self._update_run_manifest(
                status='teacher_reused_without_test',
                teacher_final_test_performed=False,
                teacher_checkpoint=teacher_checkpoint_for_student,
                teacher_checkpoint_fingerprint=file_fingerprint(teacher_checkpoint_for_student),
                teacher_checkpoint_metadata=self.active_teacher_checkpoint_metadata,
                active_teacher_hard_token_cache=getattr(
                    self.prompt_module, 'hard_token_cache_records', {}
                ),
            )
        else:
            self._update_run_manifest(
                teacher_checkpoint_used_for_student=teacher_checkpoint_for_student,
                teacher_checkpoint_fingerprint=file_fingerprint(teacher_checkpoint_for_student),
                teacher_checkpoint_metadata=self.active_teacher_checkpoint_metadata,
                active_teacher_hard_token_cache=getattr(
                    self.prompt_module, 'hard_token_cache_records', {}
                ),
            )

        # if args.if_train_teacher and epoch==0:
        # with torch.no_grad():
        #     pre_u_final_embed, pre_i_final_embed, _, _, _, _ \
        #     , _, _ \
        #     = self.teacher_model(self.ui_graph, self.iu_graph)
        with torch.no_grad():
            self.u_final_embed, self.i_final_embed, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds \
            , G_user_emb, G_item_emb, prompt_user, prompt_item \
            = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)

        # ===== TD-Distill =====
        # ===== TD-Distill =====
        if args.student_model_type in ('td_distill', 'td_distill_no_projection'):
            td_item_teacher_dim = image_item_embeds.size(-1)
            td_user_teacher_dim = image_user_embeds.size(-1)

            td_item_head_names = tuple(
                head_name for head_name, rate in (
                    ('image', float(getattr(args, 'td_item_image_rate', 1.0))),
                    ('text', float(getattr(args, 'td_item_text_rate', 1.0))),
                ) if rate > 0
            )
            td_user_head_names = tuple(
                head_name for head_name, rate in (
                    ('image', float(getattr(args, 'td_user_image_rate', 1.0))),
                    ('text', float(getattr(args, 'td_user_text_rate', 1.0))),
                ) if rate > 0
            )

            if text_item_embeds.size(-1) != td_item_teacher_dim:
                raise ValueError(
                    'TD-Distill item semantic dims do not match: image {} vs text {}'.format(
                        image_item_embeds.size(-1), text_item_embeds.size(-1)
                    )
                )
            if text_user_embeds.size(-1) != td_user_teacher_dim:
                raise ValueError(
                    'TD-Distill user semantic dims do not match: image {} vs text {}'.format(
                        image_user_embeds.size(-1), text_user_embeds.size(-1)
                    )
                )

            if args.student_model_type == 'td_distill_no_projection':
                self.td_distill_model = TDDistillNoProjectionModel(
                    self.n_users,
                    self.n_items,
                    self.student_emb_dim,
                    td_item_teacher_dim,
                    td_user_teacher_dim,
                    item_head_names=td_item_head_names,
                    user_head_names=td_user_head_names,
                ).cuda()

                self.logger.logging(
                    'TD-Distill NO-PROJECTION semantic heads initialized with item_teacher_dim=%d, user_teacher_dim=%d, item_heads=%s, user_heads=%s.' % (
                        td_item_teacher_dim,
                        td_user_teacher_dim,
                        list(td_item_head_names),
                        list(td_user_head_names),
                    )
                )
            else:
                self.td_distill_model = TDDistillModel(
                    self.n_users,
                    self.n_items,
                    self.student_emb_dim,
                    td_item_teacher_dim,
                    td_user_teacher_dim,
                    item_head_names=td_item_head_names,
                    user_head_names=td_user_head_names,
                ).cuda()

                self.logger.logging(
                    'TD-Distill semantic heads initialized with item_teacher_dim=%d, user_teacher_dim=%d, item_heads=%s, user_heads=%s.' % (
                        td_item_teacher_dim,
                        td_user_teacher_dim,
                        list(td_item_head_names),
                        list(td_user_head_names),
                    )
                )

            td_init_from_teacher = getattr(args, 'td_init_from_teacher', True)
            if td_init_from_teacher and self.u_final_embed.size(-1) == self.student_emb_dim and self.i_final_embed.size(-1) == self.student_emb_dim:
                self.td_distill_model.init_user_item_embed(self.u_final_embed, self.i_final_embed)
                self.logger.logging('TD-Distill warm start enabled from teacher embeddings.')
            else:
                self.logger.logging('TD-Distill warm start disabled.')

            self.opt_TD = optim.AdamW(
                [{'params': self.td_distill_model.parameters()}],
                lr=self.student_lr,
                weight_decay=args.student_weight_decay,
            )

            self.td_distill_full_path = os.path.join(
                self.td_distill_dir,
                'td_distill_full__%s__%s.pth' % (self.eval_protocol, self.run_name)
            )
            self.td_distill_infer_path = os.path.join(
                self.td_distill_dir,
                'td_distill_infer_only__%s__%s.pth' % (self.eval_protocol, self.run_name)
            )

            if getattr(args, 'run_efficiency_benchmark', False):
                maybe_load_student_checkpoint_for_efficiency(self, args)
                run_efficiency_benchmark(self, args, data_generator)
                return

            self.train_td_distill()
            return


            

        if args.student_model_type=='lightgcn':
            self.student_model = Student_LightGCN(self.n_users, self.n_items, self.student_emb_dim, self.student_n_layers, self.mess_dropout, self.image_feats, self.text_feats)   
            self.student_model.init_user_item_embed(self.u_final_embed, self.i_final_embed)
        elif args.student_model_type=='gcn': 
            self.student_model = Student_GCN(self.n_users, self.n_items, self.student_emb_dim, self.student_n_layers, self.mess_dropout, self.image_feats, self.text_feats)   
        elif args.student_model_type=='mlp': 
            self.student_model = Student_MLP()   
            self.student_model.init_user_item_embed(self.u_final_embed, self.i_final_embed)
        elif args.student_model_type=='mm_light':
            self.student_model = Student_MMLight(self.n_users, self.n_items, self.student_emb_dim, self.student_n_layers, self.mess_dropout, self.image_feats, self.text_feats, self.ui_graph, self.iu_graph)   
            self.student_model.init_user_item_embed(self.u_final_embed, self.i_final_embed)

        self.student_model = self.student_model.cuda()
        self.opt_S = optim.AdamW(
            [
                {'params': self.student_model.parameters()},
                {'params': self.prompt_module.parameters()},
            ],
            lr=self.student_lr,
            weight_decay=args.student_weight_decay,
        )

        if getattr(args, 'run_efficiency_benchmark', False):
            maybe_load_student_checkpoint_for_efficiency(self, args)
            run_efficiency_benchmark(self, args, data_generator)
            return

        student_selection_users, student_selection_is_val, student_selection_label = self._selection_target()
        student_selection_recall20_list = []
        student_selection_recall50_list = []
        student_selection_ndcg20_list = []
        student_selection_ndcg50_list = []
        results = None
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, kd_total_loss = 0., 0., 0., 0.
            kd_pair_loss_total, kd_list_image_loss_total, kd_list_text_loss_total, kd_feat_loss_total = 0., 0., 0., 0.
            n_batch = self.train_batch_count
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            student_batch_loss_List = []
            batch_mf_loss_List = []
            batch_emb_loss_List = []
            kd_loss_List = []
            kd_pair_loss_List = []
            kd_list_image_loss_List = []
            kd_list_text_loss_List = []
            kd_feat_loss_List = []


            sample_time = 0.
            build_item_graph = True


            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}

            for idx in tqdm(range(n_batch)):
                self.teacher_model.eval()
                self.prompt_module.eval()
                self.student_model.train()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()  # [1024], [1024], [1024] 
                sample_time += time() - sample_t1      

                with torch.no_grad():
                    self.u_final_embed, self.i_final_embed, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds \
                    , G_user_emb, G_item_emb, prompt_user, prompt_item \
                    = self.teacher_model(self.ui_graph, self.iu_graph, self.prompt_module)

                if args.student_model_type=='lightgcn':
                    u_embed, i_embed = self.student_model(self.adj, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds)
                elif args.student_model_type=='gcn': 
                    u_embed, i_embed = self.student_model(
                        self.u_final_embed, self.i_final_embed, self.ui_graph, self.iu_graph
                    )
                elif args.student_model_type=='mlp': 
                    u_embed, i_embed = self.student_model(
                        self.u_final_embed, self.i_final_embed
                    )
                elif args.student_model_type=='mm_light': 
                    u_embed, i_embed = self.student_model(image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds)


                u_embeddings = u_embed[users]
                pos_i_embeddings = i_embed[pos_items]
                neg_i_embeddings = i_embed[neg_items]
                batch_mf_loss, batch_emb_loss = self.bpr_loss(u_embeddings, pos_i_embeddings, neg_i_embeddings)
                # print(f'batch_mf_loss1: {batch_mf_loss}')  

                if args.student_model_type=='mlp':
                    batch_mf_loss = self.student_model.pairPredictwEmbeds(u_embed, i_embed, users, pos_items, neg_items).sum()
                    # print(f'batch_mf_loss2: {batch_mf_loss}')  


                # -----------------------------KD-----------------------------------
                if args.student_model_type=='mlp':
                    student_mf_loss = self.student_model.pairPredictwEmbeds(u_embed, i_embed, users, pos_items, neg_items)
                else:
                    student_mf_loss, _, _ = self.bpr_loss_for_KD(u_embeddings, pos_i_embeddings, neg_i_embeddings) 

                # --------------------------------------list-wise ranking-------------------------------------------------------------------------------------- 
                # num_nodes = u_final_embed.size(0)             
                # batch_size = 1024 
                # num_batches = (num_nodes - 1) // batch_size + 1
                # # f = lambda x: torch.exp(x / args.tau)   #       
                # indices = torch.arange(0, num_nodes)
                # list_wise_losses = []
                # for i in range(num_batches):
                #     tmp_index = indices[i * batch_size:(i + 1) * batch_size]
                #     ### constructure graph
                #     sub_x_index, sub_y_index = self.ui_graph_raw[tmp_index].nonzero()[0], self.ui_graph_raw[tmp_index].nonzero()[1]
                #     sub_dgl_g = dgl.graph((sub_x_index, sub_y_index)).to('cuda:0')
                #     ### neg sample 
                #     neg_row, neg_col = dgl.sampling.global_uniform_negative_sampling(sub_dgl_g, tmp_index.shape[0]*args.neg_sample_num)
                #     neg_row, neg_col = neg_row.reshape((tmp_index.shape[0], args.neg_sample_num)), neg_col.reshape((tmp_index.shape[0], args.neg_sample_num))  # [1024, 10] [1024, 10]
                #     # torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_items)
                #     item_index = torch.cat((torch.tensor(pos_items).cuda().unsqueeze(1), neg_col), dim=1)
                #     # u_embed, i_embed 
                #     list_wise_score = torch.mul(u_embed[users].unsqueeze(1), i_embed[item_index]).sum(-1).softmax(-1)  
                #     # list_wise_score = torch.mul(u_final_embed[users].unsqueeze(1), i_final_embed[item_index]).sum(-1).softmax(-1)  
                #     tmp_list_wise_loss = -list_wise_score[:,0].log().sum()
                #     list_wise_losses.append(tmp_list_wise_loss)
                #     del sub_x_index, sub_y_index, sub_dgl_g
                # list_wise_losses = sum(list_wise_losses)


                ### constructure graph
                sub_x_index, sub_y_index = self.ui_graph_raw[users].nonzero()[0], self.ui_graph_raw[users].nonzero()[1]
                # ===== DGL CPU Sampling Compatibility =====
                sub_dgl_g = dgl.graph((sub_x_index, sub_y_index))
                ### neg sample 
                neg_row, neg_col = dgl.sampling.global_uniform_negative_sampling(sub_dgl_g, len(users)*args.neg_sample_num, replace=True)
                neg_row = neg_row.to(self.device)
                neg_col = neg_col.to(self.device)
                neg_row, neg_col = neg_row.reshape((len(users), args.neg_sample_num)), neg_col.reshape((len(users), args.neg_sample_num))  # [1024, 10] [1024, 10]
                # torch.tensor(users), torch.tensor(pos_items), torch.tensor(neg_items)
                item_index = torch.cat((torch.as_tensor(pos_items, dtype=torch.long, device=self.device).unsqueeze(1), neg_col), dim=1)

                # s_list_wise
                list_wise_score_s = torch.mul(u_embed[users].unsqueeze(1), i_embed[item_index]).sum(-1).softmax(-1) 
                # list_wise_loss_s = -(list_wise_score_s[:,0]+1e-8).log() 
                list_wise_loss_s = -(list_wise_score_s+1e-8).log() 
                list_wise_score_t_image = torch.mul(image_user_embeds[users].unsqueeze(1), image_item_embeds[item_index]).sum(-1).softmax(-1) 
                # list_wise_loss_t_image = -(list_wise_score_t_image[:,0]+1e-8).log()
                list_wise_loss_t_image = -(list_wise_score_t_image+1e-8).log()
                list_wise_score_t_text = torch.mul(text_user_embeds[users].unsqueeze(1), text_item_embeds[item_index]).sum(-1).softmax(-1) 
                # list_wise_loss_t_text = -(list_wise_score_t_text[:,0]+1e-8).log()
                list_wise_loss_t_text = -(list_wise_score_t_text+1e-8).log()


                # list_wise_losses.append(tmp_list_wise_loss)
                del sub_x_index, sub_y_index, sub_dgl_g
                # --------------------------------------list-wise ranking-------------------------------------------------------------------------------------- 
 
                u_g_embed_mf = self.u_final_embed[users]
                pos_i_g_embed_mf = self.i_final_embed[pos_items]
                neg_i_g_embed_mf = self.i_final_embed[neg_items]

                teacher_mf_loss, _, _ = self.bpr_loss_for_KD(u_g_embed_mf, pos_i_g_embed_mf, neg_i_g_embed_mf)

                pairwise_kd_loss = self.distillation(student_mf_loss, teacher_mf_loss, temp=args.student_tau, alpha=0.7)
                list_image_kd_loss = self.distillation(list_wise_loss_s, list_wise_loss_t_image, temp=args.student_tau, alpha=0.7)
                list_text_kd_loss = self.distillation(list_wise_loss_s, list_wise_loss_t_text, temp=args.student_tau, alpha=0.7)
                # -----------------------------KD-----------------------------------

                # ----feat kd loss-----------------------------------------------------------------------------------------------------------
                # self.u_final_embed, self.i_final_embed, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds

                # reconstru_feat = decoder(feat_learned)
                if args.feat_loss_type=='mse':
                    # feature_loss = sce_criterion(reconstru_feat, features, mask_nodes_dict, alpha=args.alpha_l)
                    feature_kd_loss = self.mse_criterion(self.i_final_embed, i_embed, alpha=args.alpha_l)
                elif args.feat_loss_type=='sce':
                    feature_kd_loss = self.sce_criterion(image_item_embeds, i_embed, alpha=args.alpha_l, tip_rate=args.tip_rate_feat) + self.sce_criterion(text_item_embeds, i_embed, alpha=args.alpha_l, tip_rate=args.tip_rate_feat)
                # ----feat kd loss-----------------------------------------------------------------------------------------------------------


                weighted_pairwise_kd_loss = args.kd_loss_rate * pairwise_kd_loss
                weighted_list_image_kd_loss = args.kd_loss_list_rate * list_image_kd_loss
                weighted_list_text_kd_loss = args.kd_loss_list_rate * list_text_kd_loss
                weighted_feature_kd_loss = args.kd_loss_feat_rate * feature_kd_loss
                batch_kd_loss = (
                    weighted_pairwise_kd_loss
                    + weighted_list_image_kd_loss
                    + weighted_list_text_kd_loss
                    + weighted_feature_kd_loss
                )
                student_batch_loss = batch_mf_loss + batch_emb_loss + batch_kd_loss


                # line_var_loss.append(student_batch_loss.detach().data)
                             
                #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #+ ssl_loss2 #+ batch_contrastive_loss
                self.opt_S.zero_grad()  
                student_batch_loss.backward(retain_graph=False)
                self.opt_S.step()

                loss += student_batch_loss.detach().item()
                mf_loss += batch_mf_loss.detach().item()
                emb_loss += batch_emb_loss.detach().item()
                kd_total_loss += batch_kd_loss.detach().item()
                kd_pair_loss_total += weighted_pairwise_kd_loss.detach().item()
                kd_list_image_loss_total += weighted_list_image_kd_loss.detach().item()
                kd_list_text_loss_total += weighted_list_text_kd_loss.detach().item()
                kd_feat_loss_total += weighted_feature_kd_loss.detach().item()
        
                student_batch_loss_List.append(student_batch_loss.item())
                batch_mf_loss_List.append(batch_mf_loss.item())
                batch_emb_loss_List.append(batch_emb_loss.item())
                kd_loss_List.append(batch_kd_loss.item())
                kd_pair_loss_List.append(weighted_pairwise_kd_loss.item())
                kd_list_image_loss_List.append(weighted_list_image_kd_loss.item())
                kd_list_text_loss_List.append(weighted_list_text_kd_loss.item())
                kd_feat_loss_List.append(weighted_feature_kd_loss.item())

            # del ua_embeddings, ia_embeddings, G_ua_embeddings, G_ia_embeddings, G_u_g_embeddings, G_neg_i_g_embeddings, G_pos_i_g_embeddings
            del u_embed, i_embed, u_embeddings, pos_i_embeddings, neg_i_embeddings


            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            t2 = time()
            selection_ret = self.test(
                student_selection_users,
                is_val=student_selection_is_val,
                is_teacher=False,
            )
            training_time_list.append(t2 - t1)
            t3 = time()


            student_selection_recall20_list.append(
                selection_ret['recall'][self.primary_k_index]
            )
            student_selection_recall50_list.append(selection_ret['recall'][-1])
            student_selection_ndcg20_list.append(
                selection_ret['ndcg'][self.primary_k_index]
            )
            student_selection_ndcg50_list.append(selection_ret['ndcg'][-1])

            # student_batch_loss_List, batch_mf_loss_List, kd_loss_List, recall20_List, recall50_List, ndcg20_List, ndcg50_List= [], [], [], [], [], [], []


            tags = ["recall", "precision", "ndcg"]

            results = {
                'evaluation_protocol': self.eval_protocol,
                'selection_split': self.selection_split,
                'primary_selection_k': self.primary_k,
                'student_batch_loss_List': student_batch_loss_List,
                'batch_mf_loss_List': batch_mf_loss_List,
                'batch_emb_loss_List': batch_emb_loss_List,
                'kd_loss_List': kd_loss_List,
                'optimized_kd_loss_List': kd_loss_List,
                'kd_pair_loss_List': kd_pair_loss_List,
                'kd_list_image_loss_List': kd_list_image_loss_List,
                'kd_list_text_loss_List': kd_list_text_loss_List,
                'kd_feat_loss_List': kd_feat_loss_List,
                'selection_recall20_List': student_selection_recall20_list,
                'selection_recall50_List': student_selection_recall50_list,
                'selection_ndcg20_List': student_selection_ndcg20_list,
                'selection_ndcg50_List': student_selection_ndcg50_list,
                # Compatibility aliases; selection_split records their meaning.
                'recall20_List': student_selection_recall20_list,
                'recall50_List': student_selection_recall50_list,
                'ndcg20_List': student_selection_ndcg20_list,
                'ndcg50_List': student_selection_ndcg50_list,
            }
            self._save_pickle(results, self.converge_archive_path)
            if args.point:
                self._save_pickle(results, os.path.join(self.converge_dir, args.point))


            if args.verbose > 0:
                perf_str = 'Student %s: Epoch %d [%.1fs + %.1fs]: train==[loss %.5f = mf %.5f + emb %.5f + kd %.5f], kd_detail==[pair %.5f + list_img %.5f + list_text %.5f + feat %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (student_selection_label, epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, kd_total_loss, kd_pair_loss_total, kd_list_image_loss_total, kd_list_text_loss_total, kd_feat_loss_total, selection_ret['recall'][0], selection_ret['recall'][1], selection_ret['recall'][2],
                            selection_ret['recall'][-1],
                            selection_ret['precision'][0], selection_ret['precision'][1], selection_ret['precision'][2], selection_ret['precision'][-1], selection_ret['hit_ratio'][0], selection_ret['hit_ratio'][1], selection_ret['hit_ratio'][2], selection_ret['hit_ratio'][-1],
                            selection_ret['ndcg'][0], selection_ret['ndcg'][1], selection_ret['ndcg'][2], selection_ret['ndcg'][-1])
                self.logger.logging(perf_str)

            student_selection_recall = self._primary_recall(selection_ret)
            if student_selection_recall > student_best_selection_recall:
                student_best_selection_recall = student_selection_recall
                student_best_epoch = epoch
                self.logger.logging(
                    'Student best %s checkpoint: epoch=%d, Recall@%d=%.5f' % (
                        student_selection_label,
                        student_best_epoch,
                        self.primary_k,
                        student_best_selection_recall,
                    )
                )
                self._save_student_checkpoint(
                    student_best_selection_recall, student_best_epoch
                )
                student_stopping_step = 0
            else:
                student_stopping_step += 1
                self.logger.logging(
                    'Student early stopping step: %d/%d' % (
                        student_stopping_step, self.early_stopping_limit
                    )
                )
                if student_stopping_step >= self.early_stopping_limit:
                    self.logger.logging('Student early stop triggered.')
                    break

        if student_best_epoch is None or not os.path.exists(self.student_checkpoint_path):
            raise RuntimeError('Student training did not produce a selectable checkpoint.')
        if not self.run_final_test:
            self._load_student_checkpoint(self.student_checkpoint_path)
            self.logger.logging(
                'Student final Test skipped after restoring the best %s '
                'checkpoint; test_mat was not evaluated.' % student_selection_label
            )
            results.update({
                'best_selection_epoch': student_best_epoch,
                'best_selection_recall': student_best_selection_recall,
                'student_checkpoint': self.student_checkpoint_path,
            })
            self._save_pickle(results, self.converge_archive_path)
            if args.point:
                self._save_pickle(results, os.path.join(self.converge_dir, args.point))
            self._update_run_manifest(
                status='smoke_completed' if self.smoke_mode else 'validation_completed',
                completed_at=datetime.now().astimezone().isoformat(),
                model_stage='original_student',
                best_selection_epoch=student_best_epoch,
                best_selection_recall=student_best_selection_recall,
                final_test_performed=False,
                student_checkpoint=self.student_checkpoint_path,
            )
            return
        users_to_test = list(data_generator.test_set.keys())
        _, student_test_ret = restore_checkpoint_then_evaluate(
            self.student_checkpoint_path,
            self._load_student_checkpoint,
            lambda: self.test(users_to_test, is_val=False, is_teacher=False),
        )
        self.logger.logging(
            'Student final Test after restoring best %s checkpoint: '
            'epoch=%d, best_%s_Recall@%d=%.5f, Test Recall@%d=%.5f, '
            'precision=[%.5f], ndcg=[%.5f]' % (
                student_selection_label,
                student_best_epoch,
                self.selection_split,
                self.primary_k,
                student_best_selection_recall,
                self.primary_k,
                student_test_ret['recall'][self.primary_k_index],
                student_test_ret['precision'][self.primary_k_index],
                student_test_ret['ndcg'][self.primary_k_index],
            )
        )
        self.logger.logging(str(student_test_ret))
        results.update({
            'best_selection_epoch': student_best_epoch,
            'best_selection_recall': student_best_selection_recall,
            'final_test_result': student_test_ret,
            'student_checkpoint': self.student_checkpoint_path,
        })
        self._save_pickle(results, self.converge_archive_path)
        if args.point:
            self._save_pickle(results, os.path.join(self.converge_dir, args.point))
        self._update_run_manifest(
            status='completed',
            completed_at=datetime.now().astimezone().isoformat(),
            model_stage='original_student',
            best_selection_epoch=student_best_epoch,
            best_selection_recall=student_best_selection_recall,
            final_test_performed=True,
            final_test_result=result_to_dict(student_test_ret),
            student_checkpoint=self.student_checkpoint_path,
        )
        if args.point:
            print("########end:S###################################")
            print(args.point)
            print("###########################################")
        # ----train_student-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





    def dkd_loss(self, logits_student, logits_teacher, target, alpha, beta, temperature):

        def _get_gt_mask(logits, target):
            target = target.reshape(-1)
            mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
            return mask


        def _get_other_mask(logits, target):
            target = target.reshape(-1)
            mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
            return mask

        def _cat_mask(t, mask1, mask2):
            t1 = (t * mask1).sum(dim=1, keepdims=True)
            t2 = (t * mask2).sum(1, keepdims=True)
            rt = torch.cat([t1, t2], dim=1)
            return rt

        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher.detach() / temperature, dim=1)
        pred_student = _cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = _cat_mask(pred_teacher, gt_mask, other_mask)
        # ===== Student KD Stability =====
        log_pred_student = torch.log(pred_student.clamp_min(1e-12))
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher.detach() / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (temperature**2)
            / target.shape[0]
        )
        return alpha * tckd_loss + beta * nckd_loss







    # def distillation(self, y, teacher_scores, temp, alpha):
    #     return nn.KLDivLoss()(F.log_softmax(y.unsqueeze(0) / temp, dim=1), F.softmax(teacher_scores.unsqueeze(0) / temp, dim=1)) 


    def distillation(self, y, teacher_scores, temp, alpha):
        # ===== Student KD Stability =====
        if y.dim() == 1:
            y = y.unsqueeze(0)
            teacher_scores = teacher_scores.unsqueeze(0)
        student_log_prob = F.log_softmax(y / temp, dim=-1)
        teacher_prob = F.softmax(teacher_scores.detach() / temp, dim=-1)
        return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temp ** 2)

    # def distillation(self, y, labels, teacher_scores, temp, alpha):
    #     return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
    #             temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

    def calcRegLoss(self, params=None, model=None):
        ret = 0
        if params is not None:
            for W in params:
                ret += W.norm(2).square()
        if model is not None:
            for W in model.parameters():
                ret += W.norm(2).square()
        # ret += (model.usrStruct + model.itmStruct)
        return ret

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()        
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def bpr_loss_for_KD(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        # regularizer = 1./2*(users**2).sum() + 1./2*(pos_items**2).sum() + 1./2*(neg_items**2).sum()
        regularizer = 1./2*(users**2) + 1./2*(pos_items**2) + 1./2*(neg_items**2)
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        # mf_loss = -torch.mean(maxi)
        mf_loss = -maxi

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss   

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data).to(torch.float32)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32).coalesce()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

if __name__ == '__main__':
    select_dataset()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    # select_dataset()
    trainer = Trainer.__new__(Trainer)
    try:
        trainer.__init__(data_config=config)
        trainer.train()
    except BaseException as error:
        if hasattr(trainer, 'run_manifest') and hasattr(trainer, 'run_manifest_path'):
            try:
                trainer._update_run_manifest(
                    status='failed',
                    failed_at=datetime.now().astimezone().isoformat(),
                    failure_type=type(error).__name__,
                    failure_message=str(error),
                )
            except Exception as manifest_error:
                print(
                    'Failed to record run failure in manifest: {}'.format(
                        manifest_error
                    ),
                    file=sys.stderr,
                )
        raise
