import ast


BABY_TEACHER_PROFILE_NAME = 'baby_teacher_reference_v1'
BABY_TEACHER_PROFILE_SCOPE = 'teacher_only'
BABY_TEACHER_PROFILE_SOURCE = (
    'historical_baby_core_plus_matching_promptmm_prompt_defaults'
)
BABY_STUDENT_PROFILE_NAME = 'baby_student_reference_v1'
BABY_STUDENT_PROFILE_SCOPE = 'student_reference'
BABY_STUDENT_PROFILE_SOURCE = 'predeclared_baby_id_only_bpr_reference'

# Core teacher settings come from the dormant Baby block in parser.py. Prompt
# fields absent from that block follow the active PromptMM profile with the same
# model_cat_rate/weight_size family. This is a pinned reference baseline, not a
# claim that the parameters are validation-optimal for Baby.
BABY_TEACHER_PROFILE_DEFAULTS = {
    'seed': 2022,
    'sparse': 1,
    'Ks': '[10, 20, 40, 50]',
    'test_flag': 'part',
    'batch_size': 1024,
    'epoch': 1000,
    'cf_model': 'light_init',
    'early_stopping_patience': 7,
    'regs': '[1e-5, 1e-5, 1e-2]',
    'model_cat_rate': 0.55,
    'weight_size': '[64, 64]',
    'embed_size': 64,
    'drop_rate': 0.2,
    't_weight_decay': 0.001,
    'lr': 0.00055,
    't_feat_mf_rate': 1.0,
    'feat_reg_decay': 1e-5,
    'layers': 1,
    'mess_dropout': '[0.1, 0.1]',
    'hard_token_type': 'pca',
    'hard_token_seed': 2022,
    'soft_token_rate': 0.005,
    'feat_soft_token_rate': 1.0,
    't_prompt_rate1': 100.0,
    't_prompt_rate2': 1.0,
    't_prompt_rate3': 1.0,
    'prompt_dropout': 0.0,
}

# This profile defines the first Baby student comparison anchor. It is applied
# after the teacher/dataset profile so the frozen teacher retains its exact
# inference configuration while the student receives explicit safe defaults.
BABY_STUDENT_PROFILE_DEFAULTS = {
    'seed': 2022,
    'eval_protocol': 'val_test_once_v1',
    'Ks': '[10, 20, 40, 50]',
    'test_flag': 'part',
    'dataset_preflight': True,
    'duplicate_modalities_policy': 'error',
    'batch_size': 1024,
    'epoch': 1000,
    'smoke_train_batches': 0,
    'early_stopping_patience': 7,
    'if_train_teacher': False,
    'teacher_only': False,
    'teacher_checkpoint': 'Model/baby/teacher_model_val_test_once_v1.pt',
    'allow_teacher_alias_overwrite': False,
    'student_model_type': 'td_distill_no_projection',
    'student_embed_size': 64,
    'student_lr': 0.00006,
    'student_weight_decay': 0.01,
    'td_init_from_teacher': False,
    'td_distill_alpha': 0.0,
    'td_item_image_rate': 0.0,
    'td_item_text_rate': 0.0,
    'td_user_image_rate': 0.0,
    'td_user_text_rate': 0.0,
    'run_final_test': True,
    'run_efficiency_benchmark': False,
}

_LITERAL_FIELDS = {'Ks', 'mess_dropout', 'regs', 'weight_size'}


def dataset_profile(dataset):
    if dataset == 'baby':
        return {
            'name': BABY_TEACHER_PROFILE_NAME,
            'scope': BABY_TEACHER_PROFILE_SCOPE,
            'source': BABY_TEACHER_PROFILE_SOURCE,
            'defaults': dict(BABY_TEACHER_PROFILE_DEFAULTS),
        }
    return None


def apply_dataset_profile_defaults(parser, dataset):
    profile = dataset_profile(dataset)
    if profile is not None:
        parser.set_defaults(**profile['defaults'])
    return profile


def student_profile(dataset, profile_name):
    if not profile_name:
        return None
    if dataset != 'baby' or profile_name != BABY_STUDENT_PROFILE_NAME:
        raise ValueError(
            'Student profile {} is not valid for dataset {}'.format(
                profile_name, dataset
            )
        )
    return {
        'name': BABY_STUDENT_PROFILE_NAME,
        'scope': BABY_STUDENT_PROFILE_SCOPE,
        'source': BABY_STUDENT_PROFILE_SOURCE,
        'defaults': dict(BABY_STUDENT_PROFILE_DEFAULTS),
    }


def apply_student_profile_defaults(parser, dataset, profile_name):
    profile = student_profile(dataset, profile_name)
    if profile is not None:
        parser.set_defaults(**profile['defaults'])
    return profile


def _normalized(field_name, value):
    if field_name in _LITERAL_FIELDS and isinstance(value, str):
        return ast.literal_eval(value)
    return value


def resolved_profile_metadata(dataset, namespace):
    profile = dataset_profile(dataset)
    if profile is None:
        if dataset == 'amazon':
            return {
                'name': 'amazon_active_defaults',
                'scope': 'full',
                'source': 'active_amazon_parser_defaults',
                'overrides': {},
            }
        return {
            'name': 'unvalidated_amazon_default_fallback',
            'scope': 'unknown',
            'source': 'active_amazon_parser_defaults',
            'overrides': {},
        }

    overrides = {}
    for field_name, expected in profile['defaults'].items():
        actual = getattr(namespace, field_name)
        if _normalized(field_name, actual) != _normalized(field_name, expected):
            overrides[field_name] = {
                'expected': expected,
                'resolved': actual,
            }
    return {
        'name': profile['name'],
        'scope': profile['scope'],
        'source': profile['source'],
        'overrides': overrides,
    }


def resolved_student_profile_metadata(dataset, profile_name, namespace):
    profile = student_profile(dataset, profile_name)
    if profile is None:
        return {
            'name': 'none',
            'scope': 'none',
            'source': 'none',
            'overrides': {},
        }

    overrides = {}
    for field_name, expected in profile['defaults'].items():
        actual = getattr(namespace, field_name)
        if _normalized(field_name, actual) != _normalized(field_name, expected):
            overrides[field_name] = {
                'expected': expected,
                'resolved': actual,
            }
    return {
        'name': profile['name'],
        'scope': profile['scope'],
        'source': profile['source'],
        'overrides': overrides,
    }
