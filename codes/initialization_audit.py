"""Initialization-only controls: no extra sampling and explicit storage checks."""
import hashlib
import torch

EXPECTED_TD_TEACHER_INITIAL = {
    'users': {'shape': [35598, 64], 'dtype': 'torch.float32',
              'sha256': '81dfece426261cd6920ff64d79e8fe4b0579286185f6e7e7bb2c29ff28b76791'},
    'items': {'shape': [18357, 64], 'dtype': 'torch.float32',
              'sha256': 'e61376f4a0ed29e9c96a7d62083732ae0c991fdfcc023c51efc478b373fdd9c2'},
}


def require_same_td_initial(record):
    for side, expected in EXPECTED_TD_TEACHER_INITIAL.items():
        if record.get(side) != expected:
            raise RuntimeError(
                'Initial vectors differ from completed TD teacher-init full: '
                + side + '; expected=' + repr(expected) + '; actual=' + repr(record.get(side)))


def describe_tensor(value):
    value = value.detach().cpu().contiguous()
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError('Invalid initialization tensor.')
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())


def initialize_release(student, teacher_users, teacher_items, mode):
    if mode not in ('teacher', 'random'):
        raise ValueError('Unknown initialization mode.')
    users, items = ((teacher_users, teacher_items) if mode == 'teacher' else
                    (student.user_id_embedding.weight.detach(), student.item_id_embedding.weight.detach()))
    if users.shape != student.user_id_embedding.weight.shape or items.shape != student.item_id_embedding.weight.shape:
        raise ValueError('Initialization shape mismatch.')
    result = dict(mode=mode, source='teacher_eval' if mode == 'teacher' else 'constructor_xavier_uniform',
                  users=describe_tensor(users), items=describe_tensor(items))
    student.init_user_item_embed(users, items)
    student.assert_aliases()
    result['alias_preserved'] = True
    return result


def verify_td_teacher_copy(student, users, items):
    for actual, expected in ((student.user_id_embedding.weight, users),
                             (student.item_id_embedding.weight, items)):
        if not torch.equal(actual, expected) or actual.data_ptr() == expected.data_ptr():
            raise RuntimeError('TD initialization must copy teacher values into independent storage.')
    return dict(mode='teacher', source='teacher_eval', independent_storage=True,
                users=describe_tensor(student.user_id_embedding.weight),
                items=describe_tensor(student.item_id_embedding.weight))
