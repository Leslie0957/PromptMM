"""Pure aggregation for the fixed Sports ranking-gap diagnostic."""
import numpy as np


def thirds(counts):
    order = np.lexsort((np.arange(len(counts)), np.asarray(counts)))
    groups = np.empty(len(counts), dtype=np.uint8)
    for group, ids in enumerate(np.array_split(order, 3)):
        groups[ids] = group
    return groups


def decompose(teacher_top, student_top, val, users, user_groups, item_groups):
    n = len(users)
    arrays = {key: np.zeros(n, dtype=np.float64) for key in
              ('g', 'l', 'common', 'neither', 'overlap')}
    arrays['teacher_hits'] = np.zeros(n, dtype=np.uint8)
    arrays['student_hits'] = np.zeros(n, dtype=np.uint8)
    arrays['teacher_only_user'] = np.zeros(n, dtype=np.uint8)
    item_g = np.zeros(3, dtype=np.float64)
    item_l = np.zeros(3, dtype=np.float64)
    positive_counts = np.zeros(3, dtype=np.int64)
    user_rows = [{'users': 0, 'positives': 0, 'G_sum': 0., 'L_sum': 0.} for _ in range(3)]
    for row, uid in enumerate(users):
        positives = set(val.indices[val.indptr[uid]:val.indptr[uid + 1]].tolist())
        t = set(teacher_top[row].tolist())
        s = set(student_top[row].tolist())
        th, sh = t & positives, s & positives
        only_t, only_s = th - sh, sh - th
        common = th & sh
        denom = len(positives)
        if not denom:
            raise ValueError('Empty Validation row')
        arrays['g'][row] = len(only_t) / denom
        arrays['l'][row] = len(only_s) / denom
        arrays['common'][row] = len(common) / denom
        arrays['neither'][row] = len(positives - th - sh) / denom
        arrays['overlap'][row] = len(t & s) / len(t)
        arrays['teacher_hits'][row] = len(th)
        arrays['student_hits'][row] = len(sh)
        arrays['teacher_only_user'][row] = bool(only_t)
        ug = int(user_groups[uid])
        user_rows[ug]['users'] += 1
        user_rows[ug]['positives'] += denom
        user_rows[ug]['G_sum'] += arrays['g'][row]
        user_rows[ug]['L_sum'] += arrays['l'][row]
        for item in positives:
            positive_counts[int(item_groups[item])] += 1
        for item in only_t:
            item_g[int(item_groups[item])] += 1 / denom / n
        for item in only_s:
            item_l[int(item_groups[item])] += 1 / denom / n
    means = {key: float(np.mean(value, dtype=np.float64)) for key, value in arrays.items()
             if key not in ('teacher_hits', 'student_hits')}
    means['recall_teacher'] = float(np.mean(arrays['teacher_hits'] / np.diff(val.indptr)[users], dtype=np.float64))
    means['recall_student'] = float(np.mean(arrays['student_hits'] / np.diff(val.indptr)[users], dtype=np.float64))
    means['N'] = means['g'] - means['l']
    if abs(means['N'] - (means['recall_teacher'] - means['recall_student'])) > 1e-10:
        raise ValueError('G-L identity failed')
    if abs(sum(item_g) - means['g']) > 1e-10 or abs(sum(item_l) - means['l']) > 1e-10:
        raise ValueError('Item group contribution identity failed')
    return arrays, means, user_rows, [{'items': int(np.sum(item_groups == i)),
        'positives': int(positive_counts[i]), 'G_contribution': float(item_g[i]),
        'L_contribution': float(item_l[i])} for i in range(3)]
