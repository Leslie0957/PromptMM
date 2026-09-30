"""Label-free two-list selection and bounded exposure calibration."""
import math
import numpy as np


def merge(ids, scores, beta, k=20):
    """Arrays [users, 2, >=k], group0 target; -1 IDs mark padding."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('Nonnegative finite beta required')
    n = len(ids)
    pos = np.zeros((n, 2), dtype=np.int64)
    out = np.empty((n, k), dtype=np.int32)
    counts = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    for j in range(k):
        a, b = ids[rows, 0, pos[:,0]], ids[rows, 1, pos[:,1]]
        if np.any((a < 0) & (b < 0)):
            raise ValueError('Too few candidates')
        diff = scores[rows,1,pos[:,1]].astype(np.float64) - scores[rows,0,pos[:,0]].astype(np.float64)
        take = (a >= 0) & ((b < 0) | (beta > diff) | ((beta == diff) & (a < b)))
        out[:,j] = np.where(take, a, b)
        counts += take
        pos[:,0] += take
        pos[:,1] += ~take
    return out, counts


def calibrate(ids, scores, target, maximum, check=lambda: None):
    trace = []
    def exposure(beta):
        check()
        e = int(merge(ids,scores,beta)[1].sum())
        trace.append({'beta':float(beta), 'exposure':e})
        ordered = sorted(trace, key=lambda x:x['beta'])
        if any(a['exposure'] > b['exposure'] for a,b in zip(ordered,ordered[1:])):
            raise RuntimeError('Nonmonotone exposure')
        return e
    lo, hi = 0., float(2 * maximum + 1)
    if not np.isfinite(hi) or hi <= 0:
        raise ValueError('Invalid bound')
    el = exposure(lo)
    reason = 'nearest_endpoint'
    if target <= el:
        beta,e,reason = lo,el,'exact' if target == el else 'below_nonnegative_range'
    else:
        eh = exposure(hi)
        if target > eh:
            beta,e,reason = hi,eh,'above_range'
        else:
            for _ in range(48):
                mid = (lo + hi) / 2
                if mid == lo or mid == hi:
                    break
                em = exposure(mid)
                if em == target:
                    lo=hi=mid;el=eh=em;reason='exact';break
                if em < target:
                    lo,el = mid,em
                else:
                    hi,eh = mid,em
            beta,e = min([(lo,el),(hi,eh)],key=lambda x:(abs(x[1]-target),x[0]))
    tolerance=max(1,math.ceil(.001*target))
    return {'beta':float(beta),'exposure':e,'target':int(target),'tolerance':tolerance,
            'matched':bool(abs(e-target)<=tolerance and reason not in ('below_nonnegative_range','above_range')),
            'reason':reason,'trace':trace,'final_interval':[float(lo),float(hi)]}


def decision(rows):
    if not all(x['matched'] for x in rows):
        return 'unresolved'
    if all(x['low_Q_minus_R'] >= -3 and x['recall_Q_minus_R'] >= -.0002 for x in rows):
        return 'simple_control_sufficient'
    if all(x['low_Q_minus_R'] <= -4 and x['recall_Q_minus_R'] <= .0002 for x in rows):
        return 'simple_control_insufficient'
    return 'unresolved'
