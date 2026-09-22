"""Exact-order NumPy top-k alternative; legacy training remains unchanged."""
import heapq
import numpy as np
from utility import metrics


def rank_fast(scores, candidates, k):
    candidates = np.asarray(candidates, dtype=np.int64)
    values = scores[candidates]
    if k < 1 or len(values) < k:
        raise ValueError('Invalid candidate count or k.')
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite candidate score.')
    threshold = np.partition(values, len(values)-k)[len(values)-k]
    above = np.flatnonzero(values > threshold)
    ties = np.flatnonzero(values == threshold)[:k-len(above)]
    # Positions first: stable descending score sort retains original candidate
    # order for every tie, exactly like heapq.nlargest with a key function.
    positions = np.sort(np.concatenate((above, ties)))
    return candidates[positions[np.argsort(-values[positions], kind='stable')]].tolist()


def rank_legacy(scores, candidates, k):
    return heapq.nlargest(k, candidates, key=lambda i: scores[i])


def accumulate(totals, ranked, truth, ks, count):
    relevance = [int(i in truth) for i in ranked]
    for j,k in enumerate(ks):
        totals['recall'][j] += metrics.recall_at_k(relevance,k,len(truth))/count
        totals['precision'][j] += metrics.precision_at_k(relevance,k)/count
        totals['ndcg'][j] += metrics.ndcg_at_k(relevance,k)/count
        totals['hit_ratio'][j] += metrics.hit_at_k(relevance,k)/count


def evaluate_fast(student, adj, train, val, batch_size=256, ks=(10,20,40,50)):
    import torch
    users=np.flatnonzero(val.getnnz(axis=1)).tolist()
    if not users:raise ValueError('No Validation users.')
    totals={k:np.zeros(len(ks),dtype=np.float64) for k in ('recall','precision','ndcg','hit_ratio')}
    student.eval()
    with torch.no_grad():
        ue,ie=student(adj)
        all_items=set(range(train.shape[1]))
        for offset in range(0,len(users),batch_size):
            batch=users[offset:offset+batch_size]
            scores=(ue[batch] @ ie.T).cpu().numpy()
            if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite score.')
            for row,u in enumerate(batch):
                candidates=list(all_items-set(train.indices[train.indptr[u]:train.indptr[u+1]]))
                truth=set(val.indices[val.indptr[u]:val.indptr[u+1]])
                accumulate(totals,rank_fast(scores[row],candidates,max(ks)),truth,ks,len(users))
    if not all(np.isfinite(v).all() for v in totals.values()):
        raise FloatingPointError('Nonfinite Validation metric.')
    return {k:v.tolist() for k,v in totals.items()}
