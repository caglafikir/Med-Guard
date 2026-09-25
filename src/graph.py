import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

from .config import CONFUSION_PAIR_NAMES, INTENT_MAP


def build_confusion_pairs():
    return {
        tuple(sorted((INTENT_MAP[a], INTENT_MAP[b])))
        for a, b in CONFUSION_PAIR_NAMES
    }


def is_selective_hard_negative(intent_i, intent_j, label_i, label_j, confusion_pairs):
    if label_i == label_j:
        return False
    return tuple(sorted((intent_i, intent_j))) in confusion_pairs


def _make_row_normalized_sparse(rows, cols, wts, n, device):
    """Row-normalized adjacency — positive values only."""
    if len(rows) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            (n, n), device=device).coalesce()

    A = sp.csr_matrix((wts, (rows, cols)), shape=(n, n))
    row_sum = np.array(A.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sum)
    A_norm = D_inv @ A

    A_coo = A_norm.tocoo()
    idx_t = torch.tensor(
        np.vstack([A_coo.row, A_coo.col]), dtype=torch.long, device=device)
    val_t = torch.tensor(A_coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        idx_t, val_t, (n, n), device=device).coalesce()


def build_signed_graph_separate(X, y, intents, confusion_pairs, device,
                                 k_graph=15,
                                 pos_sim_threshold=0.75,
                                 neg_sim_threshold=0.70):
    n    = len(X)
    nbrs = NearestNeighbors(n_neighbors=k_graph + 1, metric="cosine").fit(X)
    distances, indices = nbrs.kneighbors(X)

    pos_i_list, pos_j_list = [], []
    neg_u_list, neg_s_list = [], []
    pos_weights, neg_weights = [], []

    for i in range(n):
        for idx, dist in enumerate(distances[i]):
            j   = indices[i][idx]
            if i == j:
                continue
            sim = 1 - dist

            if sim >= pos_sim_threshold:
                if y[i] == y[j]:
                    pos_i_list.append(i)
                    pos_j_list.append(j)
                    pos_weights.append(sim)

            if sim >= neg_sim_threshold:
                if is_selective_hard_negative(intents[i], intents[j], y[i], y[j], confusion_pairs):
                    if y[i] == 1 and y[j] == 0:
                        neg_u_list.append(i)
                        neg_s_list.append(j)
                        neg_weights.append(sim)
                    elif y[i] == 0 and y[j] == 1:
                        neg_u_list.append(j)
                        neg_s_list.append(i)
                        neg_weights.append(sim)

    # PyTorch edge tensors (for the pull/push loss)
    pos_i_t = torch.tensor(pos_i_list, dtype=torch.long, device=device)
    pos_j_t = torch.tensor(pos_j_list, dtype=torch.long, device=device)
    neg_u_t = torch.tensor(neg_u_list, dtype=torch.long, device=device)
    neg_s_t = torch.tensor(neg_s_list, dtype=torch.long, device=device)

    A_pos_torch = _make_row_normalized_sparse(
        rows=pos_i_list + pos_j_list,
        cols=pos_j_list + pos_i_list,
        wts=pos_weights + pos_weights,
        n=n, device=device)

    A_neg_torch = _make_row_normalized_sparse(
        rows=neg_u_list + neg_s_list,
        cols=neg_s_list + neg_u_list,
        wts=neg_weights + neg_weights,
        n=n, device=device)

    print(f"  A_pos nnz: {len(pos_i_list)*2} | A_neg nnz: {len(neg_u_list)*2}")

    return pos_i_t, pos_j_t, neg_u_t, neg_s_t, A_pos_torch, A_neg_torch
