import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def signed_embedding_propagation_separate(H, A_pos, A_neg,
                                           alpha_pos=0.2,
                                           alpha_neg=0.1,
                                           K=2,
                                           neg_lambda=0.5):

    H_0 = H

    H_pos = H.detach()
    for _ in range(K):
        msg_pos = torch.sparse.mm(A_pos, H_pos)
        H_pos   = (1 - alpha_pos) * H_0.detach() + alpha_pos * msg_pos

    msg_pos_final = torch.sparse.mm(A_pos, H_pos.detach())
    H_pos_final   = (1 - alpha_pos) * H_0 + alpha_pos * msg_pos_final

    with torch.no_grad():
        H_neg_msg = torch.sparse.mm(A_neg, H_pos_final.detach())
        delta_neg = alpha_neg * (H_neg_msg - H_pos_final.detach())

    H_final = H_pos_final - neg_lambda * delta_neg

    return H_final


def build_inductive_neighbourhood(X_query, X_train,
                                   k=10, sim_threshold=0.60):
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_train)
    distances, indices = nbrs.kneighbors(X_query)

    nbr_indices = []
    nbr_weights = []

    for i in range(len(X_query)):
        idx_list, wt_list = [], []
        for idx, dist in enumerate(distances[i]):
            sim = 1 - dist
            if sim >= sim_threshold:
                idx_list.append(indices[i][idx])
                wt_list.append(sim)
        nbr_indices.append(idx_list)
        nbr_weights.append(wt_list)

    n_connected = sum(1 for n in nbr_indices if len(n) > 0)
    avg_nbr     = np.mean([len(n) for n in nbr_indices])
    print(f"  Bağlanan: {n_connected}/{len(X_query)} | "
          f"ort. komşu: {avg_nbr:.1f}")
    return nbr_indices, nbr_weights


def inductive_embedding_propagation(H_query, H_train_pos, H_train_neg_delta,
                                     nbr_indices, nbr_weights, device,
                                     alpha_pos=0.2, alpha_neg=0.1,
                                     neg_lambda=0.5):

    H_out = H_query.clone()

    for i in range(len(H_query)):
        idx_list = nbr_indices[i]
        wt_list  = nbr_weights[i]

        if len(idx_list) == 0:
            continue

        wts   = torch.tensor(wt_list, dtype=torch.float32, device=device)
        wts   = wts / wts.sum()

        msg_pos = (wts.unsqueeze(1) * H_train_pos[idx_list]).sum(0)

        delta   = (wts.unsqueeze(1) * H_train_neg_delta[idx_list]).sum(0)

        H_out[i] = ((1 - alpha_pos) * H_query[i]
                    + alpha_pos * msg_pos
                    - neg_lambda * alpha_neg * delta)

    return H_out
