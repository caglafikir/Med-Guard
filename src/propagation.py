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
    for _ in range(K - 1):
        msg_pos = torch.sparse.mm(A_pos, H_pos)
        H_pos = (1 - alpha_pos) * msg_pos + alpha_pos * H_0.detach()

    msg_pos_final = torch.sparse.mm(A_pos, H_pos.detach())
    H_pos_final = (1 - alpha_pos) * msg_pos_final + alpha_pos * H_0

    with torch.no_grad():
        H_neg_msg = torch.sparse.mm(A_neg, H_pos_final.detach())
        delta_neg = alpha_neg * (H_neg_msg - H_pos_final.detach())

    return H_pos_final - neg_lambda * delta_neg


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
    print(f"  Connected: {n_connected}/{len(X_query)} | "
          f"avg. neighbours: {avg_nbr:.1f}")
    return nbr_indices, nbr_weights


def inductive_embedding_propagation(Hq, Htr_pos, Htr_delta,
                                     nbr_indices, nbr_weights, device,
                                     alpha_pos=0.2,
                                     neg_lambda=0.5):

    Ho = Hq.clone()
    for i in range(len(Hq)):
        il, wl = nbr_indices[i], nbr_weights[i]
        if not il:
            continue
        wts = torch.tensor(wl, dtype=torch.float32, device=device)
        wts = wts / wts.sum()
        msg   = (wts.unsqueeze(1) * Htr_pos[il]).sum(0)
        delta = (wts.unsqueeze(1) * Htr_delta[il]).sum(0)  
        Ho[i] = alpha_pos * Hq[i] + (1 - alpha_pos) * msg - neg_lambda * delta
    return Ho
