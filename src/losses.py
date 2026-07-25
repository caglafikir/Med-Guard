import torch
import torch.nn.functional as F


def positive_pull_loss(probs, pos_i, pos_j, device):
    if len(pos_i) == 0:
        return torch.tensor(0.0, device=device)
    return (probs[pos_i] - probs[pos_j]).pow(2).mean()


def negative_push_loss_v2(probs, neg_u, neg_s, device, margin=0.60):
    if len(neg_u) == 0:
        return torch.tensor(0.0, device=device)
    return F.relu(margin - (probs[neg_u] - probs[neg_s])).mean()


def propagation_consistency_loss(logits_prop, y_true, pos_weight, device):
    return F.binary_cross_entropy_with_logits(
        logits_prop, y_true,
        pos_weight=torch.tensor([pos_weight], device=device))
