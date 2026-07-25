import torch
import torch.nn.functional as F

from .losses import negative_push_loss_v2, positive_pull_loss, propagation_consistency_loss
from .propagation import signed_embedding_propagation_separate


def train_model(model, optimizer, criterion_intent,
                 X_train_t, y_train_t, int_train_t,
                 A_pos_t, A_neg_t, pos_i, pos_j, neg_u, neg_s,
                 cfg, device):

    print("="*70)
    print("TRAINING: SIGNED GRAPH + SEPARATE EMBEDDING PROPAGATION + INDUCTIVE")
    print("="*70)

    for epoch in range(cfg.EPOCHS):
        model.train()
        optimizer.zero_grad()

        H_local, safety_logits_local, intent_logits = model(X_train_t)
        p_local = torch.sigmoid(safety_logits_local)

        H_prop = signed_embedding_propagation_separate(
            H_local, A_pos_t, A_neg_t,
            alpha_pos=cfg.PROP_ALPHA_POS,
            alpha_neg=cfg.PROP_ALPHA_NEG,
            K=cfg.PROP_K,
            neg_lambda=cfg.NEG_LAMBDA)

        safety_logits_prop, _ = model.forward_from_features(H_prop)

        bce_raw = F.binary_cross_entropy_with_logits(
            safety_logits_local, y_train_t, reduction='none')
        with torch.no_grad():
            hard_fn_mask = (y_train_t == 1) & (p_local < 0.2)
            hard_fp_mask = (y_train_t == 0) & (p_local > 0.5)
        weights = torch.ones_like(y_train_t)
        weights[hard_fn_mask] = 15.0
        weights[hard_fp_mask] = 5.0
        loss_safety = (weights * bce_raw).mean()

        loss_intent = criterion_intent(intent_logits, int_train_t)
        loss_pull   = positive_pull_loss(p_local, pos_i, pos_j, device)
        loss_push   = negative_push_loss_v2(p_local, neg_u, neg_s, device, margin=cfg.PUSH_MARGIN)
        loss_prop   = propagation_consistency_loss(safety_logits_prop, y_train_t, cfg.POS_WEIGHT, device)

        total_loss = (
            loss_safety
            + cfg.LAMBDA_INTENT      * loss_intent
            + cfg.LAMBDA_PULL        * loss_pull
            + (cfg.LAMBDA_PUSH * 2.0)* loss_push
            + cfg.LAMBDA_PROP        * loss_prop
        )

        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Total: {total_loss.item():.4f} | "
                  f"Safety: {loss_safety.item():.4f} | "
                  f"Intent: {loss_intent.item():.4f} | "
                  f"Pull: {loss_pull.item():.4f} | "
                  f"Push: {loss_push.item():.4f} | "
                  f"Prop: {loss_prop.item():.4f}")

    return model
