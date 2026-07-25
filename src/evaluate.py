import torch

from .propagation import inductive_embedding_propagation, signed_embedding_propagation_separate


def run_inference(model, X_train_t, X_cal_t, X_test_t,
                   A_pos_t, A_neg_t,
                   cal_nbr_idx, cal_nbr_wts,
                   test_nbr_idx, test_nbr_wts,
                   cfg, device):

    model.eval()
    with torch.no_grad():
        H_train_local, _, _ = model(X_train_t)
        H_train_pos_final = signed_embedding_propagation_separate(
            H_train_local, A_pos_t, A_neg_t,
            alpha_pos=cfg.PROP_ALPHA_POS, alpha_neg=cfg.PROP_ALPHA_NEG,
            K=cfg.PROP_K, neg_lambda=cfg.NEG_LAMBDA)

        H_neg_msg = torch.sparse.mm(A_neg_t, H_train_pos_final)
        H_train_neg_delta = cfg.PROP_ALPHA_NEG * (H_neg_msg - H_train_pos_final)

        H_cal_local, _, _ = model(X_cal_t)
        H_cal_prop = inductive_embedding_propagation(
            H_cal_local, H_train_pos_final, H_train_neg_delta,
            cal_nbr_idx, cal_nbr_wts, device,
            alpha_pos=cfg.PROP_ALPHA_POS, alpha_neg=cfg.PROP_ALPHA_NEG,
            neg_lambda=cfg.NEG_LAMBDA)
        safety_logits_cal, intent_logits_cal = model.forward_from_features(H_cal_prop)
        probs_cal    = torch.sigmoid(safety_logits_cal).cpu().numpy()
        pred_int_cal = torch.argmax(intent_logits_cal, dim=1).cpu().numpy()

        H_test_local, _, _ = model(X_test_t)
        H_test_prop = inductive_embedding_propagation(
            H_test_local, H_train_pos_final, H_train_neg_delta,
            test_nbr_idx, test_nbr_wts, device,
            alpha_pos=cfg.PROP_ALPHA_POS, alpha_neg=cfg.PROP_ALPHA_NEG,
            neg_lambda=cfg.NEG_LAMBDA)
        safety_logits_test, intent_logits_test = model.forward_from_features(H_test_prop)
        probs_test    = torch.sigmoid(safety_logits_test).cpu().numpy()
        pred_int_test = torch.argmax(intent_logits_test, dim=1).cpu().numpy()

        _, safety_logits_test_local, _ = model(X_test_t)
        probs_test_local = torch.sigmoid(safety_logits_test_local).cpu().numpy()

        _, logits_train_local, _ = model(X_train_t)
        probs_train_local = torch.sigmoid(logits_train_local).cpu().numpy()
        logits_train_prop, _ = model.forward_from_features(H_train_pos_final)
        probs_train_prop = torch.sigmoid(logits_train_prop).cpu().numpy()

    return {
        "probs_cal": probs_cal, "pred_int_cal": pred_int_cal,
        "probs_test": probs_test, "pred_int_test": pred_int_test,
        "probs_test_local": probs_test_local,
        "probs_train_local": probs_train_local,
        "probs_train_prop": probs_train_prop,
    }
