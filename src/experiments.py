"""
Multi-seed experiments: ablation, hyperparameter sweeps, noise robustness, label efficiency.

For each (seed, scenario), the train/cal/test split, graph, and neighbourhoods are built ONCE; all
configurations are trained on the same split. This allows a PAIRED difference to be taken against
the baseline. The seed determines both the split and the model initialization.

Run (from repo root; <package> = the name of the folder containing this file):
    python -m <package>.experiments ablation --seeds 5      # replaces Table 4.4 (multi-seed)
    python -m <package>.experiments alpha    --seeds 5      # replaces Table 4.5 (alpha_pos sweep)
    python -m <package>.experiments k        --seeds 5      # number of propagation steps
    python -m <package>.experiments neg      --seeds 5      # negative delta (effective coefficient c) sweep
    python -m <package>.experiments noise    --seeds 5      # embedding and label noise
    python -m <package>.experiments label_eff --seeds 5     # label efficiency
    python -m <package>.experiments all --quick             # all of them, 2 seeds
    python -m <package>.experiments ablation --scenario feat_noise=1.0   # ablation under another scenario

Options: --data-path, --embed-cache, --out-dir, --scenario kind=level (repeatable)
    kind: clean | feat_noise (sigma) | label_noise (ratio) | label_frac (ratio)

Output (out-dir, default results/): <experiment>_raw.csv (per seed) and <experiment>_summary.csv.

Notes:
  * Metrics are reported at a fixed 0.5 threshold on the TEST set; cal-set metrics (cal_acc, cal_auroc)
    are also written. Make hyperparameter SELECTION based on cal metrics, not the test set.
  * p-values are from a paired t-test. Since many comparisons are made, do not treat a single one as
    proof by itself; look at direction consistency (wins) and effect size.
  * "C1. Random neighbours": the SAME trained model as the reference configuration, but evaluated at
    inference time with random training nodes instead of the real neighbours. This separates whether
    the gain comes from neighbourhood information.
"""
import argparse
import os
import random
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

if __package__ in (None, ""):                 # running directly via `python experiments.py`
    import importlib
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))
    __package__ = os.path.basename(_here)
    importlib.import_module(__package__)

from .config import Config, INTENT_CLASS_WEIGHTS
from .data import embed_texts, load_dataset, split_train_cal_test, to_tensors
from .evaluate import run_inference
from .graph import build_confusion_pairs, build_signed_graph_separate
from .model import SignedGraphSafetyModel
from .propagation import build_inductive_neighbourhood
from .train import train_model

ALL_SEEDS = [7, 13, 23, 45, 121, 42, 123, 3, 99, 2024]
BASE_NAME = "1.  Safety only (baseline)"
CONTROL_TAG = "C1. Random neighbours (control) <- "
METRICS = ("acc", "auroc", "rec_safe", "rec_unsafe", "cal_acc", "cal_auroc")


# =====================================================================
# Helpers
# =====================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


# =====================================================================
# Perturbations (scenarios)
# =====================================================================
def _take(v, idx):
    return v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]


def subsample_labels(split, frac, rng):
    """Keep frac of the training set (stratified by intent); dropped examples are also removed from the graph."""
    if frac >= 1.0:
        return split
    keep = []
    for c in np.unique(split["int_train"]):
        idx_c = np.where(split["int_train"] == c)[0]
        k = min(len(idx_c), max(3, int(round(frac * len(idx_c)))))
        keep.extend(rng.choice(idx_c, size=k, replace=False).tolist())
    keep = np.array(sorted(keep))
    out = dict(split)
    for key in ("X_train", "y_train", "int_train", "texts_train"):
        out[key] = _take(split[key], keep)
    return out


def flip_labels(split, rho, rng):
    """Flip rho of the training safety labels (cal/test stay clean). The intent label is unchanged."""
    if rho <= 0:
        return split
    y = split["y_train"].copy()
    idx = rng.choice(len(y), size=int(round(rho * len(y))), replace=False)
    y[idx] = 1 - y[idx]
    out = dict(split)
    out["y_train"] = y
    return out


def add_feature_noise(split, sigma, rng, emb_std):
    """Add N(0, (sigma*emb_std)^2) noise to the train/cal/test embeddings."""
    out = dict(split)
    for key in ("X_train", "X_cal", "X_test"):
        noise = rng.normal(0.0, sigma * emb_std, size=split[key].shape).astype(np.float32)
        out[key] = split[key] + noise
    return out


def knn_sims(X, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    dist, _ = nbrs.kneighbors(X)
    return 1.0 - dist[:, 1:]                       # the first neighbour is itself


def matched_thresholds(X_clean, X_noisy, cfg):
    """Similarities drop under noisy data; fixed thresholds would empty out the graph. Adjust the
    thresholds to reproduce the clean-data edge ratio (so the experiment measures 'noise', not 'graphlessness')."""
    sc = knn_sims(X_clean, cfg.K_GRAPH).ravel()
    sn = knn_sims(X_noisy, cfg.K_GRAPH).ravel()
    out = {}
    for key, thr in (("pos", cfg.POS_SIM_THRESHOLD), ("neg", cfg.NEG_SIM_THRESHOLD),
                     ("ind", cfg.INDUCTIVE_SIM_THR)):
        q = float((sc >= thr).mean())
        out[key] = float(np.quantile(sn, 1.0 - q)) if q > 0 else 1.01
    return out


def scenario_name(kind, level):
    return "clean" if kind == "clean" else f"{kind}={level:g}"


def parse_scenario(text):
    if text == "clean":
        return ("clean", 0.0)
    kind, _, level = text.partition("=")
    if kind not in ("feat_noise", "label_noise", "label_frac") or not level:
        raise SystemExit(f"Invalid scenario: {text!r}. Example: feat_noise=1.0, label_noise=0.2, label_frac=0.1")
    return (kind, float(level))


# =====================================================================
# Context (split + graph + neighbourhood) and train/evaluate
# =====================================================================
def make_context(cfg, base, kind, level, seed, device):
    split = split_train_cal_test(base["emb"], base["labels"], base["intents"], base["texts"],
                                 cfg.TEST_SIZE, cfg.CAL_SIZE_WITHIN_TRAIN, seed)
    rng = np.random.RandomState(seed)
    thr = dict(pos=cfg.POS_SIM_THRESHOLD, neg=cfg.NEG_SIM_THRESHOLD, ind=cfg.INDUCTIVE_SIM_THR)

    if kind == "label_frac":
        split = subsample_labels(split, level, rng)
    elif kind == "label_noise":
        split = flip_labels(split, level, rng)
    elif kind == "feat_noise" and level > 0:
        noisy = add_feature_noise(split, level, rng, base["emb_std"])
        thr = matched_thresholds(split["X_train"], noisy["X_train"], cfg)
        split = noisy
    elif kind not in ("clean", "feat_noise"):
        raise ValueError(f"Unknown scenario type: {kind}")

    pos_i, pos_j, neg_u, neg_s, A_pos, A_neg = build_signed_graph_separate(
        split["X_train"], split["y_train"], split["int_train"], base["confusion_pairs"], device,
        k_graph=cfg.K_GRAPH, pos_sim_threshold=thr["pos"], neg_sim_threshold=thr["neg"])
    cal_idx, cal_wts = build_inductive_neighbourhood(
        split["X_cal"], split["X_train"], k=cfg.K_INDUCTIVE, sim_threshold=thr["ind"])
    test_idx, test_wts = build_inductive_neighbourhood(
        split["X_test"], split["X_train"], k=cfg.K_INDUCTIVE, sim_threshold=thr["ind"])

    ctx = dict(to_tensors(split, device))
    ctx.update(pos_i=pos_i, pos_j=pos_j, neg_u=neg_u, neg_s=neg_s, A_pos=A_pos, A_neg=A_neg,
               cal_idx=cal_idx, cal_wts=cal_wts, test_idx=test_idx, test_wts=test_wts,
               y_cal=split["y_cal"], y_test=split["y_test"])
    ctx["diag"] = dict(n_train=int(len(split["X_train"])), pos_nnz=int(A_pos._nnz()),
                       cov_test=float(np.mean([len(x) > 0 for x in test_idx])))
    return ctx


def fit(cfg_v, ctx, seed, device):
    set_seed(seed)
    model = SignedGraphSafetyModel(in_dim=ctx["X_train_t"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_v.LR, weight_decay=cfg_v.WEIGHT_DECAY)
    criterion_intent = nn.CrossEntropyLoss(weight=torch.tensor(INTENT_CLASS_WEIGHTS, device=device))
    train_model(model, optimizer, criterion_intent,
                ctx["X_train_t"], ctx["y_train_t"], ctx["int_train_t"],
                ctx["A_pos"], ctx["A_neg"], ctx["pos_i"], ctx["pos_j"], ctx["neg_u"], ctx["neg_s"],
                cfg_v, device, verbose=False)
    return model


def evaluate(model, cfg_v, ctx, device, cal_idx=None, test_idx=None):
    out = run_inference(
        model, ctx["X_train_t"], ctx["X_cal_t"], ctx["X_test_t"], ctx["A_pos"], ctx["A_neg"],
        ctx["cal_idx"] if cal_idx is None else cal_idx, ctx["cal_wts"],
        ctx["test_idx"] if test_idx is None else test_idx, ctx["test_wts"],
        cfg_v, device)
    y_te, p_te = ctx["y_test"], out["probs_test"]
    y_ca, p_ca = ctx["y_cal"], out["probs_cal"]
    yp = (p_te >= 0.5).astype(int)
    return dict(
        acc=accuracy_score(y_te, yp),
        rec_safe=recall_score(y_te, yp, pos_label=0, zero_division=0),
        rec_unsafe=recall_score(y_te, yp, pos_label=1, zero_division=0),
        auroc=safe_auc(y_te, p_te),
        cal_acc=accuracy_score(y_ca, (p_ca >= 0.5).astype(int)),
        cal_auroc=safe_auc(y_ca, p_ca))


def random_neighbours(idx_lists, n_train, rng):
    """For each node, as many RANDOM training nodes as it has real neighbours (weights stay the same)."""
    return [rng.choice(n_train, size=len(il), replace=False).tolist() if len(il) else []
            for il in idx_lists]


# =====================================================================
# Configuration families
# =====================================================================
def base_variant(cfg):
    """Safety loss only: propagation, negative delta, and auxiliary losses are disabled."""
    return replace(cfg, LAMBDA_PROP=0.0, LAMBDA_INTENT=0.0, LAMBDA_PULL=0.0, LAMBDA_PUSH=0.0,
                   NEG_LAMBDA=0.0, NEG_LAMBDA_INFER=None, INFER_PROP=False)


def prop_variant(cfg, **kw):
    """Baseline + positive propagation only (training loss + inference-time spread)."""
    d = dict(LAMBDA_PROP=cfg.LAMBDA_PROP, INFER_PROP=True)
    d.update(kw)
    return replace(base_variant(cfg), **d)


# Each function returns: (rows, ref_name). Row = (name, cfg, add_random_neighbour_control)
def rows_ablation(cfg):
    rows = [(BASE_NAME, base_variant(cfg), False),
            ("1b. + Prop loss only (no inference prop.)", prop_variant(cfg, INFER_PROP=False), False)]
    ref_name = None
    for i, lp in enumerate(sorted({0.5, 0.8, 1.0, cfg.LAMBDA_PROP})):
        name = f"2{'abcd'[i]}. + Propagation (lambda_prop={lp:g})"
        is_ref = abs(lp - cfg.LAMBDA_PROP) < 1e-12
        rows.append((name, prop_variant(cfg, LAMBDA_PROP=lp), is_ref))
        if is_ref:
            ref_name = name
    for j, ln in enumerate((0.5, 0.8, 1.0)):
        rows.append((f"3{'abc'[j]}. + Neg delta (lambda_neg={ln:g})", prop_variant(cfg, NEG_LAMBDA=ln), False))
    with_neg = prop_variant(cfg, NEG_LAMBDA=cfg.NEG_LAMBDA)
    with_int = replace(with_neg, LAMBDA_INTENT=cfg.LAMBDA_INTENT)
    with_pull = replace(with_int, LAMBDA_PULL=cfg.LAMBDA_PULL)
    full = replace(with_pull, LAMBDA_PUSH=cfg.LAMBDA_PUSH)
    rows += [("5.  + Intent loss", with_int, False),
             ("6.  + Pull loss", with_pull, False),
             ("7.  Full model (+ Push loss)", full, False),
             ("7b. Full model without neg delta", replace(full, NEG_LAMBDA=0.0), False)]
    return rows, ref_name


def rows_alpha(cfg):
    rows = [(BASE_NAME, base_variant(cfg), False)]
    for a in (0.1, 0.2, 0.3, 0.5, 0.8, 0.9):
        rows.append((f"alpha_pos={a:g} (teleport)", prop_variant(cfg, PROP_ALPHA_POS=a), False))
    return rows, None


def rows_k(cfg):
    rows = [(BASE_NAME, base_variant(cfg), False)]
    for k in (1, 2, 3):
        rows.append((f"K={k}", prop_variant(cfg, PROP_K=k), False))
    return rows, None


def rows_neg(cfg):
    ref = prop_variant(cfg, PROP_ALPHA_NEG=1.0, NEG_LAMBDA=0.0)     # alpha_neg=1 -> effective coefficient c = NEG_LAMBDA
    ref_name = "P   neg c=0 (ref)"
    rows = [(BASE_NAME, base_variant(cfg), False), (ref_name, ref, False)]
    for c in (0.02, 0.05, 0.1, 0.3, 0.5, 1.0):
        rows.append((f"P   neg c={c:g}", replace(ref, NEG_LAMBDA=c), False))
    rows.append(("P   neg c=0.1 train only", replace(ref, NEG_LAMBDA=0.1, NEG_LAMBDA_INFER=0.0), False))
    rows.append(("P   neg c=0.1 inference only", replace(ref, NEG_LAMBDA=0.0, NEG_LAMBDA_INFER=0.1), False))
    return rows, ref_name


def rows_robust(cfg):
    alt = 0.8 if abs(cfg.PROP_ALPHA_POS - 0.8) > 1e-12 else 0.5
    rows = [(BASE_NAME, base_variant(cfg), False),
            ("P1. Prop loss only (train)", prop_variant(cfg, INFER_PROP=False), False),
            (f"P2. Prop train+inference (alpha={cfg.PROP_ALPHA_POS:g})", prop_variant(cfg), True),
            (f"P3. Prop train+inference (alpha={alt:g})", prop_variant(cfg, PROP_ALPHA_POS=alt), False)]
    return rows, None


CLEAN = [("clean", 0.0)]
EXPERIMENTS = {
    "ablation":  dict(scenarios=CLEAN, rows=rows_ablation),
    "alpha":     dict(scenarios=CLEAN, rows=rows_alpha),
    "k":         dict(scenarios=CLEAN, rows=rows_k),
    "neg":       dict(scenarios=CLEAN, rows=rows_neg),
    "noise":     dict(scenarios=CLEAN + [("feat_noise", s) for s in (0.5, 1.0, 2.0, 3.0)]
                                      + [("label_noise", r) for r in (0.05, 0.1, 0.2, 0.3)],
                      rows=rows_robust),
    "label_eff": dict(scenarios=[("label_frac", f) for f in (0.05, 0.1, 0.25, 0.5, 1.0)],
                      rows=rows_robust),
}


# =====================================================================
# Execution and summary
# =====================================================================
def run_experiment(name, spec, cfg, base, seeds, device, scenarios=None):
    scenarios = scenarios or spec["scenarios"]
    variants, ref_name = spec["rows"](cfg)
    records = []
    t0 = time.time()
    for seed in seeds:
        for kind, level in scenarios:
            scen = scenario_name(kind, level)
            ctx = make_context(cfg, base, kind, level, seed, device)
            d = ctx["diag"]
            for vname, vcfg, control in variants:
                model = fit(vcfg, ctx, seed, device)
                common = dict(experiment=name, scenario=scen, seed=seed,
                              cov_test=d["cov_test"], n_train=d["n_train"])
                records.append(dict(config=vname, **evaluate(model, vcfg, ctx, device), **common))
                if control:
                    rng = np.random.RandomState(seed + 999)
                    ci = random_neighbours(ctx["cal_idx"], d["n_train"], rng)
                    ti = random_neighbours(ctx["test_idx"], d["n_train"], rng)
                    records.append(dict(config=CONTROL_TAG + vname,
                                        **evaluate(model, vcfg, ctx, device, ci, ti), **common))
                del model
            print(f"  [{name}] seed={seed} {scen}: n_train={d['n_train']} A_pos_nnz={d['pos_nnz']} "
                  f"test_coverage={d['cov_test']:.2f}  ({time.time() - t0:.0f} s)", flush=True)
            del ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(records), ref_name


def summarize(df, ref_name=None):
    out = []
    for scen, g in df.groupby("scenario", sort=False):
        by = {n: gg.set_index("seed") for n, gg in g.groupby("config", sort=False)}
        base = by.get(BASE_NAME)
        ref = by.get(ref_name) if ref_name else None
        for cname, gg in by.items():
            row = dict(scenario=scen, config=cname, n=len(gg), cov_test=gg["cov_test"].mean())
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_std"] = gg[m].mean(), gg[m].std()
            for tag, r in (("b", base), ("r", ref)):
                if r is None:
                    continue
                for m in ("acc", "auroc", "rec_unsafe"):
                    d = (gg[m] - r[m]).dropna()
                    row[f"d{tag}_{m}"] = d.mean()
                    row[f"d{tag}_{m}_std"] = d.std()
                    row[f"d{tag}_{m}_wins"] = int((d > 0).sum())
                    row[f"d{tag}_{m}_p"] = (stats.ttest_1samp(d, 0.0).pvalue
                                            if len(d) > 2 and d.std() > 0 else np.nan)
            out.append(row)
    return pd.DataFrame(out)


def _fmt(row, tag, m):
    key = f"d{tag}_{m}"
    if key not in row or pd.isna(row[key]):
        return "-"
    p = row[f"{key}_p"]
    ps = "  -  " if pd.isna(p) else f"{p:.3f}"
    return f"{100 * row[key]:+.2f}±{100 * row[key + '_std']:.2f} p={ps} {row[key + '_wins']}/{row['n']}"


def print_summary(summary, ref_name):
    cols = ["config", "acc", "auroc", "unsafeRec", "safeRec", "d_acc (pp)", "d_auroc (pp)"]
    if ref_name:
        cols.append("d_auroc vs REF")
    for scen, g in summary.groupby("scenario", sort=False):
        print(f"\n{'=' * 118}\nScenario: {scen}   (n_seed={g['n'].iloc[0]}, test coverage={g['cov_test'].iloc[0]:.2f})"
              f"\n{'=' * 118}")
        rows = []
        for _, r in g.iterrows():
            row = [r["config"], f"{r['acc_mean']:.4f}±{r['acc_std']:.4f}", f"{r['auroc_mean']:.4f}",
                   f"{r['rec_unsafe_mean']:.4f}", f"{r['rec_safe_mean']:.4f}",
                   _fmt(r, "b", "acc"), _fmt(r, "b", "auroc")]
            if ref_name:
                row.append(_fmt(r, "r", "auroc"))
            rows.append(row)
        print(pd.DataFrame(rows, columns=cols).to_string(index=False))
    print("\nDifference: PAIRED against the baseline (same seed/split), score; mean±std, paired t-test p, "
          "wins = number of seeds where the difference is >0.")
    print("Many comparisons are made: p<0.05 alone is not proof; look at direction consistency and effect size.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-seed experiments")
    ap.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds to use (max 10)")
    ap.add_argument("--quick", action="store_true", help="quick trial with 2 seeds")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--embed-cache", default="embeddings_cache.npz")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--scenario", action="append", default=None,
                    help="override the scenario (repeatable): clean | feat_noise=1.0 | label_noise=0.2 | label_frac=0.1")
    args = ap.parse_args(argv)

    cfg = Config()
    if args.data_path:
        cfg = replace(cfg, DATA_PATH=args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = ALL_SEEDS[:2] if args.quick else ALL_SEEDS[:max(1, min(args.seeds, len(ALL_SEEDS)))]
    scenarios = [parse_scenario(s) for s in args.scenario] if args.scenario else None

    df = load_dataset(cfg.DATA_PATH)
    texts = df["request"].tolist()
    emb, _ = embed_texts(texts, cfg.EMBED_MODEL, cache_path=args.embed_cache)
    emb = np.asarray(emb, dtype=np.float32)
    base = dict(emb=emb, labels=df["label"].values, intents=df["intent_idx"].values, texts=texts,
                emb_std=float(emb.std()), confusion_pairs=build_confusion_pairs())
    print(f"Device: {device}  |  {len(df)} samples  |  seeds: {seeds}")

    os.makedirs(args.out_dir, exist_ok=True)
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        print(f"\n########## {name} ##########", flush=True)
        raw, ref_name = run_experiment(name, EXPERIMENTS[name], cfg, base, seeds, device, scenarios)
        summary = summarize(raw, ref_name)
        raw.to_csv(os.path.join(args.out_dir, f"{name}_raw.csv"), index=False, encoding="utf-8")
        summary.to_csv(os.path.join(args.out_dir, f"{name}_summary.csv"), index=False, encoding="utf-8")
        print_summary(summary, ref_name)
        print(f"\nSaved: {args.out_dir}/{name}_raw.csv, {args.out_dir}/{name}_summary.csv")


if __name__ == "__main__":
    main()