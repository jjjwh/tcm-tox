"""
per_label_5fold.py — Run 5-fold CV with per-label metrics extraction.
Saves intermediate results per fold so partial data survives crashes.
"""
import os, sys, json, random, time, copy
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AsymmetricLoss
from train_compound_ecc import (CompoundToxModel, build_compound_labels,
    train_compound_fold, compute_prior, train_compound_aggregator,
    train_drug_projection, compute_drug_prior, compute_mi_auc_chains,
    train_label_protonet, find_best_thresholds)
from model import CompoundAttentionAggregator
from model_drug_tower import DrugTower

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "per_label_results.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_SPLITS = 5
N_LABELS = 5
LABEL_NAMES = ["Hepatotoxicity","Nephrotoxicity","Cardiotoxicity","Neurotoxicity","Hematotoxicity"]

print(f"Device: {device}")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Load data
all_json = os.path.join(BASE, "dataset/output/all_herbs.json")
with open(all_json, 'r', encoding='utf-8') as f:
    all_data = json.load(f)
features = np.array([d["feature_vector"] for d in all_data], dtype=np.float32)
labels = np.array([d["label"] for d in all_data], dtype=np.float32)
with open(os.path.join(BASE, "dataset/output/compound2id.json"), 'r') as f:
    c2id = json.load(f)
n_compounds = len(c2id)
compound_mh = np.zeros((len(all_data), n_compounds), dtype=np.float32)
for i, d in enumerate(all_data):
    for cid in d["compound_ids"]:
        if cid < n_compounds: compound_mh[i, cid] = 1.0
compound_fps = np.load(os.path.join(BASE, "..", "数据爬取", "compound_fps.npy"))
label_strs = [''.join(str(int(v)) for v in d['label']) for d in all_data]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(skf.split(all_data, label_strs))

# Label prevalence (global)
prevalence = []
for li in range(N_LABELS):
    prevalence.append(float(labels[:, li].mean()))
print("=== Label Prevalence ===")
for li in range(N_LABELS):
    print(f"  {LABEL_NAMES[li]:20s}: {prevalence[li]:.1%}")

# Load Drug Tower
drug_tower = DrugTower(n_labels=8).to(device)
ckpt = torch.load(os.path.join(BASE, "dataset", "drug_tower_encoder.pt"), map_location=device)
drug_tower.load_state_dict(ckpt)
drug_tower.eval()
for p in drug_tower.parameters():
    p.requires_grad = False
print("Drug Tower loaded.")

# Load or init results
if os.path.exists(OUTPUT):
    with open(OUTPUT) as f:
        saved = json.load(f)
    fold_results = saved["fold_results"]
    completed_folds = {r["fold"] for r in fold_results}
    print(f"Resuming from {OUTPUT}: {len(fold_results)} folds complete")
else:
    fold_results = []
    completed_folds = set()

for fold, (train_idx, val_idx) in enumerate(splits):
    if fold in completed_folds:
        print(f"\nFold {fold+1}: SKIP (already complete)")
        continue

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/{N_SPLITS} (train={len(train_idx)}, val={len(val_idx)})")
    print(f"{'='*60}")

    train_herbs = [all_data[i] for i in train_idx]
    train_features = features[train_idx]; train_labels = labels[train_idx]
    val_features = features[val_idx]; val_labels = labels[val_idx]

    comp_train = compound_mh[train_idx]
    intersection = comp_train @ comp_train.T
    row_sum = comp_train.sum(axis=1, keepdims=True)
    union = row_sum + row_sum.T - intersection
    comp_sim_train = torch.tensor(intersection / np.maximum(union, 1), dtype=torch.float32)

    # Compound model + aggregator
    print("  Training compound model...")
    c_labels = build_compound_labels(train_herbs, n_compounds)
    comp_model = train_compound_fold(compound_fps, c_labels, train_herbs, epochs=40)
    aggregator = CompoundAttentionAggregator(prior_dim=5, hidden=32).to(device)
    print("  Training aggregator...")
    aggregator = train_compound_aggregator(comp_model, aggregator, compound_fps, train_herbs)
    tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data, aggregator)
    va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data, aggregator)

    # Drug prior
    print("  Training drug projection...")
    drug_proj = nn.Linear(256, 5).to(device)
    drug_proj = train_drug_projection(drug_tower, drug_proj, compound_fps, train_herbs)
    tr_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, train_idx, all_data)
    va_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, val_idx, all_data)

    # ECC chains
    print("  Computing chains...")
    chains = compute_mi_auc_chains(train_features, train_labels.astype(int), n_labels=N_LABELS, seed=SEED+fold)

    # P1
    print("  P1 chain prediction...")
    p1_ensemble = np.zeros((len(val_idx), N_LABELS), dtype=np.float32)
    p1_chain_probs = {}
    for ci, chain_order in enumerate(chains):
        val_probs = np.zeros((len(val_idx), N_LABELS), dtype=np.float32)
        for step, li in enumerate(chain_order):
            if step == 0:
                tr_inp, va_inp = train_features, val_features
            else:
                prev = chain_order[:step]
                tr_inp = np.concatenate([train_features, train_labels[:, prev]], axis=1)
                va_inp = np.concatenate([val_features, val_probs[:, prev]], axis=1)
            val_probs[:, li] = train_label_protonet(
                tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                comp_sim_train, device,
                seed=SEED + fold * 100 + li * 7 + ci * 1000,
                tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug,
                use_contrastive=True, epochs=200)
        p1_ensemble += val_probs
        p1_chain_probs[ci] = val_probs.copy()
    p1_ensemble /= len(chains)

    # P2
    print("  P2 chain prediction...")
    p2_ensemble = np.zeros((len(val_idx), N_LABELS), dtype=np.float32)
    for ci, chain_order in enumerate(chains):
        p1_val = p1_chain_probs[ci]
        val_probs2 = np.zeros((len(val_idx), N_LABELS), dtype=np.float32)
        for step, li in enumerate(chain_order):
            if step == 0:
                tr_inp, va_inp = train_features, val_features
            else:
                prev = chain_order[:step]
                soft_cond = (train_labels[:, prev] * 0.8 + 0.1).astype(np.float32)
                tr_inp = np.concatenate([train_features, soft_cond], axis=1)
                va_inp = np.concatenate([val_features, p1_val[:, prev]], axis=1)
            val_probs2[:, li] = train_label_protonet(
                tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                comp_sim_train, device,
                seed=SEED + fold * 100 + li * 7 + ci * 1000 + 9999,
                tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug,
                use_contrastive=True, epochs=200)
        p2_ensemble += val_probs2
    p2_ensemble /= len(chains)

    all_val_probs = (p1_ensemble + p2_ensemble) / 2

    # Per-label metrics
    per_label = []
    for li in range(N_LABELS):
        try:
            auc = roc_auc_score(val_labels[:, li], all_val_probs[:, li])
        except ValueError:
            auc = float('nan')
        pred_fixed = (all_val_probs[:, li] > 0.5).astype(int)
        f1_fixed = f1_score(val_labels[:, li], pred_fixed, zero_division=0)
        best_f1, best_t = 0, 0.5
        for t in np.linspace(0.05, 0.90, 18):
            p = (all_val_probs[:, li] > t).astype(int)
            f = f1_score(val_labels[:, li], p, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        per_label.append({"name": LABEL_NAMES[li], "auc": auc, "f1_fixed": f1_fixed,
                          "f1_best": best_f1, "best_threshold": best_t})

    # Macro AUC
    try:
        macro_auc = roc_auc_score(val_labels, all_val_probs, average='macro')
    except ValueError:
        macro_auc = float('nan')

    # Macro/Micro F1
    fixed_preds = (all_val_probs > 0.5).astype(int)
    macro_f1_fixed = f1_score(val_labels, fixed_preds, average='macro', zero_division=0)
    micro_f1_fixed = f1_score(val_labels, fixed_preds, average='micro', zero_division=0)
    thresholds = find_best_thresholds(val_labels, all_val_probs, N_LABELS)
    best_preds = np.zeros_like(all_val_probs, dtype=int)
    for i in range(N_LABELS):
        best_preds[:, i] = (all_val_probs[:, i] > thresholds[i]).astype(int)
    macro_f1_best = f1_score(val_labels, best_preds, average='macro', zero_division=0)
    micro_f1_best = f1_score(val_labels, best_preds, average='micro', zero_division=0)

    fold_result = {
        "fold": fold,
        "macro_auc": macro_auc,
        "macro_f1_fixed": macro_f1_fixed,
        "micro_f1_fixed": micro_f1_fixed,
        "macro_f1_best": macro_f1_best,
        "micro_f1_best": micro_f1_best,
        "per_label": per_label,
        "time_s": round(time.time() - t0, 1)
    }
    fold_results.append(fold_result)

    # Per-fold summary
    print(f"\n  Fold {fold+1} macro AUC: {macro_auc:.4f}")
    for pl in per_label:
        print(f"    {pl['name']:20s} AUC={pl['auc']:.4f}  F1@0.5={pl['f1_fixed']:.4f}  F1@best={pl['f1_best']:.4f}")
    print(f"  Fold time: {fold_result['time_s']:.0f}s")

    # Save after each fold
    with open(OUTPUT, 'w') as f:
        json.dump({"prevalence": prevalence, "label_names": LABEL_NAMES, "fold_results": fold_results}, f, indent=2)

# Compute summary
print(f"\n{'='*60}")
print(f"5-Fold CV Summary (seed={SEED})")
print(f"{'='*60}")
for metric in ['macro_auc','macro_f1_fixed','micro_f1_fixed','macro_f1_best','micro_f1_best']:
    vals = [r[metric] for r in fold_results if not (isinstance(r[metric], float) and np.isnan(r[metric]))]
    if vals:
        print(f"  {metric:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

print(f"\n{'Label':<22s} {'AUC':>10s} {'F1@0.5':>10s} {'F1@best':>10s}")
print(f"{'-'*22} {'-'*10} {'-'*10} {'-'*10}")
for li in range(N_LABELS):
    aucs = [r['per_label'][li]['auc'] for r in fold_results]
    f1f = [r['per_label'][li]['f1_fixed'] for r in fold_results]
    f1b = [r['per_label'][li]['f1_best'] for r in fold_results]
    aucs = [v for v in aucs if not np.isnan(v)]
    f1f = [v for v in f1f if not np.isnan(v)]
    f1b = [v for v in f1b if not np.isnan(v)]
    print(f"{LABEL_NAMES[li]:<22s} {np.mean(aucs):.4f}±{np.std(aucs):.2f} {np.mean(f1f):.4f}±{np.std(f1f):.2f} {np.mean(f1b):.4f}±{np.std(f1b):.2f}")

print(f"\nResults saved to {OUTPUT}")
