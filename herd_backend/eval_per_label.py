"""
Quick per-label evaluation — runs one fold of the full pipeline and extracts per-label metrics.
"""
import os, sys, json, random
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import AsymmetricLoss
from train_compound_ecc import (CompoundToxModel, build_compound_labels,
    train_compound_fold, compute_prior, train_compound_aggregator,
    train_drug_projection, compute_drug_prior, compute_mi_auc_chains,
    train_label_protonet, CompoundAttentionAggregator)
from model_drug_tower import DrugTower

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_SPLITS = 5
N_LABELS = 5
LABEL_NAMES = ["Hepatotoxicity", "Nephrotoxicity", "Cardiotoxicity", "Neurotoxicity", "Hematotoxicity"]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

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
        if cid < n_compounds:
            compound_mh[i, cid] = 1.0

compound_fps = np.load(os.path.join(BASE, "..", "数据爬取", "compound_fps.npy"))
label_strs = [''.join(str(int(v)) for v in d['label']) for d in all_data]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(skf.split(all_data, label_strs))

# Compute label prevalence
total = len(labels)
print("=== Label Prevalence ===")
for li in range(N_LABELS):
    pos = labels[:, li].sum()
    pos_rate = pos / total
    print(f"  {LABEL_NAMES[li]:20s}: {pos:3.0f} / {total} ({pos_rate:.1%})")

# Load Drug Tower
drug_tower = DrugTower(n_labels=8).to(device)
ckpt = torch.load(os.path.join(BASE, "dataset", "drug_tower_encoder.pt"), map_location=device)
drug_tower.load_state_dict(ckpt)
drug_tower.eval()
for p in drug_tower.parameters():
    p.requires_grad = False

# Run one fold for per-label metrics
print(f"\n=== Running Fold 1/{N_SPLITS} ===")
fold = 0
train_idx, val_idx = splits[fold]
train_herbs = [all_data[i] for i in train_idx]
train_features = features[train_idx]; train_labels = labels[train_idx]
val_features = features[val_idx]; val_labels = labels[val_idx]

comp_train = compound_mh[train_idx]
intersection = comp_train @ comp_train.T
row_sum = comp_train.sum(axis=1, keepdims=True)
union = row_sum + row_sum.T - intersection
comp_sim_train = torch.tensor(intersection / np.maximum(union, 1), dtype=torch.float32)

# Compound model
c_labels = build_compound_labels(train_herbs, n_compounds)
comp_model = train_compound_fold(compound_fps, c_labels, train_herbs, epochs=40)
aggregator = CompoundAttentionAggregator(prior_dim=5, hidden=32).to(device)
aggregator = train_compound_aggregator(comp_model, aggregator, compound_fps, train_herbs)
tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data, aggregator)
va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data, aggregator)

# Drug prior
drug_proj = nn.Linear(256, 5).to(device)
drug_proj = train_drug_projection(drug_tower, drug_proj, compound_fps, train_herbs)
tr_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, train_idx, all_data)
va_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, val_idx, all_data)

# Chain prediction
chains = compute_mi_auc_chains(train_features, train_labels.astype(int), n_labels=N_LABELS, seed=SEED+fold)

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
print(f"\n=== Per-Label Results (Fold 1) ===")
print(f"{'Label':<20s} {'AUC':>8s} {'F1@0.5':>8s} {'F1@best':>8s} {'Prevalence':>10s}")
print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
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
    pos_rate = val_labels[:, li].mean()
    print(f"{LABEL_NAMES[li]:<20s} {auc:8.4f} {f1_fixed:8.4f} {best_f1:8.4f} {pos_rate:10.1%}")

# Macro average
try:
    macro_auc = roc_auc_score(val_labels, all_val_probs, average='macro')
except ValueError:
    macro_auc = float('nan')
print(f"\n{'Macro Average':<20s} {macro_auc:8.4f}")
print(f"\nDone. Use seed={SEED}, fold 1/{N_SPLITS} results for Table 4 reference.")
