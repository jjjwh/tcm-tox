"""
ablation_per_label.py — Uses train_ablation.py functions (correct code path)
to compute per-label AUC/F1@0.5/F1@best for the Full Model configuration.
"""
import os, sys, random, json, argparse, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, mutual_info_score
sys.path.insert(0, '.')
from model import ProtoNet, GatedPriorProtoNet, AsymmetricLoss, knowledge_sim_loss, CompoundAttentionAggregator
from model_drug_tower import DrugTower

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT= os.path.join(BASE, "ablation_per_label_results.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
LABEL_NAMES = ["Hepatotoxicity","Nephrotoxicity","Cardiotoxicity","Neurotoxicity","Hematotoxicity"]

# ===== COPY of train_ablation.py functions (same code path as ABLATION_RESULTS) =====

def build_compound_labels(herbs, n_compounds):
    c_labels = np.zeros((n_compounds, 5), dtype=np.float32)
    for h in herbs:
        for cid in h["compound_ids"]:
            if cid < n_compounds:
                c_labels[cid] = np.maximum(c_labels[cid], h["label"])
    return c_labels

class CompoundToxModel(nn.Module):
    def __init__(self, fp_dim=1024, hidden=256, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 5)
    def forward(self, fp): return self.head(self.encoder(fp))

def train_compound_fold(compound_fps, c_labels, train_herbs, epochs=40):
    train_cids = set()
    for h in train_herbs:
        for cid in h["compound_ids"]:
            if cid < len(compound_fps) and c_labels[cid].sum() > 0:
                train_cids.add(cid)
    train_cids = sorted(train_cids)
    if len(train_cids) < 10: return None
    fps = torch.tensor(compound_fps[train_cids], dtype=torch.float32)
    lbls = torch.tensor(c_labels[train_cids], dtype=torch.float32)
    model = CompoundToxModel(dropout=0.4).to(device)
    crit = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    loader = DataLoader(TensorDataset(fps, lbls), batch_size=min(64, len(fps)), shuffle=True, drop_last=True)
    for ep in range(epochs):
        model.train()
        for fv, lb in loader:
            fv, lb = fv.to(device), lb.to(device)
            loss = crit(model(fv), lb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    model.eval()
    return model

def compute_prior(model, compound_fps, herb_indices, herbs, aggregator=None):
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    prior = np.zeros((len(herb_indices), 5), dtype=np.float32)
    if model is None: return prior
    model.eval()
    if aggregator is not None: aggregator.eval()
    with torch.no_grad():
        for j, hi in enumerate(herb_indices):
            cids = [c for c in herbs[hi]["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                probs = torch.sigmoid(model(c_fps_t[cids]))
                if aggregator is not None:
                    prior[j] = aggregator(probs)[0].cpu().numpy()
                else:
                    prior[j] = probs.cpu().numpy().mean(axis=0)
    return prior

def compute_drug_prior(drug_tower, drug_proj, compound_fps, herb_indices, herbs):
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    prior = np.zeros((len(herb_indices), 5), dtype=np.float32)
    if drug_tower is None: return prior
    drug_tower.eval(); drug_proj.eval()
    with torch.no_grad():
        for j, hi in enumerate(herb_indices):
            cids = [c for c in herbs[hi]["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                _, emb = drug_tower(c_fps_t[cids], return_embed=True)
                drug_view = emb.mean(dim=0)
                prior[j] = drug_proj(drug_view).cpu().numpy()
    return prior

def train_compound_aggregator(model, aggregator, compound_fps, train_herbs, epochs=50):
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    model.eval(); aggregator.train()
    herb_comp_probs, label_map = {}, {}
    with torch.no_grad():
        for idx, h in enumerate(train_herbs):
            cids = [c for c in h["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                herb_comp_probs[idx] = torch.sigmoid(model(c_fps_t[cids]))
                label_map[idx] = h["label"]
    crit = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
    opt = torch.optim.AdamW(aggregator.parameters(), lr=3e-3, weight_decay=1e-4)
    accum_steps = 4
    for _ in range(epochs):
        indices = list(herb_comp_probs.keys())
        random.shuffle(indices)
        opt.zero_grad()
        for i, k in enumerate(indices):
            probs, lbl = herb_comp_probs[k], torch.tensor(label_map[k], dtype=torch.float32).to(device)
            agg, _ = aggregator(probs)
            loss = crit(agg.unsqueeze(0), lbl.unsqueeze(0)) / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0 or i == len(indices) - 1:
                opt.step(); opt.zero_grad()
    aggregator.eval()
    return aggregator

def train_drug_projection(drug_tower, drug_proj, compound_fps, train_herbs, epochs=30):
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    drug_tower.eval(); drug_proj.train()
    herb_drug_views = {}
    with torch.no_grad():
        for idx, h in enumerate(train_herbs):
            cids = [c for c in h["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                _, emb = drug_tower(c_fps_t[cids], return_embed=True)
                herb_drug_views[idx] = emb.mean(dim=0)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(drug_proj.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        indices = list(herb_drug_views.keys())
        random.shuffle(indices)
        for idx in indices:
            pred = drug_proj(herb_drug_views[idx])
            lbl = torch.tensor(train_herbs[idx]["label"], dtype=torch.float32).to(device)
            loss = crit(pred, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
    drug_proj.eval()
    return drug_proj

def compute_mi_auc_chains(tr_feat, tr_l, n_labels=5, seed=42):
    mi_mat = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            mi = mutual_info_score(tr_l[:, i].astype(int), tr_l[:, j].astype(int))
            mi_mat[i, j] = mi_mat[j, i] = mi
    chains = []
    for start in range(n_labels):
        chain = [start]; remaining = list(range(n_labels)); remaining.remove(start)
        while remaining:
            best = max(remaining, key=lambda j: sum(mi_mat[j, k] for k in chain))
            chain.append(best); remaining.remove(best)
        chains.append(chain)
    return chains

def mixup_batch(fv, lb, alpha=0.2):
    if alpha <= 0: return fv, lb
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(fv.size(0))
    return lam * fv + (1 - lam) * fv[idx], lam * lb + (1 - lam) * lb[idx]

def dropadd(fv, drop_p=0.20, add_p=0.01):
    mask = (torch.rand_like(fv) > drop_p).float(); fv = fv * mask
    add_mask = (torch.rand_like(fv) < add_p) & (fv == 0)
    return torch.clamp(fv + add_mask.float(), 0, 1)

def train_label_protonet_abl(tr_inp, tr_lbl, va_inp, va_lbl, comp_sim, device, seed,
                              tr_prior=None, va_prior=None, tr_prior2=None, va_prior2=None,
                              epochs=200, patience=50, use_contrastive=True):
    """Same as train_ablation.py's train_label_protonet (no AB_FLAGS)"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    feat_dim = tr_inp.shape[1]
    tr_f = torch.tensor(tr_inp, dtype=torch.float32)
    tr_l = torch.tensor(tr_lbl, dtype=torch.float32)
    va_f = torch.tensor(va_inp, dtype=torch.float32)
    tr_p = torch.tensor(tr_prior, dtype=torch.float32) if tr_prior is not None else None
    va_p = torch.tensor(va_prior, dtype=torch.float32) if va_prior is not None else None
    tr_p2 = torch.tensor(tr_prior2, dtype=torch.float32) if tr_prior2 is not None else None
    va_p2 = torch.tensor(va_prior2, dtype=torch.float32) if va_prior2 is not None else None
    g = torch.Generator(); g.manual_seed(seed)
    loader_data = [tr_f, tr_l, torch.arange(len(tr_inp))]
    has_prior = tr_p is not None
    has_dual = has_prior and tr_p2 is not None
    if has_prior: loader_data.insert(1, tr_p)
    if has_dual: loader_data.insert(2, tr_p2)
    loader = DataLoader(TensorDataset(*loader_data), batch_size=32, shuffle=True, drop_last=True, generator=g)
    val_loader = DataLoader(TensorDataset(va_f), batch_size=64)
    model = GatedPriorProtoNet(feat_dim=feat_dim, prior_dim=5).to(device) if has_prior else ProtoNet(feat_dim=feat_dim).to(device)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    va_l_flat = va_lbl.flatten()
    best_auc, patience_cnt, best_st = 0.0, 0, None
    for _ in range(epochs):
        model.train()
        for batch in loader:
            if has_dual:
                fv, pv, pv2, lb, idxs = batch; pv, pv2 = pv.to(device), pv2.to(device)
            elif has_prior:
                fv, pv, lb, idxs = batch; pv = pv.to(device); pv2 = None
            else:
                fv, lb, idxs = batch; pv, pv2 = None, None
            fv, lb = fv.to(device), lb.to(device)
            lb = lb * 0.95 + 0.025
            fv = dropadd(fv)
            fv, lb = mixup_batch(fv, lb, alpha=0.2)
            if has_prior:
                logit, z, z_know = model(fv, prior=pv, prior2=pv2, return_proj=True)
            else:
                logit, z = model(fv, return_proj=True)
                z_know = None
            loss = criterion(logit, lb) + 0.1 * knowledge_sim_loss(z, comp_sim, idxs)
            if z_know is not None and use_contrastive:
                pos_mask = (lb @ lb.T > 0).float()
                n_pos = pos_mask.sum(dim=1).clamp(min=1)
                sim = z @ z_know.T / 0.5
                sim_max, _ = sim.max(dim=1, keepdim=True)
                sim = sim - sim_max.detach()
                log_prob = sim - torch.log(torch.exp(sim).sum(dim=1, keepdim=True) + 1e-8)
                con_loss = -(pos_mask * log_prob).sum(dim=1) / n_pos
                loss = loss + 0.05 * con_loss.mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); model.scale.data.clamp_(1.0, 30.0)
        sched.step()
        model.eval(); vps = []
        with torch.no_grad():
            for (fv,) in val_loader:
                fv = fv.to(device)
                if has_prior:
                    vp = model(fv, prior=va_p.to(device), prior2=va_p2.to(device) if va_p2 is not None else None)
                else:
                    vp = model(fv)
                vps.append(torch.sigmoid(vp).cpu().numpy())
        vp = np.concatenate(vps).flatten()
        try: auc = roc_auc_score(va_l_flat, vp)
        except ValueError: auc = 0.0
        if auc > best_auc: best_auc = auc; patience_cnt = 0; best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else: patience_cnt += 1
        if patience_cnt >= patience: break
    if best_st: model.load_state_dict(best_st)
    model.eval(); vps = []
    with torch.no_grad():
        for (fv,) in val_loader:
            fv = fv.to(device)
            if has_prior:
                vp = model(fv, prior=va_p.to(device), prior2=va_p2.to(device) if va_p2 is not None else None)
            else:
                vp = model(fv)
            vps.append(torch.sigmoid(vp).cpu().numpy())
    return np.concatenate(vps).flatten()

# ===== MAIN =====

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

all_json = os.path.join(BASE, "dataset/output/all_herbs.json")
with open(all_json, 'r', encoding='utf-8') as f: all_data = json.load(f)
features = np.array([d["feature_vector"] for d in all_data], dtype=np.float32)
labels = np.array([d["label"] for d in all_data], dtype=np.float32)
with open(os.path.join(BASE, "dataset/output/compound2id.json"), 'r') as f: c2id = json.load(f)
n_compounds = len(c2id)
compound_mh = np.zeros((len(all_data), n_compounds), dtype=np.float32)
for i, d in enumerate(all_data):
    for cid in d["compound_ids"]:
        if cid < n_compounds: compound_mh[i, cid] = 1.0
compound_fps = np.load(os.path.join(BASE, "..", "数据爬取", "compound_fps.npy"))
label_strs = [''.join(str(int(v)) for v in d['label']) for d in all_data]

# Load Drug Tower
drug_tower = DrugTower(n_labels=8).to(device)
ckpt = torch.load(os.path.join(BASE, "dataset", "drug_tower_encoder.pt"), map_location=device)
drug_tower.load_state_dict(ckpt); drug_tower.eval()
for p in drug_tower.parameters(): p.requires_grad = False

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
all_fold_results = []
fold_per_label = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, label_strs)):
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/5 (train={len(train_idx)}, val={len(val_idx)})")
    train_features = features[train_idx]; train_labels = labels[train_idx]
    val_features = features[val_idx]; val_labels = labels[val_idx]
    comp_train = compound_mh[train_idx]
    intersection = comp_train @ comp_train.T
    row_sum = comp_train.sum(axis=1, keepdims=True)
    union = row_sum + row_sum.T - intersection
    comp_sim_train = torch.tensor(intersection / np.maximum(union, 1), dtype=torch.float32)
    train_herbs = [all_data[i] for i in train_idx]

    # Compound + aggregator
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

    # ECC
    chains = compute_mi_auc_chains(train_features, train_labels.astype(int), n_labels=5, seed=SEED+fold)

    # P1
    p1_ensemble = np.zeros((len(val_idx), 5), dtype=np.float32); p1_chain_probs = {}
    for ci, chain_order in enumerate(chains):
        val_probs = np.zeros((len(val_idx), 5), dtype=np.float32)
        for step, li in enumerate(chain_order):
            if step == 0: tr_inp, va_inp = train_features, val_features
            else:
                prev = chain_order[:step]
                tr_inp = np.concatenate([train_features, train_labels[:, prev]], axis=1)
                va_inp = np.concatenate([val_features, val_probs[:, prev]], axis=1)
            val_probs[:, li] = train_label_protonet_abl(
                tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                comp_sim_train, device, seed=SEED + fold * 100 + li * 7 + ci * 1000,
                tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug,
                use_contrastive=True, epochs=200)
        p1_ensemble += val_probs; p1_chain_probs[ci] = val_probs.copy()
    p1_ensemble /= len(chains)

    # P2
    p2_ensemble = np.zeros((len(val_idx), 5), dtype=np.float32)
    for ci, chain_order in enumerate(chains):
        p1_val = p1_chain_probs[ci]
        val_probs2 = np.zeros((len(val_idx), 5), dtype=np.float32)
        for step, li in enumerate(chain_order):
            if step == 0: tr_inp, va_inp = train_features, val_features
            else:
                prev = chain_order[:step]
                soft_cond = (train_labels[:, prev] * 0.8 + 0.1).astype(np.float32)
                tr_inp = np.concatenate([train_features, soft_cond], axis=1)
                va_inp = np.concatenate([val_features, p1_val[:, prev]], axis=1)
            val_probs2[:, li] = train_label_protonet_abl(
                tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                comp_sim_train, device, seed=SEED + fold * 100 + li * 7 + ci * 1000 + 9999,
                tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug,
                use_contrastive=True, epochs=200)
        p2_ensemble += val_probs2
    p2_ensemble /= len(chains)

    probs = (p1_ensemble + p2_ensemble) / 2

    # Macro metrics
    macro_auc = roc_auc_score(val_labels, probs, average='macro')
    fixed_preds = (probs > 0.5).astype(int)
    macro_f1_fixed = f1_score(val_labels, fixed_preds, average='macro', zero_division=0)
    micro_f1_fixed = f1_score(val_labels, fixed_preds, average='micro', zero_division=0)

    # Search thresholds
    from train_compound_ecc import find_best_thresholds
    thresholds = find_best_thresholds(val_labels, probs, 5)
    best_preds = np.zeros_like(probs, dtype=int)
    for i in range(5): best_preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)
    macro_f1_best = f1_score(val_labels, best_preds, average='macro', zero_division=0)
    micro_f1_best = f1_score(val_labels, best_preds, average='micro', zero_division=0)

    # Per-label
    per_label = []
    for li in range(5):
        try: auc_li = roc_auc_score(val_labels[:, li], probs[:, li])
        except ValueError: auc_li = float('nan')
        f1f = f1_score(val_labels[:, li], (probs[:, li] > 0.5).astype(int), zero_division=0)
        f1b = f1_score(val_labels[:, li], best_preds[:, li], zero_division=0)
        per_label.append({"name": LABEL_NAMES[li], "auc": auc_li, "f1_fixed": f1f, "f1_best": f1b})

    elapsed = time.time() - t0
    print(f"Fold {fold+1}: macro_auc={macro_auc:.4f} macro_f1_fixed={macro_f1_fixed:.4f} macro_f1_best={macro_f1_best:.4f} time={elapsed:.0f}s")
    for pl in per_label:
        print(f"  {pl['name']:20s} AUC={pl['auc']:.4f} F1@0.5={pl['f1_fixed']:.4f} F1@best={pl['f1_best']:.4f}")

    all_fold_results.append({"fold": fold, "macro_auc": macro_auc, "macro_f1_fixed": macro_f1_fixed,
        "macro_f1_best": macro_f1_best, "micro_f1_fixed": micro_f1_fixed, "micro_f1_best": micro_f1_best,
        "per_label": per_label, "time_s": round(elapsed,1)})

# Summary
print(f"\n{'='*60}")
print(f"5-Fold CV Summary (ablation code path, seed={SEED})")
print(f"{'='*60}")
aucs = [r['macro_auc'] for r in all_fold_results]
mff = [r['macro_f1_fixed'] for r in all_fold_results]
mfb = [r['macro_f1_best'] for r in all_fold_results]
print(f"Macro AUC:    {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
print(f"Macro F1@0.5: {np.mean(mff):.4f} +/- {np.std(mff):.4f}")
print(f"Macro F1@best:{np.mean(mfb):.4f} +/- {np.std(mfb):.4f}")

print(f"\n{'Label':<22s} {'AUC':>12s} {'F1@0.5':>12s} {'F1@best':>12s}")
print(f"{'-'*22} {'-'*12} {'-'*12} {'-'*12}")
for li in range(5):
    aucs_li = [r['per_label'][li]['auc'] for r in all_fold_results]
    f1f_li = [r['per_label'][li]['f1_fixed'] for r in all_fold_results]
    f1b_li = [r['per_label'][li]['f1_best'] for r in all_fold_results]
    print(f"{LABEL_NAMES[li]:<22s} {np.mean(aucs_li):.4f}+/-{np.std(aucs_li):.3f} {np.mean(f1f_li):.4f}+/-{np.std(f1f_li):.3f} {np.mean(f1b_li):.4f}+/-{np.std(f1b_li):.3f}")

with open(OUTPUT, 'w') as f:
    json.dump({"config": "Full Model", "seed": SEED, "label_names": LABEL_NAMES,
               "fold_results": all_fold_results}, f, indent=2)
print(f"\nSaved to {OUTPUT}")
