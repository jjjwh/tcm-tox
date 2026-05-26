"""
train_binary.py — Hepatotoxicity & Nephrotoxicity binary classification (Full Model).
Uses Drug Tower + compound prior + attention aggregator, same as multi-label pipeline.
"""
import os, sys, random, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
sys.path.insert(0, '.')
from model import ProtoNet, GatedPriorProtoNet, AsymmetricLoss, knowledge_sim_loss, CompoundAttentionAggregator
from model_drug_tower import DrugTower

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_compound_labels(herbs, n_compounds, label_idx):
    """Binary version: only one label column."""
    c_labels = np.zeros((n_compounds, 1), dtype=np.float32)
    for h in herbs:
        for cid in h["compound_ids"]:
            if cid < n_compounds:
                c_labels[cid] = max(c_labels[cid], h["label"][label_idx])
    return c_labels


class CompoundToxModel(nn.Module):
    def __init__(self, fp_dim=1024, hidden=256, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 1)  # binary output

    def forward(self, fp, return_embed=False):
        emb = self.encoder(fp)
        logits = self.head(emb)
        if return_embed:
            return logits, emb
        return logits


def mixup_batch(fv, lb, alpha=0.2):
    if alpha <= 0: return fv, lb
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(fv.size(0))
    return lam * fv + (1 - lam) * fv[idx], lam * lb + (1 - lam) * lb[idx]


def dropadd(fv, drop_p=0.20, add_p=0.01):
    mask = (torch.rand_like(fv) > drop_p).float(); fv = fv * mask
    add_mask = (torch.rand_like(fv) < add_p) & (fv == 0)
    return torch.clamp(fv + add_mask.float(), 0, 1)


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
    model = CompoundToxModel().to(device)
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


def compute_prior(model, compound_fps, herb_indices, herbs):
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    prior = np.zeros((len(herb_indices), 1), dtype=np.float32)
    if model is None: return prior
    model.eval()
    with torch.no_grad():
        for j, hi in enumerate(herb_indices):
            cids = [c for c in herbs[hi]["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                probs = torch.sigmoid(model(c_fps_t[cids]))  # [n_c, 1]
                prior[j] = probs.cpu().numpy().mean(axis=0)  # mean pool for binary
    return prior


def compute_drug_prior(drug_tower, drug_proj, compound_fps, herb_indices, herbs):
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    prior = np.zeros((len(herb_indices), 1), dtype=np.float32)
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
            lbl = torch.tensor([train_herbs[idx]["label"][0]], dtype=torch.float32).to(device)
            loss = crit(pred, lbl)
            opt.zero_grad(); loss.backward(); opt.step()
    drug_proj.eval()
    return drug_proj


def train_binary_full(features, labels, compound_mh, compound_fps, all_data, label_idx,
                      seed=42, n_splits=5, epochs=200, patience=50):
    """Full pipeline for binary classification with Drug Tower + compound prior."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {'auroc': [], 'auprc': [], 'f1': []}

    # Load frozen Drug Tower
    drug_tower = DrugTower(n_labels=8).to(device)
    ckpt = torch.load(os.path.join(BASE, "dataset", "drug_tower_encoder.pt"), map_location=device)
    drug_tower.load_state_dict(ckpt)
    drug_tower.eval()
    for p in drug_tower.parameters(): p.requires_grad = False

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels.astype(int))):
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]
        train_herbs = [all_data[i] for i in train_idx]

        # Compound model
        c_labels = build_compound_labels(train_herbs, len(compound_fps), label_idx)
        comp_model = train_compound_fold(compound_fps, c_labels, train_herbs, epochs=40)
        tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data)
        va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data)

        # Drug prior
        drug_proj = nn.Linear(256, 1).to(device)
        drug_proj = train_drug_projection(drug_tower, drug_proj, compound_fps, train_herbs)
        tr_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, train_idx, all_data)
        va_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, val_idx, all_data)

        # Combine priors
        tr_p = np.concatenate([tr_prior, tr_drug], axis=1)
        va_p = np.concatenate([va_prior, va_drug], axis=1)

        # Herb model
        tr_f = torch.tensor(train_features, dtype=torch.float32)
        tr_l = torch.tensor(train_labels.reshape(-1, 1), dtype=torch.float32)
        va_f = torch.tensor(val_features, dtype=torch.float32)
        tr_p_t = torch.tensor(tr_p, dtype=torch.float32)
        va_p_t = torch.tensor(va_p, dtype=torch.float32)

        model = GatedPriorProtoNet(feat_dim=features.shape[1], prior_dim=2).to(device)
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        loader = DataLoader(TensorDataset(tr_f, tr_p_t, tr_l), batch_size=32, shuffle=True, drop_last=True)
        best_auc, patience_cnt, best_st = 0.0, 0, None

        for ep in range(epochs):
            model.train()
            for fv, pv, lb in loader:
                fv, pv, lb = fv.to(device), pv.to(device), lb.to(device)
                lb = lb * 0.95 + 0.025  # label smoothing
                fv = dropadd(fv); fv, lb = mixup_batch(fv, lb, alpha=0.2)
                out = model(fv, prior=pv)
                logit = out
                loss = criterion(logit, lb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); model.scale.data.clamp_(1.0, 30.0)
            sched.step()

            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(va_f.to(device), prior=va_p_t.to(device))).cpu().numpy().flatten()
            try:
                auc = roc_auc_score(val_labels, vp)
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc; patience_cnt = 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
            if patience_cnt >= patience:
                break

        if best_st: model.load_state_dict(best_st)
        model.eval()
        with torch.no_grad():
            vp_final = torch.sigmoid(model(va_f.to(device), prior=va_p_t.to(device))).cpu().numpy().flatten()

        results['auroc'].append(roc_auc_score(val_labels, vp_final))
        results['auprc'].append(average_precision_score(val_labels, vp_final))
        preds = (vp_final > 0.5).astype(int)
        results['f1'].append(f1_score(val_labels, preds, zero_division=0))
        print(f"  Fold {fold+1}: AUROC={results['auroc'][-1]:.4f}  AUPRC={results['auprc'][-1]:.4f}  F1={results['f1'][-1]:.4f}")

    for k in results:
        results[k] = (np.mean(results[k]), np.std(results[k]))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    import json
    all_json = os.path.join(BASE, "dataset/output/all_herbs.json")
    with open(all_json, 'r', encoding='utf-8') as f: all_data = json.load(f)
    features = np.array([d["feature_vector"] for d in all_data], dtype=np.float32)
    labels5 = np.load(os.path.join(BASE, "dataset", "TCM_Labels_5.npy"))
    liver_labels = np.load(os.path.join(BASE, "dataset", "TCM_Labels_Liver.npy")).flatten().astype(int)
    nephro_labels = labels5[:, 1].astype(int)

    with open(os.path.join(BASE, "dataset/output/compound2id.json"), 'r') as f: c2id = json.load(f)
    n_compounds = len(c2id)
    compound_mh = np.zeros((len(all_data), n_compounds), dtype=np.float32)
    for i, d in enumerate(all_data):
        for cid in d["compound_ids"]:
            if cid < n_compounds: compound_mh[i, cid] = 1.0
    compound_fps = np.load(os.path.join(BASE, "..", "数据爬取", "compound_fps.npy"))

    print(f"Hepatotoxicity: {liver_labels.sum()} positive / {len(liver_labels)} total")
    print(f"Nephrotoxicity: {nephro_labels.sum()} positive / {len(nephro_labels)} total")
    print()

    print("=== Hepatotoxicity (Full Model: Drug Tower + Compound + GatedPrior) ===")
    hep_res = train_binary_full(features, liver_labels, compound_mh, compound_fps, all_data,
                                label_idx=0, seed=args.seed, n_splits=args.n_splits)
    print(f"  AUROC: {hep_res['auroc'][0]:.4f} +/- {hep_res['auroc'][1]:.4f}")
    print(f"  AUPRC: {hep_res['auprc'][0]:.4f} +/- {hep_res['auprc'][1]:.4f}")
    print(f"  F1:    {hep_res['f1'][0]:.4f} +/- {hep_res['f1'][1]:.4f}")

    print()
    print("=== Nephrotoxicity (Full Model) ===")
    nep_res = train_binary_full(features, nephro_labels, compound_mh, compound_fps, all_data,
                                label_idx=1, seed=args.seed, n_splits=args.n_splits)
    print(f"  AUROC: {nep_res['auroc'][0]:.4f} +/- {nep_res['auroc'][1]:.4f}")
    print(f"  AUPRC: {nep_res['auprc'][0]:.4f} +/- {nep_res['auprc'][1]:.4f}")
    print(f"  F1:    {nep_res['f1'][0]:.4f} +/- {nep_res['f1'][1]:.4f}")

    print()
    print("=== Table 4 Summary (Full Model) ===")
    print(f"Hepatotoxicity: AUROC={hep_res['auroc'][0]:.4f}+/-{hep_res['auroc'][1]:.4f}  "
          f"AUPRC={hep_res['auprc'][0]:.4f}+/-{hep_res['auprc'][1]:.4f}  "
          f"F1={hep_res['f1'][0]:.4f}+/-{hep_res['f1'][1]:.4f}")
    print(f"Nephrotoxicity: AUROC={nep_res['auroc'][0]:.4f}+/-{nep_res['auroc'][1]:.4f}  "
          f"AUPRC={nep_res['auprc'][0]:.4f}+/-{nep_res['auprc'][1]:.4f}  "
          f"F1={nep_res['f1'][0]:.4f}+/-{nep_res['f1'][1]:.4f}")


if __name__ == "__main__":
    main()
