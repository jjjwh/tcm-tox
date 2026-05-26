"""
Quick test: Drug Tower with Morgan(1024) + RDKit descriptors(217) = 1241-dim input.
Compares TCM transfer performance vs Morgan-only baseline.
"""
import os, sys, random, numpy as np, torch, torch.nn as nn, json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from model_drug_tower import DrugTower
from model import AsymmetricLoss

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_drug_tower_quick(fps, labels, n_labels, epochs=30):
    """Quick 5-fold CV training, return best encoder."""
    label_strs = [''.join(str(int(v)) for v in row) for row in labels.astype(int)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs, best_encoder, best_val = [], None, 0.0
    for fold, (tr_idx, va_idx) in enumerate(skf.split(fps, label_strs)):
        tr_f = torch.tensor(fps[tr_idx], dtype=torch.float32)
        tr_l = torch.tensor(labels[tr_idx], dtype=torch.float32)
        va_f = torch.tensor(fps[va_idx], dtype=torch.float32)
        va_l = torch.tensor(labels[va_idx], dtype=torch.float32)
        model = DrugTower(fp_dim=fps.shape[1], n_labels=n_labels).to(device)
        crit = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        loader = DataLoader(TensorDataset(tr_f, tr_l), batch_size=64, shuffle=True, drop_last=True)
        best_auc, p_cnt, best_st = 0.0, 0, None
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
            with torch.no_grad(): vp = torch.sigmoid(model(va_f.to(device))).cpu().numpy()
            try: auc = roc_auc_score(va_l.numpy(), vp, average='macro')
            except: auc = 0.5
            if auc > best_auc: best_auc = auc; p_cnt = 0; best_st = {k:v.clone() for k,v in model.state_dict().items()}
            else: p_cnt += 1
            if p_cnt >= 10: break
        if best_st: model.load_state_dict(best_st)
        fold_aucs.append(best_auc)
        if best_auc > best_val: best_val = best_auc; best_encoder = {k:v.clone() for k,v in best_st.items()}
    return best_encoder, (np.mean(fold_aucs), np.std(fold_aucs))


def main():
    # --- Train Drug Tower with 1241-dim features ---
    drug_fps = np.load(os.path.join(BASE, "dataset", "Drug_Morgan_Desc.npy"))
    drug_labels = np.load(os.path.join(BASE, "dataset", "Drug_Labels.npy"))
    print(f"Drug Tower (1241-dim): {drug_fps.shape[0]} drugs, {drug_fps.shape[1]} features")

    encoder, drug_auc = train_drug_tower_quick(drug_fps, drug_labels, n_labels=8, epochs=30)
    print(f"Drug Tower 5-Fold CV: AUC = {drug_auc[0]:.4f} +/- {drug_auc[1]:.4f}")

    # Save encoder
    torch.save(encoder, os.path.join(BASE, "dataset", "drug_tower_desc.pt"))

    # Compare with Morgan-only baseline
    print(f"\nComparison:")
    print(f"  Morgan-only (1024d): Drug AUC=0.7399, TCM AUC=0.8054")
    print(f"  +RDKit desc (1241d): Drug AUC={drug_auc[0]:.4f}")
    print(f"\n(Full TCM transfer test requires running train_compound_ecc.py")
    print(f" with --drug_tower_ckpt drug_tower_desc.pt and fp_dim=1241)")


if __name__ == "__main__":
    main()
