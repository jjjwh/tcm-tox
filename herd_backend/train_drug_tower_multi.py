"""
train_drug_tower_multi.py — Multi-source Drug Tower pretraining + TCM transfer comparison.
Trains Drug Tower on 3 data sources, then evaluates TCM transfer performance for each.
"""
import os, sys, random, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from model_drug_tower import DrugTower
from model import AsymmetricLoss

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_drug_tower(fps, labels, n_labels, epochs=50, lr=1e-3, wd=1e-3, patience=15):
    """Train Drug Tower with 5-fold CV, return best encoder state and CV AUC."""
    label_strs = [''.join(str(int(v)) for v in row) for row in labels.astype(int)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    best_encoder, best_val_auc = None, 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(fps, label_strs)):
        tr_f = torch.tensor(fps[train_idx], dtype=torch.float32)
        tr_l = torch.tensor(labels[train_idx], dtype=torch.float32)
        va_f = torch.tensor(fps[val_idx], dtype=torch.float32)
        va_l = torch.tensor(labels[val_idx], dtype=torch.float32)

        model = DrugTower(n_labels=n_labels).to(device)
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        loader = DataLoader(TensorDataset(tr_f, tr_l), batch_size=64, shuffle=True, drop_last=True)

        best_auc, p_cnt, best_st = 0.0, 0, None
        for ep in range(epochs):
            model.train()
            for fv, lb in loader:
                fv, lb = fv.to(device), lb.to(device)
                loss = criterion(model(fv), lb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            model.eval()
            with torch.no_grad():
                vp = torch.sigmoid(model(va_f.to(device))).cpu().numpy()
            try: auc = roc_auc_score(va_l.numpy(), vp, average='macro')
            except: auc = 0.5
            if auc > best_auc: best_auc = auc; p_cnt = 0; best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else: p_cnt += 1
            if p_cnt >= patience: break

        if best_st: model.load_state_dict(best_st)
        fold_aucs.append(best_auc)
        if best_auc > best_val_auc:
            best_val_auc = best_auc
            best_encoder = {k: v.clone() for k, v in best_st.items()}

    cv_auc = (np.mean(fold_aucs), np.std(fold_aucs))
    return best_encoder, cv_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Load datasets
    # UniTox
    unitox_fps = np.load(os.path.join(BASE, "dataset", "Drug_Morgan.npy"))
    unitox_labels = np.load(os.path.join(BASE, "dataset", "Drug_Labels.npy"))
    print(f"UniTox: {unitox_fps.shape[0]} drugs, {unitox_labels.shape[1]} labels")

    # Tox21
    tox21_fps = np.load(os.path.join(BASE, "dataset", "Tox21_Morgan.npy"))
    tox21_labels = np.load(os.path.join(BASE, "dataset", "Tox21_Labels.npy"))
    print(f"Tox21:  {tox21_fps.shape[0]} drugs, {tox21_labels.shape[1]} labels")

    results = {}

    # === Experiment 1: UniTox Drug Tower (baseline) ===
    print("\n" + "="*60)
    print("Exp 1: Drug Tower on UniTox (1349 drugs, 8 labels)")
    print("="*60)
    enc1, auc1 = train_drug_tower(unitox_fps, unitox_labels, n_labels=8, epochs=args.epochs, lr=args.lr)
    torch.save(enc1, os.path.join(BASE, "dataset", "drug_tower_unitox.pt"))
    results['UniTox'] = auc1
    print(f"UniTox Drug Tower 5-Fold CV: AUC = {auc1[0]:.4f} +/- {auc1[1]:.4f}")

    # === Experiment 2: Tox21 Drug Tower ===
    print("\n" + "="*60)
    print("Exp 2: Drug Tower on Tox21 (7823 drugs, 12 labels)")
    print("="*60)
    enc2, auc2 = train_drug_tower(tox21_fps, tox21_labels, n_labels=12, epochs=args.epochs, lr=args.lr)
    torch.save(enc2, os.path.join(BASE, "dataset", "drug_tower_tox21.pt"))
    results['Tox21'] = auc2
    print(f"Tox21 Drug Tower 5-Fold CV: AUC = {auc2[0]:.4f} +/- {auc2[1]:.4f}")

    # === Experiment 3: Combined (UniTox + Tox21, shared encoder, dual head) ===
    print("\n" + "="*60)
    print("Exp 3: Drug Tower on UniTox + Tox21 combined (9172 drugs)")
    print("="*60)
    # Combine datasets: train encoder jointly on both tasks
    # Strategy: alternate batches from each dataset, same encoder, separate heads
    # For simplicity: concatenate data, use label dimension = 8+12 = 20 (zeros for missing)
    combined_fps = np.concatenate([unitox_fps, tox21_fps], axis=0)
    combined_labels = np.zeros((len(combined_fps), 20), dtype=np.float32)
    combined_labels[:len(unitox_fps), :8] = unitox_labels
    combined_labels[len(unitox_fps):, 8:] = tox21_labels
    print(f"Combined: {combined_fps.shape[0]} drugs, 20 labels (8 UniTox + 12 Tox21)")

    enc3, auc3 = train_drug_tower(combined_fps, combined_labels, n_labels=20, epochs=args.epochs, lr=args.lr)
    torch.save(enc3, os.path.join(BASE, "dataset", "drug_tower_combined.pt"))
    results['Combined'] = auc3
    print(f"Combined Drug Tower 5-Fold CV: AUC = {auc3[0]:.4f} +/- {auc3[1]:.4f}")

    # === Summary ===
    print("\n" + "="*60)
    print("DRUG TOWER PRETRAINING SUMMARY")
    print("="*60)
    print(f"{'Dataset':<20} {'Drugs':>6} {'Labels':>6} {'CV AUC':>16}")
    print("-"*52)
    print(f"{'UniTox':<20} {1349:>6} {8:>6} {results['UniTox'][0]:.4f} +/- {results['UniTox'][1]:.4f}")
    print(f"{'Tox21':<20} {7823:>6} {12:>6} {results['Tox21'][0]:.4f} +/- {results['Tox21'][1]:.4f}")
    print(f"{'UniTox+Tox21':<20} {9172:>6} {20:>6} {results['Combined'][0]:.4f} +/- {results['Combined'][1]:.4f}")


if __name__ == "__main__":
    main()
