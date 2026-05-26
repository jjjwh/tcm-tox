"""
train_drug_tower.py — Pretrain Drug Tower on UniTox Western drug data.
5-fold CV, ASL loss, save encoder for cross-domain transfer to TCM.
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from model_drug_tower import DrugTower
from model import AsymmetricLoss

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    fps = np.load(os.path.join(BASE, "dataset", "Drug_Morgan.npy"))
    labels = np.load(os.path.join(BASE, "dataset", "Drug_Labels.npy"))
    n_labels = labels.shape[1]
    label_strs = [''.join(str(int(v)) for v in row) for row in labels]

    print(f"Drug Tower Pretraining: {len(fps)} drugs, {n_labels} labels, "
          f"{args.n_splits}-fold CV, {args.epochs} epochs")

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []
    best_encoder_state = None
    best_val_auc = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(fps, label_strs)):
        print(f"\n--- Fold {fold+1}/{args.n_splits} (train={len(train_idx)}, val={len(val_idx)}) ---")

        tr_f = torch.tensor(fps[train_idx], dtype=torch.float32)
        tr_l = torch.tensor(labels[train_idx], dtype=torch.float32)
        va_f = torch.tensor(fps[val_idx], dtype=torch.float32)
        va_l = torch.tensor(labels[val_idx], dtype=torch.float32)

        model = DrugTower(n_labels=n_labels).to(device)
        criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

        loader = DataLoader(TensorDataset(tr_f, tr_l), batch_size=64, shuffle=True, drop_last=True)
        best_auc, best_state = 0.0, None
        patience, patience_limit = 0, 15

        for ep in range(args.epochs):
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
            try:
                auc = roc_auc_score(va_l.numpy(), vp, average='macro')
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= patience_limit:
                break

        if best_state:
            model.load_state_dict(best_state)
        fold_results.append(best_auc)
        print(f"  Fold {fold+1}: best val AUC = {best_auc:.4f}")

        if best_auc > best_val_auc:
            best_val_auc = best_auc
            best_encoder_state = {k: v.clone() for k, v in best_state.items()}

    aucs = fold_results
    print(f"\nDrug Tower {args.n_splits}-Fold CV: AUC = {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    save_path = os.path.join(BASE, "dataset", "drug_tower_encoder.pt")
    torch.save(best_encoder_state, save_path)
    print(f"Best encoder saved to {save_path} (val AUC={best_val_auc:.4f})")


if __name__ == "__main__":
    main()
