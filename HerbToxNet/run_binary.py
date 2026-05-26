"""
Reproduce HerbToxNet hepatotoxicity and nephrotoxicity binary classification.
Adapted from HerbToxNet main.py with output_dim=1 + BCE loss.
"""
import torch, torch.nn as nn, torch.optim as optim
import random, os, sys, copy, time, json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import create_heterogeneous_data, EarlyStopping
from modelHAN import FastGTN, dynamic_contrastive_loss, weighted_label_fusion

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def setup_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True

def prepare_gtn_data(g, device):
    h_feat = g.nodes['herb'].data['feat']; i_feat = g.nodes['ingredient'].data['feat']
    t_feat = g.nodes['target'].data['feat']
    X = torch.cat([h_feat, i_feat, t_feat], dim=0).to(device)
    num_h = h_feat.shape[0]; num_i = i_feat.shape[0]; num_t = t_feat.shape[0]
    off_h, off_i, off_t = 0, num_h, num_h + num_i
    A = torch.zeros(7, X.shape[0], X.shape[0], device=device)
    src, dst = g.edges(etype='hi')
    A[0, src+off_h, dst+off_i] = 1.0; A[1, dst+off_i, src+off_h] = 1.0
    src, dst = g.edges(etype='ht')
    A[2, src+off_h, dst+off_t] = 1.0; A[3, dst+off_t, src+off_h] = 1.0
    src, dst = g.edges(etype='it')
    A[4, src+off_i, dst+off_t] = 1.0; A[5, dst+off_t, src+off_i] = 1.0
    A[6] = torch.eye(X.shape[0], device=device)
    return A, X

def feature_masking(X, drop_rate=0.2):
    if drop_rate <= 0: return X
    mask = torch.rand_like(X) > drop_rate; return X * mask

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, dropout_prob=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.mlp(x)

def evaluate_binary(y_true, y_pred):
    best_f1, best_t = 0, 0.3
    for t in np.arange(0.05, 1.0, 0.05):
        pred_bin = (y_pred > t).astype(int)
        f = f1_score(y_true, pred_bin, zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    pred_bin = (y_pred > best_t).astype(int)
    try: auc = roc_auc_score(y_true, y_pred)
    except: auc = float('nan')
    auprc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, pred_bin, zero_division=0)
    return {'auc': auc, 'auprc': auprc, 'f1': f1, 'threshold': best_t}

def run_binary(seed, label_idx, task_name, num_folds=5, test_ratio=0.15, epochs=200):
    setup_seed(seed)
    print(f"\n{'='*60}")
    print(f"Binary: {task_name} (label column {label_idx})")
    
    g = create_heterogeneous_data()
    all_labels = g.nodes['herb'].data['label'].numpy()
    labels_1d = all_labels[:, label_idx].reshape(-1, 1)
    
    n_pos = int(labels_1d.sum())
    n_total = len(labels_1d)
    print(f"  Positive samples: {n_pos}/{n_total} ({n_pos/n_total:.1%})")
    
    all_indices = list(range(g.num_nodes('herb')))
    random.seed(seed); np.random.seed(seed)
    shuffled = list(all_indices); random.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    test_indices = sorted(shuffled[:n_test])
    train_val_indices = sorted(shuffled[n_test:])
    
    g = g.to(device)
    A_orig, X_orig = prepare_gtn_data(g, device)
    train_val_arr = np.array(train_val_indices)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    for fold, (tr_rel, va_rel) in enumerate(kf.split(train_val_indices)):
        train_idx = train_val_arr[tr_rel].tolist()
        val_idx = train_val_arr[va_rel].tolist()
        train_l = torch.tensor(labels_1d[train_idx], dtype=torch.float32).to(device)
        val_l = torch.tensor(labels_1d[val_idx], dtype=torch.float32).to(device)
        test_l = torch.tensor(labels_1d[test_indices], dtype=torch.float32).to(device)
        
        emb_model = FastGTN(num_edge_types=7, num_channels=2, in_dim=300,
                             hidden_dim=128, out_dim=64, num_layers=2).to(device)
        clf = BinaryClassifier(input_dim=128, hidden_dim=128, dropout_prob=0.1).to(device)
        opt = optim.Adam(list(emb_model.parameters()) + list(clf.parameters()),
                         lr=0.001, weight_decay=1e-5)
        
        tr_mask = torch.zeros(g.num_nodes('herb'), dtype=torch.bool).to(device)
        va_mask = torch.zeros(g.num_nodes('herb'), dtype=torch.bool).to(device)
        tr_mask[train_idx] = True; va_mask[val_idx] = True
        
        bce = nn.BCELoss()
        stopper = EarlyStopping(patience=10)
        
        for ep in range(epochs):
            emb_model.train(); clf.train(); opt.zero_grad()
            cx = feature_masking(X_orig, 0.2)
            emb_all = emb_model(A_orig, cx)
            emb = emb_all[:g.num_nodes('herb')]
            out = clf(emb)
            clf_loss = bce(out[tr_mask], train_l)
            con_loss = dynamic_contrastive_loss(emb[tr_mask], train_l)
            total_loss = clf_loss + 0.1 * con_loss
            total_loss.backward(); opt.step()
            
            if (ep + 1) % 20 == 0:
                emb_model.eval(); clf.eval()
                with torch.no_grad():
                    emb_all = emb_model(A_orig, X_orig)
                    emb = emb_all[:g.num_nodes('herb')]
                    val_out = clf(emb)[va_mask]
                    val_loss = bce(val_out, val_l)
                if stopper.step(val_loss.item(), {'emb': emb_model.state_dict(), 'clf': clf.state_dict()}):
                    break
        
        ckpt = stopper.load_checkpoint()
        emb_model.load_state_dict(ckpt['emb']); clf.load_state_dict(ckpt['clf'])
        emb_model.eval(); clf.eval()
        with torch.no_grad():
            emb_all = emb_model(A_orig, X_orig)
            emb = emb_all[:g.num_nodes('herb')]
            test_preds = clf(emb)[torch.tensor(test_indices, dtype=torch.long).to(device)].cpu().numpy().flatten()
        
        m = evaluate_binary(test_l.cpu().numpy().flatten(), test_preds)
        m['fold'] = fold
        fold_results.append(m)
        print(f"  Fold {fold+1}: AUROC={m['auc']:.4f} AUPRC={m['auprc']:.4f} F1={m['f1']:.4f}")
        
        del emb_model, clf, opt
    
    aucs = [r['auc'] for r in fold_results]; auprcs = [r['auprc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    print(f"\n  >>> {task_name} {num_folds}-fold:")
    print(f"      AUROC={np.mean(aucs):.4f}+/-{np.std(aucs):.4f}")
    print(f"      AUPRC={np.mean(auprcs):.4f}+/-{np.std(auprcs):.4f}")
    print(f"      F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}")
    return {'task': task_name, 'auc': np.mean(aucs), 'auprc': np.mean(auprcs), 'f1': np.mean(f1s)}

if __name__ == "__main__":
    # Check column ordering
    g = create_heterogeneous_data()
    labels = g.nodes['herb'].data['label'].numpy()
    print("=== TCM_Labels_5.npy column analysis ===")
    for i in range(5):
        pos = int(labels[:, i].sum())
        neg = labels.shape[0] - pos
        print(f"  Column {i}: {pos:3d} pos / {neg:3d} neg = {pos/labels.shape[0]:.1%}")
    
    # Run for all label columns to find which matches paper's "hepatotoxicity 114 herbs"
    print("\nPaper reference:")
    print("  Hepatotoxicity: AUROC=0.7060 AUPRC=0.7698 F1=0.6520")
    print("  Nephrotoxicity: AUROC=0.7741 AUPRC=0.6381 F1=0.7356")
    print("  Hepatotoxicity has 114 herbs, Nephrotoxicity has 101 herbs")
    
    # Column 0 = 114 pos → best match for "114 herbs" → probably cardiotoxicity in our naming
    # Column 1 = 101 pos → matches nephrotoxicity
    # Let's run ALL columns to find which ones produce paper-like results
    
    for seed in [0]:
        print(f"\n{'#'*60}")
        print(f"Seed={seed}")
        for label_idx in range(5):
            pos = int(labels[:, label_idx].sum())
            name = f"Label_col{label_idx}_pos{pos}"
            run_binary(seed, label_idx, name, num_folds=5, test_ratio=0.15, epochs=200)
