"""
train_ablation.py — 消融实验脚本
================================
基于 train_compound_ecc.py，通过命令行参数控制各组件的启用/禁用。
用法:
  # 完整模型
  python train_ablation.py

  # 消融: 去掉某个组件
  python train_ablation.py --no_aggregator
  python train_ablation.py --no_knowledge_reg
  python train_ablation.py --no_mixup
  python train_ablation.py --no_dropadd
  python train_ablation.py --no_smooth_first

  # 一键跑全部消融
  python train_ablation.py --run_all
"""
import os, sys, random, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, mutual_info_score
sys.path.insert(0, '.')
from model import ProtoNet, GatedPriorProtoNet, AsymmetricLoss, knowledge_sim_loss, CompoundAttentionAggregator

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Global ablation flags (set by main) ----
AB_FLAGS = {}


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


def train_compound_fold(compound_fps, c_labels, train_herb_idx, herbs, epochs=40):
    if AB_FLAGS.get('no_compound_opt', False):
        epochs = 25  # old value
    train_cids = set()
    for hi in train_herb_idx:
        for cid in herbs[hi]["compound_ids"]:
            if cid < len(compound_fps) and c_labels[cid].sum() > 0:
                train_cids.add(cid)
    train_cids = sorted(train_cids)
    if len(train_cids) < 10: return None

    fps = torch.tensor(compound_fps[train_cids], dtype=torch.float32)
    lbls = torch.tensor(c_labels[train_cids], dtype=torch.float32)

    if AB_FLAGS.get('no_compound_opt', False):
        # 旧版: train/val split + early stopping
        n = len(fps); n_tr = int(n * 0.85)
        idx = np.random.RandomState(42).permutation(n)
        tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
        model = CompoundToxModel(dropout=0.2).to(device)  # old dropout
        crit = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        loader = DataLoader(TensorDataset(fps[tr_idx], lbls[tr_idx]),
                            batch_size=min(64, n_tr), shuffle=True, drop_last=True)
        best_auc, best_state = 0.0, None
        for ep in range(epochs):
            model.train()
            for fv, lb in loader:
                fv, lb = fv.to(device), lb.to(device)
                loss = crit(model(fv), lb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            if len(va_idx) > 0:
                model.eval()
                with torch.no_grad():
                    vp = torch.sigmoid(model(fps[va_idx].to(device))).cpu().numpy()
                try: auc = roc_auc_score(lbls[va_idx].numpy(), vp, average='macro')
                except: auc = 0.0
                if auc > best_auc: best_auc = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if best_state: model.load_state_dict(best_state)
    else:
        # 新版: 全量训练, 固定epochs
        model = CompoundToxModel(dropout=0.4).to(device)
        crit = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        loader = DataLoader(TensorDataset(fps, lbls),
                            batch_size=min(64, len(fps)), shuffle=True, drop_last=True)
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


def train_compound_aggregator(model, aggregator, compound_fps, herbs, train_herb_idx, epochs=50):
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    model.eval()
    aggregator.train()
    herb_comp_probs = {}
    with torch.no_grad():
        for hi in train_herb_idx:
            cids = [c for c in herbs[hi]["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                herb_comp_probs[hi] = torch.sigmoid(model(c_fps_t[cids]))
    crit = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
    opt = torch.optim.AdamW(aggregator.parameters(), lr=3e-3, weight_decay=1e-4)
    accum_steps = 4
    for _ in range(epochs):
        indices = list(herb_comp_probs.keys())
        random.shuffle(indices)
        opt.zero_grad()
        for i, hi in enumerate(indices):
            probs = herb_comp_probs[hi]
            lbl = torch.tensor(herbs[hi]["label"], dtype=torch.float32).to(device)
            agg, _ = aggregator(probs)
            loss = crit(agg.unsqueeze(0), lbl.unsqueeze(0)) / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0 or i == len(indices) - 1:
                opt.step(); opt.zero_grad()
    aggregator.eval()
    return aggregator


def compute_mi_auc_chains(tr_feat, tr_l, n_labels=5, seed=42, strategy='mi_auc'):
    if strategy == 'fixed_forward':
        return [list(range(n_labels)) for _ in range(n_labels)]
    if strategy == 'fixed_reverse':
        return [list(reversed(range(n_labels))) for _ in range(n_labels)]
    if strategy == 'random':
        rng = np.random.RandomState(seed)
        return [[int(x) for x in rng.permutation(n_labels)] for _ in range(n_labels)]

    mi_mat = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            mi = mutual_info_score(tr_l[:, i].astype(int), tr_l[:, j].astype(int))
            mi_mat[i, j] = mi_mat[j, i] = mi

    if strategy == 'mi_only':
        standalone_aucs = np.ones(n_labels)
    else:
        standalone_aucs = np.full(n_labels, 0.5)
    for li in range(n_labels):
        try:
            scores = cross_val_score(
                LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', random_state=seed),
                tr_feat, tr_l[:, li].astype(int), cv=3, scoring='roc_auc', error_score=0.5)
            standalone_aucs[li] = float(np.mean(scores))
        except Exception: standalone_aucs[li] = 0.5
    chains = []
    for start in range(n_labels):
        chain = [start]; remaining = list(range(n_labels)); remaining.remove(start)
        while remaining:
            best = max(remaining, key=lambda j: sum(mi_mat[j, k] for k in chain) * standalone_aucs[j])
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


def find_best_thresholds(y_true, y_prob, n_labels=5):
    thresholds = []
    for i in range(n_labels):
        bt, bf = 0.5, 0.0
        for t in np.linspace(0.05, 0.90, 18):
            pred = (y_prob[:, i] > t).astype(int); f = f1_score(y_true[:, i], pred, zero_division=0)
            if f > bf: bf, bt = f, t
        thresholds.append(bt)
    return thresholds


def train_label_protonet(tr_inp, tr_lbl, va_inp, va_lbl, comp_sim, device, seed,
                          tr_prior=None, va_prior=None,
                          epochs=200, patience=50):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    feat_dim = tr_inp.shape[1]
    tr_f = torch.tensor(tr_inp, dtype=torch.float32)
    tr_l = torch.tensor(tr_lbl, dtype=torch.float32)
    va_f = torch.tensor(va_inp, dtype=torch.float32)
    tr_p = torch.tensor(tr_prior, dtype=torch.float32) if tr_prior is not None else None
    va_p = torch.tensor(va_prior, dtype=torch.float32) if va_prior is not None else None

    g = torch.Generator(); g.manual_seed(seed)
    loader_data = [tr_f, tr_l, torch.arange(len(tr_inp))]
    if tr_p is not None: loader_data.insert(1, tr_p)
    loader = DataLoader(TensorDataset(*loader_data), batch_size=32, shuffle=True, drop_last=True, generator=g)
    val_loader = DataLoader(TensorDataset(va_f), batch_size=64)

    if tr_p is not None:
        model = GatedPriorProtoNet(feat_dim=feat_dim, prior_dim=5).to(device)
    else:
        model = ProtoNet(feat_dim=feat_dim).to(device)

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    va_l_flat = va_lbl.flatten()
    best_auc, patience_cnt, best_st = 0.0, 0, None
    for _ in range(epochs):
        model.train()
        for batch in loader:
            if tr_p is not None:
                fv, pv, lb, idxs = batch
                pv = pv.to(device)
            else:
                fv, lb, idxs = batch
            fv, lb = fv.to(device), lb.to(device)

            # Ablation: smoothing order
            if AB_FLAGS.get('no_smooth_first', False):
                fv = dropadd(fv) if not AB_FLAGS.get('no_dropadd', False) else fv
                fv, lb = mixup_batch(fv, lb, alpha=0.0 if AB_FLAGS.get('no_mixup', False) else 0.2)
                lb = lb * 0.95 + 0.025  # old order: mixup before smoothing
            else:
                lb = lb * 0.95 + 0.025  # smoothing first
                fv = dropadd(fv) if not AB_FLAGS.get('no_dropadd', False) else fv
                fv, lb = mixup_batch(fv, lb, alpha=0.0 if AB_FLAGS.get('no_mixup', False) else 0.2)

            if tr_p is not None:
                logit, z = model(fv, prior=pv, return_proj=True)
            else:
                logit, z = model(fv, return_proj=True)

            if AB_FLAGS.get('no_knowledge_reg', False):
                loss = criterion(logit, lb)
            else:
                loss = criterion(logit, lb) + 0.1 * knowledge_sim_loss(z, comp_sim, idxs)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); model.scale.data.clamp_(1.0, 30.0)
        sched.step()
        model.eval(); vps = []
        with torch.no_grad():
            for (fv,) in val_loader:
                fv = fv.to(device)
                vp = model(fv, prior=va_p.to(device)) if va_p is not None else model(fv)
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
            vp = model(fv, prior=va_p.to(device)) if va_p is not None else model(fv)
            vps.append(torch.sigmoid(vp).cpu().numpy())
    return np.concatenate(vps).flatten()


def run_single_config(config_name, config_flags):
    """Run a single ablation configuration and return results."""
    global AB_FLAGS
    AB_FLAGS = config_flags
    print(f"\n{'#'*60}")
    print(f"# Ablation: {config_name}")
    print(f"# Flags: {config_flags}")
    print(f"{'#'*60}")

    seed = 42
    n_splits = 5
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_json = os.path.join(BASE, "dataset/output/all_herbs.json")
    with open(all_json, 'r', encoding='utf-8') as f: all_data = json.load(f)
    features = np.array([d["feature_vector"] for d in all_data], dtype=np.float32)
    labels = np.array([d["label"] for d in all_data], dtype=np.float32)
    with open(os.path.join(BASE, "dataset/output/compound2id.json"), 'r') as f: c2id = json.load(f)
    n_compounds = len(c2id); n_labels = 5
    compound_mh = np.zeros((len(all_data), n_compounds), dtype=np.float32)
    for i, d in enumerate(all_data):
        for cid in d["compound_ids"]:
            if cid < n_compounds: compound_mh[i, cid] = 1.0
    compound_fps = np.load(os.path.join(BASE, "..", "数据爬取", "compound_fps.npy"))
    label_strs = [''.join(str(int(v)) for v in d['label']) for d in all_data]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, label_strs)):
        train_features = features[train_idx]; train_labels = labels[train_idx]
        val_features = features[val_idx]; val_labels = labels[val_idx]

        comp_train = compound_mh[train_idx]
        intersection = comp_train @ comp_train.T
        row_sum = comp_train.sum(axis=1, keepdims=True)
        union = row_sum + row_sum.T - intersection
        comp_sim_train = torch.tensor(intersection / np.maximum(union, 1), dtype=torch.float32)

        train_herbs = [all_data[i] for i in train_idx]
        c_labels = build_compound_labels(train_herbs, n_compounds)
        compound_epochs = 25 if config_flags.get('no_compound_opt', False) else 40
        comp_model = train_compound_fold(compound_fps, c_labels, train_idx, all_data,
                                          epochs=compound_epochs)

        if config_flags.get('no_prior', False):
            # 完全不用化合物知识迁移: prior = None → 使用 ProtoNet
            tr_prior = None
            va_prior = None
        elif config_flags.get('no_aggregator', False):
            tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data)
            va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data)
        else:
            aggregator = CompoundAttentionAggregator(prior_dim=5, hidden=32).to(device)
            aggregator = train_compound_aggregator(comp_model, aggregator, compound_fps, all_data, train_idx)
            tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data, aggregator)
            va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data, aggregator)

        base_feat = train_features
        chain_strategy = config_flags.get('chain_strategy', 'mi_auc')
        chains = compute_mi_auc_chains(base_feat, train_labels.astype(int), n_labels=n_labels, seed=seed + fold, strategy=chain_strategy)

        p1_ensemble = np.zeros((len(val_idx), n_labels), dtype=np.float32)
        p1_chain_probs = {}
        for ci, chain_order in enumerate(chains):
            val_probs = np.zeros((len(val_idx), n_labels), dtype=np.float32)
            for step, li in enumerate(chain_order):
                if step == 0:
                    tr_inp = train_features; va_inp = val_features
                else:
                    prev = chain_order[:step]
                    tr_inp = np.concatenate([train_features, train_labels[:, prev]], axis=1)
                    va_inp = np.concatenate([val_features, val_probs[:, prev]], axis=1)
                val_probs[:, li] = train_label_protonet(
                    tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                    comp_sim_train, device, seed=seed + fold * 100 + li * 7 + ci * 1000,
                    tr_prior=tr_prior, va_prior=va_prior, epochs=200)
            p1_ensemble += val_probs
            p1_chain_probs[ci] = val_probs.copy()
        p1_ensemble /= len(chains)

        if config_flags.get('no_p2', False):
            all_val_probs = p1_ensemble
        else:
            p2_ensemble = np.zeros((len(val_idx), n_labels), dtype=np.float32)
            for ci, chain_order in enumerate(chains):
                p1_val = p1_chain_probs[ci]
                val_probs2 = np.zeros((len(val_idx), n_labels), dtype=np.float32)
                for step, li in enumerate(chain_order):
                    if step == 0:
                        tr_inp = train_features; va_inp = val_features
                    else:
                        prev = chain_order[:step]
                        soft_cond = (train_labels[:, prev] * 0.8 + 0.1).astype(np.float32)
                        tr_inp = np.concatenate([train_features, soft_cond], axis=1)
                        va_inp = np.concatenate([val_features, p1_val[:, prev]], axis=1)
                    val_probs2[:, li] = train_label_protonet(
                        tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                        comp_sim_train, device, seed=seed + fold * 100 + li * 7 + ci * 1000 + 9999,
                        tr_prior=tr_prior, va_prior=va_prior, epochs=200)
                p2_ensemble += val_probs2
            p2_ensemble /= len(chains)
            all_val_probs = (p1_ensemble + p2_ensemble) / 2
        try: fold_auc = roc_auc_score(val_labels, all_val_probs, average='macro')
        except ValueError: fold_auc = 0.0
        thresholds = find_best_thresholds(val_labels, all_val_probs, n_labels)
        best_preds = np.zeros_like(all_val_probs, dtype=int)
        for i in range(n_labels): best_preds[:, i] = (all_val_probs[:, i] > thresholds[i]).astype(int)
        macro_f1 = f1_score(val_labels, best_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(val_labels, best_preds, average='micro', zero_division=0)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}  Macro-F1={macro_f1:.4f}  Micro-F1={micro_f1:.4f}")
        fold_results.append((fold_auc, macro_f1, micro_f1))

    aucs = [r[0] for r in fold_results]
    macro_f1s = [r[1] for r in fold_results]
    micro_f1s = [r[2] for r in fold_results]
    result = {
        'config': config_name,
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'macro_f1_mean': np.mean(macro_f1s), 'macro_f1_std': np.std(macro_f1s),
        'micro_f1_mean': np.mean(micro_f1s), 'micro_f1_std': np.std(micro_f1s),
    }
    print(f"\n  >>> {config_name}: AUC={result['auc_mean']:.4f}+/-{result['auc_std']:.4f}  "
          f"Macro-F1={result['macro_f1_mean']:.4f}+/-{result['macro_f1_std']:.4f}  "
          f"Micro-F1={result['micro_f1_mean']:.4f}+/-{result['micro_f1_std']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_prior", action="store_true", help="Remove compound→herb knowledge transfer entirely")
    parser.add_argument("--no_aggregator", action="store_true")
    parser.add_argument("--no_smooth_first", action="store_true")
    parser.add_argument("--no_knowledge_reg", action="store_true")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--no_dropadd", action="store_true")
    parser.add_argument("--no_compound_opt", action="store_true")
    parser.add_argument("--run_all", action="store_true", help="Run all ablation configs")
    args = parser.parse_args()

    if args.run_all:
        configs = [
            ("Full Model (current best)", {}),
            ("- Compound→Herb Knowledge Transfer", {"no_prior": True}),
            ("- Aggregator (mean pool)", {"no_aggregator": True}),
            ("ECC: Fixed Forward [0,1,2,3,4]", {"chain_strategy": "fixed_forward"}),
            ("ECC: Fixed Reverse [4,3,2,1,0]", {"chain_strategy": "fixed_reverse"}),
            ("ECC: Random Order", {"chain_strategy": "random"}),
            ("ECC: MI Only (no AUC weight)", {"chain_strategy": "mi_only"}),
            ("- P2 Ensemble (P1 only)", {"no_p2": True}),
            ("- Smoothing before Mixup", {"no_smooth_first": True}),
            ("- Knowledge Regularization", {"no_knowledge_reg": True}),
            ("- Mixup Augmentation", {"no_mixup": True}),
            ("- DropAdd Augmentation", {"no_dropadd": True}),
            ("- Compound Optimizations", {"no_compound_opt": True}),
        ]
        all_results = []
        for name, flags in configs:
            result = run_single_config(name, flags)
            all_results.append(result)

        print(f"\n{'='*70}")
        print("ABLATION STUDY SUMMARY (5-Fold CV, no data leak)")
        print(f"{'='*70}")
        print(f"{'Config':<40} {'AUC':>8} {'Macro-F1':>10} {'Micro-F1':>10}")
        print(f"{'-'*40} {'-'*8} {'-'*10} {'-'*10}")
        baseline = all_results[0]
        print(f"{'Full Model':<40} {baseline['auc_mean']:.4f}±{baseline['auc_std']:.2f}  {baseline['macro_f1_mean']:.4f}±{baseline['macro_f1_std']:.2f}  {baseline['micro_f1_mean']:.4f}±{baseline['micro_f1_std']:.2f}")
        for r in all_results[1:]:
            d_auc = r['auc_mean'] - baseline['auc_mean']
            d_macro = r['macro_f1_mean'] - baseline['macro_f1_mean']
            d_micro = r['micro_f1_mean'] - baseline['micro_f1_mean']
            print(f"{r['config']:<40} {r['auc_mean']:.4f}±{r['auc_std']:.2f}  {r['macro_f1_mean']:.4f}±{r['macro_f1_std']:.2f}  {r['micro_f1_mean']:.4f}±{r['micro_f1_std']:.2f}")
            print(f"  Δ vs Full:                 {d_auc:+.4f}       {d_macro:+.4f}        {d_micro:+.4f}")
    else:
        # Single config from command-line flags
        flags = {}
        config_name = "Custom"
        if args.no_prior: flags['no_prior'] = True; config_name = "- Knowledge Transfer"
        if args.no_aggregator: flags['no_aggregator'] = True; config_name = "- Aggregator"
        if args.no_smooth_first: flags['no_smooth_first'] = True; config_name = "- Smooth Order"
        if args.no_knowledge_reg: flags['no_knowledge_reg'] = True; config_name = "- Knowledge Reg"
        if args.no_mixup: flags['no_mixup'] = True; config_name = "- Mixup"
        if args.no_dropadd: flags['no_dropadd'] = True; config_name = "- DropAdd"
        if args.no_compound_opt: flags['no_compound_opt'] = True; config_name = "- Compound Opt"
        if not flags: config_name = "Full Model"
        run_single_config(config_name, flags)


if __name__ == "__main__":
    main()
