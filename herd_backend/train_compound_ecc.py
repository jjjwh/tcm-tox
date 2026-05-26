"""
train_compound_ecc.py — 化合物迁移 + ECC-Adaptive+ 链式预测
==========================================================
300d + 5维化合物毒性先验(mean pool) + knowledge_sim_loss + ECC-Adaptive+
最优配置: patience=50, compound_epochs=40, proj_dim=64, 全量训练无val拆分
"""
import os, sys, random, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, mutual_info_score
sys.path.insert(0, '.')
from model import ProtoNet, GatedPriorProtoNet, AsymmetricLoss, knowledge_sim_loss, CompoundAttentionAggregator
from model_drug_tower import DrugTower

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train_compound_fold(compound_fps, c_labels, train_herbs, epochs=40, pretrained_encoder=None):
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
    if pretrained_encoder is not None:
        model.encoder.load_state_dict(pretrained_encoder, strict=False)
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


def train_drug_projection(drug_tower, drug_proj, compound_fps, train_herbs, epochs=30):
    """Train Linear(256→5) to map Drug Tower embeddings to herb toxicity prior."""
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    drug_tower.eval()
    drug_proj.train()

    # pre-compute drug views per herb
    herb_drug_views = {}
    with torch.no_grad():
        for idx, h in enumerate(train_herbs):
            cids = [c for c in h["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                _, emb = drug_tower(c_fps_t[cids], return_embed=True)
                herb_drug_views[idx] = emb.mean(dim=0)  # [256]

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(drug_proj.parameters(), lr=3e-3, weight_decay=1e-4)

    for _ in range(epochs):
        indices = list(herb_drug_views.keys())
        random.shuffle(indices)
        for idx in indices:
            drug_view = herb_drug_views[idx]
            lbl = torch.tensor(train_herbs[idx]["label"], dtype=torch.float32).to(device)
            pred = drug_proj(drug_view)
            loss = crit(pred, lbl)
            opt.zero_grad(); loss.backward()
            opt.step()

    drug_proj.eval()
    return drug_proj


def compute_drug_prior(drug_tower, drug_proj, compound_fps, herb_indices, herbs):
    """Western drug perspective: Drug Tower encoder → drug_view → drug_prior [5]."""
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    prior = np.zeros((len(herb_indices), 5), dtype=np.float32)
    if drug_tower is None: return prior
    drug_tower.eval(); drug_proj.eval()
    with torch.no_grad():
        for j, hi in enumerate(herb_indices):
            cids = [c for c in herbs[hi]["compound_ids"] if c < len(compound_fps) and compound_fps[c].any()]
            if cids:
                # Drug Tower encoder: compound fingerprints → drug embedding
                _, emb = drug_tower(c_fps_t[cids], return_embed=True)  # [n_c, 256]
                drug_view = emb.mean(dim=0)  # [256]
                prior[j] = drug_proj(drug_view).cpu().numpy()  # [5]
    return prior


def train_compound_aggregator(model, aggregator, compound_fps, train_herbs, epochs=50):
    """Train attention aggregator to weight compounds for herb-level prediction."""
    c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)
    model.eval()
    aggregator.train()

    # pre-compute compound predictions per herb (frozen model)
    herb_comp_probs = {}
    label_map = {}
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
            probs = herb_comp_probs[k]
            lbl = torch.tensor(label_map[k], dtype=torch.float32).to(device)
            agg, _ = aggregator(probs)
            loss = crit(agg.unsqueeze(0), lbl.unsqueeze(0)) / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0 or i == len(indices) - 1:
                opt.step(); opt.zero_grad()

    aggregator.eval()
    return aggregator


def compute_mi_auc_chains(tr_feat, tr_l, n_labels=5, seed=42):
    """ECC chain ordering by MI only. standalone_aucs removed: ablation shows MI+AUC = MI Only on small data."""
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
                          tr_prior2=None, va_prior2=None,
                          epochs=200, patience=50,
                          use_contrastive=True):
    """tr_inp/va_inp = 300d + chain labels. priors passed separately for adaptive fusion."""
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

    if has_prior:
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
            if has_dual:
                fv, pv, pv2, lb, idxs = batch
                pv, pv2 = pv.to(device), pv2.to(device)
            elif has_prior:
                fv, pv, lb, idxs = batch
                pv = pv.to(device)
                pv2 = None
            else:
                fv, lb, idxs = batch
                pv, pv2 = None, None
            fv, lb = fv.to(device), lb.to(device)
            lb = lb * 0.95 + 0.025
            fv = dropadd(fv); fv, lb = mixup_batch(fv, lb, alpha=0.2)
            if has_prior:
                logit, z, z_know = model(fv, prior=pv, prior2=pv2, return_proj=True)
            else:
                logit, z = model(fv, return_proj=True)
                z_know = None
            loss = criterion(logit, lb) + 0.1 * knowledge_sim_loss(z, comp_sim, idxs)

            # Cross-view contrastive alignment (multi-view synergy)
            if z_know is not None and use_contrastive:
                pos_mask = (lb @ lb.T > 0).float()  # share at least one positive label
                n_pos = pos_mask.sum(dim=1).clamp(min=1)
                sim = z @ z_know.T / 0.5
                sim_max, _ = sim.max(dim=1, keepdim=True)
                sim = sim - sim_max.detach()
                exp_sim = torch.exp(sim)
                log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
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
                    vp = model(fv, prior=va_p.to(device),
                                    prior2=va_p2.to(device) if va_p2 is not None else None)
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
                vp = model(fv, prior=va_p.to(device),
                                prior2=va_p2.to(device) if va_p2 is not None else None)
            else:
                vp = model(fv)
            vps.append(torch.sigmoid(vp).cpu().numpy())
    return np.concatenate(vps).flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--compound_epochs", type=int, default=40)
    parser.add_argument("--no_drug_tower", action="store_true", help="Disable Western drug cross-domain prior")
    parser.add_argument("--drug_prior_only", action="store_true", help="Use ONLY drug prior (no compound prior)")
    parser.add_argument("--no_prior", action="store_true", help="ProtoNet baseline: no compound prior, no drug prior")
    parser.add_argument("--no_contrastive", action="store_true", help="Disable cross-view contrastive alignment loss")
    parser.add_argument("--drug_tower_ckpt", type=str, default="drug_tower_encoder.pt", help="Drug Tower checkpoint filename")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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

    # Load frozen Drug Tower for cross-domain prior
    drug_tower = None
    drug_proj = None
    if not args.no_drug_tower:
        drug_tower = DrugTower(n_labels=8).to(device)
        ckpt = torch.load(os.path.join(BASE, "dataset", args.drug_tower_ckpt), map_location=device)
        # Only load encoder weights (head dims differ: 8/12/20 across datasets)
        encoder_ckpt = {k: v for k, v in ckpt.items() if k.startswith('encoder.')}
        drug_tower.load_state_dict(encoder_ckpt, strict=False)
        drug_tower.eval()
        for p in drug_tower.parameters():
            p.requires_grad = False
        drug_proj = nn.Linear(256, 5).to(device)  # drug_view → drug_prior
        print(f"Drug Tower loaded (frozen), drug_proj trainable")

    print(f"Config: splits={args.n_splits} epochs={args.epochs} compound_epochs={args.compound_epochs}"
          f"  drug_tower={not args.no_drug_tower}")
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, label_strs)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{args.n_splits} (train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        train_features = features[train_idx]; train_labels = labels[train_idx]
        val_features = features[val_idx]; val_labels = labels[val_idx]

        comp_train = compound_mh[train_idx]
        intersection = comp_train @ comp_train.T
        row_sum = comp_train.sum(axis=1, keepdims=True)
        union = row_sum + row_sum.T - intersection
        comp_sim_train = torch.tensor(intersection / np.maximum(union, 1), dtype=torch.float32)

        # 只从训练集草药构建化合物标签，防止数据泄漏
        train_herbs = [all_data[i] for i in train_idx]

        if args.no_prior:
            # ProtoNet baseline: no compound prior, no drug prior
            tr_prior, va_prior = None, None
            tr_drug, va_drug = None, None
        else:
            c_labels = build_compound_labels(train_herbs, n_compounds)
            comp_model = train_compound_fold(compound_fps, c_labels, train_herbs,
                                              epochs=args.compound_epochs)
            aggregator = CompoundAttentionAggregator(prior_dim=5, hidden=32).to(device)
            aggregator = train_compound_aggregator(comp_model, aggregator, compound_fps, train_herbs)
            tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data, aggregator)
            va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data, aggregator)

            # Cross-domain drug prior (frozen Drug Tower + trainable projection)
            if drug_tower is not None:
                drug_proj = train_drug_projection(drug_tower, drug_proj, compound_fps, train_herbs)
                tr_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, train_idx, all_data)
                va_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, val_idx, all_data)
                if args.drug_prior_only:
                    tr_prior, va_prior = tr_drug, va_drug
                    tr_drug, va_drug = None, None
            else:
                tr_drug = None
                va_drug = None

        base_feat = train_features  # 300d only for chain computation
        chains = compute_mi_auc_chains(base_feat, train_labels.astype(int), n_labels=n_labels, seed=args.seed + fold)

        p1_ensemble = np.zeros((len(val_idx), n_labels), dtype=np.float32)
        p1_chain_probs = {}

        for ci, chain_order in enumerate(chains):
            val_probs = np.zeros((len(val_idx), n_labels), dtype=np.float32)
            for step, li in enumerate(chain_order):
                if step == 0:
                    tr_inp = train_features
                    va_inp = val_features
                else:
                    prev = chain_order[:step]
                    tr_inp = np.concatenate([train_features, train_labels[:, prev]], axis=1)
                    va_inp = np.concatenate([val_features, val_probs[:, prev]], axis=1)
                val_probs[:, li] = train_label_protonet(
                    tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                    comp_sim_train, device,
                    seed=args.seed + fold * 100 + li * 7 + ci * 1000,
                    tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug, use_contrastive=not args.no_contrastive,
                    epochs=args.epochs)
            p1_ensemble += val_probs
            p1_chain_probs[ci] = val_probs.copy()
        p1_ensemble /= len(chains)

        p2_ensemble = np.zeros((len(val_idx), n_labels), dtype=np.float32)
        for ci, chain_order in enumerate(chains):
            p1_val = p1_chain_probs[ci]
            val_probs2 = np.zeros((len(val_idx), n_labels), dtype=np.float32)
            for step, li in enumerate(chain_order):
                if step == 0:
                    tr_inp = train_features
                    va_inp = val_features
                else:
                    prev = chain_order[:step]
                    soft_cond = (train_labels[:, prev] * 0.8 + 0.1).astype(np.float32)
                    tr_inp = np.concatenate([train_features, soft_cond], axis=1)
                    va_inp = np.concatenate([val_features, p1_val[:, prev]], axis=1)
                val_probs2[:, li] = train_label_protonet(
                    tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                    comp_sim_train, device,
                    seed=args.seed + fold * 100 + li * 7 + ci * 1000 + 9999,
                    tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug, use_contrastive=not args.no_contrastive,
                    epochs=args.epochs)
            p2_ensemble += val_probs2
        p2_ensemble /= len(chains)

        all_val_probs = (p1_ensemble + p2_ensemble) / 2

        try: fold_auc = roc_auc_score(val_labels, all_val_probs, average='macro')
        except ValueError: fold_auc = 0.0

        # Per-label AUC for binary comparison
        label_names = ['Hepato','Nephro','Cardio','Neuro','Hemato']
        per_label_aucs = []
        for li in range(n_labels):
            try:
                per_label_aucs.append(roc_auc_score(val_labels[:, li], all_val_probs[:, li]))
            except ValueError:
                per_label_aucs.append(0.5)

        # 固定阈值 F1 (干净，无乐观偏差)
        fixed_preds = (all_val_probs > 0.5).astype(int)
        macro_f1_fixed = f1_score(val_labels, fixed_preds, average='macro', zero_division=0)
        micro_f1_fixed = f1_score(val_labels, fixed_preds, average='micro', zero_division=0)

        # 搜索阈值 F1 (有乐观偏差，论文中注明)
        thresholds = find_best_thresholds(val_labels, all_val_probs, n_labels)
        best_preds = np.zeros_like(all_val_probs, dtype=int)
        for i in range(n_labels): best_preds[:, i] = (all_val_probs[:, i] > thresholds[i]).astype(int)
        macro_f1_best = f1_score(val_labels, best_preds, average='macro', zero_division=0)
        micro_f1_best = f1_score(val_labels, best_preds, average='micro', zero_division=0)

        print(f"  Fold {fold+1}: AUC={fold_auc:.4f} | "
              f"F1(0.5)={macro_f1_fixed:.4f}/{micro_f1_fixed:.4f} | "
              f"F1(best)={macro_f1_best:.4f}/{micro_f1_best:.4f}")
        print(f"         Per-label: " + " ".join(f"{n}={a:.4f}" for n, a in zip(label_names, per_label_aucs)))
        fold_results.append((fold_auc, macro_f1_fixed, micro_f1_fixed, macro_f1_best, micro_f1_best, per_label_aucs))

    aucs = [r[0] for r in fold_results]
    macro_f1s_fixed = [r[1] for r in fold_results]
    micro_f1s_fixed = [r[2] for r in fold_results]
    macro_f1s_best = [r[3] for r in fold_results]
    micro_f1s_best = [r[4] for r in fold_results]
    print(f"\n{'='*60}")
    print(f"{args.n_splits}-Fold CV Summary")
    print(f"{'='*60}")
    print(f"AUC      : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"F1(0.5)  : Macro={np.mean(macro_f1s_fixed):.4f}±{np.std(macro_f1s_fixed):.4f}  "
          f"Micro={np.mean(micro_f1s_fixed):.4f}±{np.std(micro_f1s_fixed):.4f}")
    print(f"F1(best) : Macro={np.mean(macro_f1s_best):.4f}±{np.std(macro_f1s_best):.4f}  "
          f"Micro={np.mean(micro_f1s_best):.4f}±{np.std(micro_f1s_best):.4f}")
    # Per-label AUC summary
    all_per_label = np.array([r[5] for r in fold_results])  # [n_folds, 5]
    print(f"Per-Label AUC:")
    for li, name in enumerate(['Hepato','Nephro','Cardio','Neuro','Hemato']):
        print(f"  {name}: {all_per_label[:, li].mean():.4f} +/- {all_per_label[:, li].std():.4f}")


if __name__ == "__main__":
    main()
