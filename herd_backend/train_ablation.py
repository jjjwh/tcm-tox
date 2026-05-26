"""
train_ablation.py — 消融实验脚本 (sync with train_compound_ecc.py)
================================================================
用法:
  python train_ablation.py --run_all          # 一键全跑
  python train_ablation.py --no_drug_tower    # 单组消融
  python train_ablation.py --no_prior         # ProtoNet 基线
"""
import os, sys, random, json, argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, mutual_info_score
sys.path.insert(0, '.')
from model import ProtoNet, GatedPriorProtoNet, AsymmetricLoss, knowledge_sim_loss, CompoundAttentionAggregator
from model_drug_tower import DrugTower

BASE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AB_FLAGS = {}

# ── Data helpers ─────────────────────────────────────────────

def build_compound_labels(herbs, n_compounds):
    c_labels = np.zeros((n_compounds, 5), dtype=np.float32)
    for h in herbs:
        for cid in h["compound_ids"]:
            if cid < n_compounds:
                c_labels[cid] = np.maximum(c_labels[cid], h["label"])
    return c_labels

# ── Compound model ───────────────────────────────────────────

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
    if AB_FLAGS.get('no_compound_opt', False):
        epochs = 25
    train_cids = set()
    for h in train_herbs:
        for cid in h["compound_ids"]:
            if cid < len(compound_fps) and c_labels[cid].sum() > 0:
                train_cids.add(cid)
    train_cids = sorted(train_cids)
    if len(train_cids) < 10: return None
    fps = torch.tensor(compound_fps[train_cids], dtype=torch.float32)
    lbls = torch.tensor(c_labels[train_cids], dtype=torch.float32)

    if AB_FLAGS.get('no_compound_opt', False):
        n = len(fps); n_tr = int(n * 0.85)
        idx = np.random.RandomState(42).permutation(n)
        tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
        model = CompoundToxModel(dropout=0.2).to(device)
        crit = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        loader = DataLoader(TensorDataset(fps[tr_idx], lbls[tr_idx]), batch_size=min(64, n_tr), shuffle=True, drop_last=True)
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
                with torch.no_grad(): vp = torch.sigmoid(model(fps[va_idx].to(device))).cpu().numpy()
                try: auc = roc_auc_score(lbls[va_idx].numpy(), vp, average='macro')
                except: auc = 0.0
                if auc > best_auc: best_auc = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if best_state: model.load_state_dict(best_state)
    else:
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

# ── Prior computation ────────────────────────────────────────

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

# ── Chain ordering ───────────────────────────────────────────

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
    standalone_aucs = np.ones(n_labels) if strategy == 'mi_only' else np.full(n_labels, 0.5)
    if strategy != 'mi_only':
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        for li in range(n_labels):
            try:
                scores = cross_val_score(LogisticRegression(max_iter=300, C=1.0, solver='lbfgs', random_state=seed),
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

# ── Augmentation ─────────────────────────────────────────────

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
            pred = (y_prob[:, i] > t).astype(int)
            f = f1_score(y_true[:, i], pred, zero_division=0)
            if f > bf: bf, bt = f, t
        thresholds.append(bt)
    return thresholds

# ── Core training ────────────────────────────────────────────

def train_label_protonet(tr_inp, tr_lbl, va_inp, va_lbl, comp_sim, device, seed,
                          tr_prior=None, va_prior=None,
                          tr_prior2=None, va_prior2=None,
                          epochs=200, patience=50,
                          use_contrastive=True):
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

            # Ablation flags
            if AB_FLAGS.get('no_smooth_first', False):
                fv = dropadd(fv) if not AB_FLAGS.get('no_dropadd', False) else fv
                fv, lb = mixup_batch(fv, lb, alpha=0.0 if AB_FLAGS.get('no_mixup', False) else 0.2)
                lb = lb * 0.95 + 0.025
            else:
                lb = lb * 0.95 + 0.025
                fv = dropadd(fv) if not AB_FLAGS.get('no_dropadd', False) else fv
                fv, lb = mixup_batch(fv, lb, alpha=0.0 if AB_FLAGS.get('no_mixup', False) else 0.2)

            if has_prior:
                logit, z, z_know = model(fv, prior=pv, prior2=pv2, return_proj=True)
            else:
                logit, z = model(fv, return_proj=True)
                z_know = None

            if AB_FLAGS.get('no_knowledge_reg', False):
                loss = criterion(logit, lb)
            else:
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

# ── Run single config ────────────────────────────────────────

def run_single_config(config_name, config_flags):
    global AB_FLAGS
    AB_FLAGS = config_flags
    print(f"\n{'#'*60}\n# Ablation: {config_name}\n# Flags: {config_flags}\n{'#'*60}")

    seed = 42
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

    # Load Drug Tower (unless disabled)
    drug_tower = None; drug_proj = None
    if not config_flags.get('no_drug_tower', False):
        drug_tower = DrugTower(n_labels=8).to(device)
        ckpt = torch.load(os.path.join(BASE, "dataset", "drug_tower_encoder.pt"), map_location=device)
        drug_tower.load_state_dict(ckpt)
        drug_tower.eval()
        for p in drug_tower.parameters(): p.requires_grad = False
        drug_proj = nn.Linear(256, 5).to(device)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
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

        if config_flags.get('no_prior', False):
            tr_prior, va_prior, tr_drug, va_drug = None, None, None, None
        else:
            c_labels = build_compound_labels(train_herbs, n_compounds)
            compound_epochs = 25 if config_flags.get('no_compound_opt', False) else 40
            comp_model = train_compound_fold(compound_fps, c_labels, train_herbs, epochs=compound_epochs)

            if config_flags.get('no_aggregator', False):
                tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data)
                va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data)
            else:
                aggregator = CompoundAttentionAggregator(prior_dim=5, hidden=32).to(device)
                aggregator = train_compound_aggregator(comp_model, aggregator, compound_fps, train_herbs)
                tr_prior = compute_prior(comp_model, compound_fps, train_idx, all_data, aggregator)
                va_prior = compute_prior(comp_model, compound_fps, val_idx, all_data, aggregator)

            if drug_tower is not None:
                drug_proj = train_drug_projection(drug_tower, drug_proj, compound_fps, train_herbs)
                tr_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, train_idx, all_data)
                va_drug = compute_drug_prior(drug_tower, drug_proj, compound_fps, val_idx, all_data)
                if config_flags.get('drug_prior_only', False):
                    tr_prior, va_prior = tr_drug, va_drug
                    tr_drug, va_drug = None, None
            else:
                tr_drug, va_drug = None, None

        chain_strategy = config_flags.get('chain_strategy', 'mi_auc')
        chains = compute_mi_auc_chains(train_features, train_labels.astype(int), n_labels=n_labels, seed=seed + fold, strategy=chain_strategy)

        use_contrastive = not config_flags.get('no_contrastive', False)

        p1_ensemble = np.zeros((len(val_idx), n_labels), dtype=np.float32)
        p1_chain_probs = {}
        for ci, chain_order in enumerate(chains):
            val_probs = np.zeros((len(val_idx), n_labels), dtype=np.float32)
            for step, li in enumerate(chain_order):
                if step == 0:
                    tr_inp, va_inp = train_features, val_features
                else:
                    prev = chain_order[:step]
                    tr_inp = np.concatenate([train_features, train_labels[:, prev]], axis=1)
                    va_inp = np.concatenate([val_features, val_probs[:, prev]], axis=1)
                val_probs[:, li] = train_label_protonet(
                    tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                    comp_sim_train, device, seed=seed + fold * 100 + li * 7 + ci * 1000,
                    tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug,
                    epochs=200, use_contrastive=use_contrastive)
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
                        tr_inp, va_inp = train_features, val_features
                    else:
                        prev = chain_order[:step]
                        soft_cond = (train_labels[:, prev] * 0.8 + 0.1).astype(np.float32)
                        tr_inp = np.concatenate([train_features, soft_cond], axis=1)
                        va_inp = np.concatenate([val_features, p1_val[:, prev]], axis=1)
                    val_probs2[:, li] = train_label_protonet(
                        tr_inp, train_labels[:, li:li+1], va_inp, val_labels[:, li:li+1],
                        comp_sim_train, device, seed=seed + fold * 100 + li * 7 + ci * 1000 + 9999,
                        tr_prior=tr_prior, va_prior=va_prior, tr_prior2=tr_drug, va_prior2=va_drug,
                        epochs=200, use_contrastive=use_contrastive)
                p2_ensemble += val_probs2
            p2_ensemble /= len(chains)
            all_val_probs = (p1_ensemble + p2_ensemble) / 2

        try: fold_auc = roc_auc_score(val_labels, all_val_probs, average='macro')
        except ValueError: fold_auc = 0.0
        fixed_preds = (all_val_probs > 0.5).astype(int)
        macro_f1_fixed = f1_score(val_labels, fixed_preds, average='macro', zero_division=0)
        micro_f1_fixed = f1_score(val_labels, fixed_preds, average='micro', zero_division=0)
        thresholds = find_best_thresholds(val_labels, all_val_probs, n_labels)
        best_preds = np.zeros_like(all_val_probs, dtype=int)
        for i in range(n_labels): best_preds[:, i] = (all_val_probs[:, i] > thresholds[i]).astype(int)
        macro_f1_best = f1_score(val_labels, best_preds, average='macro', zero_division=0)
        micro_f1_best = f1_score(val_labels, best_preds, average='micro', zero_division=0)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f} | F1(0.5)={macro_f1_fixed:.4f}/{micro_f1_fixed:.4f} | F1(best)={macro_f1_best:.4f}/{micro_f1_best:.4f}")
        fold_results.append((fold_auc, macro_f1_fixed, micro_f1_fixed, macro_f1_best, micro_f1_best))

    aucs = [r[0] for r in fold_results]
    result = {
        'config': config_name,
        'auc_mean': np.mean(aucs), 'auc_std': np.std(aucs),
        'f1_fixed_macro': np.mean([r[1] for r in fold_results]),
        'f1_best_macro': np.mean([r[3] for r in fold_results]),
        'f1_best_micro': np.mean([r[4] for r in fold_results]),
    }
    print(f"\n  >>> {config_name}: AUC={result['auc_mean']:.4f}+/-{result['auc_std']:.4f}  "
          f"F1(fixed)={result['f1_fixed_macro']:.4f}  F1(best)={result['f1_best_macro']:.4f}/{result['f1_best_micro']:.4f}")
    return result

# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_aggregator", action="store_true")
    parser.add_argument("--no_smooth_first", action="store_true")
    parser.add_argument("--no_knowledge_reg", action="store_true")
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--no_dropadd", action="store_true")
    parser.add_argument("--no_compound_opt", action="store_true")
    parser.add_argument("--no_drug_tower", action="store_true")
    parser.add_argument("--drug_prior_only", action="store_true")
    parser.add_argument("--no_prior", action="store_true")
    parser.add_argument("--no_contrastive", action="store_true")
    parser.add_argument("--no_p2", action="store_true")
    parser.add_argument("--chain_strategy", type=str, default="mi_auc")
    parser.add_argument("--run_all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        configs = [
            ("Full Model (current best)", {}),
            ("- Drug Tower (no cross-domain)", {"no_drug_tower": True}),
            ("- Drug Tower + Drug Prior Only", {"drug_prior_only": True}),
            ("- All Priors (ProtoNet baseline)", {"no_prior": True}),
            ("- Aggregator (mean pool)", {"no_aggregator": True}),
            ("- Contrastive Alignment", {"no_contrastive": True}),
            ("- P2 Ensemble (P1 only)", {"no_p2": True}),
            ("- Jaccard Knowledge Reg", {"no_knowledge_reg": True}),
            ("- Mixup Augmentation", {"no_mixup": True}),
            ("- DropAdd Augmentation", {"no_dropadd": True}),
            ("- Smooth Order", {"no_smooth_first": True}),
            ("- Compound Optimizations", {"no_compound_opt": True}),
            ("ECC: Fixed Forward [0,1,2,3,4]", {"chain_strategy": "fixed_forward"}),
            ("ECC: Fixed Reverse [4,3,2,1,0]", {"chain_strategy": "fixed_reverse"}),
            ("ECC: Random Order", {"chain_strategy": "random"}),
            ("ECC: MI Only (no AUC weight)", {"chain_strategy": "mi_only"}),
        ]
        all_results = []
        for name, flags in configs:
            result = run_single_config(name, flags)
            all_results.append(result)

        print(f"\n{'='*70}")
        print("ABLATION STUDY SUMMARY (5-Fold CV, no data leak)")
        print(f"{'='*70}")
        baseline = all_results[0]
        print(f"{'Config':<40} {'AUC':>8} {'F1(fix)':>8} {'F1(best)':>10}")
        print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*10}")
        print(f"{baseline['config']:<40} {baseline['auc_mean']:.4f}±{baseline['auc_std']:.2f}  {baseline['f1_fixed_macro']:.4f}    {baseline['f1_best_macro']:.4f}/{baseline['f1_best_micro']:.4f}")
        for r in all_results[1:]:
            d_auc = r['auc_mean'] - baseline['auc_mean']
            print(f"{r['config']:<40} {r['auc_mean']:.4f}±{r['auc_std']:.2f}  {r['f1_fixed_macro']:.4f}    {r['f1_best_macro']:.4f}/{r['f1_best_micro']:.4f}")
            print(f"  Δ: {d_auc:+.4f}")
    else:
        flags = {}
        config_name = "Custom"
        for k in ['no_aggregator','no_smooth_first','no_knowledge_reg','no_mixup','no_dropadd',
                   'no_compound_opt','no_drug_tower','drug_prior_only','no_prior','no_contrastive','no_p2']:
            if getattr(args, k, False): flags[k] = True
        if args.chain_strategy != 'mi_auc': flags['chain_strategy'] = args.chain_strategy
        if not flags: config_name = "Full Model"
        else: config_name = "Custom (" + ",".join(flags.keys()) + ")"
        run_single_config(config_name, flags)

if __name__ == "__main__":
    main()
