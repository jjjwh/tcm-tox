"""
compare_dual_view.py
====================
对比实验: 当前方法 vs 加入 DualViewModel 跨视图对比损失

三个任务:
  A. TCM 5类多标签  (train.py 基准)
  B. 肝毒性二分类   (train_tox.py 基准)
  C. 西药 8类毒性   (train_drug.py 基准)

每个任务分别跑:
  [baseline] 原始 ProtoNet (+knowledge_sim_loss, 对 A/B)
  [dual_view] 加入 DualViewModel 跨视图对比损失

Usage:
    python compare_dual_view.py
    python compare_dual_view.py --task all   # 跑全部
    python compare_dual_view.py --task A     # 只跑 TCM
    python compare_dual_view.py --task B     # 只跑肝毒性
    python compare_dual_view.py --task C     # 只跑西药
"""

import os, sys, random, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# ──────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_batch(fv, lb, alpha=0.2):
    if alpha <= 0:
        return fv, lb
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(fv.size(0))
    return lam * fv + (1 - lam) * fv[idx], lam * lb + (1 - lam) * lb[idx]


def dropadd(fv, drop_p=0.20, add_p=0.01):
    mask = (torch.rand_like(fv) > drop_p).float()
    fv = fv * mask
    add_mask = (torch.rand_like(fv) < add_p) & (fv == 0)
    return torch.clamp(fv + add_mask.float(), 0, 1)


def find_best_thresholds(y_true, y_prob, n_labels):
    thresholds = []
    for i in range(n_labels):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.90, 18):
            pred = (y_prob[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds


def knowledge_sim_loss(z, know_sim_matrix, indices):
    z_norm = F.normalize(z, dim=1)
    embed_sim = z_norm @ z_norm.T
    target_sim = know_sim_matrix[indices][:, indices].to(z.device)
    return F.mse_loss(embed_sim, target_sim)


# ──────────────────────────────────────────────────
# 模型定义
# ──────────────────────────────────────────────────

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos
        self.clip = clip; self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y; pt1 = xs_neg * (1 - y); pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            loss *= torch.pow(1 - pt, one_sided_gamma)
        return -loss.sum() / x.size(0)


class ProtoNet(nn.Module):
    """原始 ProtoNet (基准)"""
    def __init__(self, feat_dim=300, proj_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        )
        self.proj = nn.Linear(256, proj_dim)
        self.proto_pos = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.proto_neg = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, return_proj=False):
        h = self.encoder(x)
        z = F.normalize(self.proj(h), dim=1)
        pp = F.normalize(self.proto_pos.unsqueeze(0), dim=1)
        pn = F.normalize(self.proto_neg.unsqueeze(0), dim=1)
        logit = self.scale * ((z * pp).sum(1, keepdim=True) - (z * pn).sum(1, keepdim=True))
        if return_proj:
            return logit, z
        return logit


class DualViewModel(nn.Module):
    """
    双视图模型:
      - 特征视图: feat_dim → encoder → head → logit
      - 知识视图: know_dim → know_encoder → know_proj
      - 跨视图对比对齐 (CrossViewLabelConLoss)
      - 分类只用特征视图
    """
    def __init__(self, feat_dim=300, know_dim=32, proj_dim=64):
        super().__init__()
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        )
        self.feat_proj = nn.Linear(256, proj_dim)
        self.proto_pos = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.proto_neg = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.scale = nn.Parameter(torch.tensor(10.0))

        self.know_encoder = nn.Sequential(
            nn.Linear(know_dim, 64), nn.ReLU(),
            nn.Linear(64, proj_dim),
        )

    def forward(self, feat, know=None, return_proj=False):
        h = self.feat_encoder(feat)
        z_feat = F.normalize(self.feat_proj(h), dim=1)
        pp = F.normalize(self.proto_pos.unsqueeze(0), dim=1)
        pn = F.normalize(self.proto_neg.unsqueeze(0), dim=1)
        logit = self.scale * ((z_feat * pp).sum(1, keepdim=True) - (z_feat * pn).sum(1, keepdim=True))
        if return_proj and know is not None:
            z_know = self.know_encoder(know)
            return logit, z_feat, z_know
        return logit


class CrossViewLabelConLoss(nn.Module):
    """跨视图有监督对比损失 (以标签为正负对依据)"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature

    def forward(self, z_feat, z_know, labels):
        B = z_feat.size(0)
        z_feat = F.normalize(z_feat, dim=1)
        z_know = F.normalize(z_know, dim=1)
        sim = z_feat @ z_know.T / self.temp

        lab = labels.squeeze()
        # 对多标签任务用 AND 逻辑来定义"正对"（共享任意正类）
        if lab.dim() == 1:
            pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
        else:
            # 至少有一个共同阳性标签
            pos_mask = ((lab.unsqueeze(0) * lab.unsqueeze(1)).sum(-1) > 0).float()

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()


# ──────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────

def load_tcm_data():
    """加载 TCM 5类数据"""
    all_json_path = os.path.join(BASE_DIR, "dataset/output/all_herbs.json")
    with open(all_json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    with open(os.path.join(BASE_DIR, "dataset/output/compound2id.json"), 'r') as f:
        c2id = json.load(f)

    features = np.array([d["feature_vector"] for d in all_data], dtype=np.float32)
    labels = np.array([d["label"] for d in all_data], dtype=np.float32)

    n_compounds = len(c2id)
    compound_mh = np.zeros((len(all_data), n_compounds), dtype=np.float32)
    for i, d in enumerate(all_data):
        for cid in d["compound_ids"]:
            compound_mh[i, cid] = 1.0

    label_strs = [''.join(str(int(v)) for v in d['label']) for d in all_data]
    return features, labels, compound_mh, label_strs


def load_liver_data():
    """加载肝毒性数据"""
    all_json_path = os.path.join(BASE_DIR, "dataset/output/all_herbs.json")
    with open(all_json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    with open(os.path.join(BASE_DIR, "dataset/output/compound2id.json"), 'r') as f:
        c2id = json.load(f)

    features = np.array([d["feature_vector"] for d in all_data], dtype=np.float32)
    label_path = os.path.join(BASE_DIR, "dataset", "TCM_Labels_Liver.npy")
    labels = np.load(label_path).astype(np.float32).squeeze()

    n_compounds = len(c2id)
    compound_mh = np.zeros((len(all_data), n_compounds), dtype=np.float32)
    for i, d in enumerate(all_data):
        for cid in d["compound_ids"]:
            compound_mh[i, cid] = 1.0

    return features, labels, compound_mh


def load_drug_data():
    """加载西药数据"""
    feat_path = os.path.join(BASE_DIR, "dataset", "Drug_Morgan.npy")
    label_path = os.path.join(BASE_DIR, "dataset", "Drug_Labels.npy")
    features = np.load(feat_path).astype(np.float32)   # (1349, 1024)
    labels = np.load(label_path).astype(np.float32)   # (1349, 8)
    return features, labels


# ──────────────────────────────────────────────────
# 核心训练循环 (单标签)
# ──────────────────────────────────────────────────

def train_single_label(
    features, labels, compound_mh, use_dual_view,
    n_splits=5, seed=42, epochs=120, patience=25,
    batch_size=32, know_svd_dim=32,
):
    """用于肝毒性 (单标签二分类)"""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    con_loss_fn = CrossViewLabelConLoss(temperature=0.5) if use_dual_view else None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels.astype(int))):
        set_seed(seed + fold * 100)
        tr_f = torch.tensor(features[train_idx], dtype=torch.float32)
        tr_l = torch.tensor(labels[train_idx, np.newaxis], dtype=torch.float32)
        va_f = torch.tensor(features[val_idx], dtype=torch.float32)
        va_l = torch.tensor(labels[val_idx, np.newaxis], dtype=torch.float32)

        # compound Jaccard sim (for baseline reg)
        comp_train = compound_mh[train_idx]
        intersection = comp_train @ comp_train.T
        row_sum = comp_train.sum(axis=1, keepdims=True)
        union = row_sum + row_sum.T - intersection
        comp_sim_train = torch.tensor(intersection / np.maximum(union, 1), dtype=torch.float32)

        # SVD for dual view
        if use_dual_view:
            svd = TruncatedSVD(n_components=know_svd_dim, random_state=seed)
            know_train = torch.tensor(svd.fit_transform(comp_train), dtype=torch.float32)
            know_val = torch.tensor(svd.transform(compound_mh[val_idx]), dtype=torch.float32)
            train_loader = DataLoader(
                TensorDataset(tr_f, know_train, tr_l, torch.arange(len(train_idx))),
                batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(TensorDataset(va_f, know_val, va_l), batch_size=64)
            model = DualViewModel(feat_dim=features.shape[1], know_dim=know_svd_dim).to(device)
        else:
            train_loader = DataLoader(
                TensorDataset(tr_f, tr_l, torch.arange(len(train_idx))),
                batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(TensorDataset(va_f, va_l), batch_size=64)
            model = ProtoNet(feat_dim=features.shape[1]).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_score, patience_counter, best_state = 0.0, 0, None

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                if use_dual_view:
                    fv, kv, lb, idxs = batch
                    fv, kv, lb = fv.to(device), kv.to(device), lb.to(device)
                    fv = dropadd(fv)
                    fv, lb = mixup_batch(fv, lb, alpha=0.1)
                    lb_smooth = lb * (1 - 2 * 0.025) + 0.025
                    logit, z_feat, z_know = model(fv, kv, return_proj=True)
                    cls_loss = criterion(logit, lb_smooth)
                    cv_loss = con_loss_fn(z_feat, z_know, lb)
                    loss = cls_loss + 0.3 * cv_loss
                else:
                    fv, lb, idxs = batch
                    fv, lb = fv.to(device), lb.to(device)
                    fv = dropadd(fv)
                    fv, lb = mixup_batch(fv, lb, alpha=0.1)
                    lb_smooth = lb * (1 - 2 * 0.025) + 0.025
                    logit, z = model(fv, return_proj=True)
                    cls_loss = criterion(logit, lb_smooth)
                    reg_loss = knowledge_sim_loss(z, comp_sim_train, idxs)
                    loss = cls_loss + 0.1 * reg_loss

                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); model.scale.data.clamp_(1.0, 30.0)

            scheduler.step()

            model.eval()
            vp_list, vl_list = [], []
            with torch.no_grad():
                for batch in val_loader:
                    fv = batch[0].to(device)
                    if use_dual_view:
                        kv = batch[1].to(device)
                        p = torch.sigmoid(model(fv, kv)).cpu().numpy()
                    else:
                        p = torch.sigmoid(model(fv)).cpu().numpy()
                    vp_list.append(p); vl_list.append(batch[-1].numpy())
            vp = np.concatenate(vp_list).flatten()
            vl = np.concatenate(vl_list).flatten()
            try:
                score = average_precision_score(vl, vp)
            except ValueError:
                score = 0.0

            if score > best_score:
                best_score = score; patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        vp_list = []
        with torch.no_grad():
            for batch in val_loader:
                fv = batch[0].to(device)
                if use_dual_view:
                    kv = batch[1].to(device)
                    p = torch.sigmoid(model(fv, kv)).cpu().numpy()
                else:
                    p = torch.sigmoid(model(fv)).cpu().numpy()
                vp_list.append(p)
        vp = np.concatenate(vp_list).flatten()
        vl_np = labels[val_idx]

        try:
            auroc = roc_auc_score(vl_np, vp)
            auprc = average_precision_score(vl_np, vp)
        except ValueError:
            auroc = auprc = 0.0

        thresholds_arr = np.linspace(0.05, 0.90, 18)
        best_f1, best_t = 0.0, 0.5
        for t in thresholds_arr:
            f1 = f1_score(vl_np, (vp > t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        f1 = best_f1
        print(f"  Fold {fold+1}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}")
        fold_results.append((auroc, auprc, f1))

    return fold_results


# ──────────────────────────────────────────────────
# 核心训练循环 (多标签)
# ──────────────────────────────────────────────────

def train_multi_label(
    features, labels, compound_mh_or_none, use_dual_view,
    n_labels, label_strs_or_stratify,
    n_splits=5, seed=42, epochs=200, patience=30,
    batch_size=32, know_svd_dim=32,
    drop_p=0.20, add_p=0.01,
):
    """用于 TCM 5类 和 西药 8类 (多标签, per-label ProtoNet)"""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    con_loss_fn = CrossViewLabelConLoss(temperature=0.5) if use_dual_view else None

    stratify_col = label_strs_or_stratify
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, stratify_col)):
        print(f"\n  --- Fold {fold+1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) ---")
        train_f_np = features[train_idx]
        val_f_np   = features[val_idx]
        train_l_np = labels[train_idx]
        val_l_np   = labels[val_idx]

        # compound sim (只在有 compound_mh 时才用)
        comp_sim_train = None
        know_train_np = None
        know_val_np = None
        if compound_mh_or_none is not None:
            comp_train = compound_mh_or_none[train_idx]
            intersection = comp_train @ comp_train.T
            row_sum = comp_train.sum(axis=1, keepdims=True)
            union = row_sum + row_sum.T - intersection
            comp_sim_train = torch.tensor(intersection / np.maximum(union, 1), dtype=torch.float32)

            if use_dual_view:
                svd = TruncatedSVD(n_components=know_svd_dim, random_state=seed)
                know_train_np = svd.fit_transform(comp_train)
                know_val_np   = svd.transform(compound_mh_or_none[val_idx])

        all_val_probs = np.zeros((len(val_idx), n_labels), dtype=np.float32)

        for li in range(n_labels):
            set_seed(seed + fold * 100 + li * 7)

            tr_f = torch.tensor(train_f_np, dtype=torch.float32)
            tr_l = torch.tensor(train_l_np[:, li:li+1], dtype=torch.float32)
            va_f = torch.tensor(val_f_np, dtype=torch.float32)
            va_l = torch.tensor(val_l_np[:, li:li+1], dtype=torch.float32)

            if use_dual_view and know_train_np is not None:
                tr_k = torch.tensor(know_train_np, dtype=torch.float32)
                va_k = torch.tensor(know_val_np, dtype=torch.float32)
                train_loader = DataLoader(
                    TensorDataset(tr_f, tr_k, tr_l, torch.arange(len(train_idx))),
                    batch_size=batch_size, shuffle=True, drop_last=True)
                val_loader = DataLoader(TensorDataset(va_f, va_k, va_l), batch_size=64)
                model = DualViewModel(feat_dim=features.shape[1], know_dim=know_svd_dim).to(device)
            elif not use_dual_view and comp_sim_train is not None:
                train_loader = DataLoader(
                    TensorDataset(tr_f, tr_l, torch.arange(len(train_idx))),
                    batch_size=batch_size, shuffle=True, drop_last=True)
                val_loader = DataLoader(TensorDataset(va_f, va_l), batch_size=64)
                model = ProtoNet(feat_dim=features.shape[1]).to(device)
            else:
                # 西药 baseline：没有 compound_mh，纯 ProtoNet
                train_loader = DataLoader(
                    TensorDataset(tr_f, tr_l),
                    batch_size=batch_size, shuffle=True, drop_last=True)
                val_loader = DataLoader(TensorDataset(va_f, va_l), batch_size=64)
                model = ProtoNet(feat_dim=features.shape[1]).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            best_auc, patience_counter, best_state = 0.0, 0, None

            for epoch in range(epochs):
                model.train()
                for batch in train_loader:
                    if use_dual_view and know_train_np is not None:
                        fv, kv, lb, idxs = batch
                        fv, kv, lb = fv.to(device), kv.to(device), lb.to(device)
                        fv = dropadd(fv, drop_p, add_p)
                        fv, lb = mixup_batch(fv, lb, alpha=0.2)
                        lb_s = lb * 0.95 + 0.025
                        logit, z_feat, z_know = model(fv, kv, return_proj=True)
                        cls_loss = criterion(logit, lb_s)
                        cv_loss = con_loss_fn(z_feat, z_know, lb)
                        loss = cls_loss + 0.3 * cv_loss
                    elif not use_dual_view and comp_sim_train is not None:
                        fv, lb, idxs = batch
                        fv, lb = fv.to(device), lb.to(device)
                        fv = dropadd(fv, drop_p, add_p)
                        fv, lb = mixup_batch(fv, lb, alpha=0.2)
                        lb_s = lb * 0.95 + 0.025
                        logit, z = model(fv, return_proj=True)
                        cls_loss = criterion(logit, lb_s)
                        reg_loss = knowledge_sim_loss(z, comp_sim_train, idxs)
                        loss = cls_loss + 0.1 * reg_loss
                    else:
                        # 西药 baseline: 只有 (fv, lb)
                        fv, lb = batch
                        fv, lb = fv.to(device), lb.to(device)
                        fv = dropadd(fv, drop_p, add_p)
                        fv, lb = mixup_batch(fv, lb, alpha=0.2)
                        lb_s = lb * 0.95 + 0.025
                        loss = criterion(model(fv), lb_s)

                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); model.scale.data.clamp_(1.0, 30.0)

                scheduler.step()

                model.eval()
                vp_list, vl_list = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        fv = batch[0].to(device)
                        if use_dual_view and know_train_np is not None:
                            kv = batch[1].to(device)
                            p = torch.sigmoid(model(fv, kv)).cpu().numpy()
                            lb_b = batch[2]
                        else:
                            p = torch.sigmoid(model(fv)).cpu().numpy()
                            lb_b = batch[1]
                        vp_list.append(p); vl_list.append(lb_b.numpy())
                vp = np.concatenate(vp_list).flatten()
                vl = np.concatenate(vl_list).flatten()
                try:
                    val_auc = roc_auc_score(vl, vp)
                except ValueError:
                    val_auc = 0.0

                if val_auc > best_auc:
                    best_auc = val_auc; patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break

            if best_state:
                model.load_state_dict(best_state)

            model.eval()
            vp_list = []
            with torch.no_grad():
                for batch in val_loader:
                    fv = batch[0].to(device)
                    if use_dual_view and know_train_np is not None:
                        kv = batch[1].to(device)
                        p = torch.sigmoid(model(fv, kv)).cpu().numpy()
                    else:
                        p = torch.sigmoid(model(fv)).cpu().numpy()
                    vp_list.append(p)
            all_val_probs[:, li] = np.concatenate(vp_list).flatten()

        # 折内聚合
        try:
            fold_auc = roc_auc_score(val_l_np, all_val_probs, average='macro')
        except ValueError:
            fold_auc = 0.0
        thresholds = find_best_thresholds(val_l_np, all_val_probs, n_labels)
        best_preds = np.zeros_like(all_val_probs, dtype=int)
        for i in range(n_labels):
            best_preds[:, i] = (all_val_probs[:, i] > thresholds[i]).astype(int)
        macro_f1 = f1_score(val_l_np, best_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(val_l_np, best_preds, average='micro', zero_division=0)
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}  Macro-F1={macro_f1:.4f}  Micro-F1={micro_f1:.4f}")
        fold_results.append((fold_auc, macro_f1, micro_f1))

    return fold_results


# ──────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────

def print_summary(name, fold_results, metric_names=("AUC/AUROC", "Macro-F1/AUPRC", "Micro-F1/F1")):
    vals = [np.array([r[i] for r in fold_results]) for i in range(len(fold_results[0]))]
    print(f"\n  [{name}]")
    for mn, v in zip(metric_names, vals):
        print(f"    {mn}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    return [np.mean(v) for v in vals], [np.std(v) for v in vals]


def run_task_A(n_splits=5):
    print(f"\n{'='*70}")
    print("任务 A: TCM 5类多标签毒性  (train.py 对照)")
    print(f"{'='*70}")
    features, labels, compound_mh, label_strs = load_tcm_data()
    print(f"样本数: {len(features)}  特征维: {features.shape[1]}  标签数: 5")

    print("\n[Baseline] ProtoNet + knowledge_sim_loss")
    res_base = train_multi_label(
        features, labels, compound_mh, use_dual_view=False,
        n_labels=5, label_strs_or_stratify=label_strs,
        n_splits=n_splits, seed=SEED,
    )
    print("\n[DualView] DualViewModel + CrossViewLabelConLoss")
    res_dual = train_multi_label(
        features, labels, compound_mh, use_dual_view=True,
        n_labels=5, label_strs_or_stratify=label_strs,
        n_splits=n_splits, seed=SEED,
    )

    print(f"\n{'─'*50}")
    print("任务 A 汇总:")
    m_b, s_b = print_summary("Baseline", res_base, ("macro-AUC", "Macro-F1", "Micro-F1"))
    m_d, s_d = print_summary("DualView", res_dual, ("macro-AUC", "Macro-F1", "Micro-F1"))
    delta_auc = m_d[0] - m_b[0]
    print(f"\n  ΔmacroAUC  = {delta_auc:+.4f}  ({'↑改善' if delta_auc > 0 else '↓下降'})")
    return {"task": "A (TCM 5-class)", "baseline": m_b, "dual": m_d}


def run_task_B(n_splits=5):
    print(f"\n{'='*70}")
    print("任务 B: 肝毒性二分类  (train_tox.py 对照)")
    print(f"{'='*70}")
    features, labels, compound_mh = load_liver_data()
    print(f"样本数: {len(features)}  特征维: {features.shape[1]}  阳性: {int(labels.sum())}")

    print("\n[Baseline] ProtoNet + knowledge_sim_loss")
    res_base = train_single_label(
        features, labels, compound_mh, use_dual_view=False,
        n_splits=n_splits, seed=SEED,
    )
    print("\n[DualView] DualViewModel + CrossViewLabelConLoss")
    res_dual = train_single_label(
        features, labels, compound_mh, use_dual_view=True,
        n_splits=n_splits, seed=SEED,
    )

    print(f"\n{'─'*50}")
    print("任务 B 汇总:")
    m_b, s_b = print_summary("Baseline", res_base, ("AUROC", "AUPRC", "F1"))
    m_d, s_d = print_summary("DualView", res_dual, ("AUROC", "AUPRC", "F1"))
    delta_auroc = m_d[0] - m_b[0]
    print(f"\n  ΔAUROC  = {delta_auroc:+.4f}  ({'↑改善' if delta_auroc > 0 else '↓下降'})")
    return {"task": "B (Liver)", "baseline": m_b, "dual": m_d}


def run_task_C(n_splits=5):
    print(f"\n{'='*70}")
    print("任务 C: 西药 8类毒性  (train_drug.py 对照)")
    print(f"  注: 西药无化合物知识图谱，Baseline=纯ProtoNet, DualView=用Morgan指纹SVD作知识视图")
    print(f"{'='*70}")
    features, labels = load_drug_data()
    N = len(features)
    print(f"样本数: {N}  特征维: {features.shape[1]}  标签数: 8")

    # 西药: 以最稀有标签分层
    stratify_col = labels[:, np.argmin(labels.mean(axis=0))].astype(int).tolist()

    # 西药双视图: 用 Morgan 指纹本身做 SVD 降维作为"知识视图"
    # (1024d Morgan → 32d SVD)
    morgan_mh = features  # 本身就是 1024 位二进制

    print("\n[Baseline] ProtoNet (Morgan 1024d, 无知识视图)")
    res_base = train_multi_label(
        features, labels, None, use_dual_view=False,
        n_labels=8, label_strs_or_stratify=stratify_col,
        n_splits=n_splits, seed=SEED, epochs=200, patience=30, batch_size=64,
        drop_p=0.10, add_p=0.01,
    )

    print("\n[DualView] ProtoNet + Morgan SVD32 跨视图对比")
    res_dual = train_multi_label(
        features, labels, morgan_mh, use_dual_view=True,
        n_labels=8, label_strs_or_stratify=stratify_col,
        n_splits=n_splits, seed=SEED, epochs=200, patience=30, batch_size=64,
        drop_p=0.10, add_p=0.01,
    )

    print(f"\n{'─'*50}")
    print("任务 C 汇总:")
    m_b, s_b = print_summary("Baseline", res_base, ("macro-AUC", "Macro-F1", "Micro-F1"))
    m_d, s_d = print_summary("DualView", res_dual, ("macro-AUC", "Macro-F1", "Micro-F1"))
    delta_auc = m_d[0] - m_b[0]
    print(f"\n  ΔmacroAUC  = {delta_auc:+.4f}  ({'↑改善' if delta_auc > 0 else '↓下降'})")
    return {"task": "C (Drug 8-class)", "baseline": m_b, "dual": m_d}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["all", "A", "B", "C"])
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    all_results = []
    if args.task in ("all", "A"):
        all_results.append(run_task_A(args.n_splits))
    if args.task in ("all", "B"):
        all_results.append(run_task_B(args.n_splits))
    if args.task in ("all", "C"):
        all_results.append(run_task_C(args.n_splits))

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("全部任务总结")
        print(f"{'='*70}")
        print(f"{'任务':<25} {'Baseline主指标':>12} {'DualView主指标':>14} {'Δ':>8}")
        print(f"{'─'*60}")
        for r in all_results:
            b0 = r['baseline'][0]; d0 = r['dual'][0]
            delta = d0 - b0
            mark = '↑' if delta > 0 else '↓'
            print(f"{r['task']:<25} {b0:>12.4f} {d0:>14.4f} {mark}{abs(delta):>7.4f}")
