import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Shared encoder backbone: 300 → 512 → 256."""
    def __init__(self, feat_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class ProtoNet(nn.Module):
    """
    Prototype-based classification via cosine similarity.
    Maps features to a projection space, then classifies by distance
    to learnable positive/negative prototypes.
    """
    def __init__(self, feat_dim=300, proj_dim=64):
        super().__init__()
        self.encoder = Encoder(feat_dim)
        self.proj = nn.Linear(256, proj_dim)
        self.proto_pos = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.proto_neg = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, return_proj=False):
        h = self.encoder(x)
        z = F.normalize(self.proj(h), dim=1)
        pp = F.normalize(self.proto_pos.unsqueeze(0), dim=1)
        pn = F.normalize(self.proto_neg.unsqueeze(0), dim=1)
        sim_pos = (z * pp).sum(dim=1, keepdim=True)
        sim_neg = (z * pn).sum(dim=1, keepdim=True)
        logit = self.scale * (sim_pos - sim_neg)
        if return_proj:
            return logit, z
        return logit


class GatedPriorProtoNet(nn.Module):
    """
    ProtoNet + Gated Prior Injection (残差门控).
    prior 走独立门控路径，以残差方式注入 encoder 的 256 维表示。
    不改变 proj 层维度，不影响 prototype 空间。
    最坏情况 gate→0 → h=h → 等价基线。

    Design choice: gate只看prior不看features.
    在小数据(252样本)下, context-aware gate (261→128) 引入33K参数导致过拟合,
    简单设计 Linear(5→20→256)=100参数 在小数据上更稳定.
    """
    def __init__(self, feat_dim=300, prior_dim=5, proj_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        )
        self.prior_proj = nn.Sequential(
            nn.Linear(prior_dim, 64), nn.ReLU(),
            nn.Linear(64, 256),
        )
        self.gate = nn.Sequential(
            nn.Linear(prior_dim, prior_dim * 4), nn.ReLU(),
            nn.Linear(prior_dim * 4, 256),
            nn.Sigmoid(),
        )
        # 初始偏置 = 0 → sigmoid(0) ≈ 0.5，训练初期信任一半先验
        nn.init.zeros_(self.gate[-2].bias)

        self.proj = nn.Linear(256, proj_dim)
        self.proto_pos = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.proto_neg = nn.Parameter(torch.randn(proj_dim) * 0.1)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, prior=None, return_proj=False):
        h = self.encoder(x)
        if prior is not None:
            gate_val = self.gate(prior)
            prior_emb = self.prior_proj(prior)
            h = h + gate_val * prior_emb
        z = F.normalize(self.proj(h), dim=1)
        pp = F.normalize(self.proto_pos.unsqueeze(0), dim=1)
        pn = F.normalize(self.proto_neg.unsqueeze(0), dim=1)
        logit = self.scale * ((z * pp).sum(1, keepdim=True) - (z * pn).sum(1, keepdim=True))
        if return_proj:
            return logit, z
        return logit


class DualViewModel(nn.Module):
    """
    Dual-view model for cross-view contrastive feature alignment.
    Feature view: 300-dim herb features → feat_encoder → feat_proj → 128-dim
    Knowledge view: know_dim SVD vector → know_proj → 128-dim
    Classification head only uses feature view.
    """
    def __init__(self, feat_dim=300, know_dim=32, proj_dim=128):
        super().__init__()
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
        )
        self.head = nn.Linear(256, 1)
        self.feat_proj = nn.Linear(256, proj_dim)
        self.know_proj = nn.Sequential(
            nn.Linear(know_dim, 64), nn.ReLU(),
            nn.Linear(64, proj_dim),
        )

    def forward(self, feat, know=None, return_proj=False):
        h = self.feat_encoder(feat)
        logit = self.head(h)
        if return_proj and know is not None:
            z_feat = self.feat_proj(h)
            z_know = self.know_proj(know)
            return logit, z_feat, z_know
        return logit


class CrossViewLabelConLoss(nn.Module):
    """
    Cross-view contrastive loss conditioned on labels.
    Positive pair: (feat_i, know_j) where label_i == label_j.
    Negative pair: (feat_i, know_j) where label_i != label_j.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temp = temperature

    def forward(self, z_feat, z_know, labels):
        B = z_feat.size(0)
        z_feat = F.normalize(z_feat, dim=1)
        z_know = F.normalize(z_know, dim=1)

        sim = z_feat @ z_know.T / self.temp  # [B, B]

        lab = labels.squeeze()
        pos_mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()

        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=1) / n_pos
        return loss.mean()


def knowledge_sim_loss(z, know_sim_matrix, indices):
    """Regularize: if two herbs share many compounds, their embeddings should be similar."""
    z_norm = F.normalize(z, dim=1)
    embed_sim = z_norm @ z_norm.T  # [B, B]
    target_sim = know_sim_matrix[indices][:, indices].to(z.device)
    return F.mse_loss(embed_sim, target_sim)


class CompoundAttentionAggregator(nn.Module):
    """Multi-head attention aggregation: each toxicity label has its own attention over compounds.

    Replaces naive mean pool with learned per-label attention,
    providing interpretability: which compound contributes most to each toxicity type.
    """
    def __init__(self, prior_dim=5, hidden=32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(prior_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, prior_dim)  # per-label attention logits
        )

    def forward(self, compound_probs):
        """compound_probs: [n_compounds, 5], returns aggregated [5], attn_w [n_compounds, 5]"""
        attn_w = torch.softmax(self.attn(compound_probs), dim=0)  # softmax over compounds per label
        aggregated = (attn_w * compound_probs).sum(dim=0)
        return aggregated, attn_w


class AsymmetricLoss(nn.Module):
    """ASL from ICCV 2021 (Alibaba-MIIL). Operates on logits."""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.sum() / x.size(0)