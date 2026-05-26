"""
model_drug_tower.py — Drug Tower for Western drug toxicity pretraining.
Same encoder architecture as CompoundToxModel, output 8 UniTox classes.
"""
import torch, torch.nn as nn


class DrugTower(nn.Module):
    """Morgan FP (1024d) → Encoder → 8-class Western toxicity prediction."""
    def __init__(self, fp_dim=1024, hidden=256, n_labels=8, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_labels)

    def forward(self, fp, return_embed=False):
        emb = self.encoder(fp)
        logits = self.head(emb)
        if return_embed:
            return logits, emb
        return logits
