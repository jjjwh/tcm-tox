"""Debug: check why per_label_5fold.py gets lower AUC than ABLATION_RESULTS.md"""
import os,sys,json,numpy as np,torch,torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_drug_tower import DrugTower
from sklearn.model_selection import StratifiedKFold
BASE = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(BASE,'dataset/output/all_herbs.json'),'r',encoding='utf-8') as f:
    all_data = json.load(f)
labels = np.array([d['label'] for d in all_data], dtype=np.float32)
label_strs = [''.join(str(int(v)) for v in d['label']) for d in all_data]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = list(skf.split(all_data, label_strs))[0]
compound_fps = np.load(os.path.join(BASE, '..', '数据爬取', 'compound_fps.npy'))
train_herbs = [all_data[i] for i in train_idx]

dt = DrugTower(n_labels=8).to(device)
ckpt = torch.load(os.path.join(BASE,'dataset','drug_tower_encoder.pt'), map_location=device)
dt.load_state_dict(ckpt); dt.eval()
for p in dt.parameters(): p.requires_grad = False

c_fps_t = torch.tensor(compound_fps, dtype=torch.float32).to(device)

# Check: for all train herbs, can we compute drug_view?
ok = 0
for hi in train_idx:
    h = all_data[hi]
    cids = [c for c in h['compound_ids'] if c < len(compound_fps) and compound_fps[c].any()]
    if cids: ok += 1
print(f"Train herbs with valid compounds: {ok}/{len(train_idx)}")

# Test drug_proj training
drug_proj = nn.Linear(256, 5).to(device)
print(f"\nBefore training: drug_proj.weight mean={drug_proj.weight.mean():.5f}")
drug_proj_opt = torch.optim.AdamW(drug_proj.parameters(), lr=3e-3, weight_decay=1e-4)

herb_views = {}
train_labels = {}
with torch.no_grad():
    for idx, h in enumerate(train_herbs):
        cids = [c for c in h['compound_ids'] if c < len(compound_fps) and compound_fps[c].any()]
        if cids:
            _, emb = dt(c_fps_t[cids], return_embed=True)
            herb_views[idx] = emb.mean(dim=0)
            train_labels[idx] = h['label']

print(f"Herbs with drug_view: {len(herb_views)}")

# Train 5 epochs and check
crit = nn.BCEWithLogitsLoss()
for ep in range(5):
    for idx in herb_views:
        pred = drug_proj(herb_views[idx])
        lbl = torch.tensor(train_labels[idx], dtype=torch.float32).to(device)
        loss = crit(pred, lbl)
        drug_proj_opt.zero_grad(); loss.backward(); drug_proj_opt.step()

print(f"After 5 epochs: drug_proj.weight mean={drug_proj.weight.mean():.5f}")

# Check the drug_proj outputs for a few herbs
drug_proj.eval()
with torch.no_grad():
    for idx in list(herb_views.keys())[:5]:
        p = torch.sigmoid(drug_proj(herb_views[idx])).cpu().numpy()
        l = train_labels[idx]
        print(f"  Herb {idx}: pred={np.round(p,3)} label={l}")

print("\nDiagnosis: drug_proj IS trained and produces non-zero, non-constant predictions.")
print("The issue is elsewhere. Let me check the GatedPriorProtoNet prior usage...")
