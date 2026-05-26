"""Debug: binary HerbToxNet with loss tracking + prediction inspection"""
import torch, torch.nn as nn, torch.optim as optim
import random, os, sys, numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import create_heterogeneous_data, EarlyStopping
from modelHAN import FastGTN, dynamic_contrastive_loss

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_gtn_data(g, device):
    h_feat=g.nodes['herb'].data['feat'];i_feat=g.nodes['ingredient'].data['feat']
    t_feat=g.nodes['target'].data['feat']
    X=torch.cat([h_feat,i_feat,t_feat],dim=0).to(device)
    nh=h_feat.shape[0];ni=i_feat.shape[0];nt=t_feat.shape[0]
    A=torch.zeros(7,X.shape[0],X.shape[0],device=device)
    s,d=g.edges(etype='hi');A[0,s,d+ni]=1;A[1,d+ni,s]=1
    s,d=g.edges(etype='ht');A[2,s,d+nh+ni]=1;A[3,d+nh+ni,s]=1
    s,d=g.edges(etype='it');A[4,s+nh,d+nh+ni]=1;A[5,d+nh+ni,s+nh]=1
    A[6]=torch.eye(X.shape[0],device=device)
    return A,X

def feature_masking(X,drop_rate=0.2):
    return X*(torch.rand_like(X)>drop_rate)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(128,128),nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(128,128),nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(128,32),nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(32,1),nn.Sigmoid())
    def forward(self,x):return self.mlp(x)

def setup_seed(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.deterministic=True

SEED=0;setup_seed(SEED)
g=create_heterogeneous_data()
all_labels=g.nodes['herb'].data['label'].numpy()
n_herbs=g.num_nodes('herb')
g=g.to(device)
A_orig,X_orig=prepare_gtn_data(g,device)

# Try with nephrotoxicity (col 1, 101 pos) - use ALL herbs, no test split, 1 fold
label_idx=1
labels_1d=torch.tensor(all_labels[:,label_idx],dtype=torch.float32).reshape(-1,1).to(device)
n_pos=int(labels_1d.sum())

# Simple 80/20 train/val split
all_idx=list(range(n_herbs));random.shuffle(all_idx)
n_tr=int(n_herbs*0.8)
tr_idx=all_idx[:n_tr];va_idx=all_idx[n_tr:]

tr_mask=torch.zeros(n_herbs,dtype=torch.bool,device=device);tr_mask[tr_idx]=True
va_mask=torch.zeros(n_herbs,dtype=torch.bool,device=device);va_mask[va_idx]=True

print(f"Label col={label_idx}: {n_pos} pos / {n_herbs} total")
print(f"Train: {len(tr_idx)} (pos={int(labels_1d[tr_idx].sum())}) Val: {len(va_idx)} (pos={int(labels_1d[va_idx].sum())})")

emb_model=FastGTN(num_edge_types=7,num_channels=2,in_dim=300,hidden_dim=128,out_dim=64,num_layers=2).to(device)
clf=BinaryClassifier().to(device)
opt=optim.Adam(list(emb_model.parameters())+list(clf.parameters()),lr=0.001,weight_decay=1e-5)
bce=nn.BCELoss()

for ep in range(100):
    emb_model.train();clf.train();opt.zero_grad()
    cx=feature_masking(X_orig,0.2)
    emb_all=emb_model(A_orig,cx)
    emb=emb_all[:n_herbs]
    out=clf(emb)
    loss=bce(out[tr_mask],labels_1d[tr_mask])
    loss.backward();opt.step()
    
    if (ep+1)%10==0:
        emb_model.eval();clf.eval()
        with torch.no_grad():
            emb_all=emb_model(A_orig,X_orig)
            emb=emb_all[:n_herbs]
            out=clf(emb)
            val_loss=bce(out[va_mask],labels_1d[va_mask])
            # Check prediction distribution
            val_preds=out[va_mask].cpu().numpy().flatten()
            val_true=labels_1d[va_mask].cpu().numpy().flatten()
            try: auc=roc_auc_score(val_true,val_preds)
            except: auc=float('nan')
            pred_mean=val_preds.mean()
            pred_zeros=(val_preds<0.5).sum()
            print(f"Epoch {ep+1:3d}: tr_loss={loss.item():.4f} val_loss={val_loss.item():.4f} "
                  f"auc={auc:.4f} pred_mean={pred_mean:.3f} preds<0.5={pred_zeros}/{len(val_preds)}")

# Final test
emb_model.eval();clf.eval()
with torch.no_grad():
    emb_all=emb_model(A_orig,X_orig)
    emb=emb_all[:n_herbs]
    out=clf(emb)
    preds=out.cpu().numpy().flatten()
    trues=labels_1d.cpu().numpy().flatten()
    
# Search best threshold
best_f1,best_t=0,0.5
for t in np.arange(0.05,1.0,0.05):
    pb=(preds[va_idx]>t).astype(int)
    f=f1_score(trues[va_idx],pb,zero_division=0)
    if f>best_f1:best_f1,best_t=f,t

pb=(preds[va_idx]>best_t).astype(int)
try:auc=roc_auc_score(trues[va_idx],preds[va_idx])
except:auc=float('nan')
auprc=average_precision_score(trues[va_idx],preds[va_idx])

print(f"\nFinal: AUROC={auc:.4f} AUPRC={auprc:.4f} F1={best_f1:.4f} threshold={best_t:.2f}")
print(f"Pred range: [{preds.min():.4f}, {preds.max():.4f}]")
print(f"Pred > 0.5 count: {(preds>0.5).sum()}/{len(preds)}")
print(f"True pos count: {int(trues.sum())}")
