"""Quick 1-fold ablation to verify AUC scale: drug_prior vs compound_prior vs both"""
import os,sys,json,random,numpy as np,torch,torch.nn as nn
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from model import GatedPriorProtoNet,ProtoNet,CompoundAttentionAggregator,AsymmetricLoss,knowledge_sim_loss
from model_drug_tower import DrugTower
from train_compound_ecc import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,f1_score

BASE=os.path.dirname(os.path.abspath(__file__))
device='cuda'
SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)

with open(os.path.join(BASE,'dataset/output/all_herbs.json'),'r',encoding='utf-8') as f:all_data=json.load(f)
features=np.array([d['feature_vector'] for d in all_data],dtype=np.float32)
labels=np.array([d['label'] for d in all_data],dtype=np.float32)
with open(os.path.join(BASE,'dataset/output/compound2id.json'),'r') as f:c2id=json.load(f)
n_compounds=len(c2id)
compound_mh=np.zeros((len(all_data),n_compounds),dtype=np.float32)
for i,d in enumerate(all_data):
    for cid in d['compound_ids']:
        if cid<n_compounds:compound_mh[i,cid]=1.0
compound_fps=np.load(os.path.join(BASE,'..','数据爬取','compound_fps.npy'))
label_strs=[''.join(str(int(v)) for v in d['label']) for d in all_data]

dt=DrugTower(n_labels=8).to(device)
ckpt=torch.load(os.path.join(BASE,'dataset','drug_tower_encoder.pt'),map_location=device)
dt.load_state_dict(ckpt);dt.eval()
for p in dt.parameters():p.requires_grad=False

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
fold,(train_idx,val_idx)=0,list(skf.split(all_data,label_strs))[0]
print(f'Fold {fold+1}: train={len(train_idx)}, val={len(val_idx)}')

train_features,val_features=features[train_idx],features[val_idx]
train_labels,val_labels=labels[train_idx],labels[val_idx]
comp_train=compound_mh[train_idx]
inter=comp_train@comp_train.T;rs=comp_train.sum(axis=1,keepdims=True)
comp_sim_train=torch.tensor(inter/np.maximum(rs+rs.T-inter,1),dtype=torch.float32)
train_herbs=[all_data[i] for i in train_idx]

# Build compound prior
c_labels=build_compound_labels(train_herbs,n_compounds)
cm=train_compound_fold(compound_fps,c_labels,train_herbs,epochs=40)
agg=CompoundAttentionAggregator(prior_dim=5,hidden=32).to(device)
agg=train_compound_aggregator(cm,agg,compound_fps,train_herbs)
tr_compound=compute_prior(cm,compound_fps,train_idx,all_data,agg)
va_compound=compute_prior(cm,compound_fps,val_idx,all_data,agg)

# Build drug prior
dp=nn.Linear(256,5).to(device)
dp=train_drug_projection(dt,dp,compound_fps,train_herbs)
tr_drug=compute_drug_prior(dt,dp,compound_fps,train_idx,all_data)
va_drug=compute_drug_prior(dt,dp,compound_fps,val_idx,all_data)

print(f'Compound prior stats: train mean={tr_compound.mean():.4f} std={tr_compound.std():.4f}')
print(f'Drug prior stats:     train mean={tr_drug.mean():.4f} std={tr_drug.std():.4f}')

# Test 4 configurations on label 0 (Hepatotoxicity) with 100 epochs
LABEL_IDX=0
for config_name, tr_p, va_p, tr_p2, va_p2 in [
    ("No prior (ProtoNet baseline)", None, None, None, None),
    ("Compound prior only", tr_compound, va_compound, None, None),
    ("Drug prior only", tr_drug, va_drug, None, None),
    ("Both priors", tr_compound, va_compound, tr_drug, va_drug),
]:
    # For no-prior case, pass train_features directly (no extra dims)
    use_prior = (tr_p is not None)
    tr_inp_feat = train_features  # always 300d
    va_inp_feat = val_features    # always 300d
    vp = train_label_protonet(
        tr_inp_feat, train_labels[:, LABEL_IDX:LABEL_IDX+1],
        va_inp_feat, val_labels[:, LABEL_IDX:LABEL_IDX+1],
        comp_sim_train, device, seed=SEED+999,
        tr_prior=tr_p, va_prior=va_p, tr_prior2=tr_p2, va_prior2=va_p2,
        use_contrastive=use_prior, epochs=100)
    auc = roc_auc_score(val_labels[:, LABEL_IDX], vp)
    print(f'{config_name:30s}: AUC={auc:.4f}')
