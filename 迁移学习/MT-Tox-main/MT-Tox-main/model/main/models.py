import random
import os
import numpy as np
import pandas as pd
from dgl.backend import pytorch as DF
from tqdm import tqdm
import dgl
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import smiles_to_bigraph
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import copy
import argparse
from torch.autograd import Variable
from .DMPNN import * 

#Tox21 model prediction module
class Tox21_LinearPredictor(nn.Module):
    def __init__(self, node_hidden_dim=128):
        super(Tox21_LinearPredictor, self).__init__()
        
        self.lin1 = nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim//2), nn.ReLU(), nn.Dropout(0.15))
        self.lin2 = nn.Linear(node_hidden_dim//2, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        
    def forward(self, out):

        out = self.lin1(out)
        blocker_logit = self.lin2(out)
        blocker = self.softmax(blocker_logit)
        
        return (blocker_logit, blocker)

# Model class definition for Tox21 (in vitro toxicological information) training    
class Tox21_fullmodel(nn.Module):
    def __init__(self, 
                 node_input_dim=74,
                 edge_input_dim=12,
                 node_hidden_dim=128,
                 edge_hidden_dim=128,
                 num_step_message_passing=7, num_step_mha=1,
                 task_num = 12):
        super(Tox21_fullmodel, self).__init__()
        
        self.task_num = task_num
         
        self.gnn = DMPNN(node_input_dim=node_input_dim,
                                         edge_input_dim=edge_input_dim,
                                         node_hidden_dim=node_hidden_dim,
                                         edge_hidden_dim=edge_hidden_dim,
                                         num_step_message_passing=num_step_message_passing)
        self.readout = nn.ModuleList([Readout(node_hidden_dim=node_hidden_dim, num_mha=num_step_mha) for _ in range(task_num)])      
        self.task_predictor =  nn.ModuleList([Tox21_LinearPredictor(node_hidden_dim = node_hidden_dim) for _ in range(task_num)])
        
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, g, n_feat, e_feat):
        
        task_logit_list = []
        task_pred_list = []
        
        gnn_hidden = self.gnn(g, n_feat, e_feat)
               
        for idx in range(self.task_num):
            graph_out, w = self.readout[idx](g, gnn_hidden)
            graph_out = self.relu(graph_out)
            graph_out = self.dropout(graph_out)
            pred_logit, pred_score = self.task_predictor[idx](graph_out)
            task_logit_list.append(pred_logit)
            task_pred_list.append(pred_score)
            
        return task_logit_list, task_pred_list

#in vivo toxicity fine-tuning model prediction module    
class invivo_LinearPredictor(nn.Module):
    def __init__(self, node_hidden_dim=128):
        super(invivo_LinearPredictor, self).__init__()
        
        self.lin1 = nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim//4), nn.ReLU(), nn.Dropout(0.1))
        self.lin2 = nn.Sequential(nn.Linear(node_hidden_dim//4, node_hidden_dim//16), nn.ReLU(), nn.Dropout(0.1))
        self.lin3 = nn.Linear(node_hidden_dim//16, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        
    def forward(self, out):

        out = self.lin1(out)
        out = self.lin2(out)
        blocker_logit = self.lin3(out)
        blocker = self.softmax(blocker_logit)
        
        return (blocker_logit, blocker)    

def attention(queries, keys, values):
    e = torch.einsum('bxd,byd->bxy', queries, keys)
    e = e / np.sqrt(queries.shape[-1])
    alpha = torch.softmax(e, dim=-1)
    out = torch.einsum('bxy,byd->bxd', alpha, values)

    return out.squeeze(1)

class Task_fusion(nn.Module):
    def __init__(self, num_tox21_task, num_task,
                 node_hidden_dim=128):
        super(Task_fusion, self).__init__()

        self.project_q = nn.Linear(node_hidden_dim, node_hidden_dim, bias = False)
        self.project_k = nn.Linear(node_hidden_dim, node_hidden_dim, bias = False)
        self.project_v = nn.Linear(node_hidden_dim, node_hidden_dim, bias = False)
        self.num_tox21_task = num_tox21_task
        self.num_task = num_task
        self.layernorm1 = nn.LayerNorm(node_hidden_dim)
        self.layernorm2 = nn.LayerNorm(node_hidden_dim)
        self.embed_final =nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim), nn.ReLU())
        
    def forward(self, invivo_hidden, tox21_embeds):
        
        invivo_q = self.project_q(self.layernorm1(invivo_hidden))
        tox21_norms = self.layernorm2(tox21_embeds)
        tox21_k = self.project_k(tox21_norms)
        tox21_v = self.project_v(tox21_embeds)
        aggregated_feats = attention(invivo_q.unsqueeze(1), tox21_k, tox21_v)
        final_embed = self.embed_final(invivo_hidden + aggregated_feats)
        
        return final_embed

#in vivo fine-tuning module    
class MTL_invivo(nn.Module):
    def __init__(self, 
                 node_input_dim=74,
                 edge_input_dim=12,
                 node_hidden_dim=128,
                 edge_hidden_dim=128,
                 num_step_message_passing=4, num_step_mha=1,
                 tox21_task_num = 12, task_num = 3):
        super(MTL_invivo, self).__init__()
        
        self.tox21_task_num = tox21_task_num
        self.task_num = task_num
          
        self.invivo_gnn = DMPNN(node_input_dim=node_input_dim,
                                 edge_input_dim=edge_input_dim,
                                 node_hidden_dim=node_hidden_dim,
                                 edge_hidden_dim=edge_hidden_dim,
                                 num_step_message_passing=num_step_message_passing)
        self.invivo_readout = nn.ModuleList([Readout(node_hidden_dim=node_hidden_dim, num_mha=num_step_mha) for _ in range(task_num)])
        self.task_attention = nn.ModuleList([Task_fusion(node_hidden_dim = node_hidden_dim, num_tox21_task = tox21_task_num,
                                                        num_task = task_num) for _ in range(task_num)])
        self.task_predictor =  nn.ModuleList([invivo_LinearPredictor(node_hidden_dim = node_hidden_dim) for _ in range(task_num)])
        self.relu = nn.ReLU()
        
    def forward(self, g, n_feat, e_feat, tox21_embeds):
        
        gnn_hidden_invivo = self.invivo_gnn(g, n_feat, e_feat)

        task_logit_list = []
        task_pred_list = []
        
        for idx in range(self.task_num):
            graph_out, w = self.invivo_readout[idx](g, gnn_hidden_invivo)
            invivo_hidden = self.task_attention[idx](graph_out, tox21_embeds)
            pred_logit, pred_score = self.task_predictor[idx](invivo_hidden)
            task_logit_list.append(pred_logit)
            task_pred_list.append(pred_score)
            
        return task_logit_list, task_pred_list    

# Tox21 embedding generation module definition for in vivo fine-tuning step     
class Tox21_embed(nn.Module):
    def __init__(self, 
                 node_input_dim=74,
                 edge_input_dim=12,
                 node_hidden_dim=128,
                 edge_hidden_dim=128,
                 num_step_message_passing=4, num_step_mha=1,
                 task_num = 12):
        super(Tox21_embed, self).__init__()
        
        self.task_num = task_num
         
        self.gnn = DMPNN(node_input_dim=node_input_dim,
                                         edge_input_dim=edge_input_dim,
                                         node_hidden_dim=node_hidden_dim,
                                         edge_hidden_dim=edge_hidden_dim,
                                         num_step_message_passing=num_step_message_passing)
        self.readout = nn.ModuleList([Readout(node_hidden_dim=node_hidden_dim, num_mha=num_step_mha) for _ in range(task_num)])      
        self.relu = nn.ReLU()
        
    def forward(self, g, n_feat, e_feat):
        
        graph_out_list = []
        
        gnn_hidden = self.gnn(g, n_feat, e_feat)
               
        for idx in range(self.task_num):
            graph_out, w = self.readout[idx](g, gnn_hidden)
            graph_out_list.append(graph_out)
              
        tox21_embeds = torch.stack(graph_out_list, dim = 1)
        
        return tox21_embeds    
    
# Model definition for in vivo toxicity inference
class MTL_invivo_inference(nn.Module):
    def __init__(self, 
                 node_input_dim=74,
                 edge_input_dim=12,
                 node_hidden_dim=128,
                 edge_hidden_dim=128,
                 num_step_message_passing=4, num_step_mha=1,
                 tox21_task_num = 12, task_num = 3):
        super(MTL_invivo_inference, self).__init__()
        
        self.tox21_task_num = tox21_task_num
        self.task_num = task_num
         
        self.gnn = DMPNN(node_input_dim=node_input_dim,
                                         edge_input_dim=edge_input_dim,
                                         node_hidden_dim=node_hidden_dim,
                                         edge_hidden_dim=edge_hidden_dim,
                                         num_step_message_passing=num_step_message_passing)
        self.invivo_gnn = DMPNN(node_input_dim=node_input_dim,
                                 edge_input_dim=edge_input_dim,
                                 node_hidden_dim=node_hidden_dim,
                                 edge_hidden_dim=edge_hidden_dim,
                                 num_step_message_passing=num_step_message_passing)
        self.readout = nn.ModuleList([Readout(node_hidden_dim=node_hidden_dim, num_mha=num_step_mha) for _ in range(tox21_task_num)])
        self.invivo_readout = nn.ModuleList([Readout(node_hidden_dim=node_hidden_dim, num_mha=num_step_mha) for _ in range(task_num)])
        self.task_attention = nn.ModuleList([Task_fusion(node_hidden_dim = node_hidden_dim, num_tox21_task = tox21_task_num,
                                                        num_task = task_num) for _ in range(task_num)])
        self.task_predictor =  nn.ModuleList([invivo_LinearPredictor(node_hidden_dim = node_hidden_dim) for _ in range(task_num)]) 
        self.relu = nn.ReLU()
        
    def forward(self, g, n_feat, e_feat):
        
        task_out_list = []
        
        gnn_hidden = self.gnn(g, n_feat, e_feat)
        gnn_hidden_invivo = self.invivo_gnn(g, n_feat, e_feat)
        
        tox21_embeds = []       
        for idx in range(self.tox21_task_num):
            graph_out, w = self.readout[idx](g, gnn_hidden)
            tox21_embeds.append(graph_out)

        task_logit_list = []
        task_pred_list = []
        tox21_embeds = torch.stack(tox21_embeds, dim = 1)
        
        for idx in range(self.task_num):
            graph_out, w = self.invivo_readout[idx](g, gnn_hidden_invivo)
            invivo_hidden = self.task_attention[idx](graph_out, tox21_embeds)
            pred_logit, pred_score = self.task_predictor[idx](invivo_hidden)
            task_logit_list.append(pred_logit)
            task_pred_list.append(pred_score)

        return task_pred_list    
    