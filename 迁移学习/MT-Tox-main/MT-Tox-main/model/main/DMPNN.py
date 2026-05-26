# This code is adapted from https://github.com/GIST-CSBL/BayeshERG/tree/main
# Originally licensed under the MIT license


import random
import os
import numpy as np
import pandas as pd
#from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from dgl.backend import pytorch as DF
from tqdm import tqdm
import dgl
import torch
import dgllife
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import smiles_to_bigraph
import numpy as np
import torch
import torch.nn as nn
from dgl.backend import pytorch as DF
import numpy as np
import torch
import torch.nn as nn
from dgl import function as fn
import copy

class EDGE_CONV(nn.Module):
    
    def __init__(self,
                 node_dim,
                 edge_dim,
                 aggregator_type = 'mean'):
        super(EDGE_CONV, self).__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type

    def udf_sub(self, edges):
        return {'e_res': (edges.data['t'] - edges.data['rm'])}  

    def forward(self, graph, nfeat, efeat):
        with graph.local_scope():
            graph.ndata['h'] = nfeat.view(-1, self.node_dim)
            graph.edata['w'] = efeat.view(-1, self.edge_dim)
            edge_index = np.array(range(len(efeat)))
            edge_index[list(range(0, len(efeat), 2))] += 1
            edge_index[list(range(1, len(efeat), 2))] -= 1
            edge_index = torch.LongTensor(list(edge_index))
            rev_efeat = efeat[edge_index]
            graph.edata['rev_w'] = rev_efeat.view(-1, self.edge_dim)
            graph.update_all(fn.copy_e('w', 'm'), self.reducer('m', 'neigh'))
            graph.apply_edges(fn.copy_e('rev_w', 'rm'))
            graph.apply_edges(fn.copy_u('neigh', 't'))
            graph.apply_edges(self.udf_sub)
            edg_n = graph.edata['e_res']
            return edg_n



class DMPNN(nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 node_hidden_dim=64,
                 edge_hidden_dim=32,
                 num_step_message_passing=4):
        super(DMPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.edge_conv = EDGE_CONV(node_dim=node_input_dim, edge_dim=edge_hidden_dim, aggregator_type='sum')
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.init_message = nn.Sequential(nn.Linear(node_input_dim + edge_input_dim, edge_hidden_dim, bias = False), nn.ReLU())
        self.e_update = nn.Linear(edge_hidden_dim, edge_hidden_dim, bias = False)
        self.last_update = nn.Linear(node_input_dim + edge_hidden_dim, node_hidden_dim, bias = False)
        
        self.dropout = nn.Dropout(0.1)
        
    def udf_init_m(self, edges):
        return {'im': self.init_message(torch.cat((edges.src['ih'], edges.data['iw']), dim=1))}
    
    def forward(self, g, n_feat, e_feat):
        h0 = n_feat
        e0 = e_feat
        g.ndata['ih'] = h0.view(-1, self.node_input_dim)
        g.edata['iw'] = e0.view(-1, self.edge_input_dim)
        g.apply_edges(self.udf_init_m)
        e0 = g.edata['im']
        e_t = e0
        for i in range(self.num_step_message_passing):
            m_t = self.edge_conv(g, h0, e_t)
            e_t = self.relu1(e0 + self.dropout(self.e_update(m_t)))
            
        g.edata['fe'] = e_t
        g.ndata['fn'] = h0
        g.update_all(fn.copy_e('fe', 'fm'), fn.sum('fm', 'ff'))
        out = self.relu2(self.last_update(torch.cat((g.ndata['fn'], g.ndata['ff']), dim=1)))
        return out

from dgl.nn import AvgPooling, SumPooling
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_head):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.proj_q = nn.Linear(d_model, num_heads * d_head, bias = False)
        self.proj_k = nn.Linear(d_model, num_heads * d_head, bias = False)
        self.proj_v = nn.Linear(d_model, num_heads * d_head, bias = False)
        self.proj_o = nn.Linear(num_heads * d_head, d_model, bias = False)


    def forward(self, x, lengths_x):
        device = x.device
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        queries = self.proj_q(x).view(batch_size, -1, self.num_heads, self.d_head)
        keys = self.proj_k(x).view(batch_size, -1, self.num_heads, self.d_head)
        values = self.proj_v(x).view(batch_size, -1, self.num_heads, self.d_head)
        #x.view(batch_size, -1, self.num_heads, self.d_head)#


        e = torch.einsum('bxhd,byhd->bhxy', queries, keys)
        e = e / np.sqrt(self.d_head)
        mask = torch.zeros(batch_size, max_len_x + 1, max_len_x + 1).to(e.device)
        for i in range(batch_size):
            mask[i, :lengths_x[i] + 1, :lengths_x[i] + 1].fill_(1)
        mask = mask.unsqueeze(1)
        e.masked_fill_(mask == 0, -1e10)
        alpha = torch.softmax(e, dim=-1)
        out = torch.einsum('bhxy,byhd->bxhd', alpha, values)
        out = self.proj_o(
            out.contiguous().view(batch_size, (max_len_x + 1), self.num_heads * self.d_head))
#         out = out.contiguous().view(batch_size, (max_len_x + 1), self.num_heads * self.d_head)
        return out, alpha




class MultiHeadAttention_readout(nn.Module):
    def __init__(self, d_model, num_heads, d_head, num_mha=1):
        super(MultiHeadAttention_readout, self).__init__()
        self.num_mha = num_mha
        self.attblock = MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads, d_head=d_head) 
        self.class_emb = torch.nn.Embedding(1, d_model)
    def transform_feat(self, x, lengths_x):
        device = x.device
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x).cpu().numpy()
        cls_token = self.class_emb(torch.LongTensor([0]).to(device))
        x = DF.pad_packed_tensor(x, lengths_x, 0, l_min=max_len_x + 1)
        for i in range(batch_size):
            x[i, lengths_x[i], :] = cls_token
        return x

    def forward(self, x, lengths_x):
        device = x.device
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x).cpu().numpy()

        x = self.transform_feat(x, lengths_x)

        x, alpha = self.attblock(x, lengths_x)

        idx_list = []
        sum_feats = []
        bef = 0
        ind = 0
        for i in lengths_x:
            i = i.cpu().numpy()
            sum_feats.append(x[ind, 0:i, :].sum(dim = 0).unsqueeze(0))

            ind += 1
            
            bef += i
            idx_list.append(bef)
            bef += ((max_len_x + 1) - i)

        idx_list = torch.tensor(idx_list).to(device)
        x = x.view(batch_size * (max_len_x + 1), -1)
        x = torch.index_select(x, 0, idx_list.long())
        # sum_pooled = torch.cat(sum_feats, dim = 0)
        return x, alpha    
    
class Readout(nn.Module):
    def __init__(self,node_hidden_dim=64, num_mha=1):
        super(Readout, self).__init__()
        self.mha_readout = MultiHeadAttention_readout(d_model=node_hidden_dim, num_heads= 8, d_head=node_hidden_dim // 8,
                                                      num_mha=num_mha)

        self.dgl_pooling = SumPooling()
        self.dropout = nn.Dropout(0.1)

    def forward(self, g, feat):
        lengths = g.batch_num_nodes()
        # feat1 = self.dgl_pooling(g, feat)
        feat1, alpha = self.mha_readout(feat, lengths)
        
        # return feat1 + self.dropout(feat2) + self.dropout(feat3), alpha
        return feat1, alpha


class ChEMBL_fullmodel(nn.Module):
    def __init__(self, 
                 node_input_dim=74,
                 edge_input_dim=12,
                 node_hidden_dim=128,
                 edge_hidden_dim=128,
                 num_step_message_passing=7, num_step_mha=1
                ):
        super(ChEMBL_fullmodel, self).__init__()
        self.featurizer = DMPNN(node_input_dim=node_input_dim,
                 edge_input_dim=edge_input_dim,
                 node_hidden_dim=node_hidden_dim,
                 edge_hidden_dim=edge_hidden_dim)
        
        self.readout = Readout(node_hidden_dim=node_hidden_dim)
        self.lin1__ = nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim // 2), nn.ReLU())
        self.lin2__ = nn.Linear(node_hidden_dim // 2, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, g, n_feat, e_feat):
        b_length = g.batch_num_nodes()
        out = self.featurizer(g, n_feat, e_feat)
        out, w = self.readout(g, out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lin1__(out)
        out = self.dropout(out)
        blocker_logit = self.lin2__(out)
        blocker = self.softmax(blocker_logit)
        return blocker_logit, blocker

