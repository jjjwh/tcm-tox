import numpy as np
import torch
import dgl
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import dgllife
from dgllife.utils import CanonicalAtomFeaturizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Union
import numpy as np
import torch
from dgllife.utils import smiles_to_bigraph
import random
from sklearn.metrics import *
import os
import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import random
import os
import numpy as np
import pandas as pd
import copy
import dgl
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import smiles_to_bigraph
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torch.autograd import Variable
from .DMPNN import * 
from .utils import *
from .scheduler import NoamLR
from .models import *

def tox21_model_train(model, model_path, tr_data, va_data, optimizer, scheduler, loss_list, device, epochs = 100):

    path = model_path
#     os.makedirs(path + "seed" + str(seed), exist_ok = True)
          
    best_score = -1000
    task_num = model.task_num
    
    epoch_task_losses, epoch_valid_aupr, epoch_valid_auc = [], [], []
    pbar = tqdm(range(epochs))
    
    for epoch in pbar:
        model.train()
        task_epoch_loss = 0
        task_loss_list, task_count_list = [0]*task_num, [0]*task_num
        
        for num, (tox_g, tox_y, tox_mask) in enumerate(tr_data):
            tox_g, tox_y = tox_g.to(device), tox_y.to(device)
            tox_atom, tox_bond = tox_g.ndata.pop('h').to(device), tox_g.edata.pop('e').to(device)
            tox_mask = tox_mask.to(device)
            task_logits, task_preds = model(tox_g, tox_atom, tox_bond)
      
            sr_task_loss = 0
            for i in range(task_num):
                if sum(tox_mask[:, i]) != 0:                
                    loss_func = loss_list[i] 
                    task_loss = loss_func(task_logits[i][tox_mask[:, i] != 0].float(), tox_y[:, i][tox_mask[:, i] != 0].view(-1).long())
                    sr_task_loss += task_loss
                    task_loss_list[i] += task_loss.item()
                    task_count_list[i] += 1

            task_epoch_loss += (sr_task_loss/task_num).item()
            optimizer.zero_grad()
            (sr_task_loss/task_num).backward()
            optimizer.step()
            scheduler.step()

            del tox_g, tox_y, tox_mask, task_logits, task_preds, sr_task_loss
            torch.cuda.empty_cache()

            pbar.set_description(str(round(task_epoch_loss/(num+1), 4))+" ")
        task_epoch_loss /= (num + 1)
        task_specific_loss_list = [task_loss_list[i] / task_count_list[i] if task_count_list[i] != 0 else 0 for i in range(task_num)]
        
        epoch_task_losses.append(task_epoch_loss)
              
        ckpt_dict = {
            'model_state_dict': model.state_dict(),
            'Task_specific_losses': task_specific_loss_list,            
            }

        valid_bac, valid_f1, valid_aupr, valid_auc, valid_loss = tox21_model_validation(model, va_data, device)
        valid_score = np.sum(valid_bac) + np.sum(valid_f1) + np.sum(valid_aupr)
        print(f"Epoch loss : {task_epoch_loss: .4f}")
        rounded_aupr = [round(x, 4) for x in valid_aupr]
        rounded_auc = [round(x, 4) for x in valid_auc]
        
        print(f"aupr: {rounded_aupr}, auc: {rounded_auc}")
   
        if valid_score > best_score:
            best_score = valid_score
            top_epoch = epoch
#             torch.save(ckpt_dict, path + "seed" + str(seed) + "/epoch_"+str(epoch)+".pth")

    metric_dict = {
        "train_loss": epoch_task_losses,
    }
    
    return metric_dict, top_epoch

def tox21_model_validation(model, tr_va_data, device):
    results = {'loss': [], 'pre': [], 'sen': [], 'spe': [], 'acc': [], 'bac': [], 'f1': [], 'aupr': [], 'auc': []}
    with torch.no_grad():
        model.eval()

        ##################### Target ###############################
        pred_res, true_res = [], []
        
        for _, (bg, labels, mask) in enumerate(tr_va_data):
            bg, labels = bg.to(device), labels.to(device)
            atom_feats, bond_feats = bg.ndata['h'].to(device), bg.edata['e'].to(device)
            mask = mask.to(device)
            tox_logits, tox_preds = model(bg, atom_feats, bond_feats)

            tox_preds = [preds[:, 1].view(-1, 1) for preds in tox_preds]
            pred = torch.cat(tox_preds, axis = 1).reshape(tox_preds[0].shape[0], model.task_num, -1)
            pred_cls = pred.detach().to('cpu').numpy()
            true_cls = labels.detach().to('cpu').numpy()
            pred_res.append(pred_cls)
            true_res.append(true_cls)

            del bg, labels, mask, atom_feats, bond_feats, tox_logits, tox_preds
            torch.cuda.empty_cache()
            
        pred_res = np.vstack(pred_res)
        true_res = np.vstack(true_res)
        
        
        for i in range(model.task_num):
            df = pd.DataFrame()
            df['true'] = true_res[:, i]
            df['pred'] = pred_res[:, i, 0]
            df = df.dropna().reset_index(drop = True)
            metric = score(df['true'], df['pred'])
            for num, k in enumerate(results.keys()):
                    results[k].append(metric[num])

    return  results['bac'], results['f1'], results['aupr'], results['auc'], results['loss']

def tox21_model_prediction(model, tr_va_data, device):
    with torch.no_grad():
        model.eval()

        ##################### Target ###############################
        pred_res = []

        for _, (bg, labels, mask) in enumerate(tr_va_data):
            bg = bg.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            atom_feats = bg.ndata['h'].to(device)
            bond_feats = bg.edata['e'].to(device)

            tox_logits, tox_preds = model(bg, atom_feats, bond_feats)
            tox_preds = [preds[:, 1].view(-1, 1) for preds in tox_preds]
            pred = torch.cat(tox_preds, axis = 1).reshape(tox_preds[0].shape[0], model.task_num, -1)
            pred_cls = pred.detach().to('cpu').numpy()
            pred_res.append(pred_cls)

            del bg, labels, mask, atom_feats, bond_feats, tox_logits, tox_preds
            torch.cuda.empty_cache()
                
        pred_res = np.vstack(pred_res)

    return pred_res

class EMA:
    def __init__(self, model, beta=0.998):
        self.model = model
        self.beta = beta
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                old_avg = self.shadow[name]
                new_avg = self.beta * old_avg + (1.0 - self.beta) * param.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}    

def invivo_model_train(model, model_path, tr_data, va_data, optimizer, scheduler, loss_list, device, seed, epochs = 60):

    path = model_path
    os.makedirs(path + "seed" + str(seed), exist_ok = True)
          
    best_score = -1000
    task_num = model.task_num
    
    epoch_task_losses = []
    pbar = tqdm(range(epochs))

    ema = EMA(model, beta=0.998)
    
    for epoch in pbar:
        model.train()
        task_epoch_loss = 0
        task_loss_list, task_count_list = [0]*task_num, [0]*task_num
        
        for num, (tox_g, tox_y, tox_embed, tox_mask) in enumerate(tr_data):
            tox_g, tox_y = tox_g.to(device), tox_y.to(device)
            tox_atom, tox_bond = tox_g.ndata.pop('h').to(device), tox_g.edata.pop('e').to(device)
            tox_embeds, tox_mask = torch.stack(tox_embed).to(device), tox_mask.to(device)
            task_logits, task_preds = model(tox_g, tox_atom, tox_bond, tox_embeds)
      
            sr_task_loss = 0        

            for i in range(task_num):
                if sum(tox_mask[:, i]) != 0:
                    loss_func = loss_list[i]
                    task_loss = loss_func(task_logits[i][tox_mask[:, i] != 0].float(), tox_y[:, i][tox_mask[:, i] != 0].view(-1).long())     
                    sr_task_loss += task_loss
                    task_loss_list[i] += task_loss.item()
                    task_count_list[i] += 1

            task_epoch_loss += sr_task_loss.item()
            optimizer.zero_grad()
            sr_task_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            ema.update()

            del tox_g, tox_y, tox_embed, tox_embeds, tox_mask, task_logits, task_preds, sr_task_loss
            torch.cuda.empty_cache()

            pbar.set_description(str(round(task_epoch_loss/(num+1), 4))+" ")
        task_epoch_loss /= (num + 1)
        task_specific_loss_list = [task_loss_list[i] / task_count_list[i] if task_count_list[i] != 0 else 0 for i in range(task_num)]
        
        epoch_task_losses.append(task_epoch_loss)
              
        ckpt_dict = {
            'model_state_dict': model.state_dict(),
            'Task_specific_losses': task_specific_loss_list,            
            }

        ema.apply_shadow()
        valid_bac, valid_f1, valid_aupr, valid_auc, valid_loss = invivo_model_validation(model, va_data, device)
        ema.restore()        
        valid_score = np.sum(valid_bac) + np.sum(valid_auc) + np.sum(valid_aupr)
        print(f"Epoch loss : {task_epoch_loss: .4f}")
        rounded_bac = [round(x, 4) for x in valid_bac]
        rounded_f1 = [round(x, 4) for x in valid_f1]
        rounded_aupr = [round(x, 4) for x in valid_aupr]
        rounded_auc = [round(x, 4) for x in valid_auc]
        rounded_loss = [round(x, 4) for x in valid_loss]
        
        print(f"Valid bac: {rounded_bac}, f1: {rounded_f1}, aupr: {rounded_aupr}, auc: {rounded_auc}, loss: {rounded_loss}")
   
        if valid_score > best_score:
            best_score = valid_score
            top_epoch = epoch
            torch.save(ckpt_dict, path + "seed" + str(seed) + "/epoch_"+str(epoch)+".pth")

    metric_dict = {
        "train_loss": epoch_task_losses,
    }
    
    return metric_dict, top_epoch

def invivo_model_validation(model, tr_va_data, device):
    results = {'loss': [], 'pre': [], 'sen': [], 'spe': [], 'acc': [], 'bac': [], 'f1': [], 'aupr': [], 'auc': []}
    with torch.no_grad():
        model.eval()

        ##################### Target ###############################
        pred_res, true_res = [], []
        
        for _, (bg, labels, embed, mask) in enumerate(tr_va_data):
            bg, labels = bg.to(device), labels.to(device)
            atom_feats, bond_feats = bg.ndata['h'].to(device), bg.edata['e'].to(device)
            embeds, mask = torch.stack(embed).to(device), mask.to(device)
            tox_logits, tox_preds = model(bg, atom_feats, bond_feats, embeds)

            tox_preds = [preds[:, 1].view(-1, 1) for preds in tox_preds]
            pred = torch.cat(tox_preds, axis = 1).reshape(tox_preds[0].shape[0], model.task_num, -1)
            pred_cls = pred.detach().to('cpu').numpy()
            true_cls = labels.detach().to('cpu').numpy()
            pred_res.append(pred_cls)
            true_res.append(true_cls)

            del bg, labels, embed, embeds, mask, atom_feats, bond_feats, tox_logits, tox_preds
            torch.cuda.empty_cache()
            
        pred_res = np.vstack(pred_res)
        true_res = np.vstack(true_res)
        
        
        for i in range(model.task_num):
            df = pd.DataFrame()
            df['true'] = true_res[:, i]
            df['pred'] = pred_res[:, i, 0]
            df = df.dropna().reset_index(drop = True)
            metric = score(df['true'], df['pred'])
            for num, k in enumerate(results.keys()):
                    results[k].append(metric[num])

    return  results['bac'], results['f1'], results['aupr'], results['auc'], results['loss']

def invivo_model_test(model, tr_va_data, device):
    with torch.no_grad():
        model.eval()

        ##################### Target ###############################
        pred_res = []

        for _, (bg, labels, embed, mask) in enumerate(tr_va_data):
            bg = bg.to(device)
            labels = labels.to(device)
            embeds = torch.stack(embed).to(device)
            mask = mask.to(device)
            atom_feats = bg.ndata['h'].to(device)
            bond_feats = bg.edata['e'].to(device)

            tox_logits, tox_preds = model(bg, atom_feats, bond_feats, embeds)
            tox_preds = [preds[:, 1].view(-1, 1) for preds in tox_preds]
            pred = torch.cat(tox_preds, axis = 1).reshape(tox_preds[0].shape[0], model.task_num, -1)
            pred_cls = pred.detach().to('cpu').numpy()
            pred_res.append(pred_cls)

            del bg, labels, embed, embeds, mask, atom_feats, bond_feats, tox_logits, tox_preds
            torch.cuda.empty_cache()
                
        pred_res = np.vstack(pred_res)

    return pred_res

def invivo_inference(model, tr_va_data, device):
    with torch.no_grad():
        model.eval()

        ##################### Target ###############################
        pred_res = []
    
        for _, (bg, labels) in enumerate(tr_va_data):
            bg = bg.to(device)
            labels = labels.to(device)
            atom_feats = bg.ndata['h'].to(device)
            bond_feats = bg.edata['e'].to(device)

            tox_preds = model(bg, atom_feats, bond_feats)
            tox_preds = [preds[:, 1].view(-1, 1) for preds in tox_preds]
            pred = torch.cat(tox_preds, axis = 1).reshape(tox_preds[0].shape[0], model.task_num, -1)
            pred_cls = pred.detach().to('cpu').numpy()
            pred_res.append(pred_cls)
                
        pred_res = np.vstack(pred_res)

    return pred_res