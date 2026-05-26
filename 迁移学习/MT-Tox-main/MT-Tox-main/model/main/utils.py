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


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
            == len(init_lr) == len(max_lr) == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)


    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]
            
def loop_iterable(iterable):
    while True:
        yield from iterable
        
def score(df_true, df_pred):
    label = df_true
    pred_label = (df_pred > 0.5).astype(int)
    pre = precision_score(label, pred_label)
    sen = recall_score(label, pred_label)
    acc = accuracy_score(label, pred_label)
    bac =  balanced_accuracy_score(label, pred_label)
    spe = 2*bac - sen
    f1 = f1_score(label, pred_label)
    aupr = average_precision_score(label, df_pred)
    auc = roc_auc_score(label, df_pred)
    loss = log_loss(label, df_pred)
    
    return loss, pre, sen, spe, acc, bac, f1, aupr, auc

def set_seed(seed: int = 42) :
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
    
def weight_crossentropy(df, label, mode, device):
    num_class_list = []
    for class_label in range(2):
        num_samples = len(df[df[label] == class_label])
        if mode == 'tox21':
            num_class_list.append(np.sqrt(num_samples))
        else:
            num_class_list.append(num_samples)
        
    weights = [1 - (x / sum(num_class_list)) for x in num_class_list]
    weights = torch.FloatTensor(weights).to(device)
    loss_func = nn.CrossEntropyLoss(weights)
    # loss_func = nn.CrossEntropyLoss()
    
    return loss_func

def calculate_sample_weights(labels):
    
    label_counts = labels.sum(axis=0)
    
    total_count = len(labels)
    label_weights = label_counts / total_count

    sample_weights = []
    
    for label in labels:
        sample_weight = np.mean(label_weights[label == 1])
        sample_weights.append(sample_weight)
    
    return torch.tensor(np.array(sample_weights))

def evaluate_results(test_pred, df_valid, assay_list, score_func):

    perf_save_list = []

    for i, col in enumerate(assay_list):
        df_test_pred = pd.DataFrame(test_pred[:, i, :])  
        df_test_pred[col] = df_valid[col]               
        df_test_pred = df_test_pred.dropna().reset_index(drop=True) 

        metrics = score_func(df_test_pred[col], df_test_pred[0])
        perf_save_list.append(metrics)

    perf_df = pd.DataFrame(perf_save_list, columns=['loss','pre', 'sen', 'spe','acc', 'bac', 'f1', 'aupr', 'auc'])
    perf_df['task'] = assay_list
    perf_df = perf_df[['task','loss','pre', 'sen', 'spe','acc', 'bac', 'f1', 'aupr', 'auc']]

    return perf_df
