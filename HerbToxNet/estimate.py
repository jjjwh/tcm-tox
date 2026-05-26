import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import os
import time

def estimate(y_true, y_pred):
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    # print(y_true.shape, y_pred.shape)
    # j = 1
    # y_true = y_true[:,j]
    # y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    # y_pred = y_pred[:,j]
    best_f1 = 0
    best_threshold = 0.3

    for threshold in np.arange(0.05, 1.0, 0.05):
        y_pred_bin = (y_pred > threshold).astype(int)
        f1 = f1_score(y_true, y_pred_bin)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # 使用最佳阈值计算指标
    y_pred_bin = (y_pred > best_threshold).astype(int)  # 使用最佳阈值

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError as e:
        if "Only one class present in y_true. ROC AUC score is not defined in that case." in str(e):
            auc = 0  # 如果只有一个类别，则将 AUC 设置为 0
        else:
            raise  # 重新抛出其他类型的异常

    auprc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_bin)
    return{
        'auc': auc,
        'auprc': auprc,
        'f1': f1,
        'threshold': best_threshold
    }