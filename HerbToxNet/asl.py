import torch
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
class MultiLabelLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma_pos=2, gamma_neg=4, reduction='mean', class_weights=None, eps=1e-8):
        """
        多标签分类损失函数，结合了 Focal Loss 和 Asymmetric Loss 的思想
        :param alpha: 平衡正负样本比例的系数，类似 focal loss 中的 alpha
        :param gamma_pos: 正类难样本增强系数
        :param gamma_neg: 负类难样本增强系数
        :param reduction: 'mean' / 'sum' / 'none'
        :param class_weights: 每个标签的权重，shape = (m,)
        :param eps: 防止 log(0) 的小量
        """
        super(MultiLabelLoss, self).__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.class_weights = class_weights
        self.eps = eps

    def forward(self, preds, targets):
        """
        :param preds: 模型输出，sigmoid 后的概率，shape=(batch_size, m)
        :param targets: 真实标签，shape=(batch_size, m)，取值 0~1
        :return: 损失值
        """
        preds = torch.clamp(preds, min=self.eps, max=1 - self.eps)

        # 正类损失和负类损失
        pos_loss = targets * (1 - preds).pow(self.gamma_pos) * torch.log(preds)
        neg_loss = (1 - targets) * preds.pow(self.gamma_neg) * torch.log(1 - preds)

        loss = -(self.alpha * pos_loss + (1 - self.alpha) * neg_loss)

        # 应用 class weights
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device)

        # 汇总方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape=(batch_size, m)

# class AsymmetricLoss(nn.Module):
#     """非对称损失函数 (ASL)"""
#
#     def __init__(self, alpha=0.5, gamma_pos=2, gamma_neg=5, eps=1e-5):
#         super(AsymmetricLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.eps = eps
#
#     def forward(self, preds, targets):
#         # preds = torch.sigmoid(preds)
#         preds = preds.clamp(min=self.eps, max=1 - self.eps)
#
#         pos_loss = self.alpha * targets * torch.log(preds) * (1 - preds) ** self.gamma_pos
#         neg_loss = (1 - self.alpha) * (1 - targets) * torch.log(1 - preds) * preds ** self.gamma_neg
#
#         loss = -torch.mean(pos_loss + neg_loss)
#         return loss


class AsymmetricLoss(nn.Module):
    """非对称损失函数 (ASL) 多标签版本"""

    def __init__(self, alpha=0.5, gamma_pos=1, gamma_neg=2, eps=1e-5):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, preds, targets):
        """
        :param preds: 预测输出，形状为(batch_size, num_classes)
        :param targets: 目标标签，形状为(batch_size, num_classes)
        :return: 总损失
        """
        preds = preds.clamp(min=self.eps, max=1 - self.eps)

        pos_loss = self.alpha * targets * torch.log(preds) * (1 - preds) ** self.gamma_pos
        neg_loss = (1 - self.alpha) * (1 - targets) * torch.log(1 - preds) * preds ** self.gamma_neg

        # 对每个样本内的标签损失求和，然后对批次中的样本求平均
        loss = -torch.sum(pos_loss + neg_loss, dim=1).mean()
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义加权二元交叉熵损失
class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights  # 每个标签的权重

    def forward(self, inputs, targets):
        # 使用 sigmoid 将 logits 转换为概率
        # 计算加权二元交叉熵
        loss = - (self.weights * targets * torch.log(inputs + 1e-8) +
                  (1 - targets) * torch.log(1 - inputs + 1e-8))
        return torch.mean(loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 pt
        pt = inputs * targets + (1 - inputs) * (1 - targets)
        # 计算 focal loss
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(torch.clamp(pt, min=1e-8))

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss