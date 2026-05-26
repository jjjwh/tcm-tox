import numpy as np
import torch
import copy
import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.metrics import label_ranking_loss, coverage_error
import dgl
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

current_dir = os.path.dirname(__file__)

def process_data():
    """
    Process raw data from dataset/ and save processed tensors to data/
    Adapted for the provided dataset structure.
    """
    dataset_dir = os.path.join(current_dir, 'dataset')
    data_dir = os.path.join(current_dir, 'data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Observed dimensions in dataset
    num_herbs = 252
    num_ingredients = 655
    num_targets = 1540
    out_dim = 300

    print("Processing data...")

    # 1. Features
    # Herb features
    herb_feat_path = os.path.join(dataset_dir, 'herb_features_.npy')
    if os.path.exists(herb_feat_path):
        herb_features = np.load(herb_feat_path)
    else:
        print(f"Warning: {herb_feat_path} not found. Using random.")
        herb_features = np.random.randn(num_herbs, out_dim)

    # Ingredient and Target features (Random initialization as they are missing)
    ingredient_features = np.random.randn(num_ingredients, out_dim)
    target_features = np.random.randn(num_targets, out_dim)
    
    np.save(os.path.join(data_dir, f'herb_features_{out_dim}.npy'), herb_features)
    np.save(os.path.join(data_dir, f'ingredient_features_{out_dim}.npy'), ingredient_features)
    np.save(os.path.join(data_dir, f'target_features_{out_dim}.npy'), target_features)

    # 2. Edges (Matrix to EdgeIndex)
    def matrix_to_edge_index(filename):
        path = os.path.join(dataset_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return np.array([[], []], dtype=int)
        mat = np.load(path)
        rows, cols = np.where(mat != 0)
        return np.vstack([rows, cols])

    # Herb-Ingredient
    edge_hi = matrix_to_edge_index('herb_ingredient_matrix.npy')
    np.save(os.path.join(data_dir, 'edge_herb_ingredient.npy'), edge_hi)
    
    # Herb-Target
    edge_ht = matrix_to_edge_index('herb_target_matrix.npy')
    np.save(os.path.join(data_dir, 'edge_herb_target.npy'), edge_ht)
    
    # Ingredient-Target
    edge_it = matrix_to_edge_index('ingredient_target_matrix.npy')
    np.save(os.path.join(data_dir, 'edge_ingredient_target.npy'), edge_it)
    
    # 3. Labels
    label_path = os.path.join(dataset_dir, 'TCM_Labels_5.npy')
    if os.path.exists(label_path):
        labels = np.load(label_path)
        np.save(os.path.join(data_dir, 'labels.npy'), labels)
    else:
        print("Warning: TCM_Labels_5.npy not found.")

    print(f"Data processing complete. Features dim: {out_dim}")

def one_error(y_true, y_pred):
    n_samples, n_labels = y_true.shape
    one_error_score = 0.0

    for i in range(n_samples):
        # 找到预测得分最高的标签索引
        highest_score_index = np.argmax(y_pred[i])

        # 判断该标签是否属于真实标签集合
        if y_true[i, highest_score_index] == 0:
            one_error_score += 1.0

    one_error_score /= n_samples
    return one_error_score

def create_heterogeneous_data():
    """根据异构图定义和初始化特征 (DGL Version)"""
    data_dir = os.path.join(current_dir, 'data')
    
    # Ensure data exists
    if not os.path.exists(os.path.join(data_dir, 'herb_features_300.npy')):
        process_data()

    # Load edge indices
    edge_hi = np.load(os.path.join(data_dir, 'edge_herb_ingredient.npy'))
    edge_ht = np.load(os.path.join(data_dir, 'edge_herb_target.npy'))
    edge_it = np.load(os.path.join(data_dir, 'edge_ingredient_target.npy'))
    
    # Construct DGL HeteroGraph
    graph_data = {
        ('herb', 'hi', 'ingredient'): (torch.tensor(edge_hi[0]), torch.tensor(edge_hi[1])),
        ('ingredient', 'ih', 'herb'): (torch.tensor(edge_hi[1]), torch.tensor(edge_hi[0])),
        
        ('herb', 'ht', 'target'): (torch.tensor(edge_ht[0]), torch.tensor(edge_ht[1])),
        ('target', 'th', 'herb'): (torch.tensor(edge_ht[1]), torch.tensor(edge_ht[0])),
        
        ('ingredient', 'it', 'target'): (torch.tensor(edge_it[0]), torch.tensor(edge_it[1])),
        ('target', 'ti', 'ingredient'): (torch.tensor(edge_it[1]), torch.tensor(edge_it[0])),
    }
    
    g = dgl.heterograph(graph_data)
    
    # Load features (dim 300)
    g.nodes['herb'].data['feat'] = torch.tensor(np.load(os.path.join(data_dir, 'herb_features_300.npy')), dtype=torch.float32)
    g.nodes['ingredient'].data['feat'] = torch.tensor(np.load(os.path.join(data_dir, 'ingredient_features_300.npy')), dtype=torch.float32)
    g.nodes['target'].data['feat'] = torch.tensor(np.load(os.path.join(data_dir, 'target_features_300.npy')), dtype=torch.float32)

    # Labels
    g.nodes['herb'].data['label'] = torch.tensor(np.load(os.path.join(data_dir, 'labels.npy')), dtype=torch.float32)

    return g

def macro_f1_score(y_true, y_pred,thresholds):
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    thresholds = np.array(thresholds)

    # 检查维度是否匹配
    if len(thresholds) != y_true.shape[1]:
        raise ValueError("Thresholds 的长度必须与标签的数量一致！")

    # 计算每个标签的 F1 Score
    f1_scores = []
    for i in range(y_true.shape[1]):
        # 使用对应的阈值进行二值化
        binary_predictions = y_pred[:, i] >= thresholds[i]
        f1_scores.append(f1_score(y_true[:, i], binary_predictions))

    return f1_scores

def evaluate_metrics(y_true, y_pred, threshold=None):
    """
    计算多标签分类的六个评估指标，并动态选择最佳阈值
    """
    # 确保输入是 numpy 数组
    y_true = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    
    if threshold is not None:
        best_threshold = threshold
    else:
        # 动态选择最佳阈值
        best_threshold = 0.5
        best_macro_f1 = 0
        
        for thr in np.arange(0.05, 1, 0.05):  # 从 0 到 1，步长为 0.05
            y_pred_bin = (y_pred >= thr).astype(int)
            macro_f1 = f1_score(y_true, y_pred_bin, average='macro')
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_threshold = thr

    # 使用最佳阈值重新计算二值化预测
    y_pred_bin = (y_pred >= best_threshold).astype(int)
    # 计算其他指标
    micro_f1 = f1_score(y_true, y_pred_bin, average='micro')
    macro_f1 = f1_score(y_true, y_pred_bin, average='macro')
    avg_precision = precision_score(y_true, y_pred_bin, average='samples', zero_division=1)
    try:
        avg_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        avg_auc = 0.0  # 如果 AUC 计算异常（如标签类别不足），设置为 0
    rank_loss = label_ranking_loss(y_true, y_pred)
    coverage = coverage_error(y_true, y_pred) - 1  # Coverage 从 0 开始计数
    error = one_error(y_true, y_pred)
    return {
        "Macro-F1": macro_f1,
        "Micro-F1": micro_f1,
        "AvgPrec": avg_precision,
        "AvgAUC": avg_auc,
        "RankLoss": rank_loss,
        "Coverage": coverage,
        "one_error": error,
        "Best Threshold": best_threshold
    }


def cal_accuracy(val_target, final_val_out):
    threshold = 0.5

    # 将张量转换为NumPy数组
    final_val_out = final_val_out.detach().cpu().numpy()
    val_target = val_target.detach().cpu().numpy()

    # 将输出概率转换为二进制标签
    final_val_out = (final_val_out >= threshold).astype(int)

    # 计算每个样本的准确率
    sample_accuracies = []
    for true_labels, pred_labels in zip(val_target, final_val_out):
        correct_predictions = (true_labels == pred_labels).mean()
        sample_accuracies.append(correct_predictions)
    return sample_accuracies



def split_data(g, test_ratio, num_folds, random_seed=42):
    """数据集划分，固定划分方式"""
    # 设置随机种子，确保操作可重复
    all_indices = list(range(g.num_nodes('herb')))

    # 划分训练验证集和测试集
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=test_ratio, random_state=random_seed
    )

    # K 折交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    
    # split() returns indices relative to train_val_indices
    # We need to map them back to global indices
    train_val_splits = []
    train_val_indices_arr = np.array(train_val_indices)
    
    for train_idx, val_idx in kf.split(train_val_indices):
        train_global = train_val_indices_arr[train_idx].tolist()
        val_global = train_val_indices_arr[val_idx].tolist()
        train_val_splits.append((train_global, val_global))

    return {
        "train_val_indices": train_val_indices,
        "test_indices": test_indices,
        "train_val_splits": train_val_splits
    }


def cosine_similarity(a, b):
    """计算余弦相似度"""
    a_norm = a / a.norm(dim=-1, keepdim=True)
    b_norm = b / b.norm(dim=-1, keepdim=True)
    return torch.matmul(a_norm, b_norm.T)


def weighted_label_fusion(test_emb, train_embs, train_labels, threshold=0.5, tau=0.1):
    """加权标签融合策略"""
    # 计算余弦相似度
    cos_sim = cosine_similarity(test_emb, train_embs)
    neighbors = cos_sim >= threshold

    # 对邻居进行加权求和
    weights = torch.exp(cos_sim[neighbors] / tau)
    weights = weights / weights.sum()
    fused_labels = torch.sum(weights.unsqueeze(1) * train_labels[neighbors], dim=0)

    return fused_labels

class EarlyStopping(object):
    def __init__(self, patience=2):
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def step(self, loss, state_dict):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(state_dict)
        elif (loss > self.best_loss):
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(state_dict)
            self.best_loss = loss
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, state_dict):
        """Saves model when validation loss decreases."""
        self.best_model_state = copy.deepcopy(state_dict)

    def load_checkpoint(self):
        """Load the latest checkpoint."""
        return self.best_model_state



if __name__ == '__main__':
    data = create_heterogeneous_data()
    splits = split_data(data, test_ratio=0.001, num_folds=5)
