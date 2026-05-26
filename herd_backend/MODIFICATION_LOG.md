# 化合物→草药毒性预测模型 — 完整改动记录

## 概述

本文档记录对 `train_compound_ecc.py` 的全部改动过程，包括改动动机、具体变更、实验结果和最终决策。

**模型架构**: CompoundToxModel(化合物级) → Prior(草药级先验) → ProtoNet × 5 + ECC-Adaptive+ 链式预测

---

## 第一轮：基础优化 (问题0-9)

### 问题0：化合物模型去掉 train/val 拆分 [采纳]

**日期**: 2026-05-20

**改动前**: `train_compound_fold` 内部将化合物 85/15 拆分为 train/val，在 val 上算 macro-AUC 选 best epoch，保存 best_state。

**问题**: 75-90 个化合物样本上的 macro-AUC 误差极大（±0.05-0.08），early stopping 相当于随机选 epoch。

**改动**:
```python
# 去掉 train/val split
# 全量化合物用于训练
# 固定 40 epochs (原 25)
# weight_decay: 5e-4 → 1e-3
# dropout: 0.2/0.1 → 0.4
# 去掉 best_state 逻辑，直接返回最后模型
```

**结果**: 
- 旧版: AUC=0.68x (有泄漏版本，无可比基线)
- 新版训练更稳定，不再依赖随机 val 集

**文件**: `train_compound_ecc.py` → `train_compound_fold()`, `CompoundToxModel`

---

### 问题1：mixup/label_smoothing 顺序修正 [采纳]

**日期**: 2026-05-20

**改动前**:
```python
fv, lb = mixup_batch(fv, lb, alpha=0.2)  # lb 变软标签 [0.3, 0.7, ...]
lb = lb * 0.95 + 0.025                      # 在软标签上再做平滑 → 语义混乱
```

**问题**: 平滑的本意是把硬标签 0/1 推到 0.025/0.975。先 mixup 后，lb 已不是 0/1，平滑几乎失效。

**改动**: 先平滑原始硬标签，再 mixup
```python
lb = lb * 0.95 + 0.025                      # 先平滑硬标签
fv = dropadd(fv); fv, lb = mixup_batch(...)  # 再 mixup
```

**结果** (5-Fold CV, 含数据泄漏):
| 指标 | 改动前 | 改动后 | 变化 |
|------|--------|--------|------|
| AUC | 0.7074 | 0.7067 | -0.0007 |
| Macro-F1 | 0.6889 | 0.6960 | +0.0071 |
| Micro-F1 | 0.6954 | 0.7026 | +0.0072 |

**结论**: 采纳。F1 提升 ~0.7pp，标准差收窄。

**文件**: `train_compound_ecc.py` → `train_label_protonet()` L172-173

---

### 问题2：knowledge_sim_loss 在 mixup 后索引失效 [回退]

**日期**: 2026-05-20

**问题**: mixup 后 `fv[i] = lam * x_a + (1-lam) * x_b`，但 `idxs[i]` 仍指向原始样本 a。`knowledge_sim_loss` 用混合嵌入 z 去匹配原始样本的 Jaccard 相似度，语义不对应。

**尝试1 — 两次forward (原方案)**:
```python
# 拆成两次 forward
fv_aug = dropadd(fv)
_, z_clean = model(fv_aug, return_proj=True)      # 干净特征 → reg_loss
fv_mix, lb_mix = mixup_batch(fv_aug, lb, ...)
logit_mix = model(fv_mix)                          # 混合特征 → cls_loss
```

**结果**: Macro-F1 0.6941 (↓0.002), 方差翻倍 (±0.0328 vs ±0.0163)

**原因**: 两次 forward 使用不同 Dropout mask，梯度互相干扰。

**尝试2 — 相似度矩阵插值 (新方案)**:
```python
lam = np.random.beta(0.2, 0.2)
perm = torch.randperm(fv.size(0))
fv = lam * fv + (1-lam) * fv[perm]
# 知识正则目标同样插值
sim_target = lam * sim_i + (1-lam) * sim_j  
```

**结果**: Macro-F1 0.6866 (↓0.0094), Micro-F1 0.6953 (↓0.0073)

**原因**: 插值后的相似度目标信息量被稀释。

**最终决策**: 回退。原始代码虽然 indices 不完全对应，但在小数据集上作为噪声正则反而有用。

---

### 问题5：p1/p2 ensemble 权重调整 [回退]

**改动**: `0.4*p1 + 0.6*p2` 替代 `(p1+p2)/2`（认为 p2 用了 p1 的软预测，质量更高）

**结果**: AUC 0.7046 (↓0.002), Macro-F1 0.6920 (↓0.004)

**原因**: p2 虽然条件信号更好，但同时继承了 p1 的错误。固定权重无法捕捉这种依赖。等权重能让两者互相纠错。

**最终决策**: 回退。

---

### 问题4：compute_mi_auc_chains 简化 [回退]

**尝试1 — 去掉 standalone_aucs**

**改动**: 链顺序仅基于 MI 矩阵，去掉嵌套 3-fold CV 的 standalone_aucs 加权。

**结果**: AUC 0.7024 (↓0.004), Macro-F1 0.6900 (↓0.006)

**原因**: standalone_aucs 提供的标签可预测性信号对贪心链构造有用——先预测容易的标签，再条件化难的标签。

**尝试2 — 正例率替代嵌套CV**

**改动**: `standalone_aucs = 1 - 2*|pos_rate - 0.5|` 替代嵌套 CV

**结果**: AUC 0.7010 (↓0.006), Macro-F1 0.6867 (↓0.009)

**原因**: 正例率太粗糙，嵌套 CV 虽然有方差问题但 signal 更强。

**最终决策**: 两次均回退。

---

### 问题9：CompoundToxModel dropout 参数化 [采纳]

**改动**: 增加 `dropout=0.4` 参数替代硬编码 `nn.Dropout(0.4)`

**结果**: 行为等价，改善代码可维护性。

---

### 问题8：compare_dual_view.py 死代码清理 [采纳]

**改动**: 删除 `if isinstance(label_strs_or_stratify, list): stratify_col = ... else: stratify_col = ...` 无意义分支

---

### 问题3：ECC 暴露偏差 [回退]

**问题**: 训练时链条件用真实 0/1 标签，推理时用预测概率（软值），存在分布漂移。

**尝试1 — 软化训练条件**

**改动**: p1 链训练条件从 hard 改为 `train_labels * 0.9 + 0.05`，p2 从 0.8/0.1 改为 0.75/0.125

**结果**: Macro-F1 0.6902 (↓0.006), Micro-F1 0.6969 (↓0.006)。AUC 微涨 (+0.003) 但 F1 明显下跌。

**尝试2 — 二值化推理条件**

**改动**: 推理时 `(val_probs > 0.5).astype(float)` 替代原始概率

**结果**: Macro-F1 0.6798 (↓0.016), Micro-F1 0.6879 (↓0.015)。概率信息丢失严重。

**最终决策**: 两次均回退。当前 ECC 链式预测靠硬边界传递信息，软化或二值化都会损失关键信号。

---

## 第二轮：注意力聚合 (D)

### CompoundAttentionAggregator [采纳]

**日期**: 2026-05-20

**动机**: `compute_prior` 使用 `.mean(axis=0)` 对所有化合物等权平均，忽略了不同化合物对毒性的贡献差异。

**架构演进**:

**v1 — 单头注意力**:
```python
class CompoundAttentionAggregator(nn.Module):
    def __init__(self, prior_dim=5, hidden=32):
        self.attn = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(),
            nn.Linear(32, 1)  # 单头：所有毒性共享注意力
        )
```

训练: 冻结 CompoundToxModel → 预计算化合物预测 → 训练 aggregator (ASL loss, herb labels, 30 epochs)

**结果** (含泄漏): AUC 0.7413 (+3.5pp vs 问题1 baseline)

**v1.1 — 梯度累积**:
```python
accum_steps = 4  # 每4个草药累积梯度
loss = crit(agg, lbl) / accum_steps
```

**结果**: 效果持平，但标准差收窄 (Macro-F1 std ±0.0230 → ±0.0154)

**v2 — 多头注意力**:
```python
self.attn = nn.Sequential(
    nn.Linear(5, 32), nn.Tanh(),
    nn.Linear(32, 5)  # 多头：每个毒性标签独立注意力
)
# attn_w: [n_compounds, 5] — 每个化合物对每个毒性有独立权重
```

**结果** (含泄漏): AUC 0.7539 (+1.3pp vs 单头)

**v2.1 — epochs=50**:
**结果** (含泄漏): AUC 0.7543, Macro-F1 0.7250, Micro-F1 0.7311

**v3 — 消除数据泄漏后**:

发现并修复了 `build_compound_labels(all_data)` 的数据泄漏问题。
修复后真实结果:

| 配置 | AUC | Macro-F1 | Micro-F1 |
|------|-----|----------|----------|
| Mean pool (无泄漏基线) | 0.6803 | 0.6736 | 0.6779 |
| + Aggregator (无泄漏) | 0.6853 | 0.6801 | 0.6876 |
| Aggregator 真实提升 | +0.005 | +0.007 | +0.010 |

**文件**: 
- `model.py` → `CompoundAttentionAggregator`
- `train_compound_ecc.py` → `train_compound_aggregator()`, `compute_prior()`

---

## 第三轮：数据泄漏发现与修复

**日期**: 2026-05-20

**发现**: `build_compound_labels(all_data, n_compounds)` 在 5-Fold CV 之前调用，使用全部草药数据构建化合物标签。

**泄漏路径**:
1. 化合物 X 同时存在于训练草药 A 和验证草药 B
2. B 的标签 [0,1,0,0,0] 通过 `np.maximum` 写入 `c_labels[X]`
3. 化合物模型在 X 上训练时，已经"见过"验证草药的标签信息
4. 所有之前的 5-Fold CV 结果均被污染

**修复**:
```python
# 修复前 (泄漏)
c_labels = build_compound_labels(all_data, n_compounds)

# 修复后 (无泄漏)
train_herbs = [all_data[i] for i in train_idx]
c_labels = build_compound_labels(train_herbs, n_compounds)
```

**影响**:
- 旧"基线" AUC 0.707 → 真实基线 AUC 0.680
- 旧"aggregator" AUC 0.754 → 真实 aggregator AUC 0.685
- Aggregator 真实提升: +0.5~1pp (非之前看到的 +3.5pp)

---

## 最终配置总结

### 当前最优配置 (train_compound_ecc.py)

```
- 化合物模型: CompoundToxModel(fp_dim=1024, hidden=256, dropout=0.4)
  - 全量训练，固定 40 epochs, AdamW(lr=1e-3, wd=1e-3)
  - 无 train/val 拆分
  
- 训练技巧: label smoothing → dropadd → mixup (顺序! smoothing 先于 mixup)

- 先验计算: CompoundAttentionAggregator(多头, hidden=32)
  - 训练: 冻结 CompoundToxModel, 预计算化合物预测
  - AdamW(lr=3e-3, wd=1e-4), 50 epochs, 梯度累积=4
  - ASL loss (gamma_neg=2, gamma_pos=1) 对 herb labels 训练

- ECC-Adaptive+: MI × standalone_AUC 链顺序, 5 chain × 5 label × 2 pass (p1/p2)
  - p1: 训练用真实标签, 推理用 val_probs (soft)
  - p2: 训练用软化标签 (0.8*lbl+0.1), 推理用 p1_val (soft)
  - 最终: (p1 + p2) / 2 等权融合

- ProtoNet: GatedPriorProtoNet + ASL(gamma_neg=4) + knowledge_sim_loss
  - patience=50, epochs=200, lr=1e-3, wd=5e-4
  - DropAdd(drop=0.2, add=0.01), Mixup(α=0.2)

- 数据安全: c_labels 每折独立构建，只用训练草药
```

### 真实性能 (无泄漏 5-Fold CV)
| 指标 | 值 |
|------|-----|
| AUC | 0.6853 ± 0.0190 |
| Macro-F1 | 0.6801 ± 0.0204 |
| Micro-F1 | 0.6876 ± 0.0211 |

### 备份文件
| 文件 | 描述 | AUC |
|------|------|-----|
| `train_compound_ecc_070.py` | 问题0+1, mean pool, 含泄漏 | ~0.707 |
| `train_compound_ecc_075.py` | 问题0+1+aggregator, 含泄漏 | ~0.754 |
| `train_compound_ecc.py` | 当前最优, 无泄漏 | ~0.685 |

---

## 第四轮：代码安全性加固 + 评估规范

### 问题二：aggregator 传参安全化 [采纳]
**日期**: 2026-05-21
**改动**: `train_compound_aggregator` 从 `(model, aggregator, compound_fps, all_data, train_idx)` 改为 `(model, aggregator, compound_fps, train_herbs)`，内部用局部索引替代全局索引。
**结果**: 行为等价，AUC 不变。消除了潜在的数据误用风险。
**文件**: `train_compound_ecc.py` → `train_compound_aggregator()`

### 问题三：固定阈值 F1 消除评估偏差 [采纳]
**日期**: 2026-05-21
**改动**: 同时输出 F1(0.5) 和 F1(best)，前者是干净无偏指标，后者注明乐观偏差。
**结果**: 
- F1(0.5) Macro=0.6348, F1(best) Macro=0.6801 — 搜索阈值造成 +4.5pp 乐观偏差
- AUC 不受阈值影响，是干净指标
**文件**: `train_compound_ecc.py` → `main()` 评估部分

### 问题四：train_compound_fold 索引统一 [采纳]
**日期**: 2026-05-21
**改动**: `train_compound_fold` 从 `(compound_fps, c_labels, train_idx, all_data, epochs)` 改为 `(compound_fps, c_labels, train_herbs, epochs)`，直接遍历草药列表。
**结果**: 行为等价，AUC 不变。消除了全局索引与局部列表混用的问题。
**文件**: `train_compound_ecc.py` → `train_compound_fold()`

---

## 第五轮：ECC 链顺序消融

### 发现：MI Only = MI+AUC [可简化]
**日期**: 2026-05-21
**测试**: Full (MI+AUC) vs MI Only vs Random vs Fixed Forward vs Fixed Reverse
**结果**:
| 策略 | AUC |
|------|-----|
| MI + AUC (Full) | 0.6853 |
| MI Only | 0.6853 (完全一致) |
| Random | 0.6798 (-0.006) |
| Fixed Forward | 0.6823 (-0.003) |
| Fixed Reverse | 0.6824 (-0.003) |
| P1 Only | 0.6817 (-0.004) |
**结论**: standalone_aucs (嵌套3折CV) 不影响排序 → 可安全移除。P2 对 Macro-F1 贡献(+0.9pp)大于对 AUC 贡献(+0.4pp)。

### 项目基础设施
**日期**: 2026-05-21
**新增文件**:
- `CLAUDE.md` — 项目开发守则（泄漏防护、评估规范、设计决策、文档同步规则）
- `train_ablation.py` — 消融实验脚本（13 组配置，`--run_all` 一键复现）

---

## 第六轮：创新点重构 (Phase 1)

### Context-Aware Gate [回退]
**日期**: 2026-05-21
**改动**: GatedPriorProtoNet 的 gate 从 `gate(prior)` 改为 `gate(cat(h, prior))`，让门同时看特征和先验
**结果**: AUC 0.6813 (-0.4pp), 方差增大
**原因**: context-aware gate 第一层 Linear(261→128) 引入 33K 参数，在 252 样本上过拟合。原设计 Linear(5→20→256) = 100 参数在小数据上更稳定。
**文件**: `model.py` → `GatedPriorProtoNet`（已回退，docstring 保留设计选择说明）

### ECC 链排序简化 [采纳]
**日期**: 2026-05-21
**改动**: `compute_mi_auc_chains` 删除 standalone_aucs（嵌套3折CV），链排序仅用 MI 矩阵
**消融验证**: MI Only = MI+AUC（链顺序完全一致，逐折结果相同）
**结果**: AUC 不变，代码更简洁，去掉 `cross_val_score` 和 `LogisticRegression` 依赖
**文件**: `train_compound_ecc.py` → `compute_mi_auc_chains()`

### 创新点文档重写 [采纳]
**日期**: 2026-05-21
**改动**: 重写 `INNOVATION_DOC.md`，重新分级创新点：
- 主创新：化合物→草药跨粒度知识迁移框架
- 次级创新：ECC-Adaptive+ 双轮链式预测
- 辅助技术：knowledge_sim_loss + 数据增强（降级为非独立创新）
- 新增"探索过但放弃的方向"章节（5 个回退尝试及原因）
