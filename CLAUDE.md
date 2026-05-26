# CLAUDE.md — 中药毒性预测项目开发守则

## 数据泄漏防护 (最高优先级)

### 规则1: 化合物标签必须按折构建
`build_compound_labels()` 只能使用当前折的训练集草药，**禁止**传入全量数据。
```python
# 正确
train_herbs = [all_data[i] for i in train_idx]
c_labels = build_compound_labels(train_herbs, n_compounds)

# 错误 —— 会造成验证集标签通过共享化合物泄入训练
c_labels = build_compound_labels(all_data, n_compounds)
```

### 规则2: 所有标签相关计算必须在 split 之后
- `compute_mi_auc_chains` 只用训练集标签
- `find_best_thresholds` 只在验证集上搜索
- 任何用到标签的统计量（均值、分布等）必须在 split 后从训练集计算

### 规则3: 函数传参用局部列表，不用全局索引
避免 `herbs[global_idx]` 模式，直接传 `train_herbs` 列表。
```python
# 正确
def train_compound_fold(compound_fps, c_labels, train_herbs, epochs=40):
    for h in train_herbs: ...

# 错误 —— 依赖全局索引和数据，容易误用
def train_compound_fold(compound_fps, c_labels, train_idx, all_data, epochs=40):
    for hi in train_idx: h = all_data[hi]
```

## 评估规范

### 规则4: 同时报告固定阈值和搜索阈值 F1
```python
# 干净指标（无乐观偏差，论文主指标）
fixed_preds = (probs > 0.5).astype(int)
macro_f1_fixed = f1_score(labels, fixed_preds, average='macro')

# 乐观上界（论文注明"搜索阈值"）
thresholds = find_best_thresholds(labels, probs)  # 在验证集上搜索
best_preds = apply_thresholds(probs, thresholds)
macro_f1_best = f1_score(labels, best_preds, average='macro')
```
- AUC 是阈值无关的干净指标，论文优先使用
- F1(0.5) 是干净指标，F1(best) 有 ~4.5pp 乐观偏差

### 规则5: 不要用搜索阈值的 F1 来比较模型
`find_best_thresholds` 在验证集上拟合了阈值，用同一验证集的 F1 比较模型会造成乐观偏差。

## 已验证的设计决策

### 保留
- **Mixup (α=0.2)**: 移除后 AUC -4.3pp，不可移除
- **Knowledge Regularization (weight=0.1)**: 移除后 AUC -2.4pp，不可移除
- **DropAdd (drop=0.2, add=0.01)**: 移除后 AUC -1.8pp，不可移除
- **Compound→Herb Knowledge Transfer**: 移除后 AUC -0.9pp，核心创新
- **Attention Aggregator (multi-head, hidden=32)**: 移除后 AUC -0.5pp
- **P2 Ensemble**: 移除后 Macro-F1 -0.9pp
- **Label Smoothing → Mixup 顺序**: 先平滑硬标签，再 mixup
- **Compound 全量训练无 val split**: 75 样本上的 macro-AUC 误差太大，固定 40 epochs

### 已证无效/可简化
- **MI×AUC vs MI Only 链排序**: 等价，standalone_aucs 可安全移除
- **软化/二值化 ECC 条件**: 反而有害，保持原始 0/1 训练 + soft 推理
- **相似度矩阵插值 (知识正则)**: 理论上正确但小数据上效果更差
- **P1/P2 不等权**: 0.5:0.5 等权最优

## 代码修改检查清单

每次修改训练代码后，确认以下事项：
- [ ] 5 折 CV 的每折中，训练/验证是否有任何信息交叉
- [ ] 所有 `build_compound_labels` 只用训练草药
- [ ] 没有全局索引 + 全量数据的函数签名模式
- [ ] 评估同时输出 F1(0.5) 和 F1(best)
- [ ] 修改后跑完整 5-Fold CV 确认结果不退化
- [ ] 新组件加了消融实验开关

## 核心文件

| 文件 | 用途 |
|------|------|
| `herd_backend/model.py` | ProtoNet, GatedPriorProtoNet, CompoundAttentionAggregator, ASL |
| `herd_backend/train_compound_ecc.py` | 主训练脚本（当前最优，无泄漏） |
| `herd_backend/train_ablation.py` | 消融实验脚本 |
| `herd_backend/train_compound_ecc_070.py` | 备份：AUC~0.707 (含泄漏) |
| `herd_backend/train_compound_ecc_075.py` | 备份：AUC~0.754 (含泄漏) |

## 已知陷阱

1. **不要修改 knowledge_sim_loss 的调用方式**: 在 mixup 后用原始 idxs 虽然语义不完全对，但作为噪声正则在小数据上意外有益。两次 forward 或相似度插值都会破坏 Dropout 一致性或稀释信号。

2. **不要调整 P1/P2 集成权重**: 0.5/0.5 最优。P2 虽然条件信号更好但也继承了 P1 的错误。

3. **不要软化 ECC 训练条件**: 模型用硬边界传递信息。软化训练条件或二值化推理条件都会损伤 F1。

4. **小数据的特殊性**: 这个项目只有 ~252 个训练样本。很多"理论上正确"的改进（更好的正则目标、更精细的采样策略）在小数据上都会被方差淹没。在新想法落地前，先问：这个改动在 200 样本上会有足够的统计信号吗？

## 文档同步规则

每次修改代码后，**必须**同步更新对应文档。这包括消融实验、创新点变更、架构调整、超参数改动等。

### 文档对应关系

| 改动类型 | 需更新的文档 |
|----------|-------------|
| 新增/修改模型组件 | `INNOVATION_DOC.md` + `MODIFICATION_LOG.md` |
| 新增消融实验或结果 | `ABLATION_RESULTS.md` |
| 任何代码改动（含回退） | `MODIFICATION_LOG.md`（追加记录） |
| 发现新的设计原则或陷阱 | `CLAUDE.md`（本文件） |
| 新备份版本 | 文件名记录到 `MODIFICATION_LOG.md` |
| 发现数据泄漏或修复 | `CLAUDE.md` 规则1-3 + `MODIFICATION_LOG.md` |

### 更新模板

**MODIFICATION_LOG.md 追加格式**:
```markdown
### 改动名 [采纳/回退]
**日期**: YYYY-MM-DD
**改动**: 简述
**结果**: AUC x.xxxx → y.yyyy (±Δ)
**原因**: 为什么采纳/回退
```

**ABLATION_RESULTS.md 追加格式**:
- 新配置加入"消融结果总表"
- 补充"逐项分析"章节
- 更新"组件重要性排序"

**INNOVATION_DOC.md 更新触发条件**:
- 新模型架构、新损失函数、新训练策略 → 新增章节
- 已有创新点被消融验证 → 补充消融数据
- 创新点被证伪 → 移到"已探索/放弃"章节，保留记录

