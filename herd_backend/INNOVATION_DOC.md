# 中药毒性预测 — 创新点文档

## 创新点总览

| # | 创新点 | 类型 | 消融证据 |
|---|--------|------|---------|
| 1 | 跨域分子知识迁移 | 核心方法 | AUC -10.5pp |
| 2 | DropAdd 非对称分子指纹增强 | 数据增强 | AUC -1.8pp |
| 3 | 多头逐毒性化合物注意力聚合 | 架构设计 | vs 单头 -1.3pp, vs mean pool -0.5pp |
| 4 | ECC 双轮硬→软条件链式预测 | 训练策略 | Macro-F1 -0.9pp |
| 5 | 化合物共现 Jaccard 嵌入正则化 | 辅助损失 | AUC -2.4pp |

---

## 创新点 1：跨域分子知识迁移

### 核心思路

西药有大量毒性标注数据（UniTox 1349 药物，8 类毒性），中药只有 252 味草药。将西药分子知识冻结迁移到中药预测中。

### 架构

```
西药数据 (1349 drugs, 8 toxicity classes)
    ↓
Drug Tower 预训练 (Morgan FP → encoder → 8-class head)
    ↓ 冻结 encoder
中药化合物 Morgan FP → Drug Tower encoder → drug_emb [256]
    ↓ mean pool
drug_view [256] → Linear(256→5) → drug_prior [5]
    ↓
compound_prior [5] + drug_prior [5] → 融合 prior
    ↓
GatedPriorProtoNet → 最终预测
```

### 与标准迁移学习的三个结构性差异

| 维度 | 标准迁移学习 | 本方法 |
|------|------------|--------|
| **粒度** | 同粒度 (样本→样本) | **跨粒度**: 化合物→草药，需注意力聚合桥接 |
| **标签空间** | 同空间或子集 | **跨空间**: 8 类西药→5 类中药，隐式知识压缩 |
| **知识融合** | 替换 (旧→新) | **多源互补**: TCM prior + drug prior 共存，双源融合 |

### 消融证据

| 配置 | AUC | Macro-F1 |
|------|-----|----------|
| ProtoNet (无任何 prior) | 0.6791 | 0.6737 |
| TCM compound_prior only | 0.6794 | 0.6731 |
| Drug prior only | **0.7843** | 0.7363 |
| Both (hard sum) | **0.7843** | **0.7440** |

**核心洞见**：
- TCM 内部自蒸馏对 AUC 零增量（0.6794 vs 0.6791），因为不带入新信息
- Drug prior alone = Full Model AUC — 跨域知识信号主导
- 两者在 F1 上有互补：drug 擅长排序，compound 辅助校准

---

## 创新点 2：DropAdd 非对称分子指纹增强

### 核心思路

标准数据增强（高斯噪声、Bernoulli dropout）不区分"特征被错误清零"和"特征被错误置1"。分子指纹中这两种错误的概率不对称：**漏测（假阴性）远比误测（假阳性）常见**。

### 设计

```python
def dropadd(fv, drop_p=0.20, add_p=0.01):
    # 随机清零：模拟"该有的子结构没测出来"（假阴性）
    mask = (torch.rand_like(fv) > drop_p).float()
    fv = fv * mask
    # 稀疏翻转：模拟"测出了不该有的信号"（假阳性）
    add_mask = (torch.rand_like(fv) < add_p) & (fv == 0)
    return torch.clamp(fv + add_mask.float(), 0, 1)
```

### 为什么是创新

| 方面 | 标准 Dropout | DropAdd |
|------|-------------|---------|
| 方向 | 仅丢弃 | 丢弃 + 添加，双方向 |
| 对称性 | 单方向 | **非对称** (20% vs 1%) |
| Rescaling | x/(1-p) | **不做 rescaling** |
| 物理含义 | 无 | 假阴性 > 假阳性 |
| 输出约束 | 无 | clamp 到 [0,1] |

### 消融证据

移除 DropAdd → AUC **-1.8pp**, Macro-F1 **-1.8pp**

---

## 创新点 3：多头逐毒性化合物注意力聚合

### 核心思路

一味草药含 2-50 个化合物。传统做法是 mean pool（等权平均），但不同毒性类型的关键化合物不同——肝毒性看代谢酶抑制剂，肾毒性看转运体底物。

### 设计

```python
class CompoundAttentionAggregator(nn.Module):
    def __init__(self, prior_dim=5, hidden=32):
        self.attn = nn.Sequential(
            nn.Linear(5, 32), nn.Tanh(),
            nn.Linear(32, 5)  # ← 关键: 每标签独立注意力头
        )
    
    def forward(self, compound_probs):  # [n_compounds, 5]
        attn_w = torch.softmax(self.attn(compound_probs), dim=0)  # [n_compounds, 5]
        aggregated = (attn_w * compound_probs).sum(dim=0)  # [5]
        return aggregated, attn_w  # 权重直接可解释
```

### 为什么是创新

| 方面 | 标准注意力 | 本方法 |
|------|-----------|--------|
| 注意力头数 | 1（所有任务共享） | **5（每毒性独立）** |
| 药理含义 | 无 | 肝毒注意力 ≠ 肾毒注意力 |
| 可解释性 | 弱 | `attn_w[i,j]` = 化合物 i 对毒性 j 的贡献 |

### 消融证据

| 配置 | AUC |
|------|-----|
| Multi-head attention | 0.7539 |
| Single-head attention | 0.7406 (-1.3pp) |
| Mean pool | 0.6803 (-0.5pp vs single-head) |

---

## 创新点 4：ECC 双轮硬→软条件链式预测

### 核心思路

标准 ECC（Ensemble of Classifier Chains）在每轮使用相同的条件策略。本方法让两轮使用**不同的条件分布**——P1 用硬标签训练、软概率推理，P2 用软化标签训练、P1 预测推理。P2 训练时永远看不到干净的链条件，形成隐式正则化。

### 设计

```
P1 (第一轮): 训练条件 = 真实 0/1 标签     → 推理条件 = val_probs (软)
P2 (第二轮): 训练条件 = 软化标签(0.8*l+0.1) → 推理条件 = P1 预测 (软)

最终: (P1 + P2) / 2
```

### 为什么是创新

| 方面 | 标准 ECC | ECC-Adaptive+ |
|------|---------|---------------|
| 链条件 | 所有轮相同 | **两轮条件分布不同** |
| P2 训练 | 硬标签 | **软化标签（隐式计划采样）** |
| 正则化 | 无 | P2 的软条件迫使模型适应噪声 |

### 消融证据

去掉 P2 → AUC -0.4pp, **Macro-F1 -0.9pp**

P2 对 F1 的贡献远大于对 AUC 的贡献——它提升的是分类决策质量而非排序质量。

---

## 创新点 5：化合物共现 Jaccard 嵌入正则化

### 核心思路

如果两味草药共享很多化合物，它们在嵌入空间中应该相近。用草药自身的化合物共现矩阵（Jaccard 相似度）作为嵌入空间的结构先验。

### 设计

```python
def knowledge_sim_loss(z, know_sim_matrix, indices):
    embed_sim = normalize(z) @ normalize(z).T   # 模型学到的草药相似度
    target_sim = know_sim_matrix[indices][:, indices]  # 化合物共现 Jaccard 相似度
    return MSE(embed_sim, target_sim)
```

### 为什么是创新

| 方面 | 通用图正则化 | 本方法 |
|------|------------|--------|
| 知识来源 | 外部知识图谱 | **草药自身的化合物组成** |
| 相似度定义 | 通用距离度量 | **Jaccard（化合物集合交集/并集）** |
| 领域适配 | 无 | 天然适配中药的多化合物特性 |

### 消融证据

移除 knowledge_sim_loss → AUC **-2.4pp**, Macro-F1 **-2.3pp**

这个正则项的性能贡献仅次于 Drug Tower 和 Mixup，是所有辅助损失中最重要的。

---

## 完整消融证据总表

| 移除组件 | AUC Δ | Macro-F1 Δ | 对应创新点 |
|----------|-------|-----------|-----------|
| Drug Tower 跨域迁移 | **-10.5pp** | -7.1pp | 创新 1 |
| Jaccard 嵌入正则 | **-2.4pp** | **-2.3pp** | 创新 5 |
| DropAdd 增强 | -1.8pp | -1.8pp | 创新 2 |
| 多头→单头注意力 | -1.3pp | — | 创新 3 |
| Knowledge Transfer (TCM内部) | -0.9pp | -0.8pp | 创新 1 |
| P2 双轮集成 | -0.4pp | **-0.9pp** | 创新 4 |
| Aggregator→mean pool | -0.5pp | -0.7pp | 创新 3 |
| Mixup 增强 | -4.3pp | -2.4pp | 辅助 |
| Compound Opt | -0.4pp | -0.5pp | 辅助 |

---

## 探索过但放弃的方向

| 方向 | 结果 | 原因 |
|------|------|------|
| Context-aware gate | AUC -0.4pp | 33K 参数在小数据过拟合 |
| 相似度矩阵插值 | F1 -0.9pp | 插值稀释信号 |
| ECC 训练软化 | F1 -0.6pp | 破坏硬边界判别 |
| 推理二值化 | F1 -1.6pp | 丢失概率信息 |
| 自适应双先验融合 | AUC -0.6pp | 无增量，drug prior 已主导 |
| 多视图对比对齐 | AUC 持平 | 无增量，小数据无统计信号 |

---

## 局限与下一步

1. **数据规模** (252 草药)：限制复杂机制的表达空间
2. **分子表示**：仅用 Morgan FP，未利用分子图结构
3. **西药数据**：1349 药物（UniTox 子集），全量数据可能带来更大收益
4. **外部 baseline**：需与已发表方法对比
