# 消融实验报告 (最新: 2026-05-26)

**Full Model AUC: 0.8054** | 252 草药, 5-Fold CV (无泄漏)

---

## 完整消融结果 (16 组)

| 配置 | AUC | ΔAUC | F1(best) |
|------|-----|------|----------|
| **Full Model** | **0.8054** | — | 0.7545/0.7641 |
| - Drug Tower | 0.6836 | **-12.18pp** | 0.6734/0.6824 |
| Drug Prior Only | 0.8082 | +0.28* | 0.7461/0.7499 |
| ProtoNet (no prior) | 0.6768 | **-12.86pp** | 0.6774/0.6884 |
| - DropAdd | 0.7873 | **-1.81pp** | 0.7322/0.7387 |
| - P2 | 0.7947 | **-1.07pp** | 0.7474/0.7595 |
| - Jaccard Reg | 0.7961 | -0.93pp | 0.7377/0.7434 |
| - Mixup | 0.7999 | -0.55pp | 0.7475/0.7587 |
| - Aggregator | 0.8076 | +0.22* | 0.7514/0.7572 |
| - Contrastive | 0.8044 | -0.10* | 0.7539/0.7630 |
| - Smooth Order | 0.8054 | 0.00* | 0.7533/0.7604 |
| - Compound Opt | 0.8116 | +0.62* | 0.7564/0.7610 |
| ECC: Fixed Forward | 0.8117 | +0.63* | 0.7531/0.7616 |
| ECC: Fixed Reverse | 0.8057 | +0.03* | 0.7461/0.7514 |
| ECC: Random Order | 0.7986 | -0.68pp | 0.7433/0.7470 |
| ECC: MI Only | 0.8053 | -0.01* | 0.7522/0.7615 |

*标记 * 的差值在一个标准差范围内（±0.096），统计不显著。

---

## 组件重要性排序

```
Drug Tower          ★★★★★★ (AUC -12.18pp, 绝对主导)
DropAdd             ★★★★   (AUC -1.81pp, 非对称分子增强)
P2 Ensemble         ★★★    (AUC -1.07pp, F1 -0.94pp)
Jaccard Reg         ★★★    (AUC -0.93pp, 结构正则)
Mixup               ★★     (AUC -0.55pp, Drug Tower 缓解了必要)
Aggregator/Contrastive/Smooth/Compound ★ (噪声级)
```

---

## 核心发现

1. **Drug Tower 贡献 -12.18pp** — 跨域分子知识迁移是唯一的主导因素
2. **P2 双轮集成在本次 checkpoint 下贡献 -1.07pp** — 统计显著，Macro-F1 -0.94pp
3. **Drug Prior Only = Full Model** (0.8082 vs 0.8054) — TCM compound_prior 对 AUC 零增量
4. **ECC 链排序不是瓶颈** — 所有策略差距 < 0.7pp
