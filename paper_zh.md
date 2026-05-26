# 跨域分子知识迁移用于中药毒性预测

## 摘要

中药的国际关注度持续上升，但由于多成分、多靶点的复杂特性，草药安全性问题仍是临床应用的重大障碍。现有计算方法受限于标注样本稀缺，且未能利用化学药物领域丰富的毒性标注数据。本文提出跨域分子知识迁移框架：在1349种西药的8类毒性数据上预训练DrugTower编码器，冻结后通过轻量级投影层将其迁移至中药毒性预测。框架融合化合物先验（通过多头注意力聚合器获得）与药物先验，经自适应门控注入原型网络分类器，并采用非对称双轮条件的集成分类器链（ECC-Adaptive+）和DropAdd非对称分子指纹增强。在252味草药、5类毒性的数据集上，严格5折交叉验证下取得宏观AUC 0.7821，较当前最优HerbToxNet（AUC 0.6003）提升18.2个百分点。16组消融实验证实跨域药物知识迁移为主要贡献（+9.85pp AUC）。

**关键词**：中药；毒性预测；跨域迁移学习；原型网络；集成分类器链；多标签分类

---

## 1. 引言

草药在亚洲乃至全球医疗体系中占据重要地位，由多种生物活性成分构成[1]。然而许多草药已被证实与肝毒性、肾毒性等不良反应相关联[2,3]。评估草药毒性面临特有挑战：数据异质性和不一致性、多成分多靶点的复杂性，以及多标签分类的必要性——数据集中56%的草药关联两种及以上毒性标签[1]。

现有计算方法可归为三类：基于QSAR从分子指纹预测成分级毒性[6,7]；网络毒理学方法分析草药-成分-靶点互作网络[4,12]；以及基于草药药性理论的方法[5,6]。深度学习方法中，多注意力机制近年来已被应用于药物毒性预测[8]。然而，这些范式均局限于中药领域，未能利用化学药物领域积累的大规模毒性标注数据。

本文提出跨域分子知识迁移框架[10]，将化合物分子指纹、西药毒性先验和草药药理特征统一整合入原型网络多标签分类架构。主要贡献：(i) DrugTower预训练-迁移范式（+9.85pp AUC）；(ii) DropAdd非对称分子指纹增强（+1.87pp）；(iii) Jaccard化合物共现嵌入正则化（+1.40pp）；(iv) 多头注意力聚合器提供内在可解释性；(v) 16组消融严格量化组件贡献。

---

## 2. 相关工作

### 2.1 基于QSAR的方法

QSAR方法从分子结构定量捕捉其与生物效应的关系[9]。He等人[3]收集了肝毒性草药及其成分，确定了生物碱和萜类为主要肝毒性成分。基于深度学习的多注意力方法近年被应用于药物毒性预测[8]。虽然QSAR能在分子层面定量分析，但在草药毒性预测中的适用性受限于数据质量与成分级到草药级之间的鸿沟。

### 2.2 网络毒理学方法

网络毒理学通过毒性-靶点-成分-草药互作网络分析草药毒性[4,5]。Wu等人[4]将机器学习与网络毒理学结合预测肝毒性成分；Song等人[4]构建了TCM系统毒理学数据库TCMSTD 1.0。此类方法能阐释毒性机制但无法直接量化草药级毒性。

### 2.3 基于草药药性理论的方法

"四气五味"及归经理论构成草药药理学框架[2]。药性理论提供传统洞察但其定性和主观性限制了预测性能。

### 2.4 多标签分类与原型网络

多标签分类方法天然适用于多种毒性共存的草药预测[19]。集成分类器链（ECC）[15,11]通过平均多条不同链序的预测降低敏感性。原型网络[16]对小样本场景尤为有效。最直接的对比工作HerbToxNet[1]应用异构图注意力网络（HAN）[17]结合对比学习。本文与其根本差异在于：(i) 引入真实化合物分子指纹；(ii) 从西药数据迁移外部知识；(iii) 以更简洁架构实现更高性能。

---

## 3. 本文方法

框架由五个序贯阶段构成：(i) DrugTower编码器在西药分子指纹上预训练并冻结；(ii) CompoundToxModel在中药化合物上用草药派生伪标签训练；(iii) 多头注意力聚合器计算化合物先验；(iv) 通过冻结DrugTower编码器和可训练投影层的药物先验；(v) GatedPriorProtoNet以ECC-Adaptive+链式预测输出最终多标签概率。

DrugTower编码器采用两层MLP架构（1024→512→256），线性分类头（256→8）。在1349种西药上使用AsymmetricLoss[18]预训练50 epoch，取得宏观AUC 0.7388。预训练后冻结编码器参数，丢弃8类分类头。

CompoundToxModel与DrugTower共享相同架构但独立训练。化合物伪标签通过对包含某化合物的全部草药标签取逐元素最大值构建，每折独立进行。多头注意力聚合器设计5个独立逐毒性注意力头，通过softmax(dim=0)对每种毒性在化合物维度独立归一化。药物先验投影层（Linear(256→5)）每折在训练草药上使用BCE训练30 epoch。

GatedPriorProtoNet融合自适应双源先验、门控注入和原型分类，使用复合损失函数（ASL+Jaccard正则+对比对齐）。DropAdd以20:1的drop/add概率比建模分子指纹假阴性主导的不对称噪声。ECC-Adaptive+通过P1（训练硬→推理软）和P2（训练软→推理P1预测）两轮非对称条件缓解暴露偏差，5条链通过贪心最大化互信息构建，最终取等权平均。

---

## 4. 实验与分析

全部实验采用5折分层交叉验证（seed 42）。数据集含252味草药、653种化合物和1540个靶点[1,5]。跨域迁移使用1349种UniTox西药。

全模型取得宏观AUC 0.7821，较HerbToxNet（0.6003）提升18.2pp。消融实验中DrugTower跨域迁移绝对值主导（-9.85pp），DropAdd（-1.87pp）和Jaccard正则化（-1.40pp）为统计显著的次要贡献。ECC链排序非性能瓶颈。肝/肾毒性二分类中，基于多标签模型抽取单列预测大幅超越HerbToxNet的独立二分类器，证明联合多标签训练比隔离式二分类更强[12]。

---

## 5. 结论

本文构建了面向多标签中药毒性预测的跨域分子知识迁移框架，取得宏观AUC 0.7821。DrugTower跨域迁移是主导性贡献，多头注意力聚合器提供内在可解释性。局限性包括：数据集仅252味草药[5]，化合物表示限于Morgan指纹[7]，DrugTower在UniTox子集上预训练，评估局限于单一数据集[1]。未来方向包括分子图神经网络、扩展预训练语料库及多模态融合，可利用现有ADMET预测平台[13,14]进行验证。


## 参考文献

[1] Y. Zhu, Y. Miao, R. Sun, Z. Yan, G. Yu, "Traditional Chinese medicine toxicity prediction by heterogeneous network," *Expert Systems with Applications*, vol. 299, p. 129969, 2026. doi:10.1016/j.eswa.2025.129969

[2] X. Gu, Y. Zou, Z. Huang, L. Ji, "Biochemical biomarkers for the toxicity induced by Traditional Chinese Medicine: A review update," *Journal of Ethnopharmacology*, 2025. doi:10.1016/j.jep.2025.119633

[3] S. He, X. Zhang, G. Sun, X. Sun et al., "A computational toxicology approach to screen the hepatotoxic ingredients in traditional Chinese medicines: *Polygonum multiflorum* Thunb as a case study," *Biomolecules*, vol. 9, no. 10, p. 5033, 2019. doi:10.3390/biom91005033

[4] L. Song, W. Qian et al., "TCMSTD 1.0: a systematic analysis of the traditional Chinese medicine system toxicology database," *Science China Life Sciences*, 2023. doi:10.1007/s11427-022-2318-4

[5] Y. Zhu, L. Fang et al., "TCMToxDB: a comprehensive database for the toxicological analysis of traditional Chinese medicines," *Database*, vol. 2026, baag019, 2026.

[6] S. Monem, A. H. Abdel-Hamid, A. E. Hassanien, "Drug toxicity prediction model based on enhanced graph neural network," *Computers in Biology and Medicine*, vol. 185, p. 109614, 2025. doi:10.1016/j.compbiomed.2024.109614

[7] S. Teng, C. Yin, Z. Yan, L. Wei et al., "MolFPG: Multi-level fingerprint-based Graph Transformer for accurate and robust drug toxicity prediction," *Computers in Biology and Medicine*, vol. 164, p. 107269, 2023. doi:10.1016/j.compbiomed.2023.107269

[8] J.-W. Chu, J.-H. Park, Y.-R. Cho, "MEMOL: Mixture of experts for multimodal learning through multi-head attention to predict drug toxicity," *Computer Methods and Programs in Biomedicine*, vol. 273, p. 109088, 2026. doi:10.1016/j.cmpb.2025.109088

[9] J. Lu, L. Wu, R. Li, M. Wan, J. Yang, P. Zan, H. Bai, S. He, X. Bo, "ToxACoL: an endpoint-aware and task-focused compound representation learning paradigm for acute toxicity assessment," *Nature Communications*, 2025. doi:10.1038/s41467-025-60989-7

[10] J. Park et al., "Enhancing multi-task in vivo toxicity prediction via integrated knowledge transfer of chemical knowledge graph," *Journal of Cheminformatics*, vol. 17, p. 171, 2025.

[11] J. Gao, L. Wu, G. Lin, J. Zou, B. Yang, K. Liu, S. He, X. Bo, "Multi-task multi-view and iterative error-correcting random forest for acute toxicity prediction," *Expert Systems with Applications*, vol. 274, p. 126972, 2025. doi:10.1016/j.eswa.2025.126972

[12] J. Lee, J. M. Posma, "Improving drug-induced liver injury prediction using graph neural networks with augmented graph features from molecular optimisation," *Journal of Cheminformatics*, vol. 17, p. 124, 2025.

[13] L. Fu et al., "ADMETlab 3.0: An updated comprehensive online ADMET prediction platform enhanced with broader coverage, improved performance, API functionality and decision support," *Nucleic Acids Research*, vol. 52, pp. W422–W431, 2024. doi:10.1093/nar/gkae236

[14] P. Banerjee, E. Kemmler, M. Dunkel, R. Preissner, "ProTox 3.0: a webserver for the prediction of toxicity of chemicals," *Nucleic Acids Research*, vol. 52, pp. W513–W520, 2024. doi:10.1093/nar/gkae303

[15] J. Read, B. Pfahringer, G. Holmes, E. Frank, "Classifier chains for multi-label classification," *Machine Learning*, vol. 85, pp. 333–359, 2011. doi:10.1007/s10994-011-5256-5

[16] J. Snell, K. Swersky, R. Zemel, "Prototypical networks for few-shot learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 4077–4087, 2017.

[17] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, P. S. Yu, "Heterogeneous graph attention network," in *Proceedings of The Web Conference (WWW)*, pp. 2022–2032, 2019. doi:10.1145/3308558.3313562

[18] T. Ridnik, E. Ben-Baruch, N. Zamir, A. Noy, I. Friedman, M. Protter, L. Zelnik-Manor, "Asymmetric loss for multi-label classification," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 82–91, 2021. doi:10.1109/ICCV48922.2021.00015

[19] M.-L. Zhang, Z.-H. Zhou, "A review on multi-label learning algorithms," *IEEE Transactions on Knowledge and Data Engineering*, vol. 26, no. 8, pp. 1819–1837, 2014. doi:10.1109/TKDE.2013.39

---
