# 工科医学论文写作模板

> 基于 Zhu et al. (2026) "Traditional Chinese medicine toxicity prediction by heterogeneous network" (Expert Systems With Applications) 的逐章剖析

---

## 一、论文整体结构总览

| 章节 | 建议字数 | 占比 | 核心功能 |
|------|---------|------|---------|
| Title & Abstract | 200-300 | — | 独立可传播的"最小论文" |
| 1. Introduction | 1000-1500 | 15% | 建立研究必要性与贡献预告 |
| 2. Related Work | 800-1200 | 12% | 定位本文在文献中的位置 |
| 3. Methodology | 2000-3000 | 30% | 可复现的技术描述 |
| 4. Experiments | 2000-3000 | 30% | 验证主张的完整证据链 |
| 5. Conclusion | 400-600 | 8% | 总结+局限+未来方向 |
| References | — | 5% | 30-60篇 |

---

## 二、逐章详细剖析与模板

### 2.1 Abstract（摘要）— 200-300字

#### HerbToxNet 写法剖析

Zhu2026 的 Abstract 精确遵循了**五步漏斗结构**：

| 步骤 | 内容 | 原文字句 |
|------|------|---------|
| **Step 1: 背景** (1句) | 建立领域重要性 | "The clinical usage of TCMs gains increasing international attention for their distinct therapeutic effects." |
| **Step 2: 问题** (2句) | 指出当前困境 | "Traditional wet-lab based pipelines... are complex and time-consuming. ...computational models... often fails to effectively predict the toxicity of herbs." |
| **Step 3: 方法** (3-4句) | 描述你做了什么 | "We propose HerbToxNet... first constructs a heterogeneous network... leverages HAN... performs contrastive learning... uses MLP... introduces weighted label fusion." |
| **Step 4: 结果** (1-2句) | 关键数据 | "HerbToxNet outperforms competitive methods... with 96% toxicity labels confirmed for canonical herbs." |
| **Step 5: 意义** (1句) | 为什么重要 | "It can mine related toxic ingredients and targets in an interpretable way, and dissect the molecular mechanism of herb toxicity with authenticity." |

#### 模板要求

```
【字数】200-300英文词，或400-500中文字
【结构】必须包含以下5个要素，缺一不可：
  1. 背景句（1句）：该研究领域的公认重要性
  2. 问题句（1-2句）：现有方法的不足/差距
  3. 方法句（3-4句）：你的解决方案的核心技术路线
     — 必须出现模型/方法名称（如 HerbToxNet）
     — 必须出现2-4个核心技术关键词（如 heterogeneous network, contrastive learning）
  4. 结果句（1-2句）：最重要的1-3个量化指标
  5. 意义句（1句，可选）：一阶贡献总结
【关键词】4-6个，格式："关键词1；关键词2；..."
【禁止】
  — 引用文献（Abstract中不出现[1][2]等）
  — 缩写不加定义（除非极其通用如DNA、CNN）
  — "本文首次提出..."等主观夸大表述
```

---

### 2.2 Introduction（引言）— 1000-1500字

#### HerbToxNet 写法剖析

Zhu2026 的 Introduction 采用了**倒金字塔+四段式递进**结构：

**段落1（宏观背景 → 领域困境）：**
> "Herb medicine has a long history... Herbs consist of a variety of bioactive ingredients... Although they are considered as safe, many herbs have been linked to adverse effects..."
- 功能：建立"为什么要关注中药毒性"的合法性
- 技巧：用具体数字（"141 out of 252 herbs linked with ≥2 toxic labels"）建立危机感

**段落2（具体化挑战 → 四点困难）：**
> "First... Second... Third... Fourth..."
- 四个挑战以 `First, Second, Third, Fourth` 明示：
  1. 多成分复杂性问题
  2. 数据不详尽/不一致问题
  3. 多标签分类的必要性（56%草药涉及≥2毒性）
  4. 深度学习的黑箱可解释性困境
- 技巧：每个挑战由一个主题句+2-3句支撑构成

**段落3（现有方法的综述 → 局限性）：**
> "Regulatory authorities... Traditional methods... such as in vivo animal testing... With the advent of high-throughput technologies... computational approaches have gained attraction... However, existing computational approaches still encounter significant challenges..."
- 功能：走一遍"已有方案为何不够好"的逻辑
- 技巧：先肯定（"have gained attraction"）→ 转折（"However..."）→ 指出三个不足（单一信息源、无法融合多层级特征、缺乏系统框架）

**段落4（我们的方案 → 贡献列表）：**
> "To address the above challenges, we construct a heterogeneous graph... By leveraging HAN, contrastive learning, and weighted label fusion, our HerbToxNet can..."
- 功能：你的方法如何逐一解决上述问题
- 贡献通常以列表或编号形式呈现

#### 模板要求

```
【字数】1000-1500英文词（约4-5个自然段）
【结构】按以下5段顺序组织：

  第1段 | 宏观背景与必要性（~200字）
  ─────────────────────────────
  □ 第1句：领域重要性（1句大背景）
  □ 第2-3句：为何这个话题值得研究（用数据/引用支撑）
  □ 第4句：暗指本文要解决的问题
  技巧：从"所有人都同意的宏观事实"开始，逐步聚焦

  第2段 | 具体困难拆解（~250字）
  ─────────────────────────────
  □ 用"First... Second... Third..."明示3-5个具体挑战
  □ 每个挑战 = 主题句 + 解释句（1-2句）
  □ 其中至少一个挑战用数据量化（如"56%的草药..."）
  技巧：挑战的排列顺序 = 从最外部（数据）到最内部（模型特性）

  第3段 | 现有方法评述（~300字）
  ─────────────────────────────
  □ 第1-2句：传统方法简述 + 缺陷（1-2句即可，重点不是它们）
  □ 第3-5句：现有计算方法分类概述（每类1句话+1-2个引用）
    - 格式："[方法类别] have been applied... [citation1, citation2]."
  □ 第6-8句：集中指出通用缺陷（2-3个）
    - 格式："However, [方法类别] suffer from [共同问题]."
  □ 最后一句：结论性过渡 → "Given these, there is a critical need of..."
  技巧：
    - 引用要密集但不堆砌，每句不超过3个引用号
    - 引用覆盖近5-10年（该文引用2020-2025为主）
    - 与related work区分：这里只说"大类+通病"，细节留给第2章

  第4段 | 本文方案与贡献（~250字）
  ─────────────────────────────
  □ 第1-2句：直接声明本文做了什么
    - 格式："To address these challenges, we propose [方法名], which..."
  □ 用(i)(ii)(iii)(iv)列出3-5个贡献
    - 每个贡献 = 一句话 + （可选）一句补充
    - 贡献1：主要方法论创新
    - 贡献2：次要技术创新
    - 贡献3：实验/数据贡献
    - 贡献4（可选）：可解释性/应用贡献
  技巧：贡献要与第2段的挑战一一对应

【内容要求】
  — 至少引用8-15篇文献
  — 至少包含2个量化数据（如样本数、阳性率、提升幅度）
  — 必须出现你的方法名称
  — 最后一段必须以贡献列表收尾
  — 不使用"本文首次"、"填补空白"等主观措辞；改为"we propose"、"we introduce"

【禁止】
  — 大段描述方法细节（留给第3章）
  — 在引言给出实验结果（留给第4章）
  — 引用自己的工作时不标注（即使是自己的也要标注引用）
```

---

### 2.3 Related Work（相关工作）— 500-800字

#### HerbToxNet 写法剖析

Zhu2026 的 Related Work **没有子节编号**（无 2.1/2.2/2.3），而是以**连续自然段**组织。全文仅约 500-600 词，极其凝练。

**原文段落结构（4段）：**

```
第1段（3句话）：三类方法总览
  "Existing computational methods... can be divided into three categories: 
   QSAR, network toxicology, herb property theory."

第2段（~180词）：QSAR
  定义 → 如何适配草药 → He/Yang/Sun 三个代表性工作 → 局限句

第3段（~150词）：Network toxicology
  定义 → 通用流程 → Wu/Yu 两个代表性工作 → 局限句

第4段（~160词）：Herb property theory
  定义 → 四气五味详释 → Wang/Jia 两个代表性工作 → 局限句

（无独立的多标签/原型网络段落——该内容在 Introduction 中已覆盖）
```

**与常见错误写法的关键差异：**

| 要素 | 常见错误 | HerbToxNet 实际写法 |
|------|---------|-------------------|
| 子节编号 | 2.1/2.2/2.3/2.4 | **无编号**，纯段落流 |
| 段落内格式 | 粗体小标题、编号列表 | **纯正文**，无任何装饰 |
| 每类方法篇幅 | 展开写 200-300 字 | **100-180 词一个自然段** |
| 局限句 | 单独成句、加粗强调 | **自然编织在段落最后 1-2 句**，不单独标出 |
| 过渡 | "However..." 起头 | 更柔和："While QSAR-based approaches enable..., their accuracy... are limited by..." |
| 总字数 | 800-1200 | **500-700** |

#### 模板要求

```
【字数】500-800英文词（中文约800-1200字）
【格式】连续自然段，不设子节编号（若期刊要求子节编号也尽量精简，每子节不超过150词）

【段落组织】：
  第1段（~50词）：一句总述——本文涉及几类方法、分类依据
  第2-N段（每段100-150词）：每类方法一个自然段
    段内顺序：定义（1句）→ 代表性工作（2-3个，每个1句）→ 局限（1-2句）
  末段（可选，~50词）：若需讨论最接近工作或本文独特定位，单独成段

【内容要求】
  □ 每类方法 2-4 个代表性引用，每引用 1 句话（谁+做了什么+关键结果）
  □ 局限句自然嵌入段落末尾，不单独加粗或编号
  □ 分类标准清晰互斥
  □ 不贬低他人工作——用客观词汇："limited by" / "restricted to" / "often fails to"

【禁止】
  — 使用粗体小标题、编号列表、项目符号
  — 每个方法超过 200 词
  — 在此章详细描述自己的方法
  — 把相关工作写成流水账
  — 使用"no existing method can..."等绝对化表述
```

**HerbToxNet 局限句写法（可直接套用的句式）：**
```
QSAR:
  "While QSAR-based approaches enable quantitative analysis..., their accuracy 
   and applicability... are limited by the availability and quality of both 
   structural and toxicological data, as well as the incomplete characterization 
   of many ingredients."

Network toxicology:
  "While network toxicology can systematically elucidate toxic mechanisms..., 
   this kind of methods cannot directly quantify herb-level toxicity and are 
   limited by incomplete herb-ingredient-target associations, suffering biased 
   or incomplete predictions."

Herb property theory:
  "While herb property theory offers descriptive insights rooted in traditional 
   knowledge, its qualitative and subjective nature... results in ambiguous data 
   and limited predictive performance even when combined with advanced machine 
   learning methods."
```

**关键模板句式**：`"While [方法名] can/enables/offers [优点], [致命局限]."`

---

### 2.4 Methodology（研究方法论）— 2000-3000字

#### HerbToxNet 写法剖析

这是论文最核心的章节。HerbToxNet 的方法论展现出**8个可复现的写作模式**，以下逐句拆解。

**模式1：每个子节以"动机段"开头（最重要！）**

以 3.3 Contrastive learning 为例，开篇不是公式，而是：
> "The above herb representations may not effectively capture toxicity-related similarities and differences among herbs, especially under setting where each herb can possess multiple toxic labels. Therefore, we propose a contrastive learning with dynamic coefficient strategy to update the herb representations."

模式：`"The above X may not effectively capture Y. Therefore, we propose Z to [verb]..."`

再以 3.4 MLP+WLF 的动机段为例：
> "However, the above MLP may not fully utilize the valuable knowledge encoded in the training data. To further enhance the toxicity prediction, we introduce a weighted label fusion (WLF) strategy."

模式：`"However, [current approach] may not fully [achieve goal]. To [improve], we introduce [new component]."`

**每个子节的第一段必须回答"为什么需要这个模块"，不能跳过直接写公式。**

**模式2：每个设计选择必须跟一句"原因"**

HerbToxNet 的每个设计决策都有因果解释：

> "Since our objective is herb toxicity prediction, and the heterogeneous graph will be represented using HAN based on predefined meta-paths. Therefore, there is no need to assign specific initial features to the ingredient and target nodes."（3.1）

> "Homogeneous relations between ingredients or targets require explicit similarity computations or manual curation, which introduce redundancy and tedious efforts, so these relations are not used in our work."（3.1）

模式：`"Since [condition/reason], [design choice]. Therefore, [consequence]."` 或 `"[Alternative] would require [cost], so we instead [our choice]."`

**模式3：先给具体例子，再推抽象公式**

在 3.2 HAN 一节，HerbToxNet 在公式之前先给出一个具象例子：
> "For example, the meta-path of Herb-Ingredient-Herb (HIH) suggests that both herbs contain the same ingredient, implying that these two herbs may possess similar functionalities and target associations."

模式：`"For example, [concrete case] suggests that [intuitive meaning], implying that [insight]."`

**模式4：承认更简单方案的存在，再解释为何不采用**

在 3.2 语义级注意力部分：
> "A simple method for aggregating embeddings from different meta-paths is to take the average. However, due to the varying importance of meta-paths to node i in the heterogeneous graph, we adopt an attention mechanism to assign weights to each meta-path pattern and then perform aggregation."

模式：`"A simple method for [goal] is to [naive approach]. However, due to [limitation], we adopt [our approach]."`

**模式5：公式后紧跟"人话"解释其直觉含义**

在 3.3 的动态系数对比损失之后，HerbToxNet 用一整段纯文字解释公式的因果链：
> "For a given pair of herbs (h_i, h_j), a higher semantic similarity β_ij results in a larger dynamic coefficient δ_ij, which in turn leads to a larger loss L_con_ij. Consequently, the feature similarity cos(h_i, h_j) is optimized toward a higher value. Conversely, if two samples do not share any labels (β_ij = δ_ij = 0), the value L_con_ij becomes zero, and their cosine similarity cos(h_i, h_j) is optimized to a lower value."

模式：逐因果链解释：`"a higher X → a larger Y → which in turn → a larger Z. Consequently, A is optimized toward B. Conversely, if X=0, Z becomes zero, and A is optimized to a lower value."`

**模式6：每个公式必须有"where"定义从句**

无一例外：
> "where W_h and b_h denote the trainable weight matrix and bias vector, respectively. || represents the concatenation operation, and ReLU(·) is the ReLU activation function."

**模式7：超参数选择给出原因**

> "During subsequent experiments, we found that β may not sufficiently suitable to act as the dynamic coefficient for contrastive learning, as it does not provide a robust measure of relevance between two herbs. In particular, β can assign an excessively large coefficient to herb samples with low toxic label similarity. This issue can degrade the performance of herb toxicity prediction. As a result, we introduce a tunable hyperparameter t to adjust β to obtain a more appropriate dynamic coefficient δ."

模式：`"We found that [naive design] may not be suitable because [observed problem]. As a result, we introduce [hyperparameter/refinement] to [fix the problem]."`

**模式8：导言段不出现公式，仅用一句话概述每个模块**

> "The framework of HerbToxNet is illustrated in Fig. 1. HerbToxNet comprises four main modules: (i) Heterogeneous graph construction builds a herb-ingredient-target heterogeneous graph based on herb pharmacological properties, efficacy, associations; (ii) Herb representation learning employs a Heterogeneous Graph Attention Network (HAN) (Wang et al., 2019) with selected meta-paths to extract informative herb representations; (iii) Contrastive learning with dynamic coefficient optimizes the learned representations by pulling together herbs with shared toxic labels and pushing apart those with dissimilar labels; and (iv) Toxicity prediction based on weighted label fusion merges the predictions from an MLP and toxic labels from similar herbs of the target herb based on their optimized representations. These modules are organized sequentially, enabling HerbToxNet to effectively integrate heterogeneous information, learn robust representations, and achieve toxicity prediction."

分析：
- 第一句：Fig.1引用
- (i)-(iv)：每个模块=一个主动动词句（builds/employs/optimizes/merges）
- 末句：三个动词总结整体功能（integrate/learn/achieve）
- **无公式、无符号、无维度数字**

**Algorithm 伪代码规范：**
- 位置：方法论末尾、实验之前
- 格式：Input → Output → 分阶段步骤（⊳标注）
- 每行引用对应公式编号
- 约30行

#### 模板要求

```
【字数】2000-3000英文词（不含公式），中文约3000-4500字
【子节数量】4-6个模块，每模块一个子节

【章节导言段（强制，~100词）】：
  第1句："The framework of [方法名] is illustrated in Fig. 1."
  第2句："[方法名] comprises N main modules:"
  第3-N句：(i)...(ii)...(iii)... 每模块一个主动动词句
  末句："These modules are organized sequentially, enabling [方法名] to [动词1], [动词2], and [动词3]."
  ⚠ 导言段不出现公式、符号、维度数字

【每个子节的强制内部结构】：
  ┌──────────────────────────────────────────────┐
  │ 第1段：动机（~50-100词）— 必须              │
  │   "The above X may not effectively capture Y. │
  │    Therefore, we propose Z."                  │
  │   或                                         │
  │   "However, [current] may not fully [goal].   │
  │    To [improve], we introduce [new]."         │
  │                                              │
  │ 第2段：概念引入/具体例子（~50-80词）         │
  │   "For example, [concrete case] suggests      │
  │    that [intuitive meaning]."                 │
  │   先用人话解释概念，再推公式                 │
  │                                              │
  │ 第3段：方案描述+公式（~100-200词）           │
  │   正文："We define/employ/adopt X as..."       │
  │   公式 → "where"从句逐符号定义               │
  │   每个设计选择跟因果解释：                    │
  │     "Since [reason], we [choice]."            │
  │     "[Alternative] would require [cost],       │
  │      so we instead [our approach]."           │
  │                                              │
  │ 第4段（可选）：公式直觉解释                  │
  │   "For a given [input], a higher [X] results  │
  │    in a larger [Y], which in turn leads to..." │
  │   用因果链走一遍公式的直觉含义              │
  │                                              │
  │ 第5段（可选）：设计发现/超参数原因            │
  │   "We found that [naive] may not be suitable   │
  │    because [observed problem]. As a result,    │
  │    we introduce [refinement]."                 │
  └──────────────────────────────────────────────┘

【公式规范】（强制）：
  □ 公式前有自然语言铺垫，公式后有"where"从句
  □ 每个符号首次出现必须定义
  □ 公式编号连续
  □ 重要公式后必须跟一段纯文字的直觉解释（模式5）

【禁止】：
  — 子节开头直接写公式（必须先写动机段）
  — 公式堆砌无过渡文字
  — 设计选择无原因说明
  — 导言段出现公式或维度数字
  — 缺少Algorithm伪代码
  — 超参数具体值（learning rate, batch size）→ 实验章4.1
  — 数据集细节 → 实验章4.1
```

---

### 2.5 Experiments（实验）— 2000-3000字

#### HerbToxNet 写法剖析

这是论文的第二大章节，Zhu2026 分7个子节：

```
4. Experiments
  ├── 4.1 Experimental setup
  │     ├── 4.1.1 Dataset（数据来源+表1: 数据统计）
  │     ├── 4.1.2 Baselines（对比方法+分类简述）
  │     └── 4.1.3 Evaluation metrics（指标+公式）
  ├── 4.2 Results comparison with existing solutions（表2: 主结果）
  ├── 4.3 Hepatotoxicity and nephrotoxicity prediction（表3: 子任务）
  ├── 4.4 （原论文此处有内容但在补充材料中）
  ├── 4.5 Ablation study（消融实验）
  ├── 4.6 Toxicity prediction of herbs（案例研究）
  └── 4.7 Interpretability analysis
        ├── 4.7.1 Node-level attention analysis
        └── 4.7.2 Biointerpretability analysis
```

**每个子节的写法分析：**

**4.1.1 Dataset（数据描述段）：**
> "We collected pharmacological properties, efficacy, and interaction data for herbs from established databases, including ChP (2020), HERB, SymMap v2, TCMBank, and HIT 2.0. Toxicity labels were obtained from major literature and database sources such as Google Scholar, CNKI, VIP, WanFang, PubMed, and Web of Science, and were grouped into five categories: cardiotoxicity, nephrotoxicity, hepatotoxicity, systemic toxicity, and other toxicity. Table 1 gives an overview of the data sources."
- 4句话完成：数据来源 → 标签来源 → 分类 → 引用表格
- 每个数据库给出缩写 + 引用
- 用 Table 1 做统计总览

**4.2 主结果对比的写法结构：**
```
[段1] 陈述结果 + 引用Table 2
[段2] 观察点(i)：为什么超越传统方法
[段3] 观察点(ii)：与最接近方法的对比细节
[段4+] 更多观察点
```
- 每个观察点 = 观察结论（1句）+ 原因分析（2-3句）
- 使用 "(i)", "(ii)", "(iii)" 标记观察点
- 不重复表格中已有的数字，而是解释为什么

**4.3 子任务分析的写法：**
> "HerbToxNet is designed for multi-label toxicity prediction, while most methods target at single-label toxicity prediction... To assess whether HerbToxNet can be adapted to these binary tasks, we isolated hepatotoxicity and nephrotoxicity herbs from the dataset..."
- 先说为什么做这个实验（动机）
- 再说怎么做（设计）
- 然后给结果（引用表格）
- 最后解释发现

**4.5 消融实验的写法：**
> "To further study the contribution factors of HerbToxNet, we introduce five variants: (i) w/o HAN... (ii) w/o PE... (iii) w/o CL... (iv) w/o δ... (v) w/o WLF..."
- 每个消融变体命名规范：`w/o [组件缩写]`（without的简写）
- 每个变体一句话说明去掉了什么、替代成了什么
- 结果 + 分析："outperforms its variants by a clear margin, underscoring the vital role of each removed factor"

**4.6 案例研究的写法：**
> "we utilized HerbToxNet to predict and analyze the toxicity of canonical herbs, including Chai Hu, He Shou Wu, Ba Dou, Fu Zi, and Bi Ma Zi, in real-world scenarios. To rigorously evaluate the model's generalization capability, all other herbs in the dataset were used as the training set, while the five selected herbs were treated as a separate test set..."
- 结构：选了什么案例 → 怎么做的（held-out测试）→ 预测了什么 → 与文献/TCM理论对比验证 → 结论

**4.7 可解释性分析的写法：**
- 4.7.1 节点级注意力分析：从模型内部机制出发，解释为什么模型做出某个预测（以柴胡为例）
- 4.7.2 生物可解释性分析：构建"草药-成分-靶点-通路-毒性"多层解释框架

#### 模板要求

```
【字数】2000-3000英文词
【子节要求】以下6个子节必须全部包含：

┌─────────────────────────────────────────────────────┐
│ 4.1 Experimental Setup（实验设置）                   │
│ ─────────────────────────────────────────────────── │
│ □ 4.1.1 Dataset（数据集，~200字）                   │
│   - 数据来源（数据库名+引用）                       │
│   - 样本量/特征维度/标签分布                        │
│   - 引用Table 1（数据统计表）                       │
│   - 预处理步骤（在正文或指向补充材料）              │
│                                                     │
│ □ 4.1.2 Baselines（对比方法，~150字）               │
│   - 按类别分组列出对比方法（每组2-4个）             │
│   - 每组给1句概述该类方法的特点                     │
│   - 说明参数设置原则（"used recommended parameters  │
│     as starting point and fine-tuned as necessary"） │
│                                                     │
│ □ 4.1.3 Evaluation Metrics（评估指标，~100字）      │
│   - 列出所有指标的公式或定义                        │
│   - 说明为何选择这些指标                            │
│   - 明确主指标（primary metric）                    │
│                                                     │
│ □ 4.1.4 Implementation Details（实现细节，~150字）  │
│   - 框架/硬件/优化器/学习率/batch size/epoch        │
│   - 交叉验证设置（几折、随机种子）                  │
│   - 所有超参数的具体值及其选择原因                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 4.2 Main Results（主结果对比）                      │
│ ─────────────────────────────────────────────────── │
│ □ 引用结果表（Table 2）                             │
│ □ 陈述总体结论（1句）                               │
│ □ 用(i)(ii)(iii)列出3-5个关键观察                  │
│ □ 每个观察 = 结论 + 原因分析                        │
│ □ 与最强baseline的对比单独成段                      │
│ □ 统计显著性检验（t-test at 95% level）             │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 4.3 Sub-task / Specialized Analysis（子任务分析）    │
│ ─────────────────────────────────────────────────── │
│ □ 动机段：为什么做这个额外实验                      │
│ □ 设计段：怎么做（数据集切分、指标选择）            │
│ □ 结果段：引用Table 3 + 分析                        │
│ □ 对比段：与主实验结论的一致性/差异性讨论           │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 4.4 Ablation Study（消融实验）                      │
│ ─────────────────────────────────────────────────── │
│ □ 列出所有变体，命名规范：w/o [组件缩写]            │
│ □ 每个变体 = 去掉什么 + 替代方案（如果有）         │
│ □ 结果表（Table 4）：包含逐指标对比                 │
│ □ 分析：按贡献大小从高到低排列讨论                  │
│ □ 结论句：每个组件都有贡献/某些组件贡献不大         │
│ □ 如有不显著的组件，诚实报告                        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 4.5 Case Study（案例研究）                          │
│ ─────────────────────────────────────────────────── │
│ □ 选择1-5个有代表性的案例（有已知文献支撑的）       │
│ □ 说明测试方式（held-out / 独立测试集）             │
│ □ 展示预测结果，与真值/文献对比                     │
│ □ 引用外部文献验证预测的正确性                      │
│ □ 承认预测与文献不一致的地方（如有）                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 4.6 Interpretability / Error Analysis（可解释性/误差分析）│
│ ─────────────────────────────────────────────────── │
│ □ 从模型内部机制出发解释预测                        │
│ □ 与领域知识（药理/临床）交叉验证                   │
│ □ 给出具体的例子（以X草药为例）                     │
│ □ 分析错误案例（可选但推荐）                        │
└─────────────────────────────────────────────────────┘

【图表规范】（强制）：
  □ 表标题在表上方：Table N: 标题内容
  □ 图标题在图下方：Fig. N: 标题内容
  □ 每个表的Note行说明粗体含义/统计检验方法
  □ 最优结果加粗（bold font）
  □ 表中数字保留4位小数（0.xxxx）
  □ 必须报告标准差（±）
  □ 必须声明统计显著性检验方法和置信水平
```

---

### 2.6 Conclusion（结论）— 400-600字

#### HerbToxNet 写法剖析

从论文最后一页的内容来看，Conclusion 大致包含：

```
[段1] 用1-2句话重述本文做了什么
[段2] 总结主要发现/贡献（2-3句话）
[段3] 局限性讨论 + 未来方向（2-3句话）
```

#### 模板要求

```
【字数】400-600英文词（约3个自然段）

【结构】：
  第1段 | 本文做了什么（~100字）
  ─────────────────────────────
  □ 重述问题 + 方法名称 + 一句话核心思路
  □ 不要复制Abstract，用不同措辞重写

  第2段 | 主要发现（~150字）
  ─────────────────────────────
  □ 2-3个最重要的量化结论
  □ 重点突出你的方法相比baseline的gain

  第3段 | 局限性与未来（~200字）
  ─────────────────────────────
  □ 2-4个具体局限性（数据规模、方法泛化性、未探索方向）
  □ 每个局限性 = 问题陈述 + 可能的解决方向
  □ 最后以未来工作展望收尾
  □ 格式："Future work will explore [方向1], [方向2], and [方向3]."

【禁止】：
  — 引入新数据/新图表
  — 重复Abstract逐字逐句
  — 夸大结论超出数据支持范围
  — 没有局限性讨论（这是常见被拒稿原因）
```

---

## 三、全局写作规范与细节要求

### 3.1 语言与风格

| 要素 | 规范 |
|------|------|
| 时态 | 文献综述→现在时；自己工作→现在时（描述方法）/ 过去时（描述实验过程） |
| 人称 | "we" 少用 "I"，被动语态适度（不是全部被动） |
| 缩写 | 首次出现全称+缩写：Traditional Chinese Medicine (TCM)；之后只用缩写 |
| 术语 | 全文统一，不出现"毒性"有时叫toxicity有时叫toxic effect |
| 句式 | 平均句长18-25词；超过35词的句子拆成两句 |

### 3.2 图表规范

| 规范项 | 要求 |
|--------|------|
| 图片分辨率 | ≥300 DPI |
| 表格格式 | 三线表（顶线、栏目线、底线） |
| 图标题位置 | 图下方 |
| 表标题位置 | 表上方 |
| 引用方式 | Fig. 1 / Table 2（首字母大写+缩写点） |
| 正文引用 | 图/表必须在正文中被明确提及和讨论 |
| 配色 | 灰度优先（考虑黑白打印），彩色需确保灰度下可区分 |

### 3.3 引用规范

| 规范项 | 要求 |
|--------|------|
| 引用格式 | 按目标期刊要求（Elsevier用数字[1][2]，APA用(Author, Year)） |
| 引用密度 | 正文每100字约1-2个引用 |
| 自引 | 不超过总引用的10-15% |
| 时效性 | ≥60%引用来自近5年 |
| 二次引用 | 禁止——必须追溯到原始文献 |
| 引用位置 | 紧跟被引观点，不在句首堆砌 |
| 多引用 | 同一观点最多给3-4个引用号 |

### 3.4 公式与数学规范

| 规范项 | 要求 |
|--------|------|
| 编号 | 连续编号(1), (2), ... |
| 变量字体 | 标量→斜体(x)；向量→粗体(𝐱)；矩阵→大写粗体(𝐖) |
| 函数 | sin/cos/log/exp等→正体 |
| 上下标 | 描述性文字→正体；变量→斜体 |
| 括号 | 多层嵌套时外层用大括号 |
| 正文引用 | "as shown in Eq.(5)" |
| where从句 | 公式后换行，以"where"开头，逐个符号说明 |

### 3.5 补充材料（Supplementary）

```
【必须放入补充材料的内容】：
  □ 完整的超参数搜索空间和调参过程
  □ 更详细的消融实验结果
  □ 数据预处理完整pipeline
  □ 额外的案例研究细节
  □ 参数敏感性分析的全部图表
  □ 基线方法的完整参数设置

【正文vs补充的判断标准】：
  → 理解核心结论必需的 → 正文
  → 帮助复现但不影响理解结论 → 补充材料
```

### 3.6 投稿前自查清单

```
□ Abstract是否包含5要素（背景-问题-方法-结果-意义）
□ Introduction最后一段是否有明确的贡献列表
□ Related Work每个子节末尾是否有局限性陈述
□ Methodology中每个公式的每个符号是否都有定义
□ 所有图表是否在正文中被引用和讨论
□ 最优结果是否加粗
□ 是否报告了标准差
□ 是否有统计显著性检验
□ Conclusion是否有局限性和未来方向
□ 引用是否覆盖近5年文献 ≥60%
□ 补充材料是否完整
□ 所有缩写首次出现时是否全称+缩写
```

---

## 四、Zhu2026 论文的独特优点总结

基于对这篇论文的完整分析，以下写法值得在模板中特别强调：

1. **Introduction的"四重挑战"明确列出**：First/Second/Third/Fourth使读者一眼看清问题全貌，贡献列表与之一一对应
2. **Related Work每个子节末尾的批判句**：不是简单罗列文献，而是每个子类指出一个具体局限，自然过渡到本文的方案
3. **Algorithm伪代码放在方法论末尾**：给读者一个"全貌鸟瞰"，不会在公式细节中迷失
4. **命名规范**：HerbToxNet的变体都称为"w/o X"而不是"Ablation-1/Ablation-2"，增加可读性
5. **主结果表+观察点(i)(ii)(iii)的分析模式**：表格定量、正文定性，各司其职
6. **案例研究用held-out测试**：选代表性草药作为独立测试集，增强结论可信度
7. **双层可解释性**：既有模型内部的注意力分析（4.7.1），又有与生物通路交叉验证的外部分析（4.7.2）
8. **关键细节指向补充材料**：正文不冗长，但给有兴趣的读者指引了完整的细节
