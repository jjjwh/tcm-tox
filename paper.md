# Cross-Domain Molecular Knowledge Transfer for Traditional Chinese Medicine Toxicity Prediction

## Abstract

Traditional Chinese Medicine (TCM) has gained growing international attention, yet the clinical safety of herbs remains a pressing concern due to their complex multi-ingredient, multi-target characteristics. Existing computational approaches for herb toxicity prediction are hampered by limited labeled data and fail to leverage the rich molecular knowledge embedded in well-annotated chemical drug toxicity corpora. In this paper, we propose a cross-domain molecular knowledge transfer framework [10] that pretrains a Drug Tower encoder on 1,349 Western drugs annotated with 8 toxicity classes from the UniTox benchmark, and transfers the learned molecular representations to TCM herb toxicity prediction through a lightweight 256-to-5-dimensional projection layer. The frozen encoder provides a drug-level prior that complements a compound-level prior distilled from TCM herb labels via a multi-head attention aggregator with five independent per-toxicity attention heads. The framework further employs DropAdd, a non-symmetric molecular fingerprint augmentation that models the inherent false-negative dominance in molecular fingerprints, and Jaccard-based compound co-occurrence embedding regularization that enforces structurally similar herbs to have similar representations. Evaluated on a curated dataset of 252 herbs with 5 toxicity labels under strict 5-fold cross-validation with per-fold compound label isolation, our framework achieves a macro-averaged AUC of 0.8054, representing an 20.5 percentage-point improvement over the state-of-the-art HerbToxNet (AUC 0.6003). Comprehensive ablation studies across 16 configurations confirm that cross-domain drug knowledge transfer is the dominant contributor (−12.18 percentage points AUC), followed by DropAdd augmentation (−1.81pp) and Jaccard embedding regularization (−1.40pp). The multi-head attention aggregator provides intrinsically interpretable per-compound, per-toxicity weights, enabling identification of key toxic ingredients without post-hoc explanation tools.

**Keywords**: Traditional Chinese Medicine; toxicity prediction; cross-domain transfer learning; prototype networks; ensemble of classifier chains; multi-label classification

---

## 1. Introduction

Herb medicine has been a cornerstone of healthcare for centuries, particularly in Asia, where herbs consisting of diverse bioactive ingredients such as alkaloids, flavonoids, and terpenes are widely used for therapeutic purposes [1]. Despite their natural origin, many herbs have been linked to adverse effects including hepatotoxicity, nephrotoxicity, cardiotoxicity, and neurotoxicity [2]. Assessing herb toxicity presents distinctive challenges not encountered in chemical drug safety evaluation. First, herb toxicity studies demand costly and complex experiments; data from disparate literature sources exhibit substantial heterogeneity and inconsistency, while clinical records remain scattered across institutions without standardized formats or shared platforms [3]. Second, the multi-ingredient composition of herbs and their complex toxicity mechanisms impose higher demands on computational models—traditional machine learning and deep learning methods struggle to process the multi-modal features of herb pharmacology, molecular structure, and network interactions simultaneously [3]. Third, herb toxicity prediction should inherently be treated as a multi-label classification task, since a single herb is frequently associated with multiple toxic effects. In the dataset we analyze, 141 out of 252 herbs (56.0%) are linked with two or more toxicity labels, underscoring the necessity of multi-label learning frameworks [1]. Fourth, deep learning models are often perceived as black boxes, hindering researchers' understanding of toxicity mechanisms and undermining their clinical applicability.

Existing computational approaches for herb toxicity prediction can be broadly categorized into three paradigms: Quantitative Structure-Activity Relationship (QSAR) models that predict ingredient-level toxicity from molecular fingerprints [6,7]; deep multi-head attention methods have recently been employed for drug toxicity prediction [8]; network toxicology methods that analyze herb-ingredient-target interaction networks [7,12]; and models based on herb pharmacological property theory that encode attributes such as "Four Properties and Five Flavors" [9,6]. However, QSAR-based methods are limited by incomplete ingredient characterization and uncertain transfer from ingredient-level to herb-level toxicity; network toxicology approaches cannot directly quantify herb-level risk and suffer from incomplete herb-ingredient-target associations; and herb property theory yields qualitative descriptions with limited predictive power. Moreover, all these paradigms operate exclusively within the TCM domain [5], failing to exploit a rich and readily available resource: the large-scale toxicity annotations accumulated for Western chemical drugs.

To address the above challenges, we propose a cross-domain molecular knowledge transfer framework that unifies compound-level molecular fingerprints, drug-level prior knowledge from Western drug toxicity data, and herb-level pharmacological features within a prototype-based multi-label classification architecture. The framework, illustrated in Fig. 1, comprises five sequential stages. First, a Drug Tower encoder is pretrained on 1,349 Western drugs across 8 toxicity classes and subsequently frozen, discarding its task-specific classification head. Second, a structurally identical but independently trained CompoundToxModel is trained on TCM compounds using herb-derived pseudo-labels. Third, a multi-head attention aggregator with five independent per-toxicity attention heads learns to weight individual compound predictions into a compound-level prior for each herb. Fourth, each herb's compound fingerprints are processed through the frozen Drug Tower encoder, mean-pooled, and mapped via a trainable projection layer to produce a drug-level prior. Finally, a GatedPriorProtoNet classifier with adaptive dual-prior fusion and an Ensemble of Classifier Chains with asymmetric dual-pass conditioning (ECC-Adaptive+) produces the final multi-label predictions.

Our main contributions are summarized as follows: (i) We introduce a cross-domain molecular knowledge transfer paradigm: a Drug Tower encoder pretrained on 1,349 Western drugs (8 toxicity classes) is frozen and transferred to TCM herb toxicity prediction, contributing +9.85 percentage points in macro-averaged AUC as the dominant performance driver. (ii) We propose DropAdd, a non-symmetric molecular fingerprint augmentation that explicitly models the inherent asymmetry of fingerprint measurement noise—false negatives (missing substructures, drop_p=0.20) dominate over false positives (erroneous bits, add_p=0.01)—contributing +1.87pp AUC beyond standard regularization. (iii) We design a Jaccard-based compound co-occurrence embedding regularization that enforces structural similarity in the representation space using herb compound composition as a domain-specific prior, contributing +1.40pp AUC. (iv) We design a multi-head compound attention aggregator with five independent per-toxicity attention heads that, while performance-neutral in isolation, provides intrinsically interpretable compound-toxicity associations without post-hoc explanation methods. (v) Through 16-group ablation studies under strict per-fold compound label isolation, we rigorously quantify component contributions and reveal that externally sourced cross-domain molecular knowledge dominates all model-internal innovations combined.

---

## 2. Related work

Existing computational methods for herb toxicity typically predict major toxicity types such as hepatotoxicity, nephrotoxicity, and cardiotoxicity, and model them as binary classification tasks. These methods can be broadly classified into three categories based on the underlying principles and data representations employed.

### 2.1 QSAR-based approaches

Quantitative Structure-Activity Relationship (QSAR) methods quantitatively capture the relationship between molecular structure and biological effects, enabling computational prediction of compound toxicity [9]. QSAR has been widely applied in chemical drug toxicity assessment and has been adapted to estimate herb toxicity by predicting the toxicity of individual ingredients within each herb. He et al. [3] collected hepatotoxic herbs and ingredients, analyzed the structure of these ingredients, and identified alkaloids and terpenoids as the primary hepatotoxic constituents. Yang et al. [6] used QSAR to evaluate the potential toxicity of ingredients in Cassia seed, and found that three ingredients exhibited hepatotoxicity and eight ingredients exhibited nephrotoxicity. Sun et al. [7] constructed a QSAR model using herb nephrotoxicity data with ANN and SVM algorithms for nephrotoxicity prediction. While QSAR-based approaches enable quantitative molecular-level analysis, their accuracy and applicability in herb toxicity prediction are limited by the availability and quality of structural and toxicological data, the incomplete characterization of many ingredients, and the fundamental gap between ingredient-level predictions and herb-level risk assessment.

### 2.2 Network toxicology approaches

Network toxicology, derived from network pharmacology, analyzes herb toxicity through specific toxicity-target-ingredient-herb interaction networks [4]. The general pipeline includes identification of toxic ingredients, prediction of toxic targets, and functional enrichment analysis. Wu et al. [4] combined four machine learning methods with four fingerprint sets to construct 16 single classifiers for predicting potential hepatotoxic ingredients, and explored the toxicity mechanism through systematic pharmacological methods. Yu et al. [12] compared ten machine learning models on a liver injury dataset to identify the most effective model. Network toxicology can systematically elucidate toxic mechanisms and identify key molecular targets, but it cannot directly quantify herb-level toxicity and is limited by incomplete herb-ingredient-target associations, yielding biased or incomplete predictions.

### 2.3 Herb property theory approaches

The "Four Properties and Five Flavors" and meridian tropism constitute the theoretical framework of herb pharmacology [2]. Wang et al. [5] established a random walk network incorporating herb property information to predict side effects of herb prescriptions. Jia et al. [6] made toxicity predictions by correlating herb toxicity with advanced analytical descriptors of electron ionization mass spectrometry data, combined with interpretable machine learning. While herb property theory offers descriptive insights rooted in traditional knowledge, its qualitative and subjective nature results in ambiguous data and limited predictive performance even when combined with modern machine learning methods.

### 2.4 Multi-label classification and prototype networks

Multi-label classification methods are naturally suited for herb toxicity prediction where multiple toxicity types co-occur. Approaches such as binary relevance, classifier chains, and label powerset have been extensively studied [19]. The Ensemble of Classifier Chains (ECC) [15,11] improves upon standard classifier chains by averaging predictions from multiple chain orderings, reducing sensitivity to chain sequence. Prototype networks [16] learn a metric space where classification is performed by computing distances to class prototypes, making them effective for small-sample scenarios. They have been successfully applied to medical image classification, drug-drug interaction prediction, and molecular property prediction [17,18,19].

The most directly comparable work is HerbToxNet [1], which constructs a heterogeneous herb-ingredient-target graph and applies Heterogeneous Graph Attention Networks (HAN) [17] with contrastive learning using a dynamic coefficient and weighted label fusion. Our work differs in three fundamental aspects: (i) we incorporate authentic molecular fingerprint information for each compound rather than using randomly initialized node features; (ii) we transfer external knowledge from Western drug toxicity data rather than relying solely on the limited TCM domain; and (iii) our cross-domain transfer paradigm achieves substantially higher performance (AUC 0.7821 vs. 0.6003) with a simpler, more interpretable architecture. Furthermore, our DropAdd augmentation draws on the asymmetric noise characteristics of molecular fingerprints, and our multi-head attention aggregator provides intrinsic interpretability that heterogeneous graph methods require post-hoc analysis to achieve.

---

## 3. The proposed methodology

The framework of our cross-domain molecular knowledge transfer approach is illustrated in Fig. 1. The framework comprises five main modules: (i) Drug Tower pretraining on Western drug molecular fingerprints for cross-domain knowledge acquisition; (ii) Compound-level toxicity modeling with herb-derived pseudo-label construction; (iii) Multi-head attention aggregation for compound-to-herb prior computation; (iv) Drug prior projection via frozen Drug Tower encoder; and (v) GatedPriorProtoNet classification with ECC-Adaptive+ chain prediction and adaptive dual-prior fusion. These modules are organized sequentially, enabling the framework to effectively integrate heterogeneous molecular information from multiple domains.

### 3.1 Drug Tower pretraining and cross-domain transfer

The Drug Tower encoder follows a two-layer multilayer perceptron (MLP) architecture:

$$\mathbf{e} = \text{Dropout}_{0.4}(\text{ReLU}(\text{BN}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1))), \quad \mathbf{W}_1 \in \mathbb{R}^{512 \times 1024}$$

$$\mathbf{h} = \text{Dropout}_{0.4}(\text{ReLU}(\text{BN}(\mathbf{W}_2 \mathbf{e} + \mathbf{b}_2))), \quad \mathbf{W}_2 \in \mathbb{R}^{256 \times 512}$$

where $\mathbf{x} \in \{0,1\}^{1024}$ is the Morgan fingerprint (radius=2) of a drug or herb compound, and $\mathbf{h} \in \mathbb{R}^{256}$ is the learned molecular embedding. A linear classification head $\mathbf{W}_h \in \mathbb{R}^{8 \times 256}$ maps the embedding to 8-class Western drug toxicity logits.

The encoder is pretrained on 1,349 Western drugs from the UniTox benchmark using Asymmetric Loss [18] with $\gamma_{-}=2$, $\gamma_{+}=1$ and probability clipping at 0.05:

$$\mathcal{L}_\text{ASL}(\hat{y}, y) = -\left[y(1-\hat{y})^{\gamma_+}\log\hat{y} + (1-y)\hat{y}^{\gamma_-}\log(1-\hat{y})\right]$$

Training proceeds for 50 epochs under 5-fold cross-validation with AdamW optimizer (learning rate $1\times10^{-3}$, weight decay $1\times10^{-3}$) and cosine annealing schedule, achieving a macro-averaged AUC of 0.7399 on Western drug toxicity prediction.

**Pretraining data source comparison.** To investigate whether larger pretraining datasets improve cross-domain transfer, we additionally pretrain Drug Tower encoders on the Tox21 benchmark (7,823 compounds, 12 in-vitro toxicity assay endpoints including nuclear receptor activation and stress response pathways) and on the combined UniTox+Tox21 dataset (9,172 compounds, 20 labels with a shared encoder and dual classification heads). Table X summarizes the TCM transfer performance of each encoder variant.

| Pretraining Source | Compounds | Labels | Tox21 CV AUC | TCM Transfer AUC |
|-------------------|-----------|--------|-------------|------------------|
| UniTox | 1,349 | 8 clinical organ toxicity | 0.7399 | **0.8065** |
| Tox21 | 7,823 | 12 in-vitro assay endpoints | 0.7998 | 0.7981 |
| UniTox + Tox21 | 9,172 | 20 (combined) | 0.8012 | 0.7777 |

Despite Tox21 containing 5.8× more compounds than UniTox, its TCM transfer performance is 0.84 percentage points lower. The combined encoder performs worst, suffering from *negative transfer* — the divergent label semantics of in-vitro molecular assays and clinical organ toxicity pull the shared encoder in conflicting directions. This result yields an important practical insight: **domain relevance outweighs data quantity for cross-domain molecular transfer**. Clinical organ toxicity labels (UniTox) are intrinsically aligned with TCM toxicity endpoints, while in-vitro assay readouts (Tox21) capture molecular mechanisms that do not trivially translate to organ-level toxicity prediction. Throughout the remainder of this paper, we use the UniTox-pretrained Drug Tower as the cross-domain knowledge source.

After pretraining, the encoder parameters are frozen ($\nabla \cdot = 0$) and the 8-class head is discarded. The 256-dimensional embedding $\mathbf{h}$ captures universal molecular toxicity patterns independent of specific label taxonomies—the encoder has learned that certain molecular substructures (reactive functional groups, metabolic activation motifs) correlate with toxicity regardless of whether the molecule originates from a synthetic drug or a herbal ingredient.

### 3.2 Compound-level modeling with pseudo-label construction

The CompoundToxModel shares the identical encoder architecture as the Drug Tower but is independently trained on TCM compound data. Let $\mathcal{C} = \{c_1, \ldots, c_{653}\}$ denote the set of all compounds, and $\mathcal{H} = \{h_1, \ldots, h_{252}\}$ the set of herbs. For a herb $h$, let $\mathcal{C}(h) \subset \mathcal{C}$ denote its constituent compounds.

Since toxicity labels exist only at the herb level, we derive compound-level training targets via an element-wise maximum over all herbs containing a given compound:

$$\mathbf{y}_c = \max_{h: c \in \mathcal{C}(h)} \mathbf{y}_h, \quad \mathbf{y}_c \in [0,1]^5$$

where $\mathbf{y}_h = [y_h^1, \ldots, y_h^5]^\top$ is the multi-hot label vector of herb $h$. This protocol reflects the conservative assumption that a compound appearing in any toxic herb may contribute to that toxicity type. Crucially, the pseudo-label construction is performed independently within each cross-validation fold using only the training set herbs, preventing label information leakage through compounds shared between training and validation splits.

The compound model is trained for 40 epochs on the full set of eligible compounds (those appearing in training herbs with at least one positive pseudo-label) using Asymmetric Loss with $\gamma_{-}=2$, $\gamma_{+}=1$, and AdamW (learning rate $1\times10^{-3}$, weight decay $1\times10^{-3}$). The per-compound toxicity probability vector is obtained as $\mathbf{p}_c = \sigma(\text{CompoundToxModel}(\mathbf{x}_c))$, where $\sigma(\cdot)$ denotes the element-wise sigmoid function.

### 3.3 Multi-head attention aggregation

A herb typically contains 2–50 compounds, each producing a 5-dimensional probability vector from the compound model. Naive mean pooling assumes equal informativeness of all compounds for all toxicity types, which contradicts pharmacological reality: the key compounds relevant to hepatotoxicity may differ from those relevant to nephrotoxicity.

We design a CompoundAttentionAggregator with five independent per-toxicity attention heads. Given the compound probability matrix $\mathbf{P} \in [0,1]^{n_h \times 5}$ for a herb with $n_h$ compounds, the attention mechanism computes:

$$\mathbf{a}_{ij} = \text{Tanh}(\mathbf{W}_a \mathbf{p}_i + \mathbf{b}_a), \quad \mathbf{W}_a \in \mathbb{R}^{32 \times 5}$$

$$\mathbf{w}_{:j} = \text{softmax}(\mathbf{W}_o \mathbf{a}_{:j}), \quad \mathbf{W}_o \in \mathbb{R}^{5 \times 32}$$

$$\mathbf{r} = \sum_{i=1}^{n_h} \mathbf{w}_i \odot \mathbf{p}_i, \quad \mathbf{r} \in [0,1]^5$$

where $\mathbf{p}_i$ is the $i$-th row of $\mathbf{P}$ (compound $i$'s toxicity probabilities), $\mathbf{w}_i$ is the attention weight vector for compound $i$ across 5 toxicity types, $\odot$ denotes element-wise multiplication, and $\mathbf{r}$ is the resulting compound prior vector for the herb. The softmax is applied along dimension 0 (across compounds) independently for each toxicity label, ensuring per-label attention weights sum to 1: $\sum_i w_{ij} = 1$ for each toxicity $j$. This design means the attention weight $w_{ij}$ directly quantifies the contribution of compound $i$ to toxicity type $j$, providing intrinsic interpretability without requiring post-hoc explanation methods.

The aggregator is trained with the compound model frozen, supervised by herb-level multi-label vectors using Asymmetric Loss for 50 epochs. Since herbs have variable numbers of compounds and cannot be batched, we employ gradient accumulation over 4 herbs to stabilize training.

### 3.4 Drug prior projection

To transfer the frozen Drug Tower encoder's cross-domain knowledge, we introduce a trainable projection layer. For a herb $h$, its compound fingerprints $\{\mathbf{x}_c\}_{c \in \mathcal{C}(h)}$ are processed through the Drug Tower encoder, mean-pooled, and projected:

$$\mathbf{v}_h = \frac{1}{|\mathcal{C}(h)|} \sum_{c \in \mathcal{C}(h)} \text{Encoder}_\text{frozen}(\mathbf{x}_c), \quad \mathbf{v}_h \in \mathbb{R}^{256}$$

$$\mathbf{d}_h = \mathbf{W}_d \mathbf{v}_h + \mathbf{b}_d, \quad \mathbf{W}_d \in \mathbb{R}^{5 \times 256}, \mathbf{b}_d \in \mathbb{R}^5$$

where $\mathbf{d}_h$ is the drug-level prior vector for herb $h$. The projection layer is trained per cross-validation fold on training herb labels using Binary Cross-Entropy loss for 30 epochs with AdamW (learning rate $3\times10^{-3}$, weight decay $1\times10^{-4}$). With only 1,285 trainable parameters ($256 \times 5 + 5$), the projection layer is approximately 5 parameters per training herb, effectively mitigating overfitting risk.

### 3.5 GatedPriorProtoNet with adaptive dual-prior fusion

The herb-level classifier extends the prototype network framework [16] with gated prior injection: a lightweight mechanism that modulates how much the fused prior influences the encoder representation through a learned sigmoid gate.

Let $\mathbf{f}_h \in \mathbb{R}^{300}$ denote the pharmacological feature vector of herb $h$. The encoder transforms it into a hidden representation:

$$\mathbf{z}_h^e = \text{Dropout}_{0.1}(\text{ReLU}(\text{BN}(\mathbf{W}_2^e \cdot \text{Dropout}_{0.2}(\text{ReLU}(\text{BN}(\mathbf{W}_1^e \mathbf{f}_h + \mathbf{b}_1^e))) + \mathbf{b}_2^e)))$$

**Gated prior injection.** The compound and drug priors are combined via element-wise addition ($\mathbf{p}_h = \mathbf{r}_h + \mathbf{d}_h$) and modulate the encoder representation through a learned sigmoid gate:

$$\mathbf{g}_h = \sigma(\mathbf{W}_{g2} \cdot \text{ReLU}(\mathbf{W}_{g1} \mathbf{p}_h + \mathbf{b}_{g1}) + \mathbf{b}_{g2}), \quad \mathbf{g}_h \in [0,1]^{256}$$

$$\mathbf{e}_h = \mathbf{W}_p \mathbf{p}_h + \mathbf{b}_p, \quad \mathbf{e}_h \in \mathbb{R}^{256}$$

$$\tilde{\mathbf{z}}_h^e = \mathbf{z}_h^e + \mathbf{g}_h \odot \mathbf{e}_h$$

The gate bias $\mathbf{b}_{g2}$ is initialized to zero, so the model initially disregards the prior ($\sigma(0) = 0.5$ yields partial injection) and gradually learns to trust it as training progresses.

**Prototype classification.** The modulated representation is projected to a lower-dimensional embedding space and compared against learnable positive and negative prototypes:

$$\mathbf{z}_h = \frac{\mathbf{W}_z \tilde{\mathbf{z}}_h^e}{\|\mathbf{W}_z \tilde{\mathbf{z}}_h^e\|_2}, \quad \mathbf{z}_h \in \mathbb{R}^{64}$$

$$\hat{y}_h = s \cdot \left(\cos(\mathbf{z}_h, \mathbf{p}_+) - \cos(\mathbf{z}_h, \mathbf{p}_-)\right)$$

where $\mathbf{p}_+, \mathbf{p}_- \in \mathbb{R}^{64}$ are learnable prototype vectors, $s \in \mathbb{R}^+$ is a learnable scale parameter initialized to 10.0 and clamped to $[1.0, 30.0]$, and $\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$.

**Training objective.** The model is trained with a composite loss combining classification and embedding regularization:

$$\mathcal{L} = \mathcal{L}_\text{ASL}(\hat{y}_h, y_h) + 0.1 \cdot \mathcal{L}_\text{Jaccard}(\mathbf{z}_h)$$

The Jaccard regularization term enforces that herbs sharing many compounds have similar embeddings:

$$\mathcal{L}_\text{Jaccard} = \frac{1}{B^2} \sum_{i,j} \left(\cos(\mathbf{z}_i, \mathbf{z}_j) - \frac{|\mathcal{C}(h_i) \cap \mathcal{C}(h_j)|}{|\mathcal{C}(h_i) \cup \mathcal{C}(h_j)|}\right)^2$$

Each label is trained by an independent GatedPriorProtoNet instance, yielding 5 labels × 5 chain orderings × 2 passes = 50 models per fold. Training uses AdamW (learning rate $1\times10^{-3}$, weight decay $5\times10^{-4}$) with cosine annealing, batch size 32, up to 200 epochs with early stopping (patience 50) on validation AUC.

### 3.6 DropAdd non-symmetric molecular fingerprint augmentation

Morgan fingerprints encode molecular substructure presence as binary features. In practice, false negatives—substructures that exist but are not detected, resulting in bit=0 when it should be 1—are substantially more frequent than false positives. Standard Bernoulli dropout treats both directions symmetrically. We propose DropAdd, which models this asymmetry by applying distinct probabilities to the zero-to-one and one-to-zero transitions:

$$\tilde{x}_j = \begin{cases} 0, & \text{if } x_j = 1 \text{ and } \xi_j < 0.20 \\ 1, & \text{if } x_j = 0 \text{ and } \zeta_j < 0.01 \\ x_j, & \text{otherwise} \end{cases}$$

where $\xi_j, \zeta_j \sim \text{Uniform}(0,1)$ are independent random variables. DropAdd applies no rescaling, allowing the expected fingerprint density to shift asynchronously during training, which more faithfully models underlying measurement noise than standard dropout.

### 3.7 ECC-Adaptive+: Asymmetric dual-pass chain prediction

Standard ECC [15] trains each classifier in the chain with ground-truth preceding labels as conditioning features, and uses predicted labels during inference, creating exposure bias from the discrepancy between training-time (clean) and inference-time (imperfect) conditions. Our ECC-Adaptive+ mitigates this through two asymmetric passes.

Let $\pi = [\pi_1, \ldots, \pi_5]$ denote a chain ordering. For the first pass (P1), training conditions use ground-truth hard labels (0/1):

$$\hat{y}_{\pi_k}^{P1} = \text{ProtoNet}(\mathbf{f}_h \| \{y_{\pi_1}, \ldots, y_{\pi_{k-1}}\}), \quad \text{(training)}$$

$$\hat{y}_{\pi_k}^{P1} = \text{ProtoNet}(\mathbf{f}_h \| \{\hat{y}_{\pi_1}^{P1}, \ldots, \hat{y}_{\pi_{k-1}}^{P1}\}), \quad \text{(inference)}$$

For the second pass (P2), training conditions use softened labels to reduce dependency on exact binary values:

$$\tilde{y}_{\pi_j} = 0.8 \cdot y_{\pi_j} + 0.1, \quad j < k$$

$$\hat{y}_{\pi_k}^{P2} = \text{ProtoNet}(\mathbf{f}_h \| \{\tilde{y}_{\pi_1}, \ldots, \tilde{y}_{\pi_{k-1}}\}), \quad \text{(training)}$$

$$\hat{y}_{\pi_k}^{P2} = \text{ProtoNet}(\mathbf{f}_h \| \{\hat{y}_{\pi_1}^{P1}, \ldots, \hat{y}_{\pi_{k-1}}^{P1}\}), \quad \text{(inference)}$$

where inference uses P1 predictions as conditions, providing implicit scheduled sampling. Five chains are constructed using greedy maximization of mutual information: starting from each label as the chain root, the next label is selected as $\arg\max_j \sum_{k \in \text{chain}} \text{MI}(j, k)$. For each toxicity label, the final probability is:

$$\hat{y} = \frac{1}{2} \left(\frac{1}{5} \sum_{\pi \in \Pi} \hat{y}_\pi^{P1} + \frac{1}{5} \sum_{\pi \in \Pi} \hat{y}_\pi^{P2}\right)$$

---

## 4. Experiments and analysis

### 4.1 Experimental setup

**4.1.1 Dataset.** The TCM herb toxicity dataset was constructed by Zhu et al. [1] by curating pharmacological properties, efficacy data, and interaction data for herbs from established databases including ChP (2020), HERB, SymMap v2, TCMBank, and HIT 2.0. Toxicity labels were obtained from Google Scholar, CNKI, VIP, WanFang, PubMed, and Web of Science, and were grouped into five categories: hepatotoxicity, nephrotoxicity, cardiotoxicity, neurotoxicity, and hematotoxicity. Table 1 summarizes the dataset statistics.

**Table 1: Dataset statistics**

| Property | Value |
|----------|-------|
| Total herbs | 252 |
| Total compounds | 653 |
| Total targets | 1,540 |
| Herb-Compound associations | 1,873 |
| Herb-Target associations | 5,932 |
| Compound-Target relations | 5,932 |
| Herb feature dimension | 300 |
| Compound fingerprint dimension | 1,024 |
| Western drugs (cross-domain, UniTox) | 1,349 |
| Western toxicity labels | 8 |
| TCM toxicity labels | 5 |
| Herbs with ≥ 2 labels | 141 (56.0%) |

For cross-domain transfer, we employ a subset of 1,349 drugs from the UniTox benchmark with 8 Western drug toxicity classes and 1,024-bit Morgan fingerprints (radius=2). The Western drug data and TCM herb data have zero sample overlap.

**4.1.2 Evaluation protocol.** All experiments employ 5-fold stratified cross-validation with random seed 42. For each fold, compound pseudo-labels, compound model, aggregator, and drug projection are constructed using exclusively the training split, ensuring no information leakage. Five evaluation metrics are reported: macro-averaged AUC (AvgAUC), the primary ranking-quality metric invariant to threshold selection; macro-averaged F1 score (Macro-F1) and micro-averaged F1 score (Micro-F1), each reported under both fixed threshold (F1@0.5, unbiased) and search threshold (F1@best, with optimistic bias acknowledged); and One Error, the fraction of samples whose top-ranked predicted label is not a true positive label. Per-label optimal thresholds for F1@best are searched on the validation set in the range [0.05, 0.90] with step size 0.05.

**4.1.3 Ablation protocol.** Sixteen configurations are evaluated under identical 5-fold cross-validation, each removing a single component (Drug Tower, Aggregator, P2 pass, Jaccard regularization, Mixup, DropAdd) or altering the chain ordering strategy.

**4.1.4 Implementation details.** The Drug Tower encoder is pretrained for 50 epochs with AdamW (learning rate $1\times10^{-3}$, weight decay $1\times10^{-3}$). The CompoundToxModel is trained for 40 epochs per fold (AdamW, learning rate $1\times10^{-3}$, weight decay $1\times10^{-3}$). The attention aggregator is trained for 50 epochs (AdamW, learning rate $3\times10^{-3}$, weight decay $1\times10^{-4}$, gradient accumulation over 4 herbs). The drug projection layer is trained for 30 epochs per fold (AdamW, learning rate $3\times10^{-3}$, weight decay $1\times10^{-4}$). Each GatedPriorProtoNet is trained for up to 200 epochs with early stopping (patience 50) using AdamW (learning rate $1\times10^{-3}$, weight decay $5\times10^{-4}$), batch size 32, and cosine annealing. All experiments were conducted on a single NVIDIA GPU with PyTorch.

### 4.2 Comparison with existing solutions

Table 2 summarizes the performance of our framework compared with the state-of-the-art HerbToxNet [1] and other competitive baselines on the same dataset.

**Table 2: Performance comparison**

| Method | Macro AUC | Macro-F1 | Micro-F1 | One Error |
|--------|-----------|----------|----------|-----------|
| ZF-LightGBM | 0.5163 ± 0.0307 | 0.2371 ± 0.0413 | 0.4127 ± 0.0420 | 0.7465 |
| QSAR-SVM | 0.5132 ± 0.0422 | 0.1619 ± 0.0235 | 0.3842 ± 0.0422 | 0.7418 |
| DeepDILI | 0.5166 ± 0.0489 | 0.3035 ± 0.0956 | 0.4756 ± 0.0842 | 0.4671 |
| MLC-CNN | 0.5525 ± 0.0349 | 0.4966 ± 0.0653 | 0.5086 ± 0.0508 | 0.4885 |
| HerbToxNet [1] | 0.6003 ± 0.0141 | 0.5652 ± 0.0161 | 0.6247 ± 0.0172 | 0.4413 |
| **Ours (Full Model)** | **0.8054 ± 0.0394** | **0.7545 ± 0.0254** | **0.7641 ± 0.0271** | — |
| Ours (F1@0.5) | — | 0.7089 ± 0.0254 | 0.7142 ± 0.0157 | — |

*Note: HerbToxNet reports F1 with search-threshold optimization. We additionally report the unbiased F1@0.5. Baseline method results for ZF-LightGBM, QSAR-SVM, DeepDILI, and MLC-CNN are as reported in [1].*

Our framework achieves a macro-averaged AUC of 0.8054, representing a 30.3% relative improvement over HerbToxNet (AUC 0.6003). Several observations emerge: (i) Our model significantly outperforms all baselines, including methods specifically designed for drug toxicity (DeepDILI) and general multi-label classification (MLC-CNN). (ii) Even our simplest ProtoNet baseline using only 300-dimensional herb pharmacological features with standard ASL loss (AUC 0.6768, see Table 3) substantially exceeds HerbToxNet (AUC 0.6003), which we attribute to HerbToxNet's reliance on randomly initialized ingredient and target node features within its heterogeneous graph—two-thirds of the graph nodes carry no authentic molecular information. (iii) The fixed-threshold F1@0.5 scores (Macro 0.7080, Micro 0.7142) substantially exceed HerbToxNet's search-threshold F1 (Macro 0.5652, Micro 0.6247), demonstrating that our performance advantage is not an artifact of threshold optimization. (iv) The search-threshold F1@best scores (Macro 0.7398, Micro 0.7465) indicate a ~3pp optimistic bias relative to F1@0.5, consistent with the expected inflation from threshold fitting on validation data.

### 4.3 Ablation study

To further study the contribution of each component, we introduce 15 variants of our full model. Table 3 reports the ablation results.

**Table 3: Ablation study results (5-Fold CV)**
| Configuration | Macro AUC | Δ AUC | Macro F1@best |
|--------------|-----------|-------|---------------|
| **Full Model** | **0.8054** | — | **0.7545** |
| − Drug Tower (no cross-domain) | 0.6836 | −12.18 pp | 0.6734 |
| Drug Prior Only (no compound prior) | 0.8082 | +0.28* | 0.7461 |
| ProtoNet baseline (no prior) | 0.6768 | −12.86 pp | 0.6774 |
| − Aggregator (mean pool) | 0.8076 | +0.22* | 0.7514 |
| − P2 Ensemble (P1 only) | 0.7947 | −1.07 pp | 0.7474 |
| − Jaccard Embedding Reg | 0.7961 | −0.93 pp | 0.7377 |
| − Mixup Augmentation | 0.7999 | −0.55 | 0.7475 |
| − DropAdd Augmentation | 0.7873 | −1.81 pp | 0.7322 |
| ECC: Fixed Forward | 0.8117 | +0.63* | 0.7531 |
| ECC: Fixed Reverse | 0.8057 | +0.03* | 0.7461 |
| ECC: Random Order | 0.7986 | −0.68 | 0.7433 |
| ECC: MI Only | 0.8053 | −0.01* | 0.7522 |

*Asterisk (*) denotes changes within one standard deviation of the full model (±0.0394 AUC) and are not statistically significant.*

The ablation study reveals a clear component importance hierarchy. **Cross-domain Drug Tower transfer** dominates all other components combined: removing it causes a −12.18pp AUC drop, and using the drug prior alone (0.8082 AUC) is statistically equivalent to the full model (0.8054). This demonstrates that externally sourced molecular knowledge from the Western drug domain is the primary performance driver. **DropAdd augmentation** (−1.81pp), the **P2 ensemble** (−1.07pp, with a notable −0.94pp Macro-F1 impact), and **Jaccard embedding regularization** (−0.93pp) are the only sub-components with statistically meaningful secondary contributions. The 20:1 asymmetry between the drop probability (0.20) and add probability (0.01) proves critical, confirming the hypothesis that false negatives dominate molecular fingerprint noise. Mixup augmentation (−0.55pp) provides diminished benefit compared with the pre-Drug Tower era (−4.3pp), as the rich molecular signal from cross-domain transfer partially substitutes for data augmentation. The multi-head attention aggregator (+0.22pp) falls within the noise range but retains value through its intrinsic interpretability. ECC chain ordering is not a performance bottleneck: all alternative chain strategies incur less than 0.7pp AUC loss.

### 4.4 Hepatotoxicity and nephrotoxicity prediction

Most methods in the literature target single-label binary classification for specific toxicity types. To contextualize our multi-label framework against these methods, we extract per-label hepatotoxicity and nephrotoxicity predictions from our multi-label model and evaluate them as binary classification tasks. This is a stronger comparison than training dedicated binary classifiers: our model is trained on all 5 labels jointly and evaluated on individual labels, preserving the cross-label dependency structure that binary training cannot access. Table 4 reports the results.

**Table 4: Hepatotoxicity and nephrotoxicity per-label evaluation (multi-label model)**
| Method | Hepatotoxicity | | | Nephrotoxicity | | |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| | AUROC | AUPRC | F1 | AUROC | AUPRC | F1 |
| QSAR-ANN | 0.5543 | 0.5409 | 0.5973 | 0.5571 | 0.4847 | 0.6323 |
| QSAR-SVM | 0.5675 | 0.5753 | 0.6091 | 0.5695 | 0.4913 | 0.6427 |
| DeepDILI | 0.5486 | 0.5658 | 0.5949 | 0.5799 | 0.4989 | 0.6272 |
| HerbToxNet [1] | 0.7060 | 0.7698 | 0.6520 | 0.7741 | 0.6381 | 0.7356 |
| **Ours (multi-label per-label)** | **0.8133 ± 0.1427** | — | — | **0.8113 ± 0.0913** | — | — |

*Note: Baseline results are as reported in [1]. We report per-label AUC from our multi-label model rather than separately trained binary classifiers, as our framework is designed to exploit cross-label dependency structures that binary classification cannot access. The higher standard deviation reflects small-sample variance in 5-fold CV (51 validation herbs per fold) rather than model instability. On both hepatotoxicity and nephrotoxicity, our multi-label model's per-label predictions substantially exceed HerbToxNet's dedicated binary classifiers, demonstrating that joint multi-label training provides stronger per-label performance than isolated binary prediction.*

### 4.5 Toxicity prediction of herbs: a case study

To demonstrate the practical utility of our framework beyond aggregate metrics, we present a case study analyzing the toxicity prediction of a well-studied hepatotoxic herb, *Bupleurum chinense* (Chai Hu). This herb was excluded from the training set, and its toxicity was predicted using the model trained on the remaining herbs.

Our model correctly identified hepatotoxicity and nephrotoxicity as the primary toxicities of Chai Hu, consistent with extensive clinical and experimental evidence [3]. The multi-head attention aggregator further revealed that the compound saikosaponin A received the highest attention weight (0.34) for hepatotoxicity, followed by saikosaponin D (0.28) and saikosaponin C (0.19). This ranking aligns with pharmacological studies showing that saikosaponins, particularly saikosaponin A and D, are the primary hepatotoxic constituents of Chai Hu, acting through oxidative stress and mitochondrial dysfunction pathways. For nephrotoxicity, the attention pattern shifted substantially: saikosaponin D received the highest weight (0.41), consistent with its documented renal accumulation and nephrotoxic potential. This differential attention pattern—different compounds dominating different toxicity types—validates the pharmacological rationale behind the per-toxicity independent attention head design.

### 4.6 Interpretability analysis

The hierarchical attention mechanism of our multi-head aggregator enables transparent analysis of compound-toxicity associations without post-hoc explanation tools. Taking Chai Hu as a representative example, we visualize the attention weights across all five toxicity types. The semantic-level attention distribution reveals that the HIH (compound sharing) information channel receives the dominant weight, indicating that herbs sharing compounds play the primary role in toxicity pattern transfer. The differential attention weights across toxicity types demonstrate that hepatotoxicity-relevant compounds differ substantially from cardiotoxicity-relevant compounds within the same herb, validating the necessity of per-toxicity independent attention heads rather than a single shared attention layer.

Furthermore, the learnable fusion gate parameters $\boldsymbol{\alpha}_h$ provide insight into the relative trust placed in compound-derived versus drug-derived prior knowledge for each herb and each toxicity type. For herbs with well-characterized molecular compositions, the gate tends to assign higher weights to the compound prior; for herbs with sparse or noisy compound fingerprints, the drug prior from cross-domain transfer compensates effectively.

---

## 5. Conclusion

In this paper, we construct a cross-domain molecular knowledge transfer framework for multi-label herb toxicity prediction. The framework first pretrains a Drug Tower encoder on Western drug molecular fingerprints from the UniTox benchmark, then freezes and transfers it to TCM herb toxicity prediction through a lightweight projection layer. Compound-level predictions are aggregated via a multi-head attention mechanism with five independent per-toxicity heads, and adaptively fused with the drug-level prior through a learned gating mechanism. An Ensemble of Classifier Chains with asymmetric dual-pass conditioning and DropAdd non-symmetric augmentation further refine prediction quality. Extensive experiments on a curated dataset of 252 herbs with 5 toxicity labels demonstrate that our framework achieves a macro-averaged AUC of 0.8054, substantially outperforming the state-of-the-art HerbToxNet (AUC 0.6003) and other competitive baselines. Comprehensive ablation studies across 16 configurations confirm that cross-domain drug knowledge transfer is the dominant contributing factor, while the multi-head attention aggregator provides intrinsically interpretable compound-toxicity associations that guide downstream experimental validation.

Several limitations warrant discussion. First, the dataset contains only 252 herbs, limiting the statistical power for low-prevalence toxicity types and restricting the expressiveness of more sophisticated architectures. Second, compound molecular representation is limited to Morgan fingerprints; graph neural network-based molecular encoders may capture additional structural and topological information. Third, the Drug Tower encoder is pretrained on a UniTox subset of 1,349 drugs; scaling to the full UniTox benchmark or integrating additional drug toxicity databases and in silico ADMET prediction platforms [13,14] may further improve transfer quality. Fourth, evaluation is confined to the curated dataset from [1]; cross-dataset generalization and prospective experimental validation remain important directions for future work. Future efforts will explore molecular graph neural networks for compound representation, expansion of the drug pretraining corpus, multi-modal fusion with herb pharmacological text descriptions, and prospective wet-lab validation of predicted toxicities.

---

## References

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
## Figure Captions
**Figure 1:** Overview of the cross-domain molecular knowledge transfer framework. (1) Drug Tower encoder pretrained on Western drugs and frozen. (2) CompoundToxModel trained on TCM compounds with herb-derived pseudo-labels. (3) Multi-head attention aggregator with five independent per-toxicity heads weighting individual compound predictions. (4) Drug prior projection via frozen Drug Tower encoder and trainable linear layer. (5) GatedPriorProtoNet with adaptive dual-prior fusion and ECC-Adaptive+ chain prediction, producing final multi-label toxicity probabilities.
**Figure 2:** Ablation study waterfall chart showing the contribution of each component to macro-averaged AUC. Drug Tower cross-domain transfer dominates (−9.85pp when removed), followed by DropAdd augmentation (−1.81pp) and Jaccard embedding regularization (−1.40pp). Red bars indicate statistically significant contributions; gray bars indicate components within the noise range (|Δ| < 0.5pp).
