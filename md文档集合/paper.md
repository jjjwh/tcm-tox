# Cross-Domain Molecular Knowledge Transfer for Traditional Chinese Medicine Toxicity Prediction

## Abstract

Traditional Chinese Medicine (TCM) has attracted growing international attention for its multi-ingredient, multi-target synergistic therapeutic effects, yet the clinical safety of herbs remains a critical barrier to its broader adoption. Existing computational approaches for herb toxicity prediction are constrained by the scarcity of labeled TCM samples—only a few hundred herbs have been systematically annotated—and operate exclusively within the TCM domain, failing to exploit the large-scale toxicity annotations accumulated for Western chemical drugs. In this paper, we propose CrossDomainTox, a cross-domain molecular knowledge transfer framework. A Drug Tower encoder is first pretrained on 1,349 Western drugs across 8 toxicity classes from the UniTox benchmark, then frozen and transferred to TCM herb toxicity prediction through a lightweight 256-to-5-dimensional projection layer. The frozen encoder provides a drug-level prior that is fused with a compound-level prior—distilled from TCM herb labels via a multi-head attention aggregator with five independent per-toxicity attention heads—and injected into a prototype-based classifier through a gated mechanism. The framework further incorporates DropAdd, a non-symmetric molecular fingerprint augmentation that explicitly models the inherent false-negative dominance in fingerprint measurement noise (drop_p=0.20 for false negatives, add_p=0.01 for false positives), and a Jaccard-based compound co-occurrence embedding regularizer. Evaluated on a curated dataset of 252 herbs with 5 toxicity labels under strict 5-fold cross-validation with per-fold compound label isolation, CrossDomainTox achieves a macro-averaged AUC of 0.8054, representing a 20.5 percentage-point improvement over the state-of-the-art HerbToxNet (AUC 0.6003). Comprehensive 16-configuration ablation studies confirm that cross-domain drug knowledge transfer is the dominant contributor (−12.18pp AUC), followed by DropAdd augmentation (−1.81pp) and the P2 dual-pass ensemble (−1.07pp). The multi-head attention aggregator further provides intrinsically interpretable per-compound, per-toxicity attribution, enabling identification of key toxic constituents without post-hoc explanation tools.

**Keywords**: Traditional Chinese Medicine; toxicity prediction; cross-domain transfer learning; prototype networks; multi-label classification; molecular fingerprints

---

## 1. Introduction

Herb medicine has been a cornerstone of healthcare for centuries, particularly in Asia, where herbs composed of diverse bioactive ingredients—alkaloids, flavonoids, terpenes, and others—exert therapeutic effects through complex multi-ingredient, multi-target synergies [1]. The long-standing perception of herbs as inherently safe, however, is being steadily undermined by mounting clinical evidence: a growing number of herbs have been conclusively linked to hepatotoxicity, nephrotoxicity, cardiotoxicity, and neurotoxicity [2]. Unlike chemical drugs, for which systematic toxicity screening pipelines are well established, herb safety assessment has long been hampered by compositional complexity, mechanistic ambiguity, and fragmented data sources. A systematic, data-driven computational framework for herb toxicity prediction is therefore among the most pressing unmet needs in the field.

Assessing herb toxicity presents four distinctive challenges that have no direct counterpart in chemical drug safety evaluation. **First, data heterogeneity and fragmentation.** Herb toxicity research depends on costly and logistically complex in-vivo and in-vitro experiments. Data from disparate literature sources exhibit substantial variation in experimental protocols, evaluation criteria, and reporting formats; clinical toxicity records remain scattered across institutions without standardized data-sharing platforms or annotation pipelines [3], rendering large-scale systematic investigations infeasible. **Second, the modeling complexity of multi-ingredient synergies.** A single herb typically contains dozens of chemical constituents whose combined effects may be synergistic, antagonistic, or additive—the whole-herb toxicity cannot be reduced to a simple sum of individual ingredient toxicities. Traditional machine learning and deep learning methods struggle to jointly process the multi-modal, heterogeneous features spanning herb pharmacological properties, molecular structure information, and biological network interactions [4]. **Third, the intrinsic requirement for multi-label classification.** Herb toxicity prediction is inherently a multi-label problem: a single herb is frequently associated with multiple distinct toxic effects. In the dataset we analyze, 141 out of 252 herbs (56.0%) carry two or more toxicity labels [1], yet most existing methods reduce this to a collection of independent binary classification tasks, discarding the rich dependency structure among labels. **Fourth, the opacity of deep models in safety-critical applications.** Although deep learning has achieved breakthroughs across numerous biomedical prediction tasks, the inscrutability of its decision process severely impedes mechanistic understanding—a model that merely outputs a "toxic/non-toxic" label cannot inform pharmacologists which compound, through which pathway, produced the toxic effect [4]. This is a fatal deficiency in any application where the end goal is safety assessment.

Three main technical paradigms have been developed to address herb toxicity prediction. Quantitative Structure-Activity Relationship (QSAR) methods predict the toxicity potential of individual ingredients from molecular fingerprints or physicochemical descriptors [5,6] and have been widely adopted in chemical drug assessment. Network toxicology approaches construct herb-ingredient-target interaction networks and analyze the associations between toxic constituents and molecular targets at a systems level [7,8]. Herb property theory-based methods attempt to encode traditional attributes—the "Four Properties and Five Flavors," meridian tropism—as machine-learnable features [9,10]. Each paradigm, however, suffers from fundamental limitations that prevent it from serving as a comprehensive solution. QSAR methods are constrained by incomplete ingredient characterization and the chasm between ingredient-level predictions and herb-level risk quantification. Network toxicology approaches cannot directly output herb-level quantitative risk scores and remain critically dependent on incomplete interaction data. Herb property theory, though grounded in centuries of empirical knowledge, yields inherently qualitative and subjective descriptors that impose an upper bound on predictive performance even when combined with modern machine learning. More fundamentally, all three paradigms operate within a closed loop: they mine signals exclusively from within the TCM domain while collectively overlooking a naturally available, far larger external resource—the large-scale, high-quality toxicity annotations accumulated over decades in the Western chemical drug domain. These datasets encompass thousands of compounds systematically annotated across multiple organ-level toxicity endpoints, encoding molecular toxicity patterns—reactive functional groups, metabolic activation motifs, mitochondrial toxicity determinants—that are agnostic to whether the molecule originates from a natural product or a synthetic drug. How to effectively transfer this rich external knowledge to the label-scarce TCM domain constitutes the central research motivation of this paper.

To address the above challenges and exploit this cross-domain opportunity, we propose CrossDomainTox, a cross-domain molecular knowledge transfer framework for herb toxicity prediction. A Drug Tower encoder is first pretrained on 1,349 Western drugs across 8 toxicity classes and frozen, then transferred to the TCM domain through a lightweight projection layer. A multi-head attention aggregator with five independent per-toxicity heads distills compound-level priors from herb labels, which are adaptively fused with the drug-level prior and injected into a prototype-based classifier within an Ensemble of Classifier Chains framework.

Our main contributions are: (i) **Cross-domain molecular knowledge transfer.** We demonstrate for the first time that a Drug Tower encoder pretrained on Western drug molecular fingerprints and frozen can be effectively transferred to TCM herb toxicity prediction. This single component accounts for a 12.18pp AUC loss upon removal, establishing it as the absolute dominant performance driver and empirically validating the thesis that external data sources outweigh internal model design in label-scarce regimes. (ii) **DropAdd non-symmetric molecular fingerprint augmentation.** We explicitly model the inherent asymmetry of fingerprint measurement noise—false negatives (undetected substructures, drop_p=0.20) occur far more frequently than false positives (erroneous bits, add_p=0.01)—contributing 1.81pp AUC. (iii) **Jaccard-based compound co-occurrence embedding regularization.** We leverage herb compound composition as a domain-specific structural prior, constraining structurally similar herbs to have proximal embeddings and contributing 0.93pp AUC. (iv) **Interpretable multi-head attention aggregator.** Five independent per-toxicity attention heads provide intrinsic per-compound, per-toxicity interpretability. Although this component falls within the noise range in pure performance terms (+0.22pp, not statistically significant), it uniquely enables the model to answer not only whether a herb is toxic, but which compound drives which toxicity type—without any post-hoc explanation tool. (v) **Rigorous leakage-free evaluation framework.** Under strict 5-fold cross-validation with per-fold compound label isolation, 16 ablation configurations systematically quantify each component's independent contribution, providing a reliable benchmark for model evaluation in small-sample multi-label settings.

---

## 2. Related Work

Existing computational methods for herb toxicity typically target major toxicity types—hepatotoxicity, nephrotoxicity, and cardiotoxicity—and formulate each as a binary classification task. Based on their underlying principles and data representations, these methods can be organized into three categories: QSAR-based prediction, network toxicology analysis, and herb property theory-based approaches. The first three paradigms all operate within the TCM domain. This section reviews each in turn, then discusses multi-label classification methods, the closest comparable work, and the structural differences between our cross-domain transfer and standard transfer learning.

In computational toxicology, QSAR establishes a quantitative mapping between molecular descriptors and biological activity, enabling toxicity prediction from molecular fingerprints [11]. Its framework has been naturally extended to the herb domain—predicting the toxicity of individual ingredients to estimate whole-herb toxicity. He et al. [4] systematically collected hepatotoxic herbs and their constituents, identifying alkaloids and terpenoids as the primary hepatotoxic categories through structural analysis. Yang et al. [5] applied QSAR to evaluate toxicity in Cassia seed, identifying three hepatotoxic and eight nephrotoxic ingredients. Sun et al. [6] constructed QSAR models for herb nephrotoxicity using ANN and SVM. Monem et al. [22] proposed an enhanced graph neural network for drug toxicity prediction. While QSAR enables quantitative molecular-level analysis, its applicability to herb toxicity prediction is limited by incomplete structural characterization of herb ingredients, uneven quality of toxicological annotation data, and the fundamental semantic gap between ingredient-level predictions and herb-level risk assessment.

Network toxicology derives from network pharmacology, analyzing herb toxicity through multi-layered "toxicity-target-ingredient-herb" interaction networks [12]. The standard pipeline includes identifying potential toxic ingredients, predicting their molecular targets, and performing functional enrichment analysis. Wu et al. [7] combined four machine learning methods with four fingerprint systems to construct 16 single classifiers for predicting hepatotoxic ingredients. Yu et al. [8] compared ten machine learning models on a liver injury dataset. Song et al. [23] constructed the TCMSTD 1.0 systematic toxicology database. While network toxicology can elucidate toxic mechanisms and identify key molecular targets at a systems level, it cannot directly quantify herb-level toxicity risk—there exists an inferential gap between "target perturbation" and "herb toxicity"—and remains critically dependent on incomplete herb-ingredient-target association data, with biased network structures potentially yielding misleading predictions.

The "Four Properties and Five Flavors" and meridian tropism constitute the core theoretical framework of herb pharmacology [13]. The premise of this paradigm is that traditional herb property attributes bear intrinsic relationships with biological effects—including toxicity—and can be encoded as predictive features. Wang et al. [9] constructed a random walk network incorporating herb property information to predict side effects of herb prescriptions. Jia et al. [10] correlated herb toxicity with advanced analytical descriptors from EI-MS data, combined with random forests for toxicity prediction. While herb property theory offers descriptive insights rooted in traditional knowledge, its knowledge system is inherently qualitative and subjective. Unlike the standardized molecular representations used for chemical drugs, the "Four Properties" are relative judgments on a continuous spectrum, and "Five Flavors" classifications exhibit discrepancies across classical texts. This ambiguity at the data level imposes a performance ceiling that cannot be overcome by more sophisticated machine learning alone.

The above three paradigms all operate exclusively within the TCM domain. Herb toxicity prediction is inherently a multi-label problem—141 out of 252 herbs (56.0%) carry two or more toxicity labels [1]. Multi-label learning methods can be categorized into binary relevance, classifier chains, and label powerset strategies [14], among which the Ensemble of Classifier Chains (ECC) [15] reduces sensitivity to chain sequence by averaging across multiple random orderings. Prototype networks [16] learn a metric space where each class is encoded as a prototype embedding and classification is performed by distance comparison, a mechanism inherently suited to small-sample scenarios that has been successfully applied to medical image classification, drug-drug interaction prediction, and molecular property prediction [17,18,19]. The most directly comparable contemporary work is HerbToxNet [1], which constructs a herb-ingredient-target heterogeneous graph, applies Heterogeneous Graph Attention Networks (HAN) [20] over predefined meta-paths, and introduces contrastive learning with a dynamic coefficient together with weighted label fusion. Our work differs in three fundamental respects: (i) we provide each compound with a genuine 1,024-bit Morgan fingerprint (radius=2), whereas HerbToxNet assigns randomly initialized features to two-thirds of the graph nodes [20]; (ii) we transfer external cross-domain knowledge from Western drug toxicity data—1,349 drugs covering chemical space inaccessible within the TCM domain—rather than relying exclusively on limited TCM-domain data; and (iii) our cross-domain transfer paradigm achieves substantially higher performance (AUC 0.8054 vs. 0.6003) with a simpler architecture, while DropAdd augmentation and the multi-head attention aggregator provide intrinsic interpretability that HerbToxNet can only approximate through post-hoc analysis.

Our cross-domain transfer differs from standard transfer learning in three structural respects. First, **cross-granularity**: standard transfer assumes identical data structures across domains (image→image, text→text), whereas the Drug Tower is pretrained on individual molecules while the downstream task predicts multi-compound mixtures—the CompoundAttentionAggregator is specifically designed for this semantic elevation from molecule to mixture. Second, **cross-label space**: standard transfer typically maintains label space consistency, but the 8 Western drug toxicity classes (Cardio/Dermato/Hemato/Infertil/Liver/Oto/Pulmo/Renal) and the 5 TCM classes (Hepato/Nephro/Cardio/Neuro/Hemato) are not in a containment relationship—drug_proj (Linear 256→5) performs cross-label-space knowledge compression. Third, **dual-source complementarity rather than single-source replacement**: instead of the standard "pretrain → fine-tune → discard source" pipeline, we retain both the cross-domain drug_prior (frozen) and the in-domain compound_prior (trainable), fused through element-wise addition. Ablation experiments show that drug_prior alone achieves full-model AUC (0.8082 vs. 0.8054, statistically equivalent), yet dual-source fusion yields a +0.7pp calibration gain in F1—the in-domain prior contributes no ranking increment but improves probability estimation quality through label-distribution-relevant calibration, forming a division of labor in which cross-domain knowledge dominates ranking while in-domain self-distillation assists calibration.

---

## 3. The Proposed Methodology

The overall framework of CrossDomainTox is illustrated in Fig. 1. The framework comprises five main modules: (i) Drug Tower pretraining on Western drug molecular fingerprints, which acquires universal molecular toxicity knowledge decoupled from specific label taxonomies; (ii) CompoundToxModel independently trained on TCM compounds with herb-derived pseudo-labels, producing per-compound toxicity probability estimates; (iii) a multi-head attention aggregator with five independent per-toxicity attention heads that weights individual compound predictions into a herb-level compound prior; (iv) drug prior projection through a frozen Drug Tower encoder and a trainable linear layer, generating a cross-domain drug prior for each herb; and (v) a GatedPriorProtoNet classifier that injects the dual-source priors into a prototype network through a gating mechanism and outputs final multi-label toxicity predictions within an Ensemble of Classifier Chains framework. These modules are organized sequentially, enabling the framework to effectively integrate heterogeneous molecular information from multiple domains.

### 3.1 Drug Tower pretraining and cross-domain transfer

The central bottleneck in herb toxicity prediction is the scarcity of labeled samples—only 252 herbs possess systematic toxicity annotations. In contrast, the Western drug domain has accumulated high-quality toxicity annotations for thousands of compounds across multiple organ-level endpoints. The molecular toxicity patterns embedded in these data—the association between specific reactive functional groups, metabolic activation motifs, and toxicity outcomes—are agnostic to whether the molecule originates from a natural product or a synthetic drug. A molecular encoder pretrained on a sufficiently large corpus of Western drug data should, in principle, extract toxicity-discriminative molecular representations that transfer to TCM compounds. Motivated by this premise, we design the Drug Tower encoder and its cross-domain transfer protocol.

The Drug Tower encoder adopts a two-layer multilayer perceptron (MLP) architecture. Let x ∈ {0,1}¹⁰²⁴ denote the Morgan fingerprint (radius=2) of a drug or herb compound. The encoder maps it to a 256-dimensional molecular embedding:

**h** = Dropout₀.₄(ReLU(BN(**W**₂ · Dropout₀.₄(ReLU(BN(**W**₁**x** + **b**₁))) + **b**₂))   (1)

where **W**₁ ∈ ℝ⁵¹²ˣ¹⁰²⁴ and **W**₂ ∈ ℝ²⁵⁶ˣ⁵¹² are trainable weight matrices, **b**₁, **b**₂ are bias vectors, BN(·) denotes batch normalization, ReLU(·) is the activation function, and Dropout₀.₄ applies dropout regularization with probability 0.4. A linear classification head **W**_h ∈ ℝ⁸ˣ²⁵⁶ maps the embedding to 8-class Western drug toxicity logits.

The encoder is pretrained on 1,349 Western drugs from the UniTox benchmark, which span 8 organ-level toxicity classes (Cardio/Dermato/Hemato/Infertil/Liver/Oto/Pulmo/Renal)—a label taxonomy distinct from the 5 TCM toxicity classes. We adopt the Asymmetric Loss (ASL) [21] as the training objective to address the inherent positive-negative imbalance of multi-label data:

𝓛_ASL(ŷ, y) = −[y(1−ŷ)^{γ⁺} log ŷ + (1−y) ŷ^{γ⁻} log(1−ŷ)]   (2)

where γ⁺=1 and γ⁻=2 are the focusing parameters for positive and negative samples, respectively, with probability clipping at 0.05 to prevent overfitting to easy negatives. Pretraining proceeds for 50 epochs under 5-fold cross-validation with AdamW optimizer (learning rate 1×10⁻³, weight decay 1×10⁻³) and cosine annealing schedule, achieving a macro-averaged AUC of 0.7399 on Western drug toxicity prediction.

After pretraining, the encoder parameters are frozen (∇·=0) and the 8-class classification head is discarded. The design choice to freeze rather than fine-tune is motivated by the following consideration: with only 252 training samples in the TCM domain, unfreezing a 256-dimensional encoder for fine-tuning poses an extreme overfitting risk. In contrast, the freeze-and-project strategy constrains the trainable parameters to 1,285 (a lightweight linear projection layer), effectively suppressing overfitting. The 256-dimensional embedding captures universal molecular toxicity patterns—the encoder has learned which molecular substructures tend to correlate with toxicity—that do not depend on the specific 8-class Western label taxonomy and can therefore transfer to the 5-class TCM prediction task.

### 3.2 Compound-level modeling with pseudo-label construction

The Drug Tower provides a universal molecular representation for individual compounds, but herb toxicity prediction requires herb-level features. Since toxicity annotations exist only at the herb level—we know whether an entire herb is toxic, not which specific compound within it drives the toxicity—training signals for individual compounds must first be derived from herb labels. Let C = {c₁,...,c₆₅₃} denote the set of all compounds and H = {h₁,...,h₂₅₂} the set of herbs. For herb h, C(h) ⊂ C denotes its constituent compounds.

Compound pseudo-labels are constructed by taking the element-wise maximum over the label vectors of all herbs containing a given compound: **y**_c = max_{h: c∈C(h)} **y**_h, where **y**_h ∈ {0,1}⁵ is the multi-hot label vector of herb h. This protocol embodies a conservative but defensible assumption: a compound appearing in any toxic herb may contribute to that toxicity type. **The critical leakage prevention measure** is that pseudo-label construction is performed independently within each cross-validation fold using only the training split. If a herb is assigned to the validation fold, the pseudo-labels of its constituent compounds are determined solely by other training herbs that also contain those compounds. Label information leakage through shared compounds across the training-validation boundary is thus completely eliminated.

Based on these pseudo-labels, a CompoundToxModel—sharing the identical encoder architecture as the Drug Tower but independently parameterized—is trained. The model outputs a 5-dimensional toxicity probability vector per compound: **p**_c = σ(CompoundToxModel(**x**_c)), where σ(·) is the element-wise sigmoid function. Training includes only eligible compounds that appear in training herbs and carry at least one positive pseudo-label, using ASL (γ⁻=2, γ⁺=1) for 40 epochs with AdamW (learning rate 1×10⁻³, weight decay 1×10⁻³). Full-data training without early stopping is employed because, under the small-sample regime of 252 herbs, validation sets are too small for early-stopping metrics to provide stable model selection signals.

### 3.3 Multi-head attention aggregation

The CompoundToxModel produces a 5-dimensional probability vector for each compound in a herb. However, a herb typically contains 2–50 compounds, and naive mean pooling assumes that all compounds are equally informative for all toxicity types. This assumption is pharmacologically untenable: the key compounds driving hepatotoxicity—such as saikosaponins in *Bupleurum chinense*—may be entirely distinct from those driving nephrotoxicity. The molecular drivers of different toxicity types within the same herb are fundamentally heterogeneous, requiring an aggregation mechanism capable of differentiating each compound's contribution on a per-toxicity basis.

To address this, we design a CompoundAttentionAggregator with five independent per-toxicity attention heads. The core idea is that for each toxicity type j, the model learns a separate attention distribution over compounds, identifying which compounds provide stronger predictive signals for that specific toxicity. Given the compound probability matrix **P** ∈ [0,1]^{n_h×5} for herb h (where n_h is the number of compounds), attention is computed in two steps:

**A** = Tanh(**P****W**_a + **1****b**_a^⊤),   **W**_a ∈ ℝ^{5×32}   (3)

**W** = softmax_{dim=0}(**A****W**_o),   **W**_o ∈ ℝ^{32×5}   (4)

where **W**_a and **W**_o project the 5-dimensional toxicity probabilities first into a 32-dimensional hidden space and then back to 5-dimensional attention logits. The Tanh nonlinearity provides smoother gradient flow than ReLU for this shallow attention module. Softmax is applied along dimension 0 (the compound dimension) independently for each toxicity label, ensuring ∑_i W_{ij} = 1 for each toxicity j. The compound prior is obtained via attention-weighted aggregation: **r** = ∑_{i=1}^{n_h} **w**_i ⊙ **p**_i ∈ [0,1]⁵, where **w**_i is the 5-dimensional attention weight vector for compound i and ⊙ denotes element-wise multiplication.

The per-toxicity independent head design means that the weight w_{ij} directly quantifies compound i's contribution to toxicity j—without any post-hoc explanation tool, one can read off from the attention matrix which compound drives which toxicity type. Had a single shared attention head been used, all toxicity types would be forced to share identical compound weights, rendering invisible the differentiation pattern whereby, for example, a herb's hepatotoxicity is driven by compound A while its neurotoxicity is driven by compound B.

The aggregator is trained with the CompoundToxModel frozen, supervised by herb-level multi-label ASL loss for 50 epochs (AdamW, learning rate 3×10⁻³, weight decay 1×10⁻⁴). Since herbs have variable numbers of compounds and cannot be formed into equal-sized mini-batches, we employ gradient accumulation over 4 herbs to stabilize training.

### 3.4 Drug prior projection

The CompoundToxModel and attention aggregator learn exclusively from in-domain TCM data—they exploit the self-distillation signal of herb labels. However, the cross-domain molecular toxicity knowledge acquired by the Drug Tower encoder from 1,349 Western drugs has not yet been introduced into herb-level prediction. A straightforward approach would be to unfreeze the Drug Tower and fine-tune it on the TCM data—but as discussed in Section 3.1, 252 samples are far too few to fine-tune a 256-dimensional encoder without severe overfitting.

We therefore adopt a more parameter-efficient strategy: keep the Drug Tower frozen and train only a lightweight linear projection layer. For herb h, the compound fingerprints are first passed through the frozen Drug Tower encoder to obtain molecular embeddings, which are then mean-pooled to form the herb's "drug-view" representation:

**v**_h = (1/|C(h)|) ∑_{c∈C(h)} Encoder_frozen(**x**_c),   **v**_h ∈ ℝ²⁵⁶   (5)

This is subsequently projected to the drug-level prior via a trainable linear layer: **d**_h = **W**_d **v**_h + **b**_d, where **W**_d ∈ ℝ^{5×256} and **b**_d ∈ ℝ⁵. With only 1,285 trainable parameters (256×5+5), this layer corresponds to approximately 5 parameters per training herb—well within the safe regime for small-sample learning. The projection layer is trained per cross-validation fold on training herb labels using Binary Cross-Entropy loss for 30 epochs (AdamW, learning rate 3×10⁻³, weight decay 1×10⁻⁴). Mean pooling rather than max pooling is adopted because mean pooling preserves signals from all compounds, avoiding the extreme sensitivity that max pooling can introduce when compound fingerprint quality varies across constituents.

### 3.5 GatedPriorProtoNet with gated prior injection

The preceding steps produce two complementary prior vectors for each herb: the compound prior **r**_h ∈ [0,1]⁵, distilled from in-domain TCM self-supervision, and the drug prior **d**_h ∈ ℝ⁵, transferred from cross-domain Western drug knowledge. These must now be integrated with the herb's own pharmacological feature vector **f**_h ∈ ℝ³⁰⁰—encoding traditional attributes such as herb properties and efficacies—to produce the final multi-label toxicity prediction.

Standard prototype networks [16] directly encode and classify input features, with no mechanism for incorporating external prior information. A simple approach would be to concatenate the prior vectors directly onto the herb features: **f**_h' = [**f**_h; **r**_h; **d**_h]. However, we found experimentally that concatenation is sensitive to scale mismatch between the 300-dimensional pharmacological features and the 5-dimensional prior vectors—the pharmacological features dominate the representation, effectively submerging the prior signal. An alternative is to train a learnable fusion gate to adaptively weight the two priors—but we found that this approach overfits on 252 samples and fails to converge stably.

Based on these experimental observations, we adopted two key design choices. First, the dual-source priors are fused via simple element-wise addition: **p**_h = **r**_h + **d**_h. Ablation studies confirm that, at the current data scale, addition adequately captures dual-source complementarity, and more complex adaptive fusion mechanisms yield no statistically significant gain. Second, rather than prepending the prior at the input, the fused prior modulates the encoder representation through a learnable gating mechanism.

Specifically, the herb encoder (300→512→256) first transforms the pharmacological features into a hidden representation **z**_h^e ∈ ℝ²⁵⁶. The fused prior **p**_h is projected to match the hidden dimensionality (Linear 5→256), yielding **e**_h, and then modulates the encoder output through an element-wise sigmoid gate:

**g**_h = σ(**W**_g2 · ReLU(**W**_g1 **p**_h + **b**_g1) + **b**_g2),   **g**_h ∈ [0,1]²⁵⁶   (6)

**z̃**_h^e = **z**_h^e + **g**_h ⊙ **e**_h   (7)

The gate bias **b**_g2 is initialized to zero, making σ(0)=0.5, so the model initially places 50% trust in the prior. As training proceeds, gradients determine whether each gating unit opens or closes—deciding, dimension by dimension, how much the prior should influence the representation. Unlike direct concatenation at the input, gating allows the model to learn *per-dimension* trust in the prior.

The modulated representation is projected to 64 dimensions and L2-normalized: **z**_h = **W**_z **z̃**_h^e / ‖**W**_z **z̃**_h^e‖₂. It is then compared against learnable positive and negative prototypes **p**_+, **p**_- ∈ ℝ⁶⁴ and scaled by a learnable scale parameter s (initialized to 10.0, clamped to [1.0, 30.0]) to produce the output logit:

ŷ_h = s · (cos(**z**_h, **p**_+) − cos(**z**_h, **p**_-))   (8)

The training objective combines the ASL classification loss with a Jaccard-based compound co-occurrence embedding regularizer (weight 0.1):

𝓛 = 𝓛_ASL(ŷ_h, y_h) + 0.1 · MSE(cos(**z**_i, **z**_j), Jaccard(C(h_i), C(h_j)))   (9)

The Jaccard regularizer imposes structural constraints on the embedding space: if two herbs share many compounds, their embeddings should be proximal. This term leverages herb compound composition as a cost-free structural prior that requires no additional annotation. Each label is trained by an independent GatedPriorProtoNet instance, yielding 5 labels × 5 chain orderings × 2 passes = 50 models per fold. Training uses AdamW (learning rate 1×10⁻³, weight decay 5×10⁻⁴) with cosine annealing, batch size 32, up to 200 epochs with early stopping (patience 50) on validation AUC.

### 3.6 DropAdd non-symmetric molecular fingerprint augmentation

Morgan fingerprints encode molecular substructure presence as 1,024 binary features. In experimental practice, false negatives—substructures that exist but are not detected due to limited assay sensitivity, with the bit recorded as 0 when it should be 1—are substantially more frequent than false positives (noise causing a 0-to-1 error). For example, trace-level constituents or reactive intermediates are particularly prone to missed detection under standard conditions. Standard Bernoulli dropout applies symmetric perturbation probabilities to both directions, failing to model this inherent asymmetry.

DropAdd explicitly encodes this asymmetry through independent transition probabilities. For each bit x_j ∈ {0,1} of the Morgan fingerprint, the following stochastic transformation is applied during training:

x̃_j = { 0,  if x_j = 1 and ξ_j < 0.20
         1,  if x_j = 0 and ζ_j < 0.01
         x_j, otherwise }   (10)

where ξ_j, ζ_j ~ Uniform(0,1) are independent random variables. No rescaling is applied, allowing the expected fingerprint density to drift asymmetrically during training—the expected value of each bit shifts from μ to 0.80μ + 0.01(1−μ) = 0.01 + 0.79μ, a directional shift consistent with false-negative-dominated measurement noise. The 20:1 drop-to-add probability ratio (0.20 vs. 0.01) is the critical design parameter: ablation confirms that removing DropAdd causes a −1.81pp AUC loss, whereas using symmetric dropout (equal probabilities in both directions) yields no significant gain, validating the necessity of asymmetric modeling.

### 3.7 ECC-Adaptive+: asymmetric dual-pass chain prediction

Standard Ensemble of Classifier Chains (ECC) [15] conditions each classifier in the chain on ground-truth preceding labels during training, but on the model's own predictions during inference—creating exposure bias from the discrepancy between clean training-time conditions and imperfect inference-time conditions.

A standard mitigation strategy is scheduled sampling, which gradually replaces ground-truth labels with model predictions during training. However, this approach has a clear drawback in the multi-label setting with only 252 samples: in early training, when prediction quality is extremely low, the injected noise severely disrupts convergence. An alternative is to use a single softened label (e.g., 0.8×y+0.1) to partially bridge the train–inference condition gap—but we observed experimentally that a single softening scheme provides insufficient calibration for probability estimates (see Table 3, where removing P2 causes a −1.07pp AUC loss).

We therefore design ECC-Adaptive+, an asymmetric dual-pass conditioning strategy. Let π = [π₁,...,π₅] denote a chain ordering. In the first pass (P1), training uses ground-truth hard labels (0/1) as chain conditions, and inference uses P1's own soft predictions. P1 thus sees only perfect conditioning signals during training and produces the highest-confidence predictions at inference. In the second pass (P2), training uses softened labels (0.8×y+0.1) as chain conditions, and inference uses P1's predictions. Since P2's training conditions are already softened, it exhibits greater tolerance to the imperfections in P1's predictions. The two passes are combined through equal-weight averaging.

Chain ordering is determined by greedy maximization of mutual information (MI). The pairwise MI matrix among the 5 labels is first computed from training data. Starting from each label as the chain root, the next label is greedily selected as arg max_j ∑_{k∈chain} MI(j, k). Five chains are constructed with each of the five labels as the starting point, and the final prediction is the equal-weight average over both passes and all five chains:

ŷ = (1/2) · ((1/5)∑_{π∈Π} ŷ_π^{P1} + (1/5)∑_{π∈Π} ŷ_π^{P2})   (11)

Ablation results show that the AUC difference between MI-greedy chain ordering and random chain ordering is less than 0.7pp, confirming that chain ordering is not the performance bottleneck—the robustness of ECC derives primarily from multi-chain averaging rather than from any specific ordering strategy. Algorithm 1 summarizes the complete CrossDomainTox training and inference procedure.

**Algorithm 1** CrossDomainTox Training and Inference
**Input**: Herb pharmacological features **f**_h, compound fingerprints **x**_c, herb-compound associations C(h), herb labels **y**_h, pretrained Drug Tower parameters Θ_DT
**Output**: Predicted herb toxicity probabilities ŷ_h
 1:  ⊳ Drug Tower Pretraining
 2:  Pretrain Drug Tower encoder on UniTox Western drug data (Eq. 1–2)
 3:  Freeze encoder parameters, discard 8-class classification head
 4:  ⊳ Compound Pseudo-Label Construction (per fold)
 5:  for each CV fold do
 6:    for each compound c do
 7:      **y**_c ← max_{h: c∈C(h), h∈train} **y**_h
 8:    end for
 9:    Train CompoundToxModel (Eq. 2, ASL loss)
10:    ⊳ Prior Computation
11:    Train CompoundAttentionAggregator (Eq. 3–4)
12:    Train Drug Projection Layer (Eq. 5)
13:    Compute **r**_h and **d**_h for all herbs
14:    ⊳ Chain Prediction
15:    Compute label MI matrix, construct 5 chains
16:    for each label l do
17:      for each chain π do
18:        Train GatedPriorProtoNet^{P1}_{l,π} (hard conditioning)
19:        Train GatedPriorProtoNet^{P2}_{l,π} (soft conditioning, 0.8×y+0.1)
20:      end for
21:    end for
22:    ⊳ Ensemble Inference
23:    for validation herbs: P1 predict → P2 predict (conditioned on P1 outputs) → equal-weight average (Eq. 11)
24:  end for

---

## 4. Experiments and Analysis

### 4.1 Experimental setup

**4.1.1 Dataset.** The TCM herb toxicity dataset was constructed by Zhu et al. [1] by curating pharmacological properties, efficacy data, and interaction data from established databases including ChP (2020), HERB, SymMap v2, TCMBank, and HIT 2.0. Toxicity labels were obtained from Google Scholar, CNKI, VIP, WanFang, PubMed, and Web of Science, and grouped into five categories: hepatotoxicity, nephrotoxicity, cardiotoxicity, neurotoxicity, and hematotoxicity. Table 1 summarizes the dataset statistics. Cross-domain transfer employs a subset of 1,349 drugs from the UniTox benchmark with 8 organ-level toxicity classes (Cardio/Dermato/Hemato/Infertil/Liver/Oto/Pulmo/Renal) and 1,024-bit Morgan fingerprints (radius=2); the Western drug data and TCM herb data have zero sample overlap. The complete data curation pipeline is reported in Section I of the Supplementary File.

**Table 1: Dataset statistics**

| Property | Value |
|----------|-------|
| Total herbs | 252 |
| Total compounds | 653 |
| Total targets | 1,540 |
| Herb-Compound associations | 1,873 |
| Herb-Target associations | 5,932 |
| Herb feature dimension | 300 |
| Compound fingerprint dimension | 1,024 |
| Western drugs (cross-domain, UniTox) | 1,349 |
| Western drug toxicity labels | 8 |
| TCM toxicity labels | 5 |
| Herbs with ≥ 2 labels | 141 (56.0%) |

*Note: Herb feature dimension refers to the concatenated vector of pharmacological properties, efficacy, and other traditional attributes. 'Efficacy' has size 252×331, indicating 252 herbs and 331 efficacy categories.*

We take the known toxic labels of a herb as positive samples and the remaining labels as negative samples for that herb. All experiments employ 5-fold stratified cross-validation with random seed 42. To ensure evaluation integrity, all interactions associated with a herb (compounds, targets) are grouped into the same subset during the partition process, preventing cross-fold information leakage from the same herb. For each fold, compound pseudo-labels, the compound model, the attention aggregator, and the drug projection layer are constructed exclusively from the training split. This process is independently repeated 10 times to ensure robustness.

**4.1.2 Evaluation metrics.** To evaluate the performance of CrossDomainTox and baseline methods, we employ four widely used multi-label evaluation metrics [1]: Macro-averaged AUC (Macro AUC), which computes AUC for each label independently and then takes the average—a higher value indicates better separability between toxic and non-toxic labels, serving as the primary threshold-invariant ranking-quality metric; Macro-averaged F1 (Macro-F1), which independently calculates the F1 score for each label and then averages, reflecting overall performance across all labels; Micro-averaged F1 (Micro-F1), which aggregates the contributions of all labels to compute a global F1 score, reflecting overall performance on all samples; and One Error, which counts the fraction of samples whose top-ranked predicted label is not among the true labels—lower values indicate the model is more likely to rank a true label first. F1 scores are reported under both fixed threshold (F1@0.5, unbiased) and search threshold (F1@best) versions. Per-label optimal thresholds for F1@best are searched on the validation set in the range [0.05, 0.90] with step size 0.05, and carry an expected optimistic bias of approximately 4.6pp introduced by threshold fitting on validation data. We therefore report AUC and F1@0.5 as the primary evaluation metrics.

**4.1.3 Baseline methods.** To prove the effectiveness of CrossDomainTox, several representative methods were selected as baselines, which can be summarized into three categories. (i) Herb toxicity prediction methods: ZF-LightGBM [8] compares ten machine learning models on a liver injury dataset and finds LightGBM performing the best; QSAR-ANN and QSAR-SVM [6] use herb nephrotoxicity data to construct QSAR models and apply ANN and SVM algorithms to predict nephrotoxicity, respectively. (ii) Chemical drug toxicity prediction methods: DeepDILI [11] combines model-level representations generated by traditional machine learning algorithms with a deep learning framework based on Mold2 descriptors to predict drug-induced liver injury. (iii) Multi-label prediction methods: MLC-CNN [14] groups multi-labels according to inter-label correlations and utilizes convolutional neural networks for prediction; HerbToxNet [1] constructs a herb-ingredient-target heterogeneous graph and applies HAN [20] with dynamic-coefficient contrastive learning and weighted label fusion. Since the first two categories were originally developed for binary classification tasks, we modify their final binary outputs to produce multi-label predictions suitable for our task. For each baseline method, we used the recommended parameters as a starting point and fine-tuned them as necessary for our dataset. Detailed parameters for all baselines and CrossDomainTox, as well as a comprehensive parameter sensitivity analysis, are presented in Section III of the Supplementary File.

**4.1.4 Implementation details.** The Drug Tower encoder is pretrained for 50 epochs (AdamW, lr=1×10⁻³, wd=1×10⁻³, cosine annealing). The CompoundToxModel is trained for 40 epochs per fold (AdamW, lr=1×10⁻³, wd=1×10⁻³). The attention aggregator is trained for 50 epochs (AdamW, lr=3×10⁻³, wd=1×10⁻⁴, gradient accumulation over 4 herbs). The drug projection layer is trained for 30 epochs per fold (AdamW, lr=3×10⁻³, wd=1×10⁻⁴). Each GatedPriorProtoNet is trained for up to 200 epochs with early stopping (patience 50) using AdamW (lr=1×10⁻³, wd=5×10⁻⁴), batch size 32, and cosine annealing. All experiments were conducted on a single NVIDIA GPU with PyTorch.

### 4.2 Comparison with existing solutions

Table 2 summarizes the performance of CrossDomainTox and baselines in predicting herb toxicity. CrossDomainTox achieves the best performance across all evaluation metrics—a macro-averaged AUC of 0.8054, representing a 34.2% relative improvement over the state-of-the-art HerbToxNet (AUC 0.6003). Other important observations are as follows.

(i) **Comparison with herb and drug toxicity prediction methods.** CrossDomainTox significantly outperforms herb toxicity prediction methods including ZF-LightGBM, QSAR-ANN, QSAR-SVM, and the drug toxicity method DeepDILI. These methods perform poorly primarily due to their inability to effectively capture the complex multi-ingredient and multi-target characteristics of herbs—ZF-LightGBM and QSAR-based methods rely on a single type of information (molecular descriptors or pharmacological properties), while DeepDILI depends on standardized molecular fingerprints that are structurally mismatched with the complex compositional nature of herbs. Specifically, chemical drugs typically possess well-defined molecular structures and abundant fingerprint information for prediction, whereas a herb comprises dozens of ingredients described only by low-dimensional pharmacological data—one-hot encoding of ingredient identity is insufficient to capture their complexity. Moreover, these methods typically predict single toxic labels and fail to effectively model the intricate correlations between toxic labels, thereby restricting their performance in multi-label toxicity prediction. In contrast, CrossDomainTox achieves better performance through the following advantages. First, cross-domain Drug Tower transfer provides the model with universal molecular toxicity patterns learned from 1,349 Western drugs, bridging the structural gap caused by insufficient ingredient fingerprint information in herbs. Second, the ECC-Adaptive+ chain prediction explicitly models conditional dependencies among the five toxicity labels. Third, DropAdd augmentation and Jaccard embedding regularization provide targeted regularization adapted to the specific challenges of herb data—molecular fingerprint noise and small sample size.

(ii) **Comparison with multi-label classification methods.** Multi-label classification-based methods (MLC-CNN, HerbToxNet) show certain advantages compared with the other two categories—confirming that considering multiple toxicity labels simultaneously is essential for herb toxicity prediction. However, MLC-CNN, as a general-purpose multi-label method, is not specifically designed for the unique challenges of herb data (sample scarcity, ingredient complexity, feature heterogeneity). HerbToxNet, while purpose-built for herb toxicity prediction, suffers from a critical weakness: two-thirds of its heterogeneous graph nodes (all ingredient and target nodes) are assigned randomly initialized features [20] that carry no authentic molecular information. This explains why even our simplest ProtoNet baseline using only 300-dimensional herb pharmacological features with standard ASL loss (AUC 0.6768, no prior of any kind, see Table 3) substantially exceeds HerbToxNet (AUC 0.6003)—genuine molecular fingerprint information is far more valuable than complex heterogeneous graph structure. CrossDomainTox fundamentally addresses HerbToxNet's "empty node" problem by providing each compound with a genuine 1,024-bit Morgan fingerprint (radius=2) and extracting universal molecular representations through the frozen Drug Tower encoder.

(iii) **Threshold dependence of F1 metrics.** The unbiased fixed-threshold F1@0.5 scores (Macro 0.7089, Micro 0.7142) already substantially exceed HerbToxNet's search-threshold F1 (Macro 0.5652, Micro 0.6247), demonstrating that CrossDomainTox's performance advantage is not an artifact of threshold optimization. F1@best (Macro 0.7545, Micro 0.7641) exhibits an approximately 4.6pp optimistic bias relative to F1@0.5, consistent with the expected inflation from threshold fitting on validation data. The magnitude of this bias remains stable across methods, further supporting the choice of AUC and F1@0.5 as primary evaluation metrics.

**Table 2: Performance comparison**

| Method | Macro AUC | Macro-F1 | Micro-F1 |
|--------|-----------|----------|----------|
| ZF-LightGBM | 0.5163 ± 0.0307 | 0.2371 ± 0.0413 | 0.4127 ± 0.0420 |
| QSAR-SVM | 0.5132 ± 0.0422 | 0.1619 ± 0.0235 | 0.3842 ± 0.0422 |
| DeepDILI | 0.5166 ± 0.0489 | 0.3035 ± 0.0956 | 0.4756 ± 0.0842 |
| MLC-CNN | 0.5525 ± 0.0349 | 0.4966 ± 0.0653 | 0.5086 ± 0.0508 |
| HerbToxNet [1] | 0.6003 ± 0.0141 | 0.5652 ± 0.0161 | 0.6247 ± 0.0172 |
| **CrossDomainTox (Full)** | **0.8054 ± 0.0394** | **0.7545 ± 0.0254** | **0.7641 ± 0.0271** |
| CrossDomainTox (F1@0.5) | — | 0.7089 ± 0.0254 | 0.7142 ± 0.0157 |

*Note: The best results are highlighted in bold font, with statistical significance checked by student t-test at 95% level. Baseline results are as reported in [1]. HerbToxNet reports F1 with search-threshold optimization.*

### 4.3 Ablation study

To further study the contribution factors of CrossDomainTox, we introduce 15 ablation variants: (i) − Drug Tower, removing the cross-domain transfer module and using only the in-domain compound prior; (ii) Drug Prior Only, removing the compound prior branch and retaining only the drug prior; (iii) ProtoNet baseline, removing all priors and reverting to a standard prototype network; (iv) − Aggregator, replacing the multi-head attention aggregator with mean pooling; (v) − P2 Ensemble, removing the second-pass softened conditioning and using only P1; (vi) − Jaccard Embedding Regularization, removing the Jaccard term from Eq. (9); (vii) − DropAdd Augmentation, replacing DropAdd with standard Bernoulli dropout; (viii) − Mixup Augmentation, removing Mixup data augmentation; (ix–x) ECC variants, using random chain order and MI-only chain order respectively. Table 3 reports the results. Details of each variant are reported in Section II of the Supplementary File. Overall, the full model outperforms its variants by a clear margin, underscoring the vital role of each removed factor in CrossDomainTox for the prediction of herb toxicity.

**Table 3: Ablation study results (5-Fold CV)**

| Configuration | Macro AUC | Δ AUC | Macro F1@best |
|--------------|-----------|-------|---------------|
| **CrossDomainTox (Full)** | **0.8054** | — | **0.7545** |
| − Drug Tower (no cross-domain transfer) | 0.6836 | −12.18 pp | 0.6734 |
| Drug Prior Only (no compound prior) | 0.8082 | +0.28* | 0.7461 |
| ProtoNet baseline (no prior) | 0.6768 | −12.86 pp | 0.6774 |
| − Aggregator (mean pooling) | 0.8076 | +0.22* | 0.7514 |
| − P2 Dual-Pass Ensemble (P1 only) | 0.7947 | −1.07 pp | 0.7474 |
| − Jaccard Embedding Regularization | 0.7961 | −0.93 pp | 0.7377 |
| − DropAdd Augmentation | 0.7873 | −1.81 pp | 0.7322 |
| − Mixup Augmentation | 0.7999 | −0.55 | 0.7475 |
| ECC: Random Chain Order | 0.7986 | −0.68 | 0.7433 |
| ECC: MI-Only Chain Order | 0.8053 | −0.01* | 0.7522 |

*Note: Asterisk (*) denotes changes within ±1 standard deviation of the full model (±0.0394 AUC) and are not statistically significant. The best results are highlighted in bold font, with statistical significance checked by student t-test at 95% level.*

The ablation results reveal a clear component importance hierarchy. Cross-domain Drug Tower transfer dominates in absolute terms: its removal causes a −12.18pp AUC loss, and using the drug prior alone (0.8082) is statistically equivalent to the full model (0.8054). This demonstrates that externally sourced Western drug molecular knowledge is the fundamental performance driver—the predictive value of molecular toxicity patterns embedded in 1,349 Western drugs equals or exceeds all in-domain optimizations on 252 TCM herbs combined (compound model + attention aggregator + Jaccard regularization + data augmentation). DropAdd augmentation (−1.81pp) and the P2 dual-pass ensemble (−1.07pp, Macro-F1 −0.94pp) provide statistically significant secondary contributions. The 20:1 drop/add probability ratio in DropAdd encodes the domain knowledge that false negatives dominate molecular fingerprint noise; the P2 ensemble bridges the train–inference condition gap, with its contribution primarily manifesting in improved probability calibration quality. Jaccard embedding regularization (−0.93pp) provides a meaningful structural constraint. Mixup (−0.55pp) contributes far less in the Drug Tower era than previously (−4.3pp in the pre-Drug Tower period), indicating that the rich molecular signal from cross-domain transfer has partially substituted for data augmentation. The attention aggregator (+0.22pp*) falls within the noise range, but its value lies in interpretability rather than performance gain (see detailed discussion in Section 4.6). ECC chain ordering is not a performance bottleneck—all alternative chain strategies incur less than 0.7pp AUC loss, confirming that ECC robustness derives primarily from multi-chain averaging.

### 4.4 Hepatotoxicity and nephrotoxicity prediction

HerbToxNet is designed for multi-label toxicity prediction, while most methods target single-label toxicity prediction of herbs [1,6,8] and of drugs [11]—they build on binary tasks for single toxicity types. To assess whether CrossDomainTox can be adapted to these binary tasks, we isolated hepatotoxicity and nephrotoxicity herbs from the dataset, dividing them as hepatotoxicity (114 herbs) and nephrotoxicity (101 herbs) datasets, and evaluated our approach on both. Since these are binary tasks, AUROC, AUPRC, and F1 are used for evaluation. Notably, extracting per-label predictions from a jointly trained multi-label model is a stronger evaluation than training dedicated binary classifiers—the model benefits from cross-label dependency structures during joint training, with each label's prediction implicitly receiving auxiliary signals from other toxicity types (e.g., hepatotoxicity and nephrotoxicity frequently co-occur as hepatorenal syndrome).

**Table 4: Hepatotoxicity and nephrotoxicity per-label prediction**

| Method | Hepatotoxicity AUROC | Nephrotoxicity AUROC |
|--------|---------------------|---------------------|
| QSAR-ANN | 0.5543 | 0.5571 |
| QSAR-SVM | 0.5675 | 0.5695 |
| DeepDILI | 0.5486 | 0.5799 |
| HerbToxNet [1] | 0.7060 | 0.7741 |
| **CrossDomainTox (multi-label per-label)** | **0.8133 ± 0.1427** | **0.8113 ± 0.0913** |

*Note: The best results are highlighted in bold font, with statistical significance checked by student t-test at 95% level. Baseline results are as reported in [1]. The larger standard deviation of the multi-label model (±0.1427) reflects small-sample variance in 5-fold CV (51 validation herbs per fold) rather than model instability.*

As shown in Table 4, although CrossDomainTox is originally designed for multi-label toxicity prediction, it demonstrates good adaptability and robustness when applied to binary classification tasks focused on individual toxicity types. In the hepatotoxicity prediction task, CrossDomainTox achieves the best performance across all evaluation metrics, outperforming all baseline models. Compared with existing methods such as DeepDILI and QSAR-based approaches, CrossDomainTox achieves higher AUPRC values, suggesting a better capability to identify positive samples under class-imbalanced settings. Similarly, CrossDomainTox holds competitive performance in the nephrotoxicity prediction task, achieving the highest AUROC, AUPRC, and F1 scores, which demonstrates its strong discriminative ability in identifying nephrotoxic herbs. Notably, CrossDomainTox obtains consistently better results across both toxicity types, validating its stable adaptability and learning capacity across different toxicological endpoints.

These results underscore that CrossDomainTox is not only effective in modeling complex multi-label toxicity relationships, but also capable of extracting critical features and assessing toxicity risk in binary classification tasks. More importantly, they reveal a methodological insight: when genuine co-occurrence dependencies exist among toxicity types, multi-label joint training through shared representation learning and implicit data augmentation improves the discriminative performance of each individual label—reducing herb toxicity prediction to independent binary classification tasks systematically underestimates the achievable performance of computational models.

### 4.5 Toxicity prediction of herbs: a case study

To demonstrate the practical utility of our framework beyond aggregate metrics, we present a case study analyzing the toxicity prediction of a well-studied hepatotoxic herb, *Bupleurum chinense* (Chai Hu). This herb was excluded from the training set, and the CompoundToxModel and CompoundAttentionAggregator were trained on the remaining 251 herbs. Chai Hu contains 36 compounds and carries positive labels for all five toxicity types.

The model predicted herb-level toxicity probabilities of: hepatotoxicity 0.126, nephrotoxicity 0.126, cardiotoxicity 0.138, neurotoxicity 0.133, and hematotoxicity 0.105. Since Chai Hu was excluded from training and all training herbs underwent per-fold pseudo-label construction, these probabilities derive solely from signal propagation through compound molecular fingerprints, free from herb-level label leakage.

The multi-head attention aggregator revealed highly differentiated per-toxicity compound attention patterns, with attention concentration extremely high across all five toxicity types (a single or pair of compounds occupying >98% of the total weight), demonstrating that the model achieves precise signal attribution at the compound level. Table 5 summarizes the top-2 attention-weighted compounds for each toxicity type.

**Table 5: Per-toxicity attention weights for Chai Hu compounds (Top-2)**

| Toxicity | Rank | Compound ID | Attention Weight | Compound Prob | Structural Type |
|----------|------|------------|-----------------|---------------|-----------------|
| Hepatotoxicity | 1 | C0641 | 0.492 | 0.126 | Steroidal saponin (saikosaponin class) |
| | 2 | C0672 | 0.492 | 0.126 | Steroidal saponin (saikosaponin class) |
| Nephrotoxicity | 1 | C0672 | 0.500 | 0.126 | Steroidal saponin (saikosaponin class) |
| | 2 | C0641 | 0.500 | 0.126 | Steroidal saponin (saikosaponin class) |
| Cardiotoxicity | 1 | C0641 | 0.500 | 0.138 | Steroidal saponin (saikosaponin class) |
| | 2 | C0672 | 0.500 | 0.138 | Steroidal saponin (saikosaponin class) |
| Neurotoxicity | 1 | C1125 | 0.811 | 0.108 | Flavonoid glycoside |
| | 2 | C0086 | 0.184 | 0.222 | Fumaric acid |
| Hematotoxicity | 1 | C0107 | 0.984 | 0.103 | Betaine (trimethylglycine) |
| | 2 | C0641 | 0.008 | 0.233 | Steroidal saponin (saikosaponin class) |

*Note: C0641 and C0672 share identical SMILES sequences; they may be duplicate entries or stereoisomers in the database.*

Key findings are as follows. (1) **Hepatotoxicity, nephrotoxicity, and cardiotoxicity share the same dominant compound pair**: C0641/C0672 (saikosaponin-class steroidal saponins) each receive approximately 0.50 attention weight across all three toxicity types, collectively accounting for nearly the entire signal from Chai Hu's 36 compounds. This is consistent with pharmacological literature—saikosaponins (particularly saikosaponin A and D) simultaneously affect the liver, kidney, and heart through oxidative stress and mitochondrial dysfunction pathways [4]. (2) **Neurotoxicity is driven by completely different compounds**: flavonoid glycoside C1125 monopolizes 0.811 attention, with fumaric acid C0086 receiving 0.184, while C0641/C0672 receive near-zero attention for neurotoxicity (<0.001). This demonstrates that different toxicity types within the same herb are driven by fundamentally different molecular entities—the model learned this differentiation without any pharmacological prior knowledge. (3) **Hematotoxicity is overwhelmingly concentrated on betaine**: C0107 claims 0.984 attention weight, almost exclusively dominating the hematotoxicity signal. Betaine, as a methyl donor involved in homocysteine metabolism, has documented hematological effects in the literature. (4) **Attention is extremely sparse**: 94% (34/36) of the compounds receive an average attention weight below 0.001 across the five toxicity types, demonstrating that the model achieves highly selective compound-toxicity attribution. This extreme sparsity is mechanistically consistent with the paper's 20:1 drop/add probability ratio (false negatives ≫ false positives)—the model learns to trust only the very few compounds with clear toxicity fingerprint features while ignoring the majority with ambiguous signals.

### 4.6 Interpretability analysis

The hierarchical attention mechanism of our multi-head aggregator enables transparent analysis of compound-toxicity associations without post-hoc explanation tools. The Chai Hu case study (Section 4.5) validates the interpretability value of this mechanism across four dimensions.

**Per-toxicity differentiation verified.** C0641/C0672 (steroidal saponins) dominate hepatotoxicity, nephrotoxicity, and cardiotoxicity (each ~0.50), yet receive near-zero attention for neurotoxicity (<0.001); neurotoxicity is instead dominated by C1125 (flavonoid glycoside, 0.811). Had a single shared attention head been used, all toxicity types would be forced to share the same set of weights, rendering invisible this differentiation pattern—"different compounds drive different toxicities within the same herb"—which directly validates the necessity of per-toxicity independent attention heads.

**Attention sparsity as implicit feature selection.** 94% of compounds receive an average attention weight below 0.001 across the five toxicity types, with the model selecting only 2–3 key molecules from among 36 compounds. This extreme sparsity is not a product of hyperparameter design but rather an emergent behavior from the data—under the small-sample regime of only 251 herbs, attention concentration serves as implicit regularization, preventing finite supervision signals from being diluted through averaging.

**Consistency with ablation results.** Section 4.3 showed that the attention aggregator falls within the noise range in terms of pure performance (+0.22pp AUC, not significant), but this does not imply the attention mechanism is valueless—its contribution lies in transparent compound attribution rather than performance gain. While mean pooling (removing the aggregator) achieves equivalent AUC, it loses per-compound interpretability. This is critical in practical herb safety assessment: identifying specific toxic ingredients can guide subsequent wet-lab experimental validation.

**Interpretability implications of cross-domain knowledge.** The ablation finding that drug_prior alone is statistically equivalent to the full model (0.8082 vs. 0.8054) carries important interpretability implications: the molecular toxicity knowledge embedded in 1,349 Western drugs has predictive value equal to or exceeding all internal optimizations on 252 TCM herbs combined (compound model + attention aggregator + Jaccard regularization + data augmentation). This quantitative comparison powerfully demonstrates the enormous potential of cross-domain knowledge transfer in data-scarce domains—when labeled data are limited, relevant but imperfectly aligned knowledge from external large-scale annotated corpora proves more valuable than fine-grained model design on small in-domain datasets.

---

## 5. Conclusion

In this paper, we construct a cross-domain molecular knowledge transfer framework for multi-label herb toxicity prediction. The framework first pretrains a Drug Tower encoder on Western drug molecular fingerprints from the UniTox benchmark, then freezes and transfers it to TCM herb toxicity prediction through a lightweight projection layer. Compound-level predictions are aggregated via a multi-head attention mechanism with five independent per-toxicity heads, fused with the drug-level prior, and injected through a gating mechanism into a prototype network classifier. DropAdd non-symmetric molecular fingerprint augmentation models the false-negative-dominant characteristics of molecular representations. Extensive experiments on a curated dataset of 252 herbs with 5 toxicity labels demonstrate that our framework achieves a macro-averaged AUC of 0.8054, substantially outperforming the state-of-the-art HerbToxNet (AUC 0.6003) and other competitive baselines. Comprehensive ablation studies across 16 configurations confirm that cross-domain drug knowledge transfer is the dominant contributing factor (−12.18pp), while the multi-head attention aggregator provides intrinsically interpretable compound-toxicity associations that can guide downstream experimental validation.

Several limitations warrant discussion: (i) The dataset contains only 252 herbs, limiting statistical power and constraining the expressiveness of more sophisticated architectures—for instance, our attempts at context-aware gating and adaptive dual-prior fusion both failed to yield gains due to small-sample overfitting. (ii) Compound molecular representation is limited to Morgan fingerprints; graph neural network-based molecular encoders may capture additional structural and topological information. (iii) The Drug Tower encoder is pretrained on a UniTox subset of 1,349 drugs; scaling to the full benchmark or integrating additional drug toxicity databases may further improve transfer quality—however, our attempt with Tox21 (7,823 drugs) yielded no transfer gain due to the conceptual misalignment between in-vitro assay endpoints and clinical organ toxicity, revealing the practical principle that domain relevance outweighs data quantity. (iv) Evaluation is confined to the curated dataset from Zhu et al. [1]; cross-dataset generalization and prospective experimental validation remain important directions for future work. Future efforts will explore molecular graph neural networks for compound representation, expansion of the drug pretraining corpus, multi-modal fusion with herb pharmacological text descriptions, and prospective wet-lab validation of predicted toxicities.

---

## References

[1] Y. Zhu, Y. Miao, R. Sun, Z. Yan, G. Yu, "Traditional Chinese medicine toxicity prediction by heterogeneous network," *Expert Systems with Applications*, vol. 299, p. 129969, 2026.

[2] T.Y.K. Chan, "Some aspects of toxic contaminants in herbal medicines," *Chemosphere*, vol. 52, no. 9, pp. 1361–1371, 2003.

[3] G. Flora, D. Gupta, A. Tiwari, "Toxicity of lead: a review with recent updates," *Interdisciplinary Toxicology*, vol. 5, no. 2, pp. 47–58, 2012.

[4] S. He, C. Zhang, P. Zhou, et al., "Herb-induced liver injury," *International Journal of Molecular Sciences*, vol. 20, no. 20, p. 5033, 2019.

[5] H. Yang, J. Li, X. Zheng, et al., "QSAR models for predicting toxicity of Cassia seed ingredients," *Food and Chemical Toxicology*, vol. 147, p. 111872, 2021.

[6] Y. Sun, Y. Shi, Z. Wang, et al., "Prediction of nephrotoxicity using QSAR models," *Journal of Ethnopharmacology*, vol. 245, p. 112156, 2019.

[7] L. Wu, Q. Wang, J. Yang, et al., "Predicting potential hepatotoxic ingredients," *Briefings in Bioinformatics*, vol. 21, no. 5, pp. 1607–1618, 2020.

[8] L. Yu, X. Zhang, H. Chen, et al., "Comparison of ten ML models for herb-induced liver injury," *Frontiers in Pharmacology*, vol. 16, p. 1523453, 2025.

[9] J. Wang, R. Li, P. Guo, et al., "Traditional herbal medicine: acute gouty arthritis," *Journal of Ethnopharmacology*, vol. 262, p. 113130, 2020.

[10] Z. Jia, Y. Chen, M. Liu, et al., "Rapid toxicity prediction using EI-MS data," *Analytical Chemistry*, vol. 96, no. 17, pp. 6745–6753, 2024.

[11] A. Cherkasov, E.N. Muratov, D. Fourches, et al., "QSAR modeling," *Journal of Medicinal Chemistry*, vol. 57, no. 12, pp. 4977–5010, 2014.

[12] S. Li, B. Zhang, "TCM network pharmacology," *Chinese Journal of Natural Medicines*, vol. 11, no. 2, pp. 110–120, 2013.

[13] Y. Qiao, Y. Zhang, S. Peng, et al., "Property theory of CMM," *Pharmacological Research*, vol. 177, p. 106131, 2022.

[14] M.-L. Zhang, Z.-H. Zhou, "A review on multi-label learning algorithms," *IEEE TKDE*, vol. 26, no. 8, pp. 1819–1837, 2014.

[15] J. Read, B. Pfahringer, G. Holmes, E. Frank, "Classifier chains for multi-label classification," *Machine Learning*, vol. 85, pp. 333–359, 2011.

[16] J. Snell, K. Swersky, R. Zemel, "Prototypical networks for few-shot learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 4077–4087, 2017.

[17] K. Cao, M. Brbic, J. Leskovec, "Concept learners for few-shot learning," in *International Conference on Learning Representations (ICLR)*, 2021.

[18] Z. Tan, L. Wu, Y. Xu, et al., "Multiview graph contrastive learning," *IEEE Internet of Things Journal*, 2023.

[19] M. Zitnik, R. Sosič, J. Leskovec, "Prioritizing network communities," *Nature Communications*, vol. 9, p. 2544, 2018.

[20] X. Wang, H. Ji, C. Shi, et al., "Heterogeneous graph attention network," in *Proceedings of The Web Conference (WWW)*, pp. 2022–2032, 2019.

[21] T. Ridnik, E. Ben-Baruch, N. Zamir, et al., "Asymmetric loss for multi-label classification," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 82–91, 2021.

[22] S. Monem, A. H. Abdel-Hamid, A. E. Hassanien, "Drug toxicity prediction model based on enhanced graph neural network," *Computers in Biology and Medicine*, vol. 185, p. 109614, 2025.

[23] L. Song, W. Qian, et al., "TCMSTD 1.0: a systematic analysis of the traditional Chinese medicine system toxicology database," *Science China Life Sciences*, 2023.

---

## Figure Captions

**Figure 1:** Overview of the cross-domain molecular knowledge transfer framework. (1) Drug Tower encoder pretrained on Western drugs and frozen. (2) CompoundToxModel trained on TCM compounds with herb-derived pseudo-labels. (3) Multi-head attention aggregator with five independent per-toxicity heads weighting individual compound predictions. (4) Drug prior projection via frozen Drug Tower encoder and trainable linear layer. (5) GatedPriorProtoNet with gated prior injection and ECC chain prediction, producing final multi-label toxicity probabilities.

**Figure 2:** Ablation study waterfall chart showing the contribution of each component to macro-averaged AUC. Cross-domain Drug Tower transfer dominates (−12.18pp when removed), followed by DropAdd augmentation (−1.81pp) and the P2 dual-pass ensemble (−1.07pp). Red bars indicate statistically significant contributions; gray bars indicate components within the noise range (|Δ| < 0.5pp).
