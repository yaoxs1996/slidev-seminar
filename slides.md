---
theme: seriph
background: /imgs/uestc_cover.png
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
drawings:
  persist: false
title: Welcome to Slidev
---

# Time Series Anomaly Detection In ICLR-22

Xiaoshun Yao

Data Mining Lab, UESTC

yaoxs@std.uestc.edu.cn

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---

# Background

Unsupervised detection of anomaly points in time series is a challenging problem

### Current paradigms:

- density-estimation
- clustering-based
  - anomaly score formalized as the distance to cluster center
- reconstruction-based
- autoregression-based

<br>

### Drawbacks

* focus on pointwise representation or pairwise association
* insufficient to reason about the intricate dynamics

<!--
You can have `style` tag in markdown to override the style for the current page.
Learn more: https://sli.dev/guide/syntax#embedded-styles
-->

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---

# Background

Anomalies are rare and hidden by vast normal points

* classic methods: density-estimation, clustering-based
  + ignore temporal information
  + difficult to generalize to unseen real scenarios
* deep models: learning pointwise representations
  + rarity of anomalies: pointwise representation is less informative
  + vast normal time point: less distinguishable
  + pointwise cannot provide a comprehensive description of the temporal context
* explicit association modeling: GNNs, subsequence-based
  + single time point is insufficient for complex temporal patterns
  + cannot capture the fine-grained temporal association between time point and series

<!--
深度模型基于重构误差与自回归预测  
基于子序列到的模型无法捕获细粒度
-->

---

# Question Formalization

* Time series $\mathcal{X}=\{x_1,x_2,\cdots,x_N\}$
* $x_t \in \mathbb{R}^d$ represents the observation of time $t$
* Unsupervised time series anomaly detection
    + whether $x_t$ is anomalous or not without labels

---

# Motivation

Adapt Transformer to time series anomaly detection in the unsupervised regime

* temporal association of each time point can be obtained from the self-attention map
  + describe temporal context and indicate dynamic patterns
* rarity of anomalies and the dominance of normal patterns
  + anomalies is hard to build strong associations with the whole series
  + the adjacent of anomalies shall contains similar abnormal patterns
  + normal time points is highly associate with whole series

<!--
得益于Transformer在全局表征和长距离关联上的统一模型  
-->

---

# Association Discrepancy

utilize the inherent normal-abnormal distinguishability of the association distribution

quantified by Association Discrepancy: the distance between each time point's:

* prior-assocication
* series-association

### Anomaly-Attention

a two-brach self-attention models the prior-association and series-association

* prior-association: learnable Gaussian kernel
* series-association: self-attention weights learned form raw series
* minimax strategy: amplify the normal-abnormal distinguishability

<!--
异常点的关联差异应该更小  
-->

---

# Anomaly Transformer

### Overall Architecture

<!-- ![overall architechture](/imgs/overall.svg  "Overall Architechture") -->
<img 
  src="/imgs/overall.svg"
  class="h-80 m-5 rounded shadow"
/>

<p class="absolute bottom-0 left-3">Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy, ICLR'22</p>

<!--
m: margin  
h: height
w: width  
https://bootstrap-vue.org/docs/reference/spacing-classes
-->

---

# Anomaly Transformer

### Overall Architechture

stacking structure is conductive to learning underlying association

$L$ layers model and input $\mathcal{X}\in \mathbb{R}^{N\times d}$, for $l$-th layer:

$$
\begin{aligned}
\mathcal{Z}^l&=\text{Layer-Norm}(\text{Anomaly-Attention}(\mathcal{X}^{l-1})+\mathcal{X}^{l-1})  \\
\mathcal{X}^l &= \text{Layer-Norm}(\text{Feed-Forward}(\mathcal{Z}^l)+\mathcal{Z}^l)
\end{aligned}
$$

* $\mathcal{X}^l\in \mathbb{R}^{N\times d_{model}}$, output with $d_{model}$ channels
* $\mathcal{X}^0=\text{Embedding}(\mathcal{X})$
* $\mathcal{Z}^l\in \mathbb{R}^{N\times d_{model}}$ is hidden representation
* $\text{Anomaly-Attention}(\cdot)$ computer association discrepancy

---

# Anomaly-Attention

Anomaly-Attention with a two-branch structure

### prior-association

* learnable Gaussian kernel calculate temporal distance
* Gaussian kernel can pay more attention to adjancent horizon
* a learnable scale paramter $\sigma$ adapt to the various time series patterns

### series-association

* find effective association adaptive from raw series

<br>
<br>

these two forms:

* all maintain temporal dependacies
* reflect the adjacent-concentration prior
* learned association

---

# Anomaly-Attention

<p>the Anomaly-Attention in the <i>l</i>-th layer:</p>

$$
\begin{aligned}
\text{Initialization}&: Q,K,V,\sigma = X^{l-1}W_Q^l,X^{l-1}W_K^l,X^{l-1}W_V^l,X^{l-1}W_\sigma^l \\
\text{Prior-Association}&: P^l=\text{Rescale}\lparen[\frac{1}{\sqrt{2\pi}\sigma_i}\exp{-\frac{|j-i|^2}{2\sigma_i^2}}]_{i,j\in \{1,\cdots,N\}}\rparen \\
\text{Series-Association}&: \text{Softmax}(\frac{QK^T}{\sqrt{d_{model}}}) \\
\text{Reconstruction}&: \hat{Z}^l=S^lV
\end{aligned}
$$

---

# Association Discrepancy

the symmetrized KL divergence between prior- and series-associations

* average the association discrepancy from multiple layers
* combine the associations from multi-level features into one

$$
\text{AssDis}(P,S;X)=[\frac{1}{L}\sum^L_{l=1}(\text{KL}(P_{i,:}^l\Vert S_{i,:}^l)+\text{KL}(S_{i,:}^l\Vert P_{i,:}^l))]_{i=1,\cdots,N}
$$

* $\text{KL}(\cdot \| \cdot)$ computed between every row of $P^l$ and $S^l$
* $\text{AssDis}(P,S;X)\in \R^{N\times 1}$ is the point-wise association discrepancy of $X$
* $i$-th element of AssDis is the $i$-th time point of $X$

<!-- 异常点的AssDis分数应当小于正常时间点 -->

---

# Minimax Association Learning

employ the reconstruction loss for optimizing model

$$
\mathcal{L}_{Total}(\hat{X},P,S,\lambda;X)=\|X-\hat{X}\|_F^2-\lambda\times \|\text{AssDis}(P,S;X)\|_1
$$

* $\hat{X}\in \R^{N\times d}$ is the reconstruction of $X$
* when $\lambda>0$, optimization is to enlarge the association discrepacy

---

# Minimax Association Learning

minimax strategy is proposed to make the association discrepancy more distinguishable

### Minimax Strategy

* minimize phase
  + drive the prior-association $P^l$ to approximate the series-association $S^l$
  + make $P^l$ adapt to various temporal patterns
* maximize phase
  + optimize the series-association to enlarge the association discrepancy
  + force the series-association to pay more attention to the non-adjacent horizon

$$
\begin{aligned}
\text{Minimize Phase}&: \mathcal{L}_{Total}(\hat{X},P,S_{detach},-\lambda;X) \\
\text{Maximize Phase}&: \mathcal{L}_{Total}(\hat{X},P_{detach},S,\lambda;X)
\end{aligned}
$$

<!--
序列关联是从原始序列中学习获得的
-->

---

# Minimax Association Learning

### Minimax Strategy

<img 
  src="/imgs/minimax.svg"
  class="w-180 m-20 rounded shadow"
/>

---

# Minimax Association Learning

### Association-based Anomaly Criterion

* incorporate the normalized association discrepancy
* final anomaly score of $X\in \R^{N\times d}$ is

$$
\text{AnomalyScore}(X)=\text{Softmax}(-\text{AssDis}(P,S,X))\odot [\|X_{i,:}-\hat{X}_{i,:}\|_2^2]_{i=1,\cdots,N}
$$

* $\odot$ is the element-wise multiplication
* $\text{AnomalyScore}(X)\in \R^{N\times 1}$ denotes the point-wise anomaly criterion of $X$

---
layout: center
class: text-center
---

# Thanks!
