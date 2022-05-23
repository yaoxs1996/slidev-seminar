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

# Time Series Anomaly Detection With Association Discrepancy

Xiaoshun Yao

Data Mining Lab, UESTC

yaoxs@std.uestc.edu.cn

---

# Unsupervised Time Series Anomaly Detection

<br>
<br>
<br>
<br>
<br>

* Real-world systems monitored by multi-sensors
* Discovering the malfunctions is quite meaningful for
  + ensuring security
  + avoiding financial loss
* But anomalies are usually <font color=#0000FF>__rare__</font> and hidden by <font color=#0000FF>__vast normal points__</font>
  + data labeling hard and expensive

---

# Question Formalization

* Time series $\mathcal{X}=\{x_1,x_2,\cdots,x_N\}$
* $x_t \in \mathbb{R}^d$ represents the observation of time $t$
* Unsupervised time series anomaly detection
    + whether $x_t$ is anomalous or not without labels

<img 
  src="/imgs/multivariate-graph.png"
  class="h-70 ml-30"
/>

---

# Background

Unsupervised detection of anomaly points in time series is a challenging problem

### Current paradigms:

- Density-estimation
- Clustering-based
  - anomaly score formalized as the distance to cluster center
- Reconstruction-based
- Autoregression-based

<br>

### Drawbacks

* Focus on pointwise representation or pairwise association
* Insufficient to reason about the intricate dynamics

---

# Background

Anomalies are rare and hidden by vast normal points

* Classic methods: density-estimation, clustering-based
  + ignore <font color=#0000FF>__temporal information__</font>
  + difficult to generalize to unseen real scenarios
* Deep models: learning pointwise representations
  + rarity of anomalies: pointwise representation is less informative
  + vast normal time point: less distinguishable
  + pointwise cannot provide a comprehensive description of the <font color=#0000FF>__temporal context__</font>
* Explicit association modeling: GNNs, subsequence-based
  + single time point is insufficient for complex temporal patterns
  + cannot capture the fine-grained temporal association between time point and series

<!--
深度模型基于重构误差与自回归预测  
基于子序列到的模型无法捕获细粒度
-->

---

# Motivation

Adapt Transformer to time series anomaly detection in the unsupervised regime

<br>
<br>
<br>

* Temporal association of each time point can be obtained from the self-attention map
  + describe temporal context and indicate dynamic patterns
* Rarity of anomalies and the dominance of normal patterns
  + anomalies is hard to build strong associations with the whole series
  + the adjacent of anomalies shall contains <font color=#0000FF>__similar abnormal patterns__</font>
  + normal time points is <font color=#0000FF>__highly associate__</font> with whole series

<p class="absolute bottom-0 left-3">Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy, ICLR'22</p>

<!--
得益于Transformer在全局表征和长距离关联上的统一模型  
-->

---

# Association Discrepancy

Utilize the inherent normal-abnormal distinguishability of the association distribution

<br>

Quantified by Association Discrepancy: the distance between each time point's:

* Prior-assocication
* Series-association

<br>
<br>

### Anomaly-Attention

A two-brach self-attention models the prior-association and series-association

* Prior-association: learnable Gaussian kernel
* Series-association: self-attention weights learned form raw series
* Minimax strategy: amplify the normal-abnormal distinguishability

<!--
异常点的关联差异应该更小  
-->

---

# Anomaly Transformer

### Overall Architecture

<!-- ![overall architechture](/imgs/overall.svg  "Overall Architechture") -->
<img 
  src="/imgs/overall.svg"
  class="h-75 ml-10"
/>

* model the prior- and series-association simultaneously
* stop-gradient mechanism to constrain the prior- and series-associations

<!--
m: margin  
h: height
w: width  
https://bootstrap-vue.org/docs/reference/spacing-classes
-->

---

# Anomaly Transformer

### Overall Architechture

Stacking structure is conductive to learning underlying association

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

### Prior-association

* Learnable Gaussian kernel calculate temporal distance
* Gaussian kernel can pay more attention to adjancent horizon
* A learnable scale paramter $\sigma$ adapt to the various time series patterns

<br>
<br>

### Series-association

* Find effective association adaptive from raw series

<br>

These two forms:

* All maintain temporal dependacies
* Reflect the adjacent-concentration prior
* Learned association

---

# Anomaly-Attention

<br>
<br>

The Anomaly-Attention in the $l$-th layer:

$$
\begin{aligned}
\text{Initialization}&: Q,K,V,\sigma = X^{l-1}W_Q^l,X^{l-1}W_K^l,X^{l-1}W_V^l,X^{l-1}W_\sigma^l \\
\text{Prior-Association}&: P^l=\text{Rescale}\lparen[\frac{1}{\sqrt{2\pi}\sigma_i}\exp{-\frac{|j-i|^2}{2\sigma_i^2}}]_{i,j\in \{1,\cdots,N\}}\rparen \\
\text{Series-Association}&: S^l=\text{Softmax}(\frac{QK^T}{\sqrt{d_{model}}}) \\
\text{Reconstruction}&: \hat{Z}^l=S^lV
\end{aligned}
$$

$Q,K,V\in \R^{N\times d_{model}},\sigma \in \R^{N\times 1}$

---

# Anomaly-Attention

### Prior-association

$$
P^l=\text{Rescale}\lparen[\frac{1}{\sqrt{2\pi}\sigma_i}\exp{-\frac{|j-i|^2}{2\sigma_i^2}}]_{i,j\in \{1,\cdots,N\}}\rparen
$$

* $P^l\in\R^{N\times N}$ is generated based on $\sigma\in\R^{N\times 1}$
* $\sigma_i$ corresponds to the $i$-th time point
* $i$-th time point's association weight to $j$-th: Guassian kernel $G(|j-i|;\sigma_i)$
* $\text{Rescale}(\cdot)$ transform the association weights to discrete distributions $P^l$ by dividing the row sum

<br>

### Series-association

$$
S^l=\text{Softmax}(\frac{QK^T}{\sqrt{d_{model}}})
$$

* $S^l\in \R^{N \times N}$
* $\Softmax(cdot)$ normalizes the attention map and each row of $S^l$ forms a discrete distribution

---

# Anomaly-Attention

### Reconstruction

$$
\hat{Z}^l=S^lV
$$

* $\hat{Z}^l\in \R^{N\times d_{model}}$ is hidden representation after Anomaly-Attention in the $l$-th layer

<br>
<br>
<br>

### Multi-head version

* learned scale $\sigma\in\R^{N\times h}$ for $h$ heads
* $Q_m,K_m,V_m\in\R^{N\times \frac{d_{model}}{h}}$ of $m$-th head
* multiple heads output $\{\hat{Z}^l_m\in\R^{\frac{d_{model}}{h}}\}_{1\leq m\leq h}$ concat and get final result $\hat{Z}^l\in\R^{N\times d_{model}}$

---

# Association Discrepancy

<br>
<br>

The symmetrized KL divergence between prior- and series-associations

* Average the association discrepancy from multiple layers
* Combine the associations from multi-level features into one

$$
\text{AssDis}(P,S;X)=[\frac{1}{L}\sum^L_{l=1}(\text{KL}(P_{i,:}^l\Vert S_{i,:}^l)+\text{KL}(S_{i,:}^l\Vert P_{i,:}^l))]_{i=1,\cdots,N}
$$

* $\text{KL}(\cdot \| \cdot)$ computed between every row of $P^l$ and $S^l$
* $\text{AssDis}(P,S;X)\in \R^{N\times 1}$ is the point-wise association discrepancy of $X$
* $i$-th element of AssDis is the $i$-th time point of $X$

<!-- 异常点的AssDis分数应当小于正常时间点 -->

---

# Minimax Association Learning

<br>
<br>
<br>
<br>

Employ the reconstruction loss for optimizing model

$$
\mathcal{L}_{Total}(\hat{X},P,S,\lambda;X)=\|X-\hat{X}\|_F^2-\lambda\times \|\text{AssDis}(P,S;X)\|_1
$$

* $\hat{X}\in \R^{N\times d}$ is the reconstruction of $X$
* When $\lambda>0$, optimization is to enlarge the association discrepacy

---

# Minimax Association Learning

Minimax strategy is proposed to make the association discrepancy more distinguishable

<br>
<br>

### Minimax Strategy

* Minimize phase
  + drive the prior-association $P^l$ to approximate the series-association $S^l$
  + make $P^l$ adapt to various temporal patterns
* Maximize phase
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
  class="w-200 ml-10"
/>

<div grid="~ cols-2 gap-2" class="ml-10">
<div>
Minimize phase:

* Prior-association minimize AssDis within the distribution family derived by Gaussian kernel
</div>
<div>
Maximize phase:

* Series-association maximize AssDis under the reconstruction loss
</div>
</div>

---

# Minimax Association Learning

<br>
<br>
<br>
<br>

### Association-based Anomaly Criterion

* Incorporate the normalized association discrepancy
* Final anomaly score of $X\in \R^{N\times d}$ is

$$
\text{AnomalyScore}(X)=\text{Softmax}(-\text{AssDis}(P,S,X))\odot [\|X_{i,:}-\hat{X}_{i,:}\|_2^2]_{i=1,\cdots,N}
$$

* $\odot$ is the element-wise multiplication
* $\text{AnomalyScore}(X)\in \R^{N\times 1}$ denotes the point-wise anomaly criterion of $X$

---

# Experiments

<br>
<br>
<br>
<br>
<br>

* A non-overlapped sliding window to obtain a set of sub-series
  + a fixed size of 100 for all datasets
* Label anomalies if anomaly scores are larger than threshold $\delta$
  + $\delta$ is determined to make a proportion $r$ of time points of the validation dataset labeled as anomalies
  + $r=0.1\%$ for SWaT, $0.5\%$ for SMD and $1\%$ for other datasets
* If a time point in a certain successive abnormal segment is detected, all anomalies in this abnormal segment are viewed to be correctly detected

---

# Experiments

### Results

<img 
  src="/imgs/result.svg"
  class="w-170 ml-25"
/>

---

# Experimants

<br>
<br>
<br>
<br>

### ROC

<img 
  src="/imgs/roc.svg"
  class="w-260"
/>

---
layout: center
class: text-center
---

# Thanks!
