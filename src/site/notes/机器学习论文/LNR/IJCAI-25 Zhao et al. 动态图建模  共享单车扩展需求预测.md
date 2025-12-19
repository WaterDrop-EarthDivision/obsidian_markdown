---
{"dg-publish":true,"dg-home":true,"permalink":"//lnr/ijcai-25-zhao-et-al/","tags":["gardenEntry"],"dgPassFrontmatter":true,"created":"2025-11-12T11:55:25.508+08:00","updated":"2025-12-19T13:20:49.480+08:00"}
---


# BGM: Demand Prediction for Expanding Bike-Sharing Systems with Dynamic Graph Modeling BGM：基于动态图建模的共享单车扩展需求预测

## Abstract

Accurate demand prediction is crucial for the equitable and sustainable expansion of bike-sharing systems, which help reduce urban congestion, promote low-carbon mobility, and improve transportation access in underserved areas. However, expanding these systems presents societal challenges, particularly in ensuring fair resource distribution and operational efficiency. A major hurdle is the difficulty of demand prediction at new stations, which lack historical usage data and are heavily influenced by the existing network. Additionally, new stations dynamically reshape demand patterns across time and space, complicating efforts to balance supply and accessibility in evolving urban environments. Existing methods model relationships between new and existing stations but often assume static patterns, overlooking how new stations reshape demand dynamics over time and space. To tackle these challenges, we propose a novel demand prediction framework for expanding bikesharing systems, namely BGM, which leverages dynamic graph modeling to capture the evolving inter-station correlations while accounting for spatial and temporal heterogeneity. Specifically, we develop a knowledge transfer approach that studies the embeddings transformation across existing and new stations through a learnable orthogonal mapping matrix. We further design a gated selecting vector-based feature fusion mechanism to integrate the transferred embeddings and the intrinsic features of stations for precise predictions. Experiments on real-world bike-sharing data demonstrate that BGM outperforms existing methods.

准确的需求预测对于共享单车系统的公平和可持续扩张至关重要，这有助于减少城市拥堵，促进低碳机动性，并改善服务不足地区的交通便利。然而，扩大这些系统带来了社会挑战，特别是在确保公平的资源分配和业务效率方面。一个主要的障碍是新车站需求预测的困难，它们缺乏历史使用数据，并且受到现有网络的严重影响。。此外，新的车站动态地重塑了跨越时间和空间的需求模式，使在不断变化的城市环境中平衡供应和可达性的努力复杂化。现有的方法对新车站和现有车站之间的关系进行建模，但往往假设模式是静态的，忽略了新车站如何重塑时间和空间上的需求动态。为了应对这些挑战，我们提出了一种新的扩展共享单车系统的需求预测框架，即BGM，它利用动态图建模来捕捉不断演变的站间相关性，同时考虑了时空异质性。具体地说，我们开发了一种知识转移方法，通过一个**可学习的正交映射矩阵**来研究现有站点和新站点之间的嵌入变换。我们进一步设计了一种基于门控选择向量的特征融合机制，将传递的嵌入和站点的固有特征结合起来，以实现准确的预测。对真实世界共享单车数据的实验表明，BGM的性能优于现有方法。

> 主要改变：
> 1.之前的方法都是静态的 该论文提出的方法是 动态建模的
> 2.基于正交映射的知识迁移机制
> 3.门控选择向量驱动的特征融合策略

如何理解基于正交映射的知识迁移机制和门控选择向量驱动的特征融合策略？举例说明？ To通义千问
```blindfold
## 一、基于正交映射的知识迁移机制（Orthogonal Mapping-based Knowledge Transfer）

### 🎯 背景问题：

新设立的共享单车站点没有历史使用数据（即“冷启动”问题），但它的需求模式很可能和某些已有站点相似（比如都靠近地铁站、大学或商业区）。如何把老站点的“经验”迁移到新站点？

### 🔧 技术核心思想：

- 每个已有站点都有一个**嵌入向量**（embedding），这个向量编码了该站点的历史使用模式、周边环境等信息。
- 新站点虽然没有历史数据，但我们可以根据它的**地理位置、POI（兴趣点）、人口密度等静态特征**生成一个初始表示。
- 为了将老站点的“知识”迁移到新站点，模型学习一个**可学习的正交映射矩阵 Q**，使得：
    
    enew​=Q⋅fnew​
    
    其中：
    - fnew​ 是新站点的原始特征向量（如 POI 向量），
    - enew​ 是映射到“老站点语义空间”中的嵌入，
    - Q 是正交矩阵（满足 Q⊤Q=I）。

### ✅ 为什么用“正交”？

- 正交变换**保持向量长度和角度不变**，避免扭曲原始特征的语义结构。
- 类比：就像把一张地图从北京坐标系旋转平移到上海坐标系，但城市之间的相对位置关系不变。

### 🌰 举例说明：

假设：

- 老站点 A（靠近大学）的嵌入是 `[0.8, -0.3]`，日均借车 200 次；
- 新站点 X 也建在一所大学旁，其原始特征（如“教育类 POI 密度高”）为 `[1.0, 0.0]`。

通过训练，模型学到的正交矩阵 Q 可能将 `[1.0, 0.0]` 映射为接近 `[0.8, -0.3]` 的向量。这样，即使 X 站点刚开张，模型也能“猜”它可能像 A 站点一样高需求——这就是**知识迁移**。

---

## 二、门控选择向量驱动的特征融合策略（Gated Selecting Vector-based Fusion）

### 🎯 背景问题：

新站点的最终预测不能只靠迁移来的知识（可能不完全准确），也不能只靠本地静态特征（缺乏动态行为信息）。需要**智能融合两者**。

### 🔧 技术核心思想：

- 模型有两个信息源：
    1. **迁移嵌入** etransferred​（来自正交映射）
    2. **本地特征嵌入** elocal​（由新站点自身属性编码得到）
- 引入一个**门控选择向量** g∈[0,1]d，通过 sigmoid 函数动态决定每个维度上“更相信谁”：
    
    efinal​=g⊙etransferred​+(1−g)⊙elocal​
    
    （⊙ 表示逐元素相乘）

### 🌰 举例说明：

继续上面的例子：

- 迁移嵌入认为新站点 X 应该高需求（因为像大学旁的老站 A）→ etransferred​=[0.75,−0.25]
- 但本地特征发现：X 虽在大学旁，但**位于封闭校区内，校外无法进入** → 本地嵌入 elocal​=[−0.1,0.6]（暗示低需求）

此时，门控向量 g 可能学习为 `[0.2, 0.8]`，意味着：

- 在第1维（代表“借车活跃度”），更信任本地信息（权重 0.8 给 local）；
- 在第2维（可能代表“还车倾向”），部分参考迁移知识。

最终融合后的嵌入会更平衡，避免盲目照搬老站模式，从而做出更精准的预测。
```

## Introduction

### 简单介绍背景 提出问题和解决办法

Bike-sharing systems have gained popularity worldwide, offering short-term bike rentals through networks of stations distributed across urban areas. These systems provide an ecofriendly solution for last-mile commuting, reduce traffic congestion, and promote sustainable urban mobility [Macioszek et al., 2020]. To amplify these benefits, providers have rapidly expanded their bike-sharing networks, as seen in Figure 1(a) and Figure 1(b), which illustrate the network expansion in New York City from January 2018 to January 2024 [Mahajan and Argota S´anchez-Vaquerizo, 2024]. While network expansion amplifies the environmental and societal benefits of bike-sharing systems, it also introduces critical societal challenges. One pressing issue is ensuring equitable access to mobility across socioeconomically diverse neighborhoods, as underserved areas often face limited access to transportation infrastructure. Another challenge lies in resource allocation—determining where and when to deploy bikes and docking stations to balance supply and demand effectively. Addressing these challenges requires accurate demand prediction, particularly for new stations, to optimize operations and support the sustainable and inclusive expansion of bikesharing systems, particularly in the absence of historical data for new stations. Moreover, new stations will also change usage patterns across the network, creating complex spatiotemporal dependencies. Therefore, it is crucial to model the dynamic relationships between new and existing stations to ensure equitable access and efficient resource allocation.

自行车共享系统在全球范围内广受欢迎，通过分布在城市地区的车站网络提供短期自行车租赁。这些系统为最后一英里的通勤提供了一种生态友好的解决方案，减少了交通拥堵，促进了可持续的城市机动性[Macioszek等人，2020]。为了放大这些好处，提供商迅速扩展了他们的共享单车网络，如图1(A)和图1(B)所示，这两个图说明了2018年1月至2024年1月纽约市的网络扩展[马哈扬和阿尔戈塔·S·安切斯-瓦奎里佐，2024年]。虽然网络扩张放大了共享单车系统的环境和社会效益，但也带来了关键的社会挑战。一个紧迫的问题是确保在社会经济不同的社区公平获得交通工具，因为服务不足的地区获得交通基础设施的机会往往有限。另一个挑战在于资源分配--确定何时何地部署自行车和停靠站，以有效平衡供需。应对这些挑战需要准确的需求预测，特别是对新站点的需求预测，以优化运营并支持共享单车系统的可持续和包容性扩展，特别是在没有新站点的历史数据的情况下。此外，新的站点还将改变整个网络的使用模式，产生复杂的时空依赖关系。因此，至关重要的是要对新站点和现有站点之间的动态关系进行建模，以确保公平接入和有效的资源分配。

> ques:
> 1.如何确保在社会经济不同的社区公平获得交通工具？--> 传统就能解决
> 2.如何确定何时何地部署自行车和停靠站，以有效平衡供需？ --> 动态建模
> 3.新站点没有历史数据如何预测放在哪里？ --> 知识迁移+特征融合
> 4.新站点会改变网络关系？ -->动态建模

![Pasted image 20251112120849.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112120849.png)

### 介绍之前的方法的局限性

Various methods have been developed to address these challenges. Traditional regression-based models leverage historical data to identify demand patterns, providing a foundational understanding of user behavior [Chen et al., 2015; Liu et al., 2015]. Functional zone-based approaches improve this by incorporating diverse urban characteristics, such as population density and land use, allowing for localized and precise analyses of demand [Liu et al., 2017]. Machine learning models, such as random forests and support vector machines, further enhance prediction accuracy by integrating multiple external factors, including temporal features and points of interest [Kou and Cai, 2021]. Recently, advanced graph neural networks have been employed to capture complex spatial dependencies and mitigate data scarcity, enabling more robust and reliable predictions for new station deployment [Chen et al., 2020]. However, these methods assume the relationships between new and existing stations are static, failing to capture how demand shifts with expansion. This can lead to inefficient station deployment and inequitable service distribution, particularly in underserved areas. Addressing these challenges requires advanced models that adapt to evolving demand patterns, supporting more inclusive and data-driven urban mobility planning.

已经开发了各种方法来应对这些挑战。传统的基于回归的模型利用历史数据来识别需求模式，提供了对用户行为的基础性理解[Chen等人，2015年；Liu等人，2015年]。基于功能区的方法通过纳入不同的城市特征，如人口密度和土地使用，允许对需求进行本地化和精确的分析，从而改进了这一点[Liu等，2017]。机器学习模型，如随机森林和支持向量机，通过整合包括时间特征和兴趣点在内的多种外部因素，进一步提高预测精度[Kou和Cai，2021]。最近，先进的图形神经网络被用来捕捉复杂的空间依赖关系并缓解数据稀缺，从而能够对新站点的部署进行更稳健和可靠的预测[Chen等人，2020]。然而，这些方法假设新车站和现有车站之间的关系是静态的，未能捕捉到需求如何随着扩张而变化。这可能导致站点部署效率低下和服务分配不公平，特别是在服务不足的地区。应对这些挑战需要适应不断变化的需求模式的先进模式，支持更具包容性和数据驱动型的城市交通规划。

> 回归模型 -> 历史数据 -> 需求模式 --> 可能只是简单的对客流量的预测，什么时候哪个站点应该增多或减少
> 基于功能区的模型 -> 人口密度和土地使用 --> 进一步扩展需求模式特征，增强单个区域预测精度
> 随机森林和支持向量机 -> 整合 时间特征和兴趣点  --> 提高预测精度
> 图神经网络 --> 静态的数据特征 --> 可以对新站点预测了
 
> 什么叫做基于功能区的方法？
>“**基于功能区的方法**”（Functional Zone-based Approaches）是指在共享单车（或其他城市服务）需求预测中，**将城市划分为具有特定土地利用或社会经济功能的区域**（如商业区、住宅区、教育区、工业区等），并**利用这些功能区的特征来建模和解释用户出行需求的空间分布规律**。

### 指出方法的先进性

To tackle these challenges, we design a novel and unified Demand Prediction Framework in Bike-Sharing Systems Expansion with Dynamic Graph Modeling (BGM), which dynamically adapts to the evolving network structure and diverse contextual factors. Specifically, BGM leverages a dynamic graph architecture that continuously updates its structure and node attributes to reflect the evolving relationships between stations, such as spatial proximity, functional similarity, and temporal demand patterns. This dynamic modeling enables the framework to capture the shifting demand patterns. Building upon this, it incorporates two key modules: i) a knowledge transfer approach that studies the transformation of embeddings across existing and new stations through a learnable orthogonal mapping matrix, ensuring the adaptation and alignment of spatio-temporal patterns to dynamic conditions; and ii) a gated selecting vector-based feature fusion mechanism that selectively combines the transferred embeddings with the intrinsic features of stations, generating enriched and context-aware embeddings for precise predictions. The contributions of our work are summarized as follows:
- We present a unified dynamic graph framework that updates its structure and node attributes to capture the evolving spatial and temporal relationships in bikesharing networks.
- We propose a novel knowledge transfer approach that studies the embeddings transformation across existing and new stations by a learnable orthogonal mapping matrix to address data sparsity in newly added stations.
- We design a gated selecting vector-based feature fusion mechanism to seamlessly integrate transferred embeddings from existing stations with the intrinsic features of stations. This mechanism minimizes negative transfer while dynamically balancing contributions from both sources to ensure precise predictions.
- Extensive experiments demonstrate that our model outperforms other state-of-the-art methods on real-world public datasets (NYC’s Citi-Bike), showcasing superior prediction accuracy and robustness for new station demand predictions.

为了应对这些挑战，我们设计了一种新颖的、统一的共享单车系统扩展需求预测框架，该框架可以动态地适应不断变化的网络结构和不同的上下文因素。具体地说，BGM利用动态图体系结构不断更新其结构和节点属性，以反映站点之间不断演变的关系，如**空间邻近度、功能相似性和时间需求模式**。这种动态建模使框架能够捕获不断变化的需求模式。在此基础上，它包含了两个关键模块：i)知识转移方法，通过可学习的正交映射矩阵研究跨现有站点和新站点的嵌入的转换，确保时空模式对动态条件的适应和对齐；以及ii)基于门控选择向量的特征融合机制，其选择性地将转移的嵌入与站点的固有特征相结合，生成丰富的上下文感知嵌入以实现精确预测。我们工作的贡献概括如下：

- 我们提出了一个统一的动态图框架，该框架更新其结构和节点属性，以捕捉共享单车网络中不断变化的空间和时间关系。
- 我们提出了一种新的知识转移方法，通过一个可学习的正交映射矩阵来研究现有站点和新站点之间的嵌入变换，以解决新添加站点的数据稀疏性问题。
- 我们设计了一种基于门控选择向量的特征融合机制，将现有站点的转移嵌入与站点的固有特征无缝地结合在一起。这一机制最大限度地减少了负转移，同时动态平衡了来自两个来源的贡献，以确保准确的预测。
- 大量的实验表明，我们的模型在真实的公共数据集(纽约市的Citi-Bike)上的性能优于其他最先进的方法，表现出对新站点需求预测的卓越预测精度和稳健性。

>-> 主要创新点就俩：
>1.提出了一种知识转移方法，可以将新站点快速映射到老站点框架中
>2.提出了门控机制，让模型选择到底现有站点的转移嵌入更重要还是站点的固有特征更重要。

## Preliminaries

### Expanding Bike-Sharing System.
 
An Expanding bikesharing system consists of a set of stations V, including existing stations VA with historical data and new stations VB without historical data. A station v ∈ V is associated with a feature vector xv encoding its spatio-temporal and contextual attributes (location, Points of Interest, rental records) and external factors.

扩展的自行车共享系统由一组站点V组成，包括具有历史数据的现有站点VA和没有历史数据的新站点VB。站点V∈V与编码其时空和上下文属性(位置、兴趣点、租赁记录)和外部因素的特征向量XV相关联。

>如何理解？
```blindfold
### 原文：

> **An expanding bike-sharing system consists of a set of stations V, including existing stations VA​ with historical data and new stations VB​ without historical data.**

#### ✅ 理解：

- 整个共享单车网络用一个**站点集合 V** 表示。
- 这个集合被分成两部分：
    - **VA​**：已经运营一段时间的“老站点”，有**历史借还记录**（比如过去6个月每天每小时借了多少车）；
    - **VB​**：刚新增的“新站点”，**还没有任何使用数据**（冷启动问题）。

> 🌰 举例：  
> 到2024年1月，纽约Citi Bike共有2000个站点。其中1800个是2023年之前就有的（属于 VA​），200个是2024年1月刚新增的（属于 VB​）。

---

### 原文：

> **A station v∈V is associated with a feature vector xv​ encoding its spatio-temporal and contextual attributes (location, Points of Interest, rental records) and external factors.**

#### ✅ 理解：

- 每个站点 v（无论是老站还是新站）都对应一个**特征向量 xv​**，这个向量用来描述该站点的各种信息。
- 这些信息分为几类：

|类别|内容|是否新站可用？|
|---|---|---|
|**空间属性**|经纬度、所在区域（如是否在市中心）|✅ 是|
|**上下文属性**（Contextual）|周边POI（如地铁站、学校、商场数量）、人口密度、道路连通性|✅ 是|
|**时序/租赁记录**|过去每小时的借车数、还车数|❌ 新站没有|
|**外部因素**|天气、节假日、温度、是否工作日等|✅ 是（全局或区域级）|

> ⚠️ 注意：虽然所有站点都有 xv​，但**内容完整性不同**：
> 
> - 对于 v∈VA​（老站）：xv​ 包含**完整的历史租赁记录 + 静态特征**；
> - 对于 v∈VB​（新站）：xv​ **缺少租赁记录**，只有静态特征（位置、POI等）和外部因素。
```

### Bike Demand.

Bike demand refers to the number of rentals at a station within a given time interval and is influenced by spatial and temporal factors. To model these variations, we represent bike-sharing systems as a dynamic graph, stations are represented as nodes vi ∈ V and time steps are t. The demand at station vi at time t is denoted as dt i. The graph Gt = (Vt, Et,Wt) evolves, where edges (i, j) ∈ Et capture demand dependencies with weights wt ij reflecting spatial, proximity, and temporal correlations.

自行车需求量是指一个站点在给定时间间隔内的租赁次数，受空间和时间因素的影响。为了对这些变化进行建模，我们将共享单车系统表示为动态图，站点被表示为节点vi∈V，时间步长为t。站点vi在时间t的需求被表示为dt i。图Gt=(Vt，Et，Wt)演变，其中边(i，j)∈et捕捉需求依赖关系，其中权重wt ij反映空间、邻近和时间相关性。

> 举例说明
> 假设时间 t=周一 8:00：
>- 站点集合 V={A（住宅区）,B（地铁站）,C（写字楼）}
>- 需求：dAt​=60（大量人从A出发），dBt​=10，dCt​=5
>- 图 Gt 中：
    - 边 (A,B) 权重高（很多人从A骑到地铁站B）；
    - 边 (A,C) 权重中（部分人直接骑到公司）；
    - 边 (B,C) 权重低（地铁站和写字楼之间步行即可，很少骑车）
>
>到了 t′=周日 15:00：
>
>- 需求模式完全不同（休闲出行为主）；
>- 可能 (B,公园站) 的权重变得很高；
>- 图结构 Gt′ 也随之改变。


> 如何理解？
```blindfold
### 1. **什么是“Bike demand”？**

> **Bike demand refers to the number of rentals at a station within a given time interval and is influenced by spatial and temporal factors.**

- **单车需求**（Bike demand）：指在某个时间段内（比如早上8:00–9:00），某个站点被**借出的单车数量**（有时也包括还车数，但这里明确说是 rentals，即借车）。
- 它受两类因素影响：
    - **空间因素**（spatial）：比如站点是否靠近地铁站、办公楼、公园；
    - **时间因素**（temporal）：比如是否是工作日、早晚高峰、下雨天等。

> 🌰 举例：  
> 周一早8点，位于写字楼区的A站借出50辆车；而周日晚上8点，同一站点只借出5辆。这就是时空因素共同作用的结果。

---

### 2. **为什么用图**（graph）

> **To model these variations, we represent bike-sharing systems as a dynamic graph...**

- 共享单车网络天然具有**图结构**：
    - **站点 ↔ 节点**（nodes）
    - **站点之间的关联 ↔ 边**（edges）
- 但关键在于：这种关联**不是固定不变的**。例如：
    - 早高峰时，住宅区站点向地铁站站点“输出”大量用户；
    - 晚高峰则反向流动；
    - 新增一个站点后，可能分流附近老站的需求。
- 因此，需要用**动态图**（dynamic graph）——即图的结构或边权重随时间变化。

---

### 3. **动态图的形式化定义**

> **Stations are represented as nodes vi​∈V and time steps are t. The demand at station vi​ at time t is denoted as dit​.**

- V：所有站点的集合（包括老站和新站）；
- vi​：第 i 个站点；
- t：离散的时间步（如每小时一个时间步）；
- dit​：站点 i 在时间 t 的需求（借车数）→ **这是我们要预测的目标**。

---

### 4. **图是如何“动态演化”的**？

> **The graph Gt=(Vt,Et,Wt) evolves, where edges (i,j)∈Et capture demand dependencies with weights wijt​ reflecting spatial, proximity, and temporal correlations.**

这个公式是核心，我们拆开看：

|符号|含义|说明|
|---|---|---|
|Gt|时间 t 对应的图快照|整个系统在时刻 t 的状态|
|Vt|节点集合|通常 Vt=V（站点集合不变），但如果系统在扩张，也可能随时间增加新节点|
|Et|边集合|表示哪些站点对在时间 t 存在“需求依赖关系”|
|wijt​|边 (i,j) 的权重|衡量站点 i 和 j 在时间 t 的关联强度|

#### 🔍 权重 wijt​ 反映什么？

它综合了三类信息：

1. **空间相关性**（spatial correlation）：比如两个站点都在大学城，需求模式相似；
2. **地理邻近性**（proximity）：距离越近，用户越可能在两者之间骑行或选择其一；
3. **时间协同性**（temporal correlation）：比如两个站点总在同一时段出现高峰（如都服务于通勤）。

> 💡 注意：这些权重**可以随时间变化**！  
> 例如：
> 
> - 白天：住宅区A和地铁站B高度相关（wAB8am​ 很大）；
> - 晚上：A和附近商场C更相关（wAC7pm​ 变大）；
> - 新增站点D后，原本A→B的流量部分转向A→D，导致 wABt​ 下降，wADt​ 上升。

---

### 🧩 举个完整例子

假设时间 t=周一 8:00：

- 站点集合 V={A（住宅区）,B（地铁站）,C（写字楼）}
- 需求：dAt​=60（大量人从A出发），dBt​=10，dCt​=5
- 图 Gt 中：
    - 边 (A,B) 权重高（很多人从A骑到地铁站B）；
    - 边 (A,C) 权重中（部分人直接骑到公司）；
    - 边 (B,C) 权重低（地铁站和写字楼之间步行即可，很少骑车）

到了 t′=周日 15:00：

- 需求模式完全不同（休闲出行为主）；
- 可能 (B,公园站) 的权重变得很高；
- 图结构 Gt′ 也随之改变。
```

### Problem Formulation.

Given an expanding bike-sharing system with existing stations VA that have historical data and newly deployed stations VB, the problem of demand prediction is to accurately estimate the hourly demand for all stations in the bike-sharing network.

假设共享单车系统不断扩大，现有站点VA有历史数据，新部署站点VB，需求预测的问题是准确估计共享单车网络中所有站点的每小时需求。

需求：
- 系统正在**扩张**（expanding）：既有老站点，也有刚新增的站点。
- **VA​**：已运营一段时间的站点集合，拥有**完整的历史借还记录**（如过去几个月每小时的租车数）。
- **VB​**：最近新增的站点集合，**没有历史使用数据**（冷启动问题）。
预测目标：
    对网络中**每一个站点**（包括 VA​ 和 VB​），预测其**未来每小时的租车数量**（即需求）。

## Methodology

To predict the bike demand in expanding networks, we propose a framework that combines spatio-temporal feature encoding, dynamic graph construction, embedding transformation, and feature fusion. Illustrated in Figure 2. The following subsections provide detailed explanations of each component.

为了预测扩展网络中的自行车需求，我们提出了一种结合时空特征编码、动态图形构建、嵌入变换和特征融合的框架。下面的小节提供了每个组件的详细说明。

### Spatio-Temporal Feature Encoding 时空特征编码

To effectively represent the spatio-temporal characteristics of bike-sharing systems, we encode spatio-temporal features of stations as the inputs for the graph-based learning framework.

为了有效地表示共享单车系统的时空特征，我们将站点的时空特征编码为基于图的学习框架的输入。

#### Spatial features
Xt,spatial i include Points of Interest (POI) distributions, distance metrics, and road network connectivity, as depicted in the spatial feature extraction module in Figure 2. Edges between nodes capture interdependent relationships such as spatial proximity or functional similarity and are represented by an adjacency matrix At, which varies temporally over time to reflect dynamic changes. POIs within a fixed radius (e.g., 500 meters) are categorized and normalized into a weighted vector representing the station’s functional environment. Distance metrics, modeled using a Gaussian decay function, emphasize closer station interactions, while road network connectivity is captured as binary indicators for links to key infrastructure.

X^(t,spatial)_ I  包括兴趣点(POI)分布、距离度量和道路网络连通性，如图2中的空间特征提取模块所示。节点之间的边捕捉相互依赖的关系，如空间接近或功能相似性，并由邻接矩阵表示，该邻接矩阵随时间变化以反映动态变化。固定半径(例如，500米)内的POI被分类并归一化为表示站的功能环境的加权向量。使用高斯衰减函数建模的距离度量强调更紧密的站点交互，而道路网络连通性则被捕获为关键基础设施链接的二进制指标。

> 1.500m半径范围被分类 并且判断其中有什么环境（医院，公司等）从而进行加权
> 2.用高斯衰减函数模拟自行车，公式如下

$$ w_{ij}^{\text{dist}} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right) $$
>越近值越大，且不超过1，当d_ij超过一定范围后，值会迅速增大从而快速衰减，符合人类自行车出行方式
>3.站点是否可以直接与其它基础设施进行联通，比如地铁等

> 如何理解？为什么要这么做？
```blindfold
## 一、空间特征 Xi,spatialt​ 包含什么？

虽然写作 Xi,spatialt​，但注意：**这些特征本质上是静态的**（不随时间剧烈变化），所以“t”可能只是为了统一符号，或表示在时间 t 的上下文中使用。

### 1. **兴趣点分布**（POI distributions）

- **做法**：以站点为中心，划定一个固定半径（如500米），统计该范围内各类 POI 的数量（如地铁站1个、咖啡店3家、办公楼5栋等）。
- 然后将这些计数**归一化**（如除以总数或使用TF-IDF思想），形成一个**加权向量**，代表该站点的“功能环境”。

> 🌰 举例：  
> 站点A周围：学校×2、住宅×10 → 向量偏向“居住+教育”；  
> 站点B周围：商场×3、餐厅×8、地铁×1 → 向量偏向“商业+交通”。

- **目的**：用 POI 向量刻画站点的**城市功能语义**，帮助模型理解“这里的人可能为什么骑车”。

---

### 2. **距离度量**（Distance metrics）

- **做法**：不是简单用欧氏距离，而是用**高斯衰减函数**（Gaussian decay function）建模站点间的影响强度：
    
    $ w_{ij}^{\text{dist}} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right) $
    
    其中 dij​ 是站点 i 和 j 的距离，σ 控制衰减速度。
    
- **效果**：距离越近，权重越高；超过一定距离后影响迅速趋近于0。
    
- **目的**：反映“用户更可能在近距离站点间骑行或选择替代站点”，符合人类出行行为。
    

---

### 3. **路网连通性**（Road network connectivity）

- **做法**：检查站点是否通过道路直接连接到关键基础设施（如地铁口、公交枢纽、主干道入口），用**二值指标**（0/1）表示。

> 🌰 例如：
> 
> - 若站点出口直连地铁闸机 → 连通性 = 1；
> - 若需绕行天桥或穿过封闭小区 → 连通性 = 0。

- **目的**：捕捉“**可达性**”（accessibility）——即使两个站点地理距离近，若被河流、围墙隔断，实际交互也很弱。

---

## 二、图的边（Edges）如何构建？为什么动态？

> “Edges between nodes capture interdependent relationships such as spatial proximity or functional similarity and are represented by an adjacency matrix At, which varies temporally over time to reflect dynamic changes.”

### ✅ 边代表什么？

边不仅表示“物理接近”，还表示“**需求上的相互依赖**”，包括：

- **空间邻近性**（Spatial proximity）：距离近 → 可能互为备选；
- **功能相似性**（Functional similarity）：都是大学站 → 需求模式同步（如课间高峰）；
- **流量转移关系**：新增一站后，老站需求下降 → 存在负相关依赖。

### ✅ 为什么邻接矩阵 At 要随时间变化？

因为**站点间的关系不是固定的**！例如：

|时间|场景|邻接关系变化|
|---|---|---|
|工作日早高峰|住宅区 → 地铁站|住宅站与地铁站之间边权重高|
|周末下午|地铁站 → 公园|地铁站与公园站边权重升高|
|新增站点D|D靠近老站C|C与其他站的边权重下降，C-D边权重上升|

> 🔁 这就是“**动态图**”的核心：**邻接矩阵 At 随时间演化**，以捕捉时空上下文驱动的需求依赖变化。
```


#### Temporal features
Xt,temporal i include normalized historical demand, weather attributes, and periodic time encodings. Hourly demand data is normalized across stations to ensure consistency, while weather features, such as temperature and precipitation, are processed similarly. The final feature vector for each station at time t is defined as: 
$$ \mathbf{X}_i^t = \left[ \mathbf{X}_{i}^{t,\text{spatial}}, \, \mathbf{X}_{i}^{t,\text{temporal}}, \, \text{Taxi}_i^t \right] \tag{1} $$
where Taxit i denotes external mobility data, such as taxi trip records, associated with the station at time t.

{X}_ i^t 包括规格化历史需求、天气属性和周期性时间编码。各站点的每小时需求数据被标准化，以确保一致性，而天气特征，如温度和降雨量，也被类似地处理。每个站点在时间t的最终特征向量被定义为：
$$ \mathbf{X}_i^t = \left[ \mathbf{X}_{i}^{t,\text{spatial}}, \, \mathbf{X}_{i}^{t,\text{temporal}}, \, \text{Taxi}_i^t \right] \tag{1} $$其中Taxit i表示与时间t的站点相关联的外部机动性数据，例如出租车出行记录。

>?为什么要外部机动数据？天气特征是当天的还是之前的？还要预测之后的天气吗？周期性时间编码如何实现？

|问题|回答|
|---|---|
|**为什么用出租车数据**？|作为城市出行需求的外部代理信号，帮助模型理解“何时何地有人要移动”，尤其对新站点或异常事件至关重要。|
|**天气是哪天的**？|使用**对应预测时段的天气预报**（非历史真实值，也非不可知的未来值），在实际系统中通过天气 API 获取。|
|**周期性时间编码怎么做**？|用 **sin/cos 映射**将小时、星期等周期变量转换为连续、平滑、保距的特征，避免模型误解时间顺序。|
```blindfold
### ❓1. **为什么要引入外部机动性数据**（如出租车出行记录）？

#### ✅ 目的：**捕捉城市整体出行需求的“宏观信号”**

- 共享单车不是孤立系统，而是**城市多模式交通网络的一部分**。
- 出租车、网约车、地铁等其他交通方式的活跃程度，往往与共享单车需求**高度相关**：
    - 地铁故障 → 共享单车/出租车需求激增；
    - 大型活动结束 → 出租车和单车同时出现高峰；
    - 恶劣天气 → 出租车需求上升，单车需求下降。

#### 🌰 举例：

> 某站点附近举办演唱会，晚上10点散场。
> 
> - 历史单车数据可能从未见过这种场景（冷启动或罕见事件）；
> - 但若发现**出租车订单在该区域突然飙升**，模型可推断“此处有异常高需求”，从而提升单车调度或预测准确性。

#### 💡 技术价值：

- **弥补历史数据不足**（尤其对新站点 VB​）；
- **增强模型对突发事件的鲁棒性**；
- **实现跨模态知识迁移**：用出租车数据“告诉”单车系统“这里有人要出行”。

> 📌 注意：`Taxi_i^t` 并非指“从该站点打车”，而是**以该站点为中心一定范围内**（如500米）的出租车上下客量、订单密度等聚合指标。

---

### ❓2. **天气特征是当天的还是之前的**？需要预测未来天气吗？

#### ✅ 答案：**取决于任务设定——通常是“已知的未来天气预报”**。

在**实际部署的需求预测系统中**：

- 如果预测的是**未来几小时或明天的需求**，天气特征使用的是**气象部门提供的天气预报数据**（而非真实观测值）；
- 这些预报数据在预测时刻是**可获得的外部输入**，就像你知道“明天下午有雨”一样。

#### 📊 两种常见场景：

|预测任务|天气数据来源|是否可行|
|---|---|---|
|**回溯实验**（论文评估）|使用历史真实天气（如 NOAA 数据）|✅ 可行，用于验证模型上限|
|**在线部署**（实际系统）|使用实时天气预报 API（如 OpenWeather, AccuWeather）|✅ 可行，工业界标准做法|

#### ⚠️ 关键原则：

> **所有输入特征必须在预测时刻“已知”或“可获取”**。  
> 因此，**不能使用“未来的实际天气”**（那是数据泄露！），但可以使用“对未来天气的预测”。

> 📌 所以：天气特征是**预测时间窗口对应的预报值**，不是历史值，也不是神秘的“真实未来值”。

---

### ❓3. **周期性时间编码**（Periodic Time Encodings）如何实现？

人类出行具有强烈周期性：

- **日内周期**（Diurnal）：早晚高峰；
- **周周期**（Weekly）：工作日 vs 周末；
- **节假日效应**等。

但直接用“小时=8”、“星期=1”作为数值输入，会让模型误以为“23点和0点相差23”，而实际上它们**相邻**！

#### ✅ 解决方案：**将周期性变量映射到圆上**（sin/cos 编码）

##### 方法：对每个周期变量 t∈[0,T)，生成两个特征：

sin_feat=sin(T2πt​),cos_feat=cos(T2πt​)

##### 常见应用：

|时间变量|周期 T|编码方式|
|---|---|---|
|小时（0–23）|24|sin(2π⋅hour/24),cos(2π⋅hour/24)|
|星期几（0–6）|7|sin(2π⋅weekday/7),cos(2π⋅weekday/7)|
|年内天数|365|同理（用于季节性）|

##### 优点：

- 保留周期性：23点和0点在圆上相邻 → 距离小；
- 模型可学习任意相位的周期模式；
- 不引入虚假顺序关系。

#### 🌰 举例：

- `hour = 0` → (sin≈0, cos≈1)
- `hour = 23` → (sin≈−0.26, cos≈0.97)
- 两者在二维空间中**距离很近**，符合现实。

> 📌 这些 sin/cos 特征会作为 `X_i^{t,temporal}` 的一部分，与其他时间特征（如是否节假日、是否工作日）一起输入模型。
```

This encoding process represents each station as a node embedding that encapsulates its spatial context, temporal variations, and external mobility influences, providing a robust foundation for downstream graph-based learning tasks.
这种编码过程将每个站点表示为一个节点嵌入，封装了其空间背景、时间变化和外部流动性影响，为下游基于图的学习任务提供了可靠的基础。
{ #c3feff}


![Pasted image 20251112131113.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112131113.png)
### Dynamic Graph Construction

Dynamic graph modeling captures evolving relationships between bike-sharing stations by dynamically updating nodes and edges over time, as represented in the dynamic graph construction layer of Figure 2. Unlike static graphs, dynamic graphs adapt to real-time changes in demand patterns and external factors, ensuring relationships remain relevant. Nodes and edges, as introduced in Section 2, form the foundation of this model. Each node represents a station, with features ht i derived from Xt i as defined in Equation (1). Each edge weight at ij is computed as:
$$ a_{ij}^t = \alpha \cdot S_{ij} + \beta \cdot D_{ij} + \gamma \cdot T_{ij}^t \tag{2} $$
where Sij is the cosine similarity of POI vectors, Dij is spatial proximity based on a Gaussian decay function, and Tt ij is temporal demand similarity. Parameters α, β, and γ balance these factors. Here, i and j denote the indices of stations, and at ij captures the relationship between station i and j.

动态图建模通过随时间动态更新节点和边来捕获共享单车站点之间不断变化的关系，如图2的动态图构建层所示。与静态图不同，动态图适应需求模式和外部因素的实时变化，确保关系保持相关性。第2节中介绍的节点和边构成了此模型的基础。每个节点表示一个站点，如公式(1)中定义的那样，具有从Xt_i导出的特征Ht_i。Ij处的每个边权重计算为：
$$ a_{ij}^t = \alpha \cdot S_{ij} + \beta \cdot D_{ij} + \gamma \cdot T_{ij}^t \tag{2} $$
其中，sij是POI向量的余弦相似度，dij是基于高斯衰减函数的空间邻近度，tTij是时间需求相似度。参数α，β和γ平衡了这些因素。这里，i和j表示站的索引，并且at ij捕捉站i和j之间的关系。

![Pasted image 20251112131344.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112131344.png)

| arg   |                                                     |
| ----- | --------------------------------------------------- |
| aijt​ | 时间 t 时，站点 i 与站点 j 之间的**动态边权重**（即邻接矩阵 At 中的元素）       |
| α,β,γ | **可学习或预设的权重参数**，用于平衡三类关系的重要性（通常满足 α+β+γ=1 或通过归一化处理） |
| Sij​  | **功能相似性**：基于 POI 向量的余弦相似度（静态，不随时间变）                 |
| Dij​  | **空间邻近性**：基于高斯衰减函数的距离权重（静态或缓慢变化）                    |
| Tijt​ | **时序需求相似性**：在时间 t 附近，两站点历史需求模式的相似度（**动态变化**）        |
![Pasted image 20251112133130.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112133130.png)

At each time step, At is updated to reflect changes in demand or external factors. For example, connections between residential and business district stations strengthen during peak hours due to increased commuter trips. External factors, such as weather, further influence edge weights, dynamically adjusting relationships in real time.

在每个时间步，A^t都会更新，以反映需求或外部因素的变化。例如，由于通勤次数增加，住宅区和商业区车站之间的联系在高峰时间加强。外部因素，如天气，进一步影响边权重，实时动态调整关系。

New stations are integrated into the graph by initializing spatial features based on POI distributions, as shown in Figure 2. Edges for new stations are created based on functional similarity and spatial proximity:
$$ a_{i_{new}}^t = \alpha \cdot S_{i_{new}} + \beta \cdot D_{i_{new}} \tag{3} $$

where at inew represents the connection strength between an existing station i and a new station. Node embeddings ht i are updated using a graph convolutional network (GCN):

$$ \mathbf{h}_i^{t+1} = \sigma\left( \sum_{j \in \mathcal{N}(i)} a_{ij}^t \mathbf{W} \mathbf{h}_j^t + \mathbf{b} \right) $$

where N(i) is the set of neighbors (including i itself), at ij is the edge weight, W is a learnable weight matrix, and σ is a non-linear activation function such as ReLU. This ensures that node embeddings dynamically evolve to reflect both static and dynamic relationships.

如图2所示，通过基于POI分布初始化空间特征，将新站点整合到图中。基于功能相似性和空间邻近度创建新站点的边：
$$ a_{i_{new}}^t = \alpha \cdot S_{i_{new}} + \beta \cdot D_{i_{new}} \tag{3} $$
其中at inew表示现有站点i与新站点之间的连接强度。使用图卷积网络(Gcn)来更新节点嵌入HTi：
$$ \mathbf{h}_i^{t+1} = \sigma\left( \sum_{j \in \mathcal{N}(i)} a_{ij}^t \mathbf{W} \mathbf{h}_j^t + \mathbf{b} \right) $$
其中N(I)是邻居的集合(包括i本身)，at ij是边权重，W是可学习的权重矩阵，以及σ是诸如RELU的非线性激活函数。这确保节点嵌入动态发展以反映静态和动态关系。

> 如何理解？
```blindfold
### 一、创建新节点的过程

#### 1. **初始化空间特征**

- 新站点首先基于其周围兴趣点（POI）分布来初始化**空间特征**。
    - 例如：如果新站周围有3个学校、2个商场，则根据这些POI类型和数量生成一个特征向量。
- 这些特征帮助模型理解“这个新站的功能环境”，如是否位于商业区、居民区等。

#### 2. **建立边权重**

- 对于每个新站 inew​，需要计算它与已有站点 i 之间的**连接强度** ai,inew​​：
    
    ai,inew​​=α⋅Si,inew​​+β⋅Di,inew​​
    
    - **Si,inew​​**：功能相似性（Function Similarity），即两个站点周围的POI向量的余弦相似度；
    - **Di,inew​​**：空间邻近性（Spatial Proximity），基于高斯衰减函数的距离权重；
    - **α,β**：平衡这两个因素的系数。

> 🌟 **关键点**：通过这种方式，即使新站没有历史需求数据，也能基于其“城市角色”和地理位置与现有网络建立合理的联系。

---

### 二、公式 (4) 解释：GCN 更新节点嵌入

#### 1. **公式形式**

hit+1​=σ​j∈N(i)∑​aijt​Whjt​+b​

#### 2. **符号解释**

|符号|含义|
|---|---|
|hit+1​|节点 i 在时间步 t+1 的隐藏状态（或称嵌入表示）|
|σ(⋅)|非线性激活函数（如ReLU），增加模型表达能力|
|∑j∈N(i)​|对节点 i 的所有邻居（包括自身）求和|
|aijt​|时间 t 时从节点 j 到 i 的边权重（动态变化）|
|W|可学习的权重矩阵，用于变换邻居特征|
|hjt​|邻居节点 j 在时间 t 的隐藏状态|
|b|偏置项|
```

### Embedding Transformation

As illustrated in Figure 2, the knowledge transfer process leverages a learnable orthogonal mapping matrix to transform embeddings from existing stations (VA) to new stations (VB), facilitating the transfer of spatial and temporal knowledge across the stations. The process begins with identifying similarities between new and existing stations. For each time step t, the similarity between a new station i ∈ VB and an existing station j ∈ VA is measured using cosine similarity:

$$ S_{ij}^{\text{spatial}} = \frac{\mathbf{h}_i^{\text{spatial}} \cdot \mathbf{h}_j^{\text{spatial}}}{\|\mathbf{h}_i^{\text{spatial}}\| \, \|\mathbf{h}_j^{\text{spatial}}\|} $$
where hspatial i and hspatial j are spatial embeddings derived from POI distributions, road network connectivity, and taxi trip data. Cosine similarity is employed to capture proportional relationships in embedding spaces and adapt to dynamic temporal changes effectively. Based on St ij , the top 3 existing stations with the highest similarity are selected for each time step t. Their embeddings are aggregated as follows:
$$ \mathbf{u}_i^{B(t)} = \frac{\sum_{j \in \text{Top-3}} S_{ij}^t \, \mathbf{h}_j^A}{\sum_{j \in \text{Top-3}} S_{ij}^t} $$
To transfer and align the aggregated embeddings to the new stations, a learnable orthogonal mapping matrix X is applied:
$$ \mathbf{h}_i^{B(t),\text{trans}} = \mathbf{X} \cdot \mathbf{u}_i^{B(t)} $$

where X is optimized under the orthogonality constraint X⊤X = I. This ensures the transferred embeddings preserve structural integrity and contextual info from existing stations while adapting to the unique characteristics of new stations. The knowledge transfer approach leverages transformed embedding to capture shared patterns of new stations in the context of the existing network. By dynamically identifying and aligning spatio-temporal demand similarities, this transformation mechanism effectively mitigates the challenges posed by data sparsity and evolving station relationships.

如图2所示，知识转移过程利用可学习的正交映射矩阵来将嵌入从现有站(VA)转换到新站(VB)，从而促进跨站的空间和时间知识的转移。这一过程首先确定新站和现有站之间的相似之处。对于每个时间步长t，使用余弦相似度来测量新站i∈Vb和现有站j∈Va之间的相似性：
$$ S_{ij}^{\text{spatial}} = \frac{\mathbf{h}_i^{\text{spatial}} \cdot \mathbf{h}_j^{\text{spatial}}}{\|\mathbf{h}_i^{\text{spatial}}\| \, \|\mathbf{h}_j^{\text{spatial}}\|} $$
其中，h^spatial_i和h^spatial_j是从POI分布、道路网络连通性和出租车出行数据导出的空间嵌入。利用余弦相似性来捕捉嵌入空间中的比例关系，并有效地适应动态的时间变化。基于ST ij，为每个时间点t选择相似度最高的前3个现有站点。它们的嵌入聚合如下：
$$ \mathbf{u}_i^{B(t)} = \frac{\sum_{j \in \text{Top-3}} S_{ij}^t \, \mathbf{h}_j^A}{\sum_{j \in \text{Top-3}} S_{ij}^t} $$

为了将聚集的嵌入传输和对准到新站，应用可学习的正交映射矩阵X：
$$ \mathbf{h}_i^{B(t),\text{trans}} = \mathbf{X} \cdot \mathbf{u}_i^{B(t)} $$
其中X是在正交性约束X⊤X=I下优化的。这确保了所传输的嵌入保持来自现有站的结构完整性和上下文信息，同时适应新站的独特特征。
该知识转移方法利用变换的嵌入来捕获现有网络环境中的新站点的共享模式。通过动态识别和调整时空需求相似性，该转换机制有效地缓解了数据稀疏和站点关系演变带来的挑战。

>详细过程
```blindfold
![Pasted image 20251112140535.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112140535.png)
```


> 为什么要做正交变换？
> **正交映射是一种“只旋转、不扭曲”的线性变换，它确保从老站迁移过来的知识，在新站的表示中保持原有的几何关系和语义结构，从而提升冷启动性能和模型稳定性**。
```blindfold
![Pasted image 20251112140833.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112140833.png)
```

### Feature Fusion for Expanded Stations

As illustrated in Figure 2, the feature fusion mechanism integrates transferred embeddings, existing features, and temporal dependencies to generate representations for new stations. This process dynamically balances contributions from different feature sources, ensuring the final representation adapts to both spatial and temporal heterogeneity.
The fused feature ft i dynamically balances transferred embeddings (ui B(t)) and original features (hi B) through a gating mechanism. It is computed as:
$$ \mathbf{f}_i^t = \mathbf{g}_i^t \odot \mathbf{u}_i^{B(t)} + (1 - \mathbf{g}_i^t) \odot \mathbf{h}_i^B \tag{8} $$
where gt i ∈ Rd is the gating vector at time t, ⊙ denotes element-wise multiplication, and hi B includes spatial features such as POIs, road networks, and taxi trip records. The gating vector gt i adjusts the balance between transferred embeddings and original features based on their joint features. It is computed as:
$$ \mathbf{g}_i^t = \sigma\left( \mathbf{W}_g \left[ \mathbf{h}_i^B; \, \mathbf{u}_i^{B(t)} \right] + \mathbf{b}_g \right) \tag{9} $$

where [hi B; ui B(t)] represents the concatenation of the two feature vectors, Wg is a learnable weight matrix, bg is a bias term, and σ is the sigmoid activation function. A temporal attention mechanism dynamically assigns varying weights to features from different time steps to effectively capture temporal dependencies:
$$ \alpha_t = \frac{\exp(\mathbf{q}_t \cdot \mathbf{k}_t)}{\sum_{t' \in T} \exp(\mathbf{q}_{t'} \cdot \mathbf{k}_{t'})} $$
where qt and kt are the query and key vectors derived from temporal embeddings, and T denotes the set of time steps. The final fused feature integrates spatial, temporal, and dynamic relationships, ensuring robustness in predicting demand at stations. The fused feature ft i serves as the input to the demand prediction module, combining transferred knowledge (ui B(t)) and station-specific features (hi B). This representation effectively addresses data sparsity while preserving station-specific characteristics.


如图2所示，特征融合机制集成了传输嵌入、现有特征和时间依赖关系，以生成新站点的表示。该过程动态地平衡来自不同要素来源的贡献，确保最终的表示适应空间和时间的异质性。
融合特征ft i通过选通机制动态地平衡传送的嵌入(ui B(T))和原始特征(Hi B)。它的计算公式为：
$$ \mathbf{f}_i^t = \mathbf{g}_i^t \odot \mathbf{u}_i^{B(t)} + (1 - \mathbf{g}_i^t) \odot \mathbf{h}_i^B \tag{8} $$

其中GT i∈Rd是时间t处的选通向量，⊙表示逐元素乘法，而hi B包括诸如POI、道路网络和出租车行程记录的空间特征。
门控向量GT i基于传递的嵌入和原始特征的联合特征来调整它们之间的平衡。它的计算公式为：
$$ \mathbf{g}_i^t = \sigma\left( \mathbf{W}_g \left[ \mathbf{h}_i^B; \, \mathbf{u}_i^{B(t)} \right] + \mathbf{b}_g \right) \tag{9} $$
其中[hi B；ui B(T)]表示两个特征向量的级联，Wg是可学习的权重矩阵，Bg是偏置项，σ是Sigmoid激活函数。时间注意机制动态地向来自不同时间步长的特征分配不同的权重，以有效地捕获时间依赖性：
$$ \alpha_t = \frac{\exp(\mathbf{q}_t \cdot \mathbf{k}_t)}{\sum_{t' \in T} \exp(\mathbf{q}_{t'} \cdot \mathbf{k}_{t'})} $$
其中，qt和kt是从时间嵌入中导出的查询和关键字向量，并且T表示时间步长集合。
最终的融合特征集成了空间、时间和动态关系，确保了对车站需求预测的健壮性。融合后的特征ft i用作需求预测模块的输入，将传送的知识(ui B(T))和车站特定特征(Hi B)组合在一起。这种表示有效地解决了数据稀疏性问题，同时保留了站点特定的特征。
![Pasted image 20251112141519.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112141519.png)

> 这里主要进行门控筛选
> $g_i^t$ 是 传递的嵌入（3个老站点的嵌入）和原始特征（新站点的特征）concat后 --> MLP --> $g_i^t$ 
> 然后通过$g_i^t$ 对传递的特征和原始特征进行门控筛选 判断哪个更重要形成$f^t_i$
> 之后对之前的时间进行查询从而获得时间依赖性

> ?但是新站点哪来的之前的时间序列呢？
> 
<div class="transclusion internal-embed is-loaded"><a class="markdown-embed-link" href="//lnr/ijcai-25-zhao-et-al/#c3feff" aria-label="Open link"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="svg-icon lucide-link"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg></a><div class="markdown-embed">



This encoding process represents each station as a node embedding that encapsulates its spatial context, temporal variations, and external mobility influences, providing a robust foundation for downstream graph-based learning tasks.
这种编码过程将每个站点表示为一个节点嵌入，封装了其空间背景、时间变化和外部流动性影响，为下游基于图的学习任务提供了可靠的基础。 

</div></div>

> 所以实际上使用的是老节点的时间变化信息

```blindfold
![Pasted image 20251112142044.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112142044.png)
```


### Loss Function

The total loss function consists of spatial alignment loss Ls, temporal alignment loss Lt, and prediction loss Lp. The spatial alignment loss Ls aligns embeddings with similar demand patterns using a contrastive approach:
$$ \mathcal{L}_s = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} q_{ik} \log \frac{\exp(\mathbf{z}_i^t \cdot \mathbf{h}_k / \tau)}{\sum_{j=1}^{K} \exp(\mathbf{z}_i^t \cdot \mathbf{h}_j / \tau)} \tag{11} $$
where zt i is the node embedding of station i, hk is a prototype vector, and τ is a temperature parameter. The temporal alignment loss Lt ensures consistency across consecutive time steps:
$$ \mathcal{L}_t = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma(\mathbf{z}_i^t \cdot \mathbf{z}_i^{t+1}) \tag{12} $$

where σ is the sigmoid function. 
The prediction loss Lp minimizes the mean squared error (MSE) between predicted and actual demand:
$$ \mathcal{L}_p = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{y}_i^t - \hat{\mathbf{y}}_i^t\|_2 \tag{13} $$
where yt i is the actual demand and ˆyt i is the predicted demand.

总损失函数由空间对准损失Ls、时间对准损失Lt和预测损失Lp组成。
空间线形损失LS采用对比的方法将埋设与相似的需求模式对齐：
$$ \mathcal{L}_s = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} q_{ik} \log \frac{\exp(\mathbf{z}_i^t \cdot \mathbf{h}_k / \tau)}{\sum_{j=1}^{K} \exp(\mathbf{z}_i^t \cdot \mathbf{h}_j / \tau)} \tag{11} $$
其中zt i是站点i的节点埋入，Hk是原型向量，τ是温度参数。
时间对准损失Lt确保连续时间步长上的一致性：
$$ \mathcal{L}_t = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma(\mathbf{z}_i^t \cdot \mathbf{z}_i^{t+1}) \tag{12} $$
其中σ是Sigmoid函数。
预测损失Lp使预测需求和实际需求之间的均方误差最小化：
$$ \mathcal{L}_p = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{y}_i^t - \hat{\mathbf{y}}_i^t\|_2 \tag{13} $$
其中yt i是实际需求，ˆyt i是预测需求。

> 三个损失：
> 1.空间 --> 相似的空间嵌入应该相似
> 2.时间 --> 相似的时间时间应该相似
> 3.预测 --> 预测值和真实值应该相似

```blindfold
![Pasted image 20251112144945.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112144945.png)
![Pasted image 20251112145526.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112145526.png)
```

## Experiments

In this section, we present the implementation details and dataset descriptions, followed by the evaluation metrics and baseline methods used for comparison. We then conduct extensive experiments to assess the performance of BGM, including comparisons with baseline models, ablation studies, and robustness analysis under different expansion scenarios.

在本部分中，我们将介绍实施细节和数据集描述，然后介绍用于比较的评估指标和基线方法。然后，我们进行了大量的实验来评估BGM的性能，包括与基线模型的比较、烧蚀研究以及在不同扩展场景下的稳健性分析。

### Implementation Details
Training. The training process involves three steps: (i) computing embeddings using the dynamic graph, (ii) refining embeddings with spatial and temporal alignment losses, and (iii) predicting demand using a multi-layer perceptron (MLP). The spatial alignment loss (Ls) encourages stations with similar demand to share representations, while the temporal alignment loss (Lt) ensures consistency across consecutive time steps. The refined embeddings are then fed into an MLP for demand prediction at t + 1, represented as xˆ t+1,n = MLP(hn), where hn is the station embedding. The overall training objective is given by Ljoint = Lp + Ls + Lt. The model is trained using the Adam optimizer, with a batch size of 64, the learning rate of 1 × 10−3, and 200 epochs with early stopping, running on an NVIDIA RTX 4090 GPU. The model implementation is available at: https://github.com/ YixuanColt/BSS-BGM.git.

训练。训练过程包括三个步骤：
(I)使用动态图计算嵌入，
(Ii)在空间和时间对齐损失的情况下细化嵌入，
(Iii)使用多层感知器(MLP)预测需求。
空间对齐损失(LS)鼓励具有相似需求的站点共享表示，而时间对齐损失(LT)确保连续时间步长的一致性。然后将改进后的嵌入值送入t+1处的最大似然比用于需求预测，表示为xˆt+1，n=mLP(Hn)，其中Hn为站点嵌入值。
模型采用ADAM优化器进行训练，训练批大小为，学习速率为1×10−3，提前停止时间为200个历元，运行在NVIDIA RTX4090GPU上。模型实现可在以下网址获得：https://github.com/yi xuanColt/bss-bgm.git。

> optim:Adam
> batch_size:64
> lr:1e-3
> eapoch:200 with earlystop
> github:https://github.com/yixuanColt/bss-bgm.git
##### code
```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataloader import DemandDataset, load_data_from_csv, preprocess_features, preprocess_targets
from lib.logger import ExperimentLogger
from lib.metrics import torch_rmse, torch_mae, torch_evaluate_metrics
from lib.utils import set_seed
from model.DemandPredictionModel import DemandPredictionModel
import numpy as np


# Configuration
config = {
    "seed": 42,
    "learning_rate": 0.001,
    "batch_size": 8,
    "num_epochs": 20,
    "input_dim": 16,
    "hidden_dim": 32,
    "time_steps": 4,
    "spatial_decay": 0.1,
    "temporal_decay": 0.1,
    "log_dir": "./logs"
}

# Set random seed for reproducibility
set_seed(config["seed"])

# Initialize logger
logger = ExperimentLogger(config["log_dir"])
logger.log_config(config)

# Load and preprocess data
print("Loading data...")
# **新站点（或所有待预测站点）的固有静态特征**
intrinsic_features = np.random.rand(50, config["input_dim"])  # Replace with actual features
# **已有站点（source stations）的嵌入或代表性特征**，用于知识迁移。
existing_features = np.random.rand(5, config["input_dim"])  # Replace with actual features
# **每个站点过去 T 个时间步的动态特征**（temporal covariates）
temporal_features = np.random.rand(50, config["time_steps"], config["input_dim"])  # Replace with actual temporal features
# **新站点与已有站点之间的空间相似性或距离度量**。
spatial_distances = np.random.rand(50, 5)  # Replace with actual spatial distances
# **新站点与已有站点在时间动态上的相似程度**。
temporal_similarities = np.random.rand(50, 5)  # Replace with actual temporal similarities
# **每个新站点在未来某个时间步（如 t+1）的真实需求值**
targets = np.random.rand(50, 1)  # Replace with actual demand values

# Create Dataset and DataLoader
dataset = DemandDataset(
    intrinsic_features=torch.tensor(intrinsic_features, dtype=torch.float32),
    existing_features=torch.tensor(existing_features, dtype=torch.float32),
    temporal_features=torch.tensor(temporal_features, dtype=torch.float32),
    spatial_distances=torch.tensor(spatial_distances, dtype=torch.float32),
    temporal_similarities=torch.tensor(temporal_similarities, dtype=torch.float32),
    targets=torch.tensor(targets, dtype=torch.float32)
)
train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize model
model = DemandPredictionModel(
    num_nodes=existing_features.shape[0],
    input_dim=config["input_dim"],
    hidden_dim=config["hidden_dim"],
    time_steps=config["time_steps"],
    spatial_decay=config["spatial_decay"],
    temporal_decay=config["temporal_decay"]
)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Define loss function
def compute_loss(predictions, targets):
    mse_loss = torch.mean((predictions - targets) ** 2)  # MSE
    mae_loss = torch.mean(torch.abs(predictions - targets))  # MAE
    return mse_loss + 0.5 * mae_loss  # Weighted loss

# Training loop
print("Starting training...")
for epoch in range(config["num_epochs"]):
    model.train()
    total_loss = 0

    for batch in train_loader:
        intrinsic_features = batch["intrinsic_features"]
        existing_features = batch["existing_features"]
        temporal_features = batch["temporal_features"]
        spatial_distances = batch["spatial_distances"]
        temporal_similarities = batch["temporal_similarities"]
        targets = batch["targets"]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(
            intrinsic_features,
            existing_features,
            temporal_features,
            spatial_distances,
            temporal_similarities
        )

        # Compute loss
        loss = compute_loss(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Log metrics
    logger.log_metrics(epoch + 1, {"Loss": total_loss / len(train_loader)})
    print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
print("Evaluating model...")
model.eval()
all_predictions, all_targets = [], []

with torch.no_grad():
    for batch in train_loader:
        predictions = model(
            batch["intrinsic_features"],
            batch["existing_features"],
            batch["temporal_features"],
            batch["spatial_distances"],
            batch["temporal_similarities"]
        )
        all_predictions.append(predictions.numpy())
        all_targets.append(batch["targets"].numpy())

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

# Compute evaluation metrics
metrics = torch_evaluate_metrics(torch.tensor(all_predictions), torch.tensor(all_targets))
print(f"RMSE: {metrics['RMSE'].item():.4f}, MAE: {metrics['MAE'].item():.4f}")

# Log final metrics
logger.log_metrics("Final Evaluation", {"RMSE": metrics["RMSE"].item(), "MAE": metrics["MAE"].item()})
```

### Data Description

To evaluate our model, we collected five datasets 1, integrating bike-sharing operational data with meteorology, POI, road network, and NYC taxi trip records. While these sources collectively provide essential input features for model training, our performance evaluation primarily focuses on bikesharing operational data from NYC Citi-Bike and Chicago Divvy-Bike, as presented in Table 2. Bike-Sharing Data. The dataset includes operational records from the NYC Citi-Bike sharing systems over one month (January 1 to 31, 2018). The NYC Citi-Bike data consists of two primary components: i) trip records for rentals and returns at each station, and ii) station-level details, e.g., spatial and temporal information. Meteorology Data. The meteorological data was collected from the NYC Mesowest database, providing historical daily weather records. Each record describes the weather conditions, categorized into four main types: sunny, overcast, rainy/snowy, and extreme weather conditions. POI & Road Network Data. The POI data were collected from online mapping service providers in NYC (Google Place API). Table 1 shows the details. Besides, road network data was collected from NYC Open Data for further analysis. NYC-Taxi. The dataset contains detailed trip records of yellow taxis in New York City. Each trip record includes the pick-up and drop-off times and the corresponding latitude and longitude of these locations.

为了评估我们的模型，我们收集了五个数据集1，将共享单车运营数据与气象、POI、公路网和纽约市出租车出行记录整合在一起。虽然这些来源共同为模型培训提供了基本的输入功能，但我们的绩效评估主要关注来自NYC Citi-Bike和Chicago Divvy-Bike的共享单车运营数据，如表2所示。
共享单车数据。该数据集包括纽约市Citi-Bike共享系统一个月(2018年1月1日至31日)的运营记录。
纽约市Citi-Bike数据由两个主要部分组成：i)每个站点的租金和回报的出行记录，以及ii)站点级别的详细信息，例如空间和时间信息。
气象数据。气象数据收集自NYC Mesowest数据库，提供历史每日天气记录。每个记录都描述了天气状况，分为四种主要类型：晴天、阴天、雨雪和极端天气状况。
POI和道路网数据。POI数据是从纽约市的在线地图服务提供商(Google Place API)收集的。表1显示了详细信息。
此外，道路网数据是从NYC Open Data收集的，以供进一步分析。纽约出租车。该数据集包含纽约市黄色出租车的详细行程记录。每一次出行记录都包括接送时间以及这些地点对应的纬度和经度。


![Pasted image 20251112150455.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112150455.png)![Pasted image 20251112150539.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112150539.png)

###  Evaluation Metrics and Compared Methods Evaluation metrics.

There are two performance metrics employed for evaluating our model, the Root Mean Square Error (RMSE) and Mean Absolute Error (MAE). These metrics collectively provide a robust framework to assess our model’s capability to capture subtle variations in the data. Compared methods. We select seven methods for comparison in this study. These methods are described as follows:
Linear Regression (LR) [Singhvi et al., 2015]: Predicts bike demand using taxi, weather, and spatial features with neighborhood-level aggregation. 
• Spatial Regression (SR) [Faghih-Imani and Eluru, 2016]: Models spatial lag and error structures to capture spatio-temporal dependencies in demand. 
• Function Zone (FZ) [Liu et al., 2017]: Clusters stations into functional zones and predicts zone-level transitions using Random Forest and ridge regression. 
• GraphSAGE (SAGE) [Hamilton et al., 2017]: Aggregates neighborhood features inductively, enabling generalization to unseen stations. 
• IGNNK (N2K) [Wu et al., 2020]: Learns spatial message passing via subgraph reconstruction to interpolate unsampled nodes and generalize across dynamic graphs.
• TrafficStream (TS) [Chen et al., 2021]: Integrates continual learning with spatio-temporal graph networks for adaptive model updating over dynamic traffic data.
• Spatial-MGAT(MGAT) [Liang et al., 2023a]: Captures spatial dependencies through multi-graph attention on proximity and built environment similarity.

有两个性能指标用于评估我们的模型，均方根误差(RMSE)和平均绝对误差(MAE)。这些指标共同提供了一个健壮的框架来评估我们的模型捕获数据中细微变化的能力。比较研究方法。在这项研究中，我们选择了七种方法进行比较。这些方法如下所述：
- 线性回归(LR)[Singhvi等人，2015]：使用出租车、天气和空间特征以及邻居级聚合来预测自行车需求。
- 空间回归(SR)[Faghih-Imani和Eluru，2016]：对空间滞后和误差结构进行建模，以捕捉需求中的时空依赖关系。
- 功能区(FZ)[Liu等人，2017]：将站点划分为功能区，并使用随机森林和岭回归预测区域级别的过渡。
- GraphSAGE(SAGE)[Hamilton等人，2017]：对社区要素进行归纳聚合，将其推广到看不见的站点。
- IGNNK(N2K)[Wu等人，2020]：通过子图重构学习空间消息传递，以内插未采样节点并在动态图中泛化。
- TrafficStream(TS)[Chen等人，2021]：将连续学习与时空图网络相结合，以实现动态交通数据上的自适应模型更新。
- Space-MGAT(MGAT)[梁等人，2023a]：通过关注邻近和已建环境相似性的多图捕获空间依赖关系。

### Comparison with Baselines

The comparative analysis of demand prediction models for bike-sharing system expansion, using data from NYC CitiBike and Chicago Divvy-Bike, highlights the superior performance of advanced spatio-temporal models over traditional baselines. As shown in Table 2, traditional models like linear regression and spatial regression exhibit higher RMSE and MAE values at both existing and new stations, indicating limitations in capturing complex and dynamic bike-sharing demand patterns. The Function Zone model performs better by clustering stations based on spatial features, reflecting the importance of functional heterogeneity in urban demand. GraphSAGE improves upon traditional methods by enabling inductive learning over unseen stations, but limited temporal modeling yields suboptimal results. IGNNK enhances prediction accuracy via subgraph reconstruction and spatial message passing, showing strong performance especially at new stations. TrafficStream integrates continual learning with spatio-temporal graph networks, offering robust results across dynamic settings but limited transferability to newly added stations. Spatial-MGAT further advances performance by capturing fine-grained spatial dependencies through multigraph attention, especially under sparse training data. Overall, while each method contributes uniquely, none achieves the consistent accuracy and adaptability demonstrated by BGM across all scenarios.

使用NYC CitiBike和Chicago Divvy-Bike的数据，对共享单车系统扩张的需求预测模型进行了比较分析，突显了先进的时空模型相对于传统基线的优越性能。如表2所示，线性回归和空间回归等传统模型在现有站点和新站点都显示出较高的RMSE和MAE值，这表明在捕捉复杂和动态的共享单车需求模式方面存在局限性。功能区模型通过基于空间特征的站点聚类来表现得更好，反映了功能异质性在城市需求中的重要性。GraphSAGE通过允许在看不见的站点上进行归纳学习来改进传统方法，但有限的时间建模产生的结果不是最优的。IGNNK通过子图重建和空间信息传递来提高预测精度，在新站点上表现出很强的性能。TrafficStream将持续学习与时空图形网络相结合，在动态设置中提供稳健的结果，但对新添加的站点的可传输性有限。Space-MGAT通过多图注意力捕获细粒度的空间依赖关系，进一步提高了性能，特别是在稀疏训练数据的情况下。总体而言，虽然每种方法都有独特的贡献，但没有一种方法在所有情况下都能达到BGM所展示的一致的准确性和适应性。

The proposed BGM framework demonstrates the best overall performance, achieving the lowest RMSE and MAE values across all NYC and Chicago datasets. Its ability to dynamically model evolving network structures, integrate spatio-temporal interactions, and facilitate knowledge transfer between existing and new stations makes it a robust solution for enhancing transportation accessibility, optimizing bike-sharing resource allocation, and supporting sustainable urban growth. As shown in Table 2, BGM highlights the importance of embedding transformation, feature fusion, and dynamic graph modeling in designing intelligent, fair, and scalable urban mobility systems. By improving demand prediction, BGM supports equitable infrastructure expansion and better decision making for urban planners, addressing the broader societal goal of inclusive, data-driven, and futureready transportation development.

建议的BGM框架展示了最佳的整体性能，在所有纽约和芝加哥数据集中实现了最低的RMSE和MAE值。它能够动态模拟不断演变的网络结构，整合时空互动，并促进现有站点和新站点之间的知识转移，使其成为提高交通可达性、优化共享单车资源配置和支持城市可持续发展的强大解决方案。如表2所示，BGM强调了嵌入变换、功能融合和动态图形建模在设计智能、公平和可扩展的城市移动系统中的重要性。通过改进需求预测，BGM支持城市规划者公平的基础设施扩张和更好的决策，满足包容性、数据驱动型和未来就绪交通发展的更广泛社会目标。

### Ablation Studies

In this section, we conduct ablation studies to evaluate the contributions of key components in the BGM model by testing three variants: W/O DM (without dynamic graph modeling), W/O ET (without feature embedding transformation), and W/O FF (without feature fusion). By systematically removing each component, we analyze their impact on model performance. As shown in Figure 3(a) and Figure 3(b), BGM achieves the lowest RMSE and MAE values across all datasets, with a 13.8% improvement over W/O ET, confirming the critical role of embedding transformation in demand prediction. Among the variants, W/O ET exhibits the most significant performance drop, underscoring the crucial role of embedding transformation in refining feature representations and enhancing predictive accuracy. The W/O DM and W/O FF variants also show declines, highlighting the importance of dynamic graph modeling and feature fusion in capturing spatial-temporal dependencies and optimizing prediction performance. These results confirm that all three components are essential for maintaining robustness and precision.

在这一部分中，我们通过测试三种变量来评估BGM模型中关键组件的贡献：W/O DM(不带动态图建模)、W/O ET(不带特征嵌入变换)和W/O FF(不带特征融合)。通过系统地移除每个组件，分析了它们对模型性能的影响。如图3(A)和图3(B)所示，BGM在所有数据集中实现了最低的RMSE和MAE值，比W/O ET提高了13.8%，证实了嵌入转换在需求预测中的关键作用。在这些变体中，W/O ET表现出最显著的性能下降，这突显了嵌入变换在改进特征表示和提高预测精度方面的关键作用。W/O DM和W/O FF变量也出现下降，突显了动态图形建模和特征融合在捕获时空相关性和优化预测性能方面的重要性。这些结果证实，所有这三个组件对于保持稳健性和精确度都是必不可少的。

![Pasted image 20251112151012.png](/img/user/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87/LNR/Pasted%20image%2020251112151012.png)

### Robustness Analysis

To evaluate the performance and robustness of the BGM model under different expansion scenarios, we designed two experiments focusing on growth rates and expansion patterns. This analysis not only evaluates how the model and baseline methods perform under different network growth speeds but also addresses a critical societal challenge: ensuring the scalability and sustainability of bike-sharing systems in the face of rapid urbanization. The first experiment investigates the impact of varying growth rates while maintaining a fixed expansion pattern. As shown in Figure 3(c) and Figure 3(d), BGM consistently outperforms all baselines across existing and new stations. For instance, BGM achieves an RMSE of 1.78 and an MAE of 0.90 at a 15% growth rate. Even at 30% growth, BGM maintains a low RMSE of 1.75 and 0.87, outperforming all baselines. These results confirm that BGM adapts effectively to varying expansion rates, ensuring stable demand predictions in dynamic urban environments.

为了评估BGM模型在不同扩展场景下的性能和稳健性，我们设计了两个专注于增长率和扩展模式的实验。这一分析不仅评估了模型和基线方法在不同网络增长速度下的表现，还解决了一个关键的社会挑战：在快速城市化的情况下确保共享单车系统的可扩展性和可持续性。
第一个实验调查了在保持固定扩张模式的同时，不同增长率的影响。如图3(C)和图3(D)所示，BGM在现有站点和新站点上的表现始终优于所有基线。例如，BGM在15%的增长率下实现了1.78的RMSE和0.90的MAE。即使增长30%，BGM仍保持1.75和0.87的低RMSE，表现优于所有基线。这些结果证实了BGM有效地适应了不同的扩张速度，确保了在动态的城市环境中稳定的需求预测。

The second experiment examines the impact of different expansion patterns, directly addressing a key societal challenge: ensuring equitable access to bike-sharing services across diverse urban areas. Two patterns are considered: localized expansion, as shown in Figure 3(e), where new stations are distributed near existing networks to enhance connectivity, and regional expansion, as shown in Figure 3(f), where new stations are introduced in disconnected areas to improve accessibility in underserved regions. Results in Figures 3(g) and 3(h) show that BGM achieves the lowest RMSE and MAE in both scenarios, demonstrating robustness. Specifically, for localized expansion, BGM records an RMSE of 1.51 and an MAE of 0.83, outperforming the second-best model. It still performs well under regional expansion, with RMSE 1.73 and MAE 1.10, where historical data is scarce. These findings highlight the model’s ability to support equitable and efficient mobility resource distribution, advancing broader goals of urban sustainability and inclusion.

第二个实验考察了不同扩张模式的影响，直接解决了一个关键的社会挑战：确保在不同的城市地区公平获得共享单车服务。考虑了两种模式：局部扩展，如图3(E)所示，新站点分布在现有网络附近，以加强连通性；区域扩展，如图3(F)所示，在不相连的地区引入新站点，以改善服务不足区域的可达性。图3(G)和图3(H)中的结果表明，BGM在这两种情况下都实现了最低的均方根误差和最小均方误差，显示了稳健性。具体地说，就本地化扩张而言，BGM的RMSE为1.51，MAE为0.83，表现优于第二好的型号。它在地区扩张下的表现仍然很好，RMSE为1.73，MAE为1.10，这两个地方的历史数据都很稀缺。这些发现突显了该模型支持公平和高效的流动资源分配的能力，推进了城市可持续性和包容性的更广泛目标。

## Related Work

Demand Prediction for Bike-Sharing Systems. In recent years, bike-sharing demand prediction has evolved significantly. Early studies using time-series models like ARIMA and SARIMA captured temporal patterns but neglected spatial dynamics [Li et al., 2019]. Cluster-based methods improved short-term forecasts by grouping stations with similar demands, incorporating geographic and historical features [Chen et al., 2016; Lahoorpoor et al., 2019]. Station-level models integrated external data such as weather and events but faced challenges at data-sparse locations [Yu et al., 2023]. Advanced approaches, including spatio-temporal graph convolutional networks (STGCNs), effectively modeled spatialtemporal dependencies [Xiao et al., 2021; Ma et al., 2022; Tang et al., 2021], while TrafficStream introduced streaming GNN frameworks with continual learning for large-scale forecasting [Chen et al., 2021]. Reinforcement learning optimized operations in real-time settings [Demizu et al., 2023].

共享单车系统需求预测。近年来，共享单车需求预测有了显著的演变。使用ARIMA和SARIMA等时间序列模型的早期研究捕捉了时间模式，但忽略了空间动力学[Li等人，2019年]。基于分组的方法通过将具有类似需求的站点分组，纳入地理和历史特征，改进了短期预测[Chen等人，2016；Lahoorprep等人，2019年]。站级模型集成了天气和事件等外部数据，但在数据稀疏的位置面临挑战[Yu等人，2023]。包括时空图卷积网络(STGCN)在内的先进方法有效地对时空依赖进行了建模[肖等人，2021；Ma等人，2022；Tang等人，2021]，而TrafficStream引入了具有连续学习的流式GNN框架用于大规模预测[Chen等人，2021]。强化学习在实时环境中优化操作[Demizu等人，2023]。

Bike-Sharing Systems Expansion. There is a substantial body of research dedicated to modeling the expansion of bike-sharing systems, such as planning optimal locations for new stations [Li and Zheng, 2020; Liang et al., 2024] or enhancing the capacity of existing stations [Liang et al., 2023b; Liang et al., 2023a]. However, much of this existing work assumes that the demand patterns for new stations will closely resemble those of existing stations, which differs significantly from our proposed approach. For example, the method in [Liu et al., 2017] proposed a zone-based hierarchical demand model to estimate average demand at newly added stations during different stages of expansion.

共享单车系统的扩张。有大量的研究致力于对共享单车系统的扩张进行建模，例如为新站点规划最优位置[Li和郑，2020；梁等人，2024]或增强现有站点的能力[梁等人，2023b；梁等人，2023a]。然而，这些现有工作大多假设新车站的需求模式将与现有车站的需求模式非常相似，这与我们建议的方法有很大不同。例如，[Liu et al.，2017]中的方法提出了基于区域的分层需求模型，以估计不同扩张阶段新增车站的平均需求。

Dynamic Graph Modeling. Spatio-temporal graph networks capture dynamic spatial and temporal station relationships. RNN-based methods [Yu et al., 2017; Kapoor et al., 2020; Roy et al., 2021; Ghosh et al., 2020] use recurrent and graph convolutions but suffer inefficiency and gradient issues. CNN-based approaches [Mohamed et al., 2020; Zhang et al., 2022] enhance efficiency via graph and 1D convolutions yet risk oversmoothing. Inductive GNNs like GraphSAGE [Hamilton et al., 2017] and IGNN [Wu et al., 2020] enable scalable learning on large or incomplete graphs. Adaptive graph learning [Zheng et al., 2023] and transformer architectures [Jin et al., 2023] further improve modeling of dynamic and long-range dependencies.

动态图形建模。时空图网络捕捉动态的空间和时间站点关系。基于RNN的方法[Yu等人，2017；Kapoor等人，2020；Roy等人，2021；Ghosh等人，2020]使用递归和图卷积，但存在效率低下和梯度问题。基于CNN的方法[Mohamed等人，2020；Zhang等人，2022]通过图和一维卷积提高了效率，但存在过度平滑的风险。像GraphSAGE[Hamilton等人，2017]和IGNN[Wu等人，2020]这样的归纳GNN实现了对大图或不完整图的可伸缩学习。自适应图学习[郑等人，2023]和变压器架构[金等人，2023]进一步改进了动态和远程依赖关系的建模。

## Conclusion

In this paper, we proposed a novel framework BGM, designed to address the challenges of demand prediction in expanding bike-sharing systems. Built on dynamic graph modeling, BGM captures evolving spatio-temporal dependencies and enhances knowledge transfer to new stations through embedding transformation. The gated feature fusion mechanism optimally integrates transferred and intrinsic features, reducing negative transfer and ensuring accurate predictions with minimal historical data. Beyond technical advancements, BGM directly addresses societal challenges such as optimizing resource allocation, promoting equitable access to mobility, and supporting the sustainable expansion of urban transportation networks. Experiments on real-world datasets demonstrate that BGM outperforms existing methods, providing actionable insights for urban mobility planning. Future work will explore extending BGM to other urban systems and incorporating multi-modal data, further enhancing its societal impact in dynamic and diverse urban contexts.

在本文中，我们提出了一个新的框架BGM，旨在解决扩展共享单车系统时需求预测的挑战。BGM建立在动态图建模的基础上，捕获不断变化的时空依赖关系，并通过嵌入变换增强知识向新站点的转移。门控特征融合机制优化地集成了转移特征和内在特征，减少了负转移，并确保了用最少的历史数据进行准确的预测。除技术进步外，BGM还直接应对社会挑战，如优化资源分配、促进公平获得交通工具以及支持城市交通网络的可持续扩展。在真实数据集上的实验表明，BGM的性能优于现有方法，为城市交通规划提供了可操作的见解。未来的工作将探索将BGM扩展到其他城市系统，并纳入多模式数据，进一步增强其在动态和多样化城市背景下的社会影响。