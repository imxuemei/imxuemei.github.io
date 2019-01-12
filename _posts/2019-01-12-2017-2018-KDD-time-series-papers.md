---
layout: post
title: "2017-2018 KDD Time Series相关论文"
categories: Paper-Reading
tags: KDD time-series
---

[TOC]
SIGKDD：CCF A类
# 2017 KDD
> [https://www.kdd.org/kdd2017/accepted-papers](https://www.kdd.org/kdd2017/accepted-papers)
## [1] Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data
David Hallac (Stanford University);Sagar Vare (Stanford University);Stephen Boyd (Stanford University);Jure Leskovec (Stanford University)  

多变量时间序列的子序列聚类是用于发现时间数据中的重复模式的有用工具。一旦发现了这些模式，看似复杂的数据集就可以被解释为只有少数状态或集群的时间序列。例如，来自健身追踪应用的原始传感器数据可以表示为选择的几个动作（即，步行，坐着，跑步）的时间线。然而，发现这些模式具有挑战性，因为它需要同时分割和聚类时间序列。此外，解释所得到的聚类是困难的，尤其是当数据是高维的时。在这里，我们提出了一种基于模型的聚类的新方法，我们称之为Toeplitz基于逆协方差的聚类（TICC）。 TICC方法中的每个聚类由相关网络或马尔可夫随机场（MRF）定义，表征该聚类的典型子序列中的不同观察之间的相互依赖性。基于该图形表示，TICC同时对时间序列数据进行分段和聚类。我们通过期望最大化（EM）算法解决TICC问题。我们分别通过动态规划和乘法器的交替方向法（ADMM）推导出封闭形式的解决方案，以可扩展的方式有效地解决E和M步骤。我们通过在一系列合成实验中将TICC与几个最先进的基线进行比较来验证我们的方法，然后我们在汽车传感器数据集上演示TICC如何用于在现实场景中学习可解释的聚类。
<!-- more -->
## [2] Mixture Factorized Ornstein-Uhlenbeck Processes for Time-Series Forecasting
Author(s):   
Guo-Jun Qi (UCF);Jiliang Tang (MSU);Jingdong Wang (Microsoft);Jiebo Luo (University of Rochester)  

可以通过对观测数据的趋势和波动进行建模来预测未来对时间序列数据的观测。已经开发了许多经典的时间序列分析模型，如自回归模型（AR）及其变体，以实现这种预测能力。虽然它们通常基于白噪声假设来模拟数据波动，但是采用了更一般的布朗运动，导致了Ornstein-Uhlenbeck（OU）过程。 OU过程在预测未来对许多类型的时间序列的观察中取得了巨大的成功，然而，它仍然受限于建模由单个持久因子驱动的简单扩散动力学，该因素从未随时间演变。然而，在许多实际问题中，通常存在隐藏因素的混合，并且它们出现或消失的时间和频率是未知的。这带来了挑战，激励我们开发混合因子化OU过程（MFOUP）来模拟不断变化的因素。新模型能够捕捉多个混合隐藏因子的变化状态，从中可以推断出它们在推动时间序列运动中的作用。我们对三个预测问题进行了实验，包括传感器和市场数据流。结果表明，与其他算法相比，它在预测未来观测和捕捉隐藏因子的演化模式方面具有竞争性。

## [3] A Framework for Guided Time Series Motif Discovery
Author(s): Hoang Anh Dau (University of California, Riverside);Eamonn Keogh (University of California, Riverside)  

时间序列图案发现可能是时间序列数据挖掘中最常用的原语，并且已经应用于机器人，医学和气候学等各种领域。 最近在motif发现的可扩展性方面取得了重大进展。 然而，我们认为目前对主题发现的定义是有限的，并且可能在用户的意图/期望和主题发现搜索结果之间产生不匹配。 在这项工作中，我们解释了这些问题背后的原因，并介绍了一个新的和一般的框架来解决它们。 我们的想法可以与当前最先进的算法一起使用，几乎没有时间或空间开销，并且足够快，允许对大量数据集进行实时交互和假设测试。 我们展示了我们的想法在地震学和癫痫发作监测等多种领域的实用性。

## [4] Tripoles: A New Class of Relationships in Time Series Data
Author(s): Saurabh Agrawal (University of Minnesota);Gowtham Atluri (University of Cincinnati);Anuj Karpatne (University of Minnesota);William Haltom (University of Minnesota);Stefan Liess (University of Minnesota);Snigdhansu Chatterjee (University of Minnesota);Vipin Kumar (Univesity of Minnesota)  
时间序列数据中的关系挖掘是几个学科极为关注的研究方向之一。传统上，在时空数据中研究的关系是在成对的远程位置或区域之间。在这项工作中，我们在三个时间序列中定义了一种新的关系模式，我们称之为一个涉及三个时间序列的\ textit {tripole}。我们展示了三极可以捕获数据中的有趣关系，这些关系是使用传统研究的成对关系无法捕获的。我们提出了一种新方法，用于在给定的时间序列数据集中查找三极杆，并证明其与来自气候科学领域的真实世界数据集的蛮力搜索相比具有计算效率。此外，我们还展示了可以在包括气候科学和神经科学在内的各个领域的真实数据集中找到三极。此外，我们发现大多数发现的三极在多个数据集中具有统计显着性和可重现性，这些数据集完全独立于用于查找三极的原始数据集。其中一个发现的气候数据中的三极导致西伯利亚和太平洋之间新的气候遥相关，这在气候领域以前是未知的。

## TrioVecEvent: Embedding-Based Online Local Event Detection in Geo-Tagged Tweet Streams
Author(s): Chao Zhang (University of Illinois at Urbana-Champaign);Liyuan Liu (University of Illinois at Urbana-Champaign);Dongming Lei (University of Illinois at Urbana-Champaign);Quan Yuan (University of Illinois at Urbana-Champaign);Honglei Zhuang (University of Illinois at Urbana-Champaign);Tim Hanratty (U.S. Army Research Lab);Jiawei Han (University of Illinois at Urbana-Champaign)  
在其发起时检测本地事件（例如，抗议，灾难）是广泛应用的重要任务，从灾害控制到犯罪监控和场所推荐。近年来，人们越来越关注利用地理标记的推文流进行在线本地事件检测。尽管如此，现有方法的准确性仍然不能令人满意地建立可靠的本地事件检测系统。我们提出TrioVecEvent，这是一种利用多模式嵌入来实现准确的在线本地事件检测的方法。 TrioVecEvent的有效性得益于其两步检测方案。首先，它通过将查询窗口中的推文划分为连贯的地理主题群集来确保底层本地事件的高覆盖率。为了生成高质量的地理主题聚类，我们通过学习位置，时间和文本的多模式嵌入来捕获短文本语义，然后使用新的贝叶斯混合模型执行在线聚类。其次，TrioVecEvent将地理主题群集视为候选事件，并提取一组用于对候选者进行分类的功能。利用多模式嵌入作为背景知识，我们引入了可以很好地表征局部事件的判别特征，这使得能够利用少量训练数据精确定位候选池中的真实本地事件。我们使用众包来评估TrioVecEvent，发现它将最先进方法的检测精度从36.8％提高到80.4％，伪回忆从48.3％提高到61.2％。

## Effective and Real-time In-App Activity Analysis in Encrypted Internet Traffic Streams
Author(s): Junming Liu (Rutgers University);Yanjie Fu (Missouri University of Science and Technology);Jingci Ming (Rutgers University);Yong Ren (Futurewei Tech. Inc);Leilei Sun (Dalian University of Technology);Hui Xiong (Rutgers University)  

旨在将移动互联网流量分类为不同类型的服务使用的移动应用内服务分析由于越来越多地采用针对应用内服务的安全协议而成为移动服务提供商的挑战性和紧急任务。虽然已经对移动互联网流量的分类做了一些努力，但是现有的方法回复了复杂的特征构造和大的存储缓存，这导致了低处理速度，因此对于在线实时场景不实用。为此，我们开发了一种迭代分析器，用于以实时方式对加密的移动流量进行分类。具体而言，我们首先从通过最大化最大化内部活动相似性和最小化不同活动相似性（MIMD）测量的交通分组序列中提取的原始特征中选择一组最佳判别特征。

为了开发在线分析器，我们首先用一系列时间窗来表示交通流，其中每个时间窗由最佳特征向量描述，并在数据包级迭代更新。我们的特征元素不是从一系列原始流量包中提取特征元素，而是在观察到新的流量包并且不需要存储原始流量包时更新。

从相同的服务使用活动生成的时间窗口按照我们提出的方法分组，即递归时间连续性约束的KMeans聚类（rCKC）。然后将聚类中心的特征向量馈送到随机森林分类器中以识别相应的服务使用。最后，我们提供了来自微信，Whatsapp和Facebook的实际交通数据的广泛实验，以证明我们的方法的有效性和效率。结果表明，该分析仪在实际场景下具有较高的精度，存储缓存要求低，处理速度快。

## Multi-Aspect Streaming Tensor Completion
Author(s): Qingquan Song (Texas A&M University);Xiao Huang (Texas A&M University);Hancheng Ge (Texas A&M University);James Caverlee (Texas A&M University);Xia Hu (Texas A&M University)

## DenseAlert: Incremental Dense-Subtensor Detection in Tensor Streams
Author(s): Kijung Shin (Carnegie Mellon University);Bryan Hooi (Carnegie Mellon University);Jisu Kim (Carnegie Mellon University);Christos Faloutsos (Carnegie Mellon University)

## Anarchists, Unite: Practical Entropy Approximation for Distributed Streams
Author(s): Moshe Gabel (Technion);Daniel Keren (Haifa University);Assaf Schuster (Technion)

## Anomaly Detection in Streams with Extreme Value Theory
Author(s): Alban Siffer (IRISA);Pierre-Alain Fouque (IRISA);Alexandre Termier (IRISA);Christine Largouët (IRISA)

## Collecting and Analyzing Millions of mHealth Data Streams
Author(s): Thomas Quisel (Evidation Health);Luca Foschini (Evidation Health);Alessio Signorini (Evidation Health);David Kale (USC Information Sciences Institute)

## Extremely Fast Decision Tree Mining for Evolving Data Streams
Author(s): Albert Bifet (Telecom ParisTech);Jiajin Zhang (Noah's Ark Lab, Huawei);Wei Fan (Huawei Noah’s Ark Lab);Cheng He (Noah's Ark Lab, Huawei);Jianfeng Zhang (Noah's Ark Lab, Huawei);Jianfeng Qian (Huawei Noah's Ark Lab);Geoffrey Holmes (University of Waikato);Bernhard Pfahringer (University of Waikato)


# 2018 KDD
> [https://www.kdd.org/kdd2018/accepted-papers](https://www.kdd.org/kdd2018/accepted-papers)
## Deep r-th Root Rank Supervised Joint Binary Embedding for Multivariate Time Series Retrieval
Dongjin Song (NEC Labs America); Ning Xia (NEC Labs America); Wei Cheng (NEC Labs America); Haifeng Chen (NEC Labs America); Dacheng Tao (The University of Sydney)

多变量时间序列数据在许多现实世界的应用中变得越来越普遍，例如，电厂监控，医疗保健，可穿戴设备，汽车等。因此，多变量时间序列检索，即给定当前多变量时间序列段，如何在历史数据（或数据库）中获取其相关的时间序列段，吸引了许多领域的大量兴趣。然而，构建这样的系统是具有挑战性的，因为它需要原始时间序列的紧凑表示，其可以明确地编码时间动态以及不同时间序列对（传感器）之间的相关性（相互作用）。此外，它需要查询效率，并期望返回的排名列表在顶部具有高精度。尽管已经开发了各种方法，但很少有方法可以共同解决这两个挑战。为了解决这个问题，在本文中，我们提出了一个深度第r个秩监督联合二进制嵌入（Deep r-RSJBE）来执行多变量时间序列检索。给定原始多变量时间序列片段，我们使用长短期记忆（LSTM）单元来编码时间动态并利用卷积神经网络（CNN）来编码不同时间序列对（传感器）之间的相关性（相互作用）。随后，追求联合二进制嵌入以结合时间动态和相关性。最后，我们开发了一种新的第r个根排名损失，以优化汉明距离排名列表顶部的精度。基于三个公开可用的时间序列数据集的彻底的实证研究证明了Deep r-RSJBE的有效性和效率。

## Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis
Jingyuan Wang (Beihang University); Ze Wang (Beihang University); Jianfeng Li (Beihang University); Junjie Wu (Beihang University)  

近年来，几乎所有学术和工业领域都出现了前所未有的时间序列。各种类型的深度神经网络模型已被引入时间序列分析，但重要的频率信息缺乏有效的建模。鉴于此，本文提出了一种基于小波的神经网络结构，称为多级小波分解网络（mWDN），用于构建时间序列分析的频率感知深度学习模型。 mWDN保留了频率学习中多级离散小波分解的优势，同时能够在深度神经网络框架下对所有参数进行微调。基于mWDN，我们进一步提出了两种深度学习模型，分别称为残差分类流（RCF）和多频长期短期记忆（mLSTM），用于时间序列分类和预测。这两个模型将不同频率的全部或部分mWDN分解子系列作为输入，并采用反向传播算法全局学习所有参数，从而实现基于小波的频率分析无缝嵌入深度学习框架。对40个UCR数据集和实际用户体积数据集进行了大量实验，证明了基于mWDN的时间序列模型的出色性能。特别是，我们提出了一种基于mWDN的模型的重要性分析方法，它成功地识别了对时间序列分析至关重要的时间序列元素和mWDN层。这确实表明了mWDN的可解释性优势，并且可以被视为对可解释的深度学习的深入探索。

## Interpretable Representation Learning for Healthcare via Capturing Disease Progression through Time
Tian Bai (Temple University); Shanshan Zhang (Temple University); Brian Egleston (Fox Chase Cancer Center); Slobodan Vucetic (Temple University)  

最近已将各种深度学习模型应用于电子健康记录（EHR）的预测建模。在作为特定类型的EHR数据的医疗索赔数据中，每个患者被表示为对健康提供者进行时间排序的不规则抽样访问的序列，其中每次访问被记录为指定患者在期间提供的诊断和治疗的无序医疗代码集。访问。基于不同患者状况具有不同时间进展模式的观察结果，本文提出了一种新的可解释的深度学习模型，称为时间轴。时间轴的主要新颖之处在于它具有学习每个医疗代码的时间衰减因子的机制。这使得时间表可以了解慢性病对未来就诊的影响比急性病更长。时间轴还有一个注意机制，可以改善访问的矢量嵌入。通过分析时间轴的注意力和疾病进展功能，可以解释预测并了解未来访问的风险如何随时间而变化。我们在两个大型现实世界数据集上评估了时间轴。具体任务是预测在之前的访问中下一次住院就诊的主要诊断类别。我们的结果表明，时间轴比基于RNN的最先进的深度学习模型具有更高的准确性。此外，我们证明时间线学习的时间衰减因子和注意力与医学知识一致，时间轴可以提供对其预测的有用见解。

## HeavyGuardian: Separate and Guard Hot Items in Data Streams
Tong Yang (Peking University); Junzhi Gong (Peking University); Haowei Zhang (Peking University); Lei Zou (Peking University); Lei Shi (SKLCS, Institute of Software, Chinese Academy of Sciences); Xiaoming Li (Peking University)

数据流处理是许多领域的基本问题，例如数据挖掘，数据库，网络流量测量。 数据流处理中有五个典型任务：频率估计，重型击球手检测，重度变化检测，频率分布估计和熵估计。 针对不同的任务提出了不同的算法，但它们很少同时实现高精度和高速度。 为了解决这个问题，我们提出了一种名为HeavyGuardian的新型数据结构。 关键的想法是智能地分离和保护热门物品的信息，同时大致记录冷物品的频率。 我们在上述五个典型任务中部署了HeavyGuardian。 大量实验结果表明，对于五种典型任务中的每一项，HeavyGuardian都能获得比最先进解决方案更高的精度和更高的速度。 GITHub提供了HeavyGuardian的源代码和其他相关算法。

## SpotLight: Detecting Anomalies in Streaming Graphs
Dhivya Eswaran (Carnegie Mellon University); Christos Faloutsos (Carnegie Mellon University); Sudipto Guha (Amazon); Nina Mishra (Amazon)

我们如何从电子邮件或运输日志中发现有趣的事件？我们如何从IP-IP通信数据中检测端口扫描或拒绝服务攻击？一般来说，给定一系列加权图，有向图或二分图，每个图总结一个时间窗中活动的快照，我们如何发现包含近实际中大密集子图（例如近混合图）突然出现或消失的异常图 - 使用次线性内存？为此，我们提出了一种名为SpotLight的随机草图绘制方法，该方法可以保证异常图形远离草图空间中的“正常”实例，并且具有适当选择参数的高概率。对现实世界数据集的大量实验表明，SpotLight（a）与先前方法相比，精度提高了至少8.4％，（b）速度快，可在几分钟内处理数百万条边，（c）与数量成线性关系边缘和草图尺寸和（d）在实践中导致有趣的发现。

## Model-based Clustering of Short Text Streams
Jianhua Yin (School of Computer Science and Technology, Shandong University); Daren Chao (School of Computer Science and Technology, Shandong University); Zhongkun Liu (School of Computer Science and Technology, Shandong University); Wei Zhang (Shanghai Key Laboratory of Trustworthy Computing, East China Normal University); Xiaohui Yu (School of Computer Science and Technology, Shandong University); Jianyong Wang (Tsinghua University)  

由于各种社交媒体中短文的爆炸式增长，短文本流聚类已成为一个日益重要的问题。 在本文中，我们提出了一种基于模型的短文本流聚类算法（MStream），它可以自然地处理概念漂移问题和稀疏性问题。 MStream算法只需一次传递即可实现最先进的性能，并且当我们允许每批次多次迭代时，可以获得更好的性能。 我们进一步提出了一种改进的MStream算法，其遗忘规则称为MStreamF，它可以通过删除过时批次的集群来有效地删除过时的文档。 我们广泛的实验研究表明，MStream和MStreamF可以在几个真实数据集上实现比三个基线更好的性能。

## DILOF: Effective and Memory Efficient Local Outlier Detection in Data Streams
Gyoung S. Na (Pohang University of Science and Technology (POSTECH)); Donghyun Kim (Pohang University of Science and Technology (POSTECH),); Hwanjo Yu (Pohang University of Science and Technology (POSTECH),)

随着对检测数据流中的异常值的需求急剧增长，已经进行了许多研究，旨在为数据流开发称为局部异常因子（LOF）的众所周知的异常检测算法的扩展。但是，现有的基于LOF的数据流算法仍然存在两个固有的局限性：1）需要大量的存储空间。 2）未检测到长序列的异常值。在本文中，我们提出了一种新的数据流离群检测算法，称为DILOF，有效地克服了这些限制。为此，我们首先开发了一种新的基于密度的采样算法来总结过去的数据，然后提出一种检测异常值序列的新策略。值得注意的是，我们的提议算法不需要任何关于数据分布的先验知识或假设。此外，通过开发强大的距离近似技术，我们将DILOF的执行时间加速了大约15倍。我们对实际数据集的全面实验表明，DILOF在准确性和执行时间方面明显优于最先进的竞争对手。所提算法的源代码可在我们的网站上找到：http：//di.postech.ac.kr/DILOF。

## xStream: Outlier Detection in Feature-Evolving Data Streams
Emaad Ahmed Manzoor (Carnegie Mellon University); Hemank Lamba (Carnegie Mellon University); Leman Akoglu (Carnegie Mellon University)

这项工作解决了以前没有研究过的特征演化流的异常检测问题。在此设置中，（1）数据点可能会随着特征值的变化而发展，以及（2）特征空间可能会随着时间的推移随着新出现的特征而发展。这与行流明显不同，其中具有固定特征的点一次到达一个。我们提出了一种基于密度的集合异常值检测器，称为xStream，用于这种更极端的流设置，它具有以下关键属性：（1）它是一个恒定空间和恒定时间（每个传入更新）算法，（2）它测量多个尺度或粒度的离群值，它可以通过距离保持预测处理（3 i）高维度，并通过$ O（1）$  - 时间模型更新随着流的进展处理（3 $ ii $）非平稳性。此外，xStream可以解决（较不常见的）磁盘驻留静态以及行流设置的异常值检测问题。我们在所有三种设置中对许多真实数据集严格评估xStream：静态，行流和特征演化流。在静态和行流场景下的实验表明，xStream与最先进的探测器一样具有竞争力，并且在高维噪声方面特别有效。我们还证明了我们的解决方案快速准确，适用于不断变化的流的适度空间开销。


