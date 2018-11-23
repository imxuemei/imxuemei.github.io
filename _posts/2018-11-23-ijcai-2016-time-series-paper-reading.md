---
layout: post
title: "2016 IJCAI中6篇Time Series相关论文概读"
categories: Paper-Reading
tags: IJCAI time-series
---

### [1] Are Spiking Neural Networks Useful for Classifying and Early Recognition of Spatio-Temporal Patterns?
#### meta-info

| meta | info |
| --- | --- |
| Title | Are Spiking Neural Networks Useful for Classifying and Early Recognition of Spatio-Temporal Patterns? |
| Authors | Banafsheh Rekabdar |
| Citation | 1 |
| Application | human-robot交互，目前该论文还是初级阶段，仅在数字识别、手势识别上具备实验 |
| Model/Method | SNN的4个变种方法 |
| Related works | 暂无 |
| dataset | 数字识别、手势识别 |
| baselines | SVM、LR、ENN |

<!-- more -->
#### Abstract
Learning and recognizing spatio-temporal patterns is an important problem for all biological systems. 
Gestures, movements and activities, all encompass both spatial and temporal information that is critical for implicit communication and learning. 
This paper presents a novel, unsupervised approach for learning, recognizing and early classifying spatio-temporal patterns using spiking neural networks for human robotic domains. 
The proposed spiking approach has four variations which have been validated on images of handwritten digits and human hand gestures and motions. 
The main contributions of this work are as follows: 
i) it requires a very small number of training examples, 
ii) it enables early recognition from only partial information of the pattern, 
iii) it learns patterns in an unsupervised manner, 
iv) it accepts variable sized input patterns, 
v) it is invariant to scale and translation, 
vi) it can recognize patterns in real-time and,
vii) it is suitable for human-robot interaction applications and has been successfully tested on a PR2 robot. 
We also compared all variations of this approach with well-known supervised machine learning methods including support vector machines (SVM), regularized logistic regression (LR) and ensemble neural networks (ENN). 
Although our approach is unsupervised, it outperforms others and in some cases, provides comparable results with other methods.  

摘要：学习和识别时空模式对所有生物系统来说都是一个重要的问题。手势、移动和活动，所有包含空间和时间信息，这些时空信息对隐式的沟通和学习是非常关键的。本文提出了一种新的，无监督的方法使用尖峰神经网络来学习、识别、尽早分类时空模式以用于人类机器人领域。所提出的尖峰方法有四个变种，这些变种方法已经在手写数字图像和人类手势和移动上得到了验证。本文的主要贡献如下：i)它只需要非常少的训练样例；ii)它仅从模式的局部信息中就能够实现早期识别；iii)它以无监督的方式学习了模式；iv)它可接受多种大小的输入；v)它可以不变地伸缩和翻译；vi)它能够实时识别模式；vii)它适用于human-robot交互应用并且在PR2 robot上得到了测试。我们将这个方法的变种与目前主流的有监督方法进行了对比，包括SVM、LR、ENN。尽管我们的方法是无监督的，但它在某些案例上也提供了具有可比性的结果。

> Univeristy of Nevada, Reno  
brekabdar@unr.edu

### [2] Clustering Financial Time Series: How Long Is Enough?
#### meta-info

| meta | info |
| --- | --- |
| Title | Clustering Financial Time Series: How Long Is Enough? |
| Authors | Gautier Marti,S'ebastien Andler,rank Nielsen,Philippe Donnat |
| Citation | 8 |
| Application |  |
| Model/Method |  |
| Related works |  |
| dataset |  |
| baselines |  |

#### Abstract
Researchers have used from 30 days to several years of daily returns as source data for clustering financial time series based on their correlations.
This paper sets up a statistical framework to study the validity of such practices. 
We first show that clustering correlated random variables from their observed values is statistically consistent. 
Then, we also give a first empirical answer to the much debated question: How long should the time series
be? 
If too short, the clusters found can be spurious; if too long, dynamics can be smoothed out.  

摘要：研究者们使用了从30天到几年的每天的返回作为源数据，基于它们的相关性来聚类金融时序数据。本文建立了一个统计框架来研究这些实践的有效性。我们先展示了从他们观察到的值中聚类相关性随机变量在统计学上是一致的。然后，我们给出了第一个实验结果答案来解答很有争论性的问题：应该处理多长时间的时序数据？如果太短，聚类后的簇可能具有欺诈性，如果太长，动态性可能会趋于平滑而丢失。

> Gautier Marti, Hellebore Capital Ltd, Ecole Polytechnique  
S'ebastien Andler, ENS de Lyon, Hellebore Capital Ltd  
Frank Nielsen, Ecole Polytechnique, LIX - UMR 7161  
Philippe Donnat, Hellebore Capital Ltd, Michelin House, London

### [3] Real-Time Web Scale Event Summarization Using Sequential Decision Making
#### meta-info

| meta | info |
| --- | --- |
| Title | Real-Time Web Scale Event Summarization Using Sequential Decision Making |
| Authors | Chris Kedzie,Fernando Diaz,Kathleen McKeown |
| Citation |  |
| Application |  |
| Model/Method |  |
| Related works |  |
| dataset |  |
| baselines |  |

#### Abstract
We present a system based on sequential decision making for the online summarization of massive document streams, such as those found on the web. Given an event of interest (e.g. “Boston marathon bombing”), our system is able to filter the stream for relevance and produce a series of short text updates describing the event as it unfolds over time. Unlike previous work, our approach is able to jointly model the relevance, comprehensiveness, novelty, and timeliness required by time-sensitive queries. We demonstrate a 28.3% improvement in summary F1 and a 43.8% improvement in timesensitive F1 metrics.  

摘要：我们为大量文档流的在线总结提供了一个基于顺序决策的系统，例如那些在网页上找到的文档。给定一个有趣的事件后（例如：波士顿马拉松爆炸），我们的系统能够过滤这些流以关联和生成这个短文本升级的一个序列，随时间展开式地描述这个事件。与之前方法不同，我们的方法能够联合建模时间敏感序列所需要的相关性、理解力、创新性和时间线。我们证明了在总结F1上有28.3%的提升，在时间敏感F1度量上有43.8%的提升。


> Chris Kedzie, Columbia University,Dept. of Computer Science,kedzie@cs.columbia.edu  
Fernando Diaz,Microsoft Research,fdiaz@microsoft.com,  
Kathleen McKeown,Columbia University,Dept. of Computer Science,kathy@cs.columbia.edu  

### [4] Resolving Over-Constrained Conditional Temporal Problems Using Semantically Similar Alternatives
#### meta-info

| meta | info |
| --- | --- |
| Title | Resolving Over-Constrained Conditional Temporal Problems Using Semantically Similar Alternatives |
| Authors | Peng Yu,Jiaying Shen and Peter Z. Yeh,Brian Williams |
| Citation |  |
| Application |  |
| Model/Method |  |
| Related works |  |
| dataset |  |
| baselines |  |

#### Abstract
In recent literature, several approaches have been developed to solve over-constrained travel planning problems, which are often framed as conditional temporal problems with discrete choices. These approaches are able to explain the causes of failure and recommend alternative solutions by suspending or weakening temporal constraints. While helpful, they may not be practical in many situations, as we often cannot compromise on time. In this paper, we present an approach for solving such over-constrained problems, by also relaxing non-temporal variable domains through the consideration of additional options that are semantically similar. Our solution, called Conflict-Directed Semantic Relaxation (CDSR), integrates a knowledge base and a semantic similarity calculator, and is able to simultaneously enumerate both temporal and domain relaxations in best-first order. When evaluated empirically on a range of urban trip planning scenarios, CDSR demonstrates a substantial improvement in flexibility compared to temporal relaxation only approaches.  

摘要：近期文献中，已经开发了几种方法来解决过度约束的交通规划问题，这些问题常常被框架化为带离散选择的条件时间问题。这些方法能够解释失败的原因，并通过中断或减弱时间条件来推荐替代方案。虽然有帮助，但是他们在很多场景下可能不够实用，因为我们常常不能在时间上妥协。本文中，我们提出了一种解决这类过度约束问题的方法，通过对语义相似的额外选项的考虑来放松非时间变量域。我们的方法称为直接冲突语义放松（CDSR），集成了一个知识基底、语义相似计算器，能够同时罗列时间和领域放松，以最佳第一的顺序。

> Peng Yu, MIT, yupeng@mit.edu  
Jiaying Shen and Peter Z. Yeh, Nuance Communications, Inc. {Jiaying.Shen,Peter.Yeh}@nuance.com  
Brian Williams, MIT, williams@mit.edu

### [5] ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data
#### meta-info

| meta | info |
| --- | --- |
| Title | ST-MVL: Filling Missing Values in Geo-Sensory Time Series Data |
| Authors | Xiuwen Yi1,2,*, Yu Zheng2,3,1,+, Junbo Zhang2, Tianrui Li1 |
| Citation | 17 |
| Application | 传感器缺失数据处理 |
| Model/Method | ST-MVL，4种view分别处理后，放入多view算法中学习(简单线性模型) |
| Related works | 空间模型：反向距离权重、线性回归、Kriging<br>基于特征的时间模型：图模型、回归模型等<br>无特征的时间模型：SES、ARMA、SARIMA |
| dataset | 北京市空气质量和气象数据2014.5.1-2015.4.30 |
| baselines | ARMA,Kriging,SARIMA,stKNN,DESM,AKE,IDW+SES,CF,NMF,NMF-MVL |

#### Abstract
Many sensors have been deployed in the physical world, generating massive geo-tagged time series data.
In reality, readings of sensors are usually lost at various unexpected moments because of sensor or communication errors. 
Those missing readings do not only affect real-time monitoring but also compromise the performance of further data analysis.
In this paper, we propose a spatio-temporal multiview-based learning (ST-MVL) method to collectively fill missing readings in a collection of geosensory time series data, considering 1) the temporal correlation between readings at different timestamps in the same series and 2) the spatial correlation between different time series. 
Our method combines empirical statistic models, consisting of Inverse Distance Weighting and Simple Exponential Smoothing, with data-driven algorithms, comprised of Userbased and Item-based Collaborative Filtering.
The former models handle general missing cases based on empirical assumptions derived from history data over a long period, standing for two global views from spatial and temporal perspectives respectively.
The latter algorithms deal with special cases where empirical assumptions may not hold, based on recent contexts of data, denoting two local views from spatial and temporal perspectives respectively. 
The predictions of the four views are aggregated to a final value in a multi-view learning algorithm. 
We evaluate our method based on Beijing air quality and meteorological data, finding advantages to our model compared with ten baseline approaches.  

摘要：物理世界中部署了许多传感器，它们每天产生大量geo-tagged时序数据。现实中，由于传感器或通信错误，传感器的数据读取常常在多种异常时刻丢失。这些丢失的读取不仅影响实时监控也妨碍了进一步开展数据分析。本文中，我们提出了一中时空的基于多面的学习方法（ST-MVL）来整体补齐在一系列地理传感器时序数据中丢失的读取，方法考虑了：1）相同序列中不同时间戳的读取之间的时间相关性，2）不同时序的空间相关性。我们的方法结合了经验统计模型和数据驱动的算法，经验统计模型包括反向距离权重和简单指数平滑，数据驱动的算法包括基于用户的和基于项的协同过滤。前述模型用来处理在长期的历史数据中基于经验假设的通用丢失案例，代表了两个全局view：独立的空间、时间角度。后面的算法用来处理经验假设不能解决的特殊案例，算法基于数据的近期上下文，代表了两个局部view：独立的空间、时间角度。这4个view的预测综合起来得到了多view学习算法的最终值。我们基于北京市空气质量和气象数据进行了评估，与10个基准方法相比我们的模型具有优势。

> Xiuwen Yi1,2,*, Yu Zheng2,3,1,+, Junbo Zhang2, Tianrui Li1  
1School of Information Science and Technology, Southwest Jiaotong University, Chengdu, China  
2Microsoft Research, Beijing, China  
3Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences  
xiuwenyi@foxmail.com, {yuzheng, junbo.zhang}@microsoft.com, trli@swjtu.edu.cn  


### [6] Unsupervised Feature Learning from Time Series
#### meta-info

| meta | info |
| --- | --- |
| Title | Unsupervised Feature Learning from Time Series |
| Authors | Qin Zhang⇤, Jia Wu⇤, Hong Yang§, Yingjie Tian†,‡, Chengqi Zhang⇤ |
| Citation |  |
| Application |  |
| Model/Method |  |
| Related works |  |
| dataset |  |
| baselines |  |

#### Abstract
In this paper we study the problem of learning discriminative features (segments), often referred to as shapelets [Ye and Keogh, 2009] of time series, from unlabeled time series data. 
Discovering shapelets for time series classification has been widely studied, where many search-based algorithms are proposed to efficiently scan and select segments from a pool of candidates.
However, such types of search-based algorithms may incur high time cost when the segment candidate pool is large.
Alternatively, a recent work [Grabocka et al., 2014] uses regression learning to directly learn, instead of searching for, shapelets from time series.
Motivated by the above observations, we propose a new Unsupervised Shapelet Learning Model (USLM) to efficiently learn shapelets from unlabeled time series data.
The corresponding learning function integrates the strengths of pseudo-class label, spectral analysis, shapelets regularization term and regularized least-squares to auto-learn shapelets, pseudo-class labels and classification boundaries simultaneously.
A coordinate descent algorithm is used to iteratively solve the learning function.
Experiments show that USLM outperforms searchbased algorithms on real-world time series data.  

摘要：
本文中，我们研究了从未标注的时序数据中区别特征（片段）学习的问题，常常涉及时序的shapelets。在时序分类中发现shapelets已经得到了广泛的研究，许多基于搜索的算法致力于有效的扫描，并从一池候选中选择片段。然而，这些基于搜索的算法在片段候选池很大时可能需要花费很高的时间成本。替而代之的是，近期的研究使用了回归学习替代搜索来直接学习时序中的shapelets。基于上述研究的驱动，我们提出了一种新的无监督shapelet学习模型（USLM）来有效地从未标注时序数据中学习shapelets。与前面方法类似地集成了pseudo-class标注的优势、频谱分析、shapelets正则化术语和正则的least-squares来同时自动学习shapelets、pseudo-class标注和分类边界。一个协同下降算法用来迭代地解决学习问题。实验表明USLM在真实世界时序数据集上优于基于搜索的方法。

> 时间序列 shapelets 是时间序列中能够最大限度地表示一个类别的子序列。

> Qin Zhang⇤, Jia Wu⇤, Hong Yang§, Yingjie Tian†,‡, Chengqi Zhang⇤  
⇤ Quantum Computation & Intelligent Systems Centre, University of Technology Sydney, Australia  
† Research Center on Fictitious Economy & Data Science, Chinese Academy of Sciences, Beijing, China.  
‡ Key Lab of Big Data Mining & Knowledge Management, Chinese Academy of Sciences, Beijing, China.  
§ MathWorks, Beijing, China  
{qin.zhang@student.,jia.wu@, chengqi.zhang@}uts.edu.au, hong.yang@mathworks.cn, tyj@ucas.ac.cn  

