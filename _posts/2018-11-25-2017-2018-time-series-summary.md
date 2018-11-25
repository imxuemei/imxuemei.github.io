---
layout: post
title:  "2017&2018 ICML中time series相关论文汇总"
categories: Paper-Reading
tags: ICML time-series
---

> 2017论文Proceedings参见 [http://proceedings.mlr.press/v70/](http://proceedings.mlr.press/v70/)  
2018论文Proceedings参见 [http://proceedings.mlr.press/v80/](http://proceedings.mlr.press/v80/)


### 概述
Time series相关论文2017年共计7篇，2018年共计16篇。  

### 论文汇总
#### 2017年
[1] Nonnegative Matrix Factorization for Time Series Recovery From a Few Temporal Aggregates  
Jiali Mei, Yohann De Castro, Yannig Goude, Georges Hébrail ; PMLR 70:2382-2390  
非负矩阵因子分解法实现从少量时间聚合中进行时序恢复，需要阅读  

[2] Coherent Probabilistic Forecasts for Hierarchical Time Series  
Souhaib Ben Taieb, James W. Taylor, Rob J. Hyndman ; PMLR 70:3348-3357  
分层时序的一致可能性预测，需要阅读  

[3] Prediction and Control with Temporal Segment Models  
Nikhil Mishra, Pieter Abbeel, Igor Mordatch ; PMLR 70:2459-2468  
使用时间分片模型的预测和控制，需要阅读  

[4] Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs  
Rakshit Trivedi, Hanjun Dai, Yichen Wang, Le Song ; PMLR 70:3462-3471  
认识进化：深度时间推理实现动态知识图  

[5] Soft-DTW: a Differentiable Loss Function for Time-Series  
Marco Cuturi, Mathieu Blondel ; PMLR 70:894-903  
Soft-DTW：对时序的一种可微的损失函数  

[6] iSurvive: An Interpretable, Event-time Prediction Model for mHealth  
Walter H. Dempsey, Alexander Moreno, Christy K. Scott, Michael L. Dennis, David H. Gustafson, Susan A. Murphy, James M. Rehg ; PMLR 70:970-979  
iSurvive：一种可解释的事件时间模型实现mHealth  

[7] Bidirectional Learning for Time-series Models with Hidden Units  
Takayuki Osogami, Hiroshi Kajino, Taro Sekiyama ; PMLR 70:2711-2720  
对带隐藏单元的时序模型的双向学习  

#### 2018年
[1] Autoregressive Convolutional Neural Networks for Asynchronous Time Series  
Mikolaj Binkowski, Gautier Marti, Philippe Donnat ; PMLR 80:580-589  
异步时序的自动回归卷积神经网络  

[2] Adversarial Time-to-Event Modeling  
Paidamoyo Chapfuwa, Chenyang Tao, Chunyuan Li, Courtney Page, Benjamin Goldstein, Lawrence Carin Duke, Ricardo Henao ; PMLR 80:735-744  
对抗时间-时间建模  

[3] Hierarchical Deep Generative Models for Multi-Rate Multivariate Time Series  
Zhengping Che, Sanjay Purushotham, Guangyu Li, Bo Jiang, Yan Liu ; PMLR 80:784-793
对多速率多样时序的分层深度生成模型  

[4] Continuous-Time Flows for Efficient Inference and Density Estimation  
Changyou Chen, Chunyuan Li, Liqun Chen, Wenlin Wang, Yunchen Pu, Lawrence Carin Duke ; PMLR 80:824-833  
对有效推理和密度估计的连续时间流，有空可读  

[5] Time Limits in Reinforcement Learning  
Fabio Pardo, Arash Tavakoli, Vitaly Levdik, Petar Kormushev ; PMLR 80:4045-4054
增强学习的时间限制，有空可读  

[6] Constant-Time Predictive Distributions for Gaussian Processes  
Geoff Pleiss, Jacob Gardner, Kilian Weinberger, Andrew Gordon Wilson ; PMLR 80:4114-4123  
对高斯过程的常量时间可预测分布，有空可读  

[7] PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning  
Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang, Philip S Yu ; PMLR 80:5123-5132  
PredRNN++：在时空可预测学习的Deep-in-Time困境的解决方法  

[8] Continuous and Discrete-time Accelerated Stochastic Mirror Descent for Strongly Convex Functions  
Pan Xu, Tianhao Wang, Quanquan Gu ; PMLR 80:5492-5501  
对强凸函数的连续和离散时间加速随机镜像下降，暂不相关  

[9] Generative Temporal Models with Spatial Memory for Partially Observed Environments  
Marco Fraccaro, Danilo Rezende, Yori Zwols, Alexander Pritzel, S. M. Ali Eslami, Fabio Viola ; PMLR 80:1549-1558  
局部观察环境下带空间内存的生成时间模型，暂不相关   

[10] Temporal Poisson Square Root Graphical Models  
Sinong Geng, Zhaobin Kuang, Peggy Peissig, David Page ; PMLR 80:1714-1723  
时间泊松平方根图的模型，暂不相关  

[11] Spatio-temporal Bayesian On-line Changepoint Detection with Model Selection  
Jeremias Knoblauch, Theodoros Damoulas ; PMLR 80:2718-2727  
带模型选择的时空贝叶斯在线变化点检测  

[12] Learning Localized Spatio-Temporal Models From Streaming Data  
Muhammad Osama, Dave Zachariah, Thomas Schön ; PMLR 80:3927-3935  
从流数据中学习局部时空模型  

[13] TACO: Learning Task Decomposition via Temporal Alignment for Control  
Kyriacos Shiarlis, Markus Wulfmeier, Sasha Salter, Shimon Whiteson, Ingmar Posner ; PMLR 80:4654-4663  
TACO：通过时间分配学任务分解进行控制，暂不相关  

[14] Learning Low-Dimensional Temporal Representations  
Bing Su, Ying Wu ; PMLR 80:4761-4770  
学习低维时间表示  

[15] Least-Squares Temporal Difference Learning for the Linear Quadratic Regulator  
Stephen Tu, Benjamin Recht ; PMLR 80:5005-5014  
对线性二次调节器的最小平方时间差异学习  

[16] Semi-Supervised Learning on Data Streams via Temporal Label Propagation  
Tal Wagner, Sudipto Guha, Shiva Kasiviswanathan, Nina Mishra ; PMLR 80:5095-5104  
通过时间标签传播的流式数据的半监督学习  