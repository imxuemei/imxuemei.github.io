---
layout: post
title:  "2018 ICJAI Time Series相关论文阅读记录"
categories: paper-reading
tags: time series
---
### [1]Causal Inference in Time Series via Supervised Learning
#### Meta-Info
| meta | info |
| --- | --- |
| Authors | Yoichi Chikahara and Akinori Fujino, NTT Communication Science Laboratories |
| Citation | 1 |
| Application | 时序中的因果推理 |
| Method/Model | 设计了一个三元分类器代替回归模型实现因果推理，SIGC |
| Related works | RCC随机因果系数 |
| dataset | 人造数据集：线性和非线性，<br>真实数据集：Cause-Effect Pairs数据库中5项 |
| baselines | RCC |
#### Abstract
时序中的因果推理是很多领域中一个重要的问题。传统方法使用回归模型来解决这个问题。这些方法的准确性非常依赖于模型中数据拟合程度，因此需要选择一个合适的回归模型，这在实践中是困难的。本文提出了一种有监督的学习框架使用分类器来代替回归模型。我们提出了一种特征表示，它使用了给定的历史变量值的假定分布的距离，实验表明这种特征表示能够为带有不同因果推理的时序提供足够的不同向量。进一步的，我们将我们的框架推广到了多变量时序，在i.i.d.数据上实验结果优于基于模型和有监督学习的方法。
#### Problems or challenges
传统方法需要理解数据并为每种时序数据选择不同的模型，实践中较为困难。本文的目标是构建一种方法直接在时序数据上开展因果推理，而不需要深入理解数据分析。
#### Contribution or Method
1) 针对格兰杰因果关系识别，训练了一个分类器对时序进行三元分类，即结果标注为3元标注：X->Y, Y->X, No Causation。  
2) 提出了一种特征表示：提供足够的不同特征向量表示时序的不同因果关系，这些特征向量是基于格兰杰因果关系定义识别出的：如果Y的紧随的两个future value的假定分布是不同的，一个future value是由Y的past value给出的，一个future value是由X和Y的past value共同给出的，则X是Y的原因。计算特征向量时使用了两个分布之间的距离。计算距离时使用了kernel mean embedding，将每个分布映射到了RKHS（再生核Hilbert空间）的特征空间，使用点之间的距离进行度量，术语为最大平均差异（maximum mean discrepancy）。

### [2]GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction
#### Meta-Info
meta | info
---|---
Authors | Yuxuan Liang1;2, Songyu Ke3;2, Junbo Zhang2;4, Xiuwen Yi4;2, Yu Zheng2;1;3;4
Citation | 3
Application | 时空结合的地理传感器数据时序预测
Method/Model | 多层注意力网络及LSTM
Related works | 地理传感器时序预测：自回归模型。<br>深度学习：RNN、LSTM。<br>注意力机制：多层注意力模型、结合RNN的注意力模型
dataset | 人造数据集：线性和非线性，<br>真实数据集：Cause-Effect Pairs数据库中5项
baselines | RCC
#### Abstract
无数的传感器被部署在不同的地理位置来持续和协同地监控周边环境，例如空气质量。这些传感器产生了多种的地理传感时间序列，它们的读取之间具备空间相关性。预测地理传感时间序列是非常重要但具有挑战的，因为它被多种复杂因素影响，例如：动态空间-时间相关性和外部因素。本文中，我们使用多层基于注意力的RNN来预测在未来几个小时的地理传感器的读取，该RNN考虑了多层传感器读取，气象数据和空间数据。更具体地，我们的模型包括以下两个主要部分：1）一种多层注意力机制来对动态空间-时间依赖进行建模；2）一种通用融合模块来合并来自不同域的外部因素。在两种真实世界数据集上的实验，即：空气质量数据、水的质量数据，表明我们的方法由于9种基线方法。
#### Problems or Challenges
本文主要解决地理传感器时序预测问题。地理传感器时序预测有2项挑战：
1）动态时空相关性，包括动态的同位置时序相关和动态的位置间时序相关；
2）外部因素，例如气象、高峰时间、土地使用等会影响传感器数据读取。
#### Contribution or Method
本文有3个贡献：
1. 多层注意力机制：其中第一层提供了一个注意力机制，包括局部注意力、全部注意力，主要处理空间相关性。第二层应用了时间注意力处理同一时序数据的不同时间间隔的动态时间相关性。
2. 外部因素融合模型：将外部因素的潜在表示输入到多层注意力网络中来增强外部因素的重要性。
3. 真实实验：使用了真实世界数据集来实测结果。

> 作者来自：  
1 School of Computer Science and Technology, Xidian University, Xi’an, China  
2 Urban Computing Business Unit, JD Finance, Beijing, China  
3 Zhiyuan College, Shanghai Jiao Tong University, Shanghai, China  
4 School of Information Science and Technology, Southwest Jiaotong University, Chengdu, China  
fyuxliang, songyu-ke, msjunbozhang, xiuwyi, msyuzhengg@outlook.com

### [3]Online Pricing for Revenue Maximization with Unknown Time Discounting Valuations
#### Meta-Info
meta | info
---|---
Authors | Weichao Mao1, Zhenzhe Zheng1, Fan Wu1∗, Guihai Chen2
Citation | 0
Application | 在线估价
Method/Model | Biased-UCB，非静态多臂老虎机优化
Related works | 在线拍卖相关、随机需求模型相关，UCB框架
dataset | iPinYou公司在线竞拍日志
baselines | UCB1，Rexp3

#### Abstract
在线定价机制在多代理系统中广泛用于资源分配。然而，大多现有在线定价机制假设购买者的价格不会随时间变化，因而不能捕获随应用涌现而导致的估价动态变化的本质。本文中，我们研究了在在线拍卖中随未知时间折扣估价时的收入最大化问题，并建模为非静态的多臂老虎机优化问题。我们设计了基于折扣估价的独特特征在线估价机制，命名为Biased-UCB。我们使用竞争分析对我们的定价机制进行了理论分析，生成了竞争比率。结果数值表明我们的设计能够在真实的拍卖数据集上在最大收入的指标中最优。
#### Problems or Challenges
当前定价机制没有考虑未来价格的浮动，本文提出的算法考虑了价格随未来潜在价值的动态变化后，卖方可获得的最大收入。
#### Contribution or Method
3个贡献：
1. 将动态价格变动下的最大收入问题建模为非静态的多臂老虎机优化问题。
2. 充分探索了估价中时间折扣特性，提出了Biased-UCB，并进行了理论分析。
3. 基于真实世界数据集进行了实验，在最大收入指标上效果最优。

> 作者来自：  
1 Shanghai Key Laboratory of Scalable Computing and Systems, Shanghai Jiao Tong University, China  
2 Department of Computer Science and Technology, Nanjing University, China  
fmaoweichao,zhengzhenzheg@sjtu.edu.cn, ffwu,gcheng@cs.sjtu.edu.cn

### [4]Spatio-Temporal Check-in Time Prediction with Recurrent Neural Network based Survival Analysis
#### Meta-Info
meta | info
---|---
Authors | Guolei Yang1, Ying Cai2 and Chandan K. Reddy3
Citation | 3
Application | 用户签到时间预测
Method/Model | 循环审查回归模型RCR
Related works | 签到地点预测：Periodic Mobility模型/Periodic Social Mobility模型；NextPlace Framework；Spatial Temporal RNN（ST-RNN）<br>事件时间预测：Cox模型、参数审查回归、审查线性回归
dataset | New York City和东京Foursquare签到数据、Gowalla全球签到数据
baselines | Linear、DNN、AFT、Cox、RCR

#### Abstract
我们介绍了一种新的签到时间预测问题，目标是预测用户在一个给出的地点的签到时间。我们把签到预测记为生存分析问题，并提出了循环审查回归模型（RCR）。我们认为关键的挑战是签到数据的稀缺性，是用户/地点中签到分布的不均匀导致的。我们的想法是丰富潜在访问者的签到数据，即那些以前没有访问过这个地点但是有可能访问的用户。RCR使用RNN从真实和潜在的访问者的历史签到数据学习出潜在表示，然后与审查回归组合做出预测。实验表明在真实世界数据集上RCR性能优于目前最好的事件时间预测技术。
#### Problems or Challenges
基于位置的社交网络（LBSN）app应用已经很广，可以收集用户的位置信息，预测移动轨迹。大量的文献对移动预测进行了研究，但是没有对到达预测位置的时间的预测。本文主要是签到时间预测问题，即基于用户的已知签到路径预测到某个地点的签到时间，时间预测有助于规划精确营销和旅行者服务。
#### Contribution or Method
本文主要贡献：1）提出了人类行动研究的一个开放性问题：签到时间预测问题。2）解决了数据稀缺问题。3）基于真实世界数据评估模型。
本文问题定义是：给出所有用户的历史check-in数据和设定的目的地点，求解出某用户u访问目的地点l的具体时间t。
第一步：针对数据稀缺问题，本文选择2类数据作为样本，1类为实际在l地签到的用户数据，1类为通过模型选择的潜在用户。首先使用矩阵因子分解方法得到用户u是否有可能访问l。同时符合以下3个条件的用户作为潜在用户：生活在l所在的城市；朋友（fb的fellows等）去过l；矩阵因子分解法得到的结果预测了可能去l的用户。
第二步：使用第一步中选择的用户的历史数据来产生轨迹片段。设定一个超参数表示数据截取的观察窗口大小。对于实际访问过l的用户数据，在观察窗口大小内截取用户访问l及之前的具有相关性的签到数据。对于未访问过l的用户数据，在观察窗口大小内从最后一次签到开始取由n次连续签到构成的片段。
第三步：2类特征提取：用户移动特征、签到特征，特征全部为人工定义特征，根据历史数据计算得到特征值。
第四步：RCR模型：时间计算方法采用了生存分析方法，使用带有门函数的LSTM进行数据训练，并使用后续时间传播算法训练模型。损失函数为负似然函数。
> 作者来自：  
1 Facebook, Inc.  
2 Department of Computer Science, Iowa State University  
3 Department of Computer Science, Virginia Tech.  
glyang@fb.com, yingcai@iastate.edu, reddy@cs.vt.edu

### [5]Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
#### Meta-Info
meta | info
---|---
Authors | Bing Yu ⇤1, Haoteng Yin⇤2,3, Zhanxing Zhu †3,4
Citation | 2
Application | 预测交通拥堵
Method/Model | Graphic网络的深度学习方法，时空图形卷积网络STGCN。近似计算方法：Chebyshev多项式近似、1<sup>st</sup>order近似
Related works | GCRN、GRU
dataset | BJER4、PeMSD7
baselines | HA、LSVR、ARIMA、FNN、FC-LSTM、GCGRU

#### Abstract
及时准确的交通预报在城市交通控制和引导中是非常重要的。由于交通流的高度非线性和复杂性，传统方法不能满足中长期预测任务的需求且其常常忽略空间和时间的依赖性。本文中，我们提出了一种新型的deep learning框架，时空图形卷积网络（STGCN），来解决交通领域的时序预测问题。与使用常规卷积和循环单元不同，我们把问题用图来表示并建立了带有完整卷积结构的模型，这可以实现更快的训练速度和更少的参数。实验表明，我们的STGCN模型能够通过多种规模的交通网络的建模来有效地捕获可理解的时空相关性，在多种真实世界交通数据集上全部优于目前最好的基准方法。
#### Problems or Challenges
本文主要针对交通流进行中长期预测（即30分钟之后的预测）。传统统计学方法主要预测5-30分钟的短期，长期预测有两类方法：动态模型、数据驱动的预测。动态模型主要使用数学工具进行仿真计算，计算量大，且由于做了大量的假设和简化降低了预测准确性。所以目前转向了数据驱动的预测。数据驱动的方法里面，部分没有处理空间信息，部分组合了时间、空间信息，但是是以类似图像、视频的网格的形式。
本文将交通网络建模为一般图模型，而不是像网络或分片一样独立对待。为了处理循环网络中固有的相关性，在时间轴上使用了一个全卷积结构。
#### Contribution or Method
本文提出的新的深度学习架构由几个时空卷积块和全连接的输出层构成，时空卷积块包括位于中间的一个图卷积层和两端的临时门卷积层。Graph-Conv中，使用了Chebyshev多项式近似、1st-order近似算法来近似计算获得图模型，降低了计算复杂度。

> 作者来自：  
1 School of Mathematical Sciences, Peking University, Beijing, China  
2 Academy for Advanced Interdisciplinary Studies, Peking University, Beijing, China  
3 Center for Data Science, Peking University, Beijing, China  
4 Beijing Institute of Big Data Research (BIBDR), Beijing, China  
{byu, htyin, zhanxing.zhu}@pku.edu.cn

### [6]A Non-Parametric Generative Model for Human Trajectories
#### Meta-Info
meta | info
---|---
Authors | Kun Ouyang1;2, Reza Shokri1, David S. Rosenblum1, Wenzhuo Yang2
Citation | 0
Application | 预测人类移动轨迹/位置跟踪
Method/Model | 非参数生成模型，生成对抗网络GAN
Related works | order-K MC、time dependent MC、input-output HMM、HSMM、LSTM
dataset | Nokia Lausanne location trajectories
baselines | First-order MC、Time-dependent MC、HMM、LSTM-MLE

#### Abstract
人类移动轨迹建模和结合现实的位置轨迹生成在很多分析（隐私意识）中起到了基础作用，并设计为基于位置数据进行实现。本文中，我们提出了一种非参生成模型来生成人类移动的位置轨迹和语义特征。我们为位置跟踪设计了一种简单、直接的有效嵌入，使用生成对抗网络来产生此空间中的数据点。我们基于现实位置轨迹评估了我们的方法，并比较了我们的综合轨迹和多种现存方法，在它们提供的真实轨迹的地理和语义特征上独立和集成的比较。我们的实验结果证明我们的生成模型能够保留真实数据中多样有效的属性。

#### Others
相关工作中提及的模型：  
- order-K MC(2004)、time dependent MC(2011)：基于Markov Chain，将位置作为状态进行建模，当前状态仅依赖前序K-step信息。  
- HMM(2016/2017)：缺点是没有完整地对状态的时间依赖性进行建模。
- HSMM(2014)：把状态持续时间放入隐藏变量。
- MC和HMM/HSMM：都需要隐藏变量的预定义参数概率。
- LSTM(2016/2017)：性能优于HMM/HSMM，但由于使用最大log-likelihood方法，存在exposure bias问题。

> 作者来自：  
> 1 National University of Singapore  
2 SAP Innovation Center Singapore  
fouyangk, reza, davidg@comp.nus.edu.sg, wenzhuo.yang01@sap.com  

### [7]Deep into Hypersphere: Robust and Unsupervised Anomaly Discovery in Dynamic Networks
#### Meta-Info
meta | info
---|---
Authors | Xian Teng1, Muheng Yan1, Ali Mert Ertugrul1;2, Yu-Ru Lin1∗
Citation | 1
Application | 异常行为识别
Method/Model | DeepSphere，包括hypersphere learning、LSTM-AE
Related works | Dynamic networks、Deep Learning
dataset | 人造数据、真实数据（New York City Taxi Trips、Tweets & News Collection、restaurant video）
baselines | RAE、LSTM-AE、TSHL、LOF、OC-SVM

#### Abstract
在很多领域中匿名系统不断增长和灵活地应用--从智能传输系统、信息系统，到商业交易管理--使得这些系统对“正常”和“异常”行为的理解成为挑战。因为系统可能由内部区域、子系统间的关系构成，系统不仅需要在异常情形时警告用户，还需要提供透明性，透明性是识别异常是如何偏离了正常行为以便做出更恰当的处理。我们提供了一个统一的异常发现框架“DeepSphere”，它可以同时满足以上两种需求，即：识别异常用例并进一步地在时空上下文中探测这个用例的异常结构。DeepSphere利用了deep autoencoder和hypersphere学习方法，具备隔离异常污染和重构异常行为的能力。DeepSphere不依赖人类标注的样例，能够泛化至未见过数据。在人造和真实数据集的大量实验证明了所提出方法的稳定/一致性和鲁棒性。

#### Others
相关工作中提及的模型：  
* Dynamic Networks：近年针对异常行为检测主要是动态图方法。GraphScope、EventTree+、NetSpot、NPHGS、Time-Series Hyperesphere Learning。这些方法的缺点是不能同时从高层和低层角度进行分析，使用的特征都不相同。DeepSphere能够学习隐藏的非线性特征。
* Deep Learning：
  * Deep AutoEncoder：RDA、RCAE、RPCA等，缺点是假定输入矩阵是低秩矩阵，输出是稀疏矩阵，同时损失函数计算困难。DeepSphere是端到端学习模型，不需要优化算法。
  * RNN：LSTM-AD、EncDec-AD，缺点是有监督学习，需要干净的正常行为数据集。DeepSphere通过Hypersphere learning组件阻止了异常行为数据，保证了学习数据集的干净度。

> 作者来自：  
1 School of Computing and Information, University of Pittsburgh, USA  
2 Graduate School of Informatics, Middle East Technical University, Turkey  
fxian.teng, yanmuheng, ertugrul, yuruling@pitt.edu

### [8]Dynamically Hierarchy Revolution: DirNet for Compressing Recurrent Neural Network on Mobile Devices
#### Meta-Info
meta | info
---|---
Authors | Jie Zhang1∗, Xiaolong Wang2y, Dawei Li2, Yalin Wang1
Citation | 0
Application | 模型压缩，DirNet
Method/Model | 权重矩阵W进行稀疏编码实现压缩
Related works | RNN的压缩方法：剪枝、low-rank近似、知识提取、低精度量化，LSTM剪枝
dataset | Penn Tree Bank dataset(PTB)、LibriSpeech语料库
baselines | LSTM-SVD、LSTM-ODL

#### Abstract
RNN在各种问题上实现了优异的性能。然而，由于其高计算和内存要求，在资源受限设备上部署RNN是一项挑战性任务。为了保证高压缩率下最小准确度损失和受移动资源需求驱动，我们提出了一种基于快速字典学习算法新的DirNet压缩方法，该方法1）动态挖掘带有层的预测字典矩阵的字典原子来调整压缩比。2）自适应地改变跨层稀疏码的稀疏性。在语言模型和训练了1000h对话数据集的ASR模型的实验结果表明我们的方法显著优于以前的方法。在现有移动设备上评估显示，我们在实时模型推理上能够把原始模型大小减小了8倍，且准确度损失可忽略不计。

#### Others
主要思路：W近似等于D*Z，通过自适应学习优化得D，自适应网络结构优化Z，最后得到优化的W。

> 作者来自：  
1Arizona State University, Tempe, AZ, USA  
2Samsung Research America, Mountain View, CA, USA  
fjiezhang.joena, ylwangg@asu.edu, fxiaolong.w, dawei.lg@samsung.com