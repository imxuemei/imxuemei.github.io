---
layout: post
title: "2017-2018半监督学习、迁移学习、在线学习4大会议论文汇总"
categories: Paper-Reading
tags: IJCAI ICML AAAI NIPS semi-supervised-learning online-learning transfer-learning
---

# ICML
## 2018: semi-supervised learning
### Semi-Supervised Learning via Compact Latent Space Clustering
- Konstantinos Kamnitsas, Daniel Castro, Loic Le Folgoc, Ian Walker, Ryutaro Tanno, Daniel Rueckert, Ben Glocker, Antonio Criminisi, Aditya Nori ; PMLR 80:2459-2468
- 使用紧凑潜在空间集的半监督学习方法

### Semi-Supervised Learning on Data Streams via Temporal Label Propagation
- Tal Wagner, Sudipto Guha, Shiva Kasiviswanathan, Nina Mishra ; PMLR 80:5095-5104
- 使用时间标注传播的数据流上的半监督学习

## 2018: transfer learning
### Explicit Inductive Bias for Transfer Learning with Convolutional Networks
- Xuhong LI, Yves Grandvalet, Franck Davoine ; PMLR 80:2825-2834
- 卷积网络中用于迁移学习的明确归纳偏置

### Transfer Learning via Learning to Transfer
- Ying WEI, Yu Zhang, Junzhou Huang, Qiang Yang ; PMLR 80:5085-5094
- 通过学习实现迁移的迁移学习，来自香港科技大学团队

### Curriculum Learning by Transfer Learning: Theory and Experiments with Deep Networks
- Daphna Weinshall, Gad Cohen, Dan Amir ; PMLR 80:5238-5246
- 通过迁移学习的课程体系学习：深度网络的理论和实验
<!-- more -->

## 2018: online learning
### Online Learning with Abstention
- Corinna Cortes, Giulia DeSalvo, Claudio Gentile, Mehryar Mohri, Scott Yang ; PMLR 80:1059-1067
- 带弃权的在线学习

## 2017: semi-supervised learning
### Semi-Supervised Classification Based on Classification from Positive and Unlabeled Data
- Tomoya Sakai, Marthinus Christoffel Plessis, Gang Niu, Masashi Sugiyama ; PMLR 70:2998-3006
- 基于半监督分类方法从正例和未标注数据中进行分类

## 2017: online learning
### The Price of Differential Privacy for Online Learning
- Naman Agarwal, Karan Singh ; PMLR 70:32-40
- 用于在线学习的差别隐私的价格

### Follow the Compressed Leader: Faster Online Learning of Eigenvectors and Faster MMWU
- Zeyuan Allen-Zhu, Yuanzhi Li ; PMLR 70:116-125
- 追随被压缩的领导者：特征向量的更快地在线学习和更快地MMWU

### Emulating the Expert: Inverse Optimization through Online Learning
- Andreas Bärmann, Sebastian Pokutta, Oskar Schneider ; PMLR 70:400-410
- 仿真专家：使用在线学习的反转优化

### Online Learning with Local Permutations and Delayed Feedback
- Ohad Shamir, Liran Szlak ; PMLR 70:3086-3094
- 使用局部置换和延迟反馈的在线学习

### Model-Independent Online Learning for Influence Maximization
- Sharan Vaswani, Branislav Kveton, Zheng Wen, Mohammad Ghavamzadeh, Laks V. S. Lakshmanan, Mark Schmidt ; PMLR 70:3530-3539
- 用于影响最大化的模型独立的在线学习

### Projection-free Distributed Online Learning in Networks
- Wenpeng Zhang, Peilin Zhao, Wenwu Zhu, Steven C. H. Hoi, Tong Zhang ; PMLR 70:4054-4062
- 网络中自有规划的分布式在线学习

### Online Learning to Rank in Stochastic Click Models
- Masrour Zoghi, Tomas Tunys, Mohammad Ghavamzadeh, Branislav Kveton, Csaba Szepesvari, Zheng Wen ; PMLR 70:4199-4208
- 用于随机点击模型中评分的在线学习

## 2017: transfer learning
### Deep Transfer Learning with Joint Adaptation Networks
- Mingsheng Long, Han Zhu, Jianmin Wang, Michael I. Jordan ; PMLR 70:2208-2217
- 使用联合自适应网络的深度迁移学习

# NIPS
## 2018: semi-supervised
### Bayesian Semi-supervised Learning with Graph Gaussian Processes
- Yin Cheng Ng, Nicolò Colombo, Ricardo Silva
- 使用图高斯过程的贝叶斯半监督学习

### The Pessimistic Limits and Possibilities of Margin-based Losses in Semi-supervised Learning
- Jesse Krijthe, Marco Loog
- 半监督学习中的悲观限制和基于边距损失的可能性

### Realistic Evaluation of Deep Semi-Supervised Learning Algorithms
- Avital Oliver, Augustus Odena, Colin A. Raffel, Ekin Dogus Cubuk, Ian Goodfellow
- 深度半监督学习算法的真实评估

### Semi-Supervised Learning with Declaratively Specified Entropy Constraints 
- Haitian Sun, William W. Cohen, Lidong Bing
- 带声明指定熵约束的半监督学习

### Semi-supervised Deep Kernel Learning: Regression with Unlabeled Data by Minimizing Predictive
- Variance Neal Jean, Sang Michael Xie, Stefano Ermon
- 半监督深度核学习：通过最小化预测实现未标注数据的回归算法

### The Sample Complexity of Semi-Supervised Learning with Nonparametric Mixture Models
- Chen Dan, Liu Leqi, Bryon Aragam, Pradeep K. Ravikumar, Eric P. Xing
- 带非参混合模型的半监督学习的样本复杂性

## 2018: transfer learning
### Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning
- Tyler Scott, Karl Ridgeway, Michael C. Mozer
- 适配的深度嵌入：K-shot归纳迁移学习的合成方法

### Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis 
- Ye Jia, Yu Zhang, Ron Weiss, Quan Wang, Jonathan Shen, Fei Ren, zhifeng Chen, Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu
- 从声纹识别到多重声线语音合成的迁移学习，Google出品

### Scalable Hyperparameter Transfer Learning
- Valerio Perrone, Rodolphe Jenatton, Matthias W. Seeger, Cedric Archambeau
- 可扩展的超参数迁移学习

### Transfer Learning with Neural AutoML
- Catherine Wong, Neil Houlsby, Yifeng Lu, Andrea Gesmundo
- 使用神经AutoML的迁移学习

### Hardware Conditioned Policies for Multi-Robot Transfer Learning 
- Tao Chen, Adithyavairavan Murali, Abhinav Gupta
- 用于多robot迁移学习的硬件约束策略

## 2018: online learning
### Generalized Inverse Optimization through Online Learning
- Chaosheng Dong, Yiran Chen, Bo Zeng
- 通过在线学习的泛化反转优化

### Adaptive Online Learning in Dynamic Environments 
- Lijun Zhang, Shiyin Lu, Zhi-Hua Zhou
- 动态环境中的自适应在线学习

### Online Learning with an Unknown Fairness Metric
- Stephen Gillen, Christopher Jung, Michael Kearns, Aaron Roth
- 使用未知公平度量的在线学习
### Faster Online Learning of Optimal Threshold for Consistent F-measure Optimization
- Xiaoxuan Zhang, Mingrui Liu, Xun Zhou, Tianbao Yang
- 用于一致F度量优化的优化阈值的快速在线学习
### Community Exploration: From Offline Optimization to Online Learning 
- Xiaowei Chen, Weiran Huang, Wei Chen, John C. S. Lui
- 社区探索：从离线优化到在线学习
### Online Learning of Quantum States
- Scott Aaronson, Xinyi Chen, Elad Hazan, Satyen Kale, Ashwin Nayak
- 量子态的在线学习

## 2017: semi-supervised learning
### Semi-Supervised Learning for Optical Flow with Generative Adversarial Networks
- Wei-Sheng Lai, Jia-Bin Huang, Ming-Hsuan Yang
- 用于视觉流的使用GAN的半监督学习
### Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results
- Antti Tarvainen, Harri Valpola
- 公平的老师是更好的角色模型：平均权重一致性目标提升半监督深度学习效果
### ExtremeWeather: A large-scale climate dataset for semi-supervised detection, localization, and understanding of extreme weather events
- Evan Racah, Christopher Beckham, Tegan Maharaj, Samira Ebrahimi Kahou, Mr. Prabhat, Chris Pal
- 极端天气：大规模气候数据集下半监督学习检测、定位和极端天气事件的识别
### Semi-supervised Learning with GANs: Manifold Invariance with Improved Inference
- Abhishek Kumar, Prasanna Sattigeri, Tom Fletcher
- 使用GAN的半监督学习：使用改进推理的多种不变性
### Learning Disentangled Representations with Semi-Supervised Deep Generative Models
- Siddharth Narayanaswamy, T. Brooks Paige, Jan-Willem van de Meent, Alban Desmaison, Noah Goodman, Pushmeet Kohli, Frank Wood, Philip Torr
- 使用半监督深度生成模型的解耦表示的学习
### Good Semi-supervised Learning That Requires a Bad GAN 
- Zihang Dai, Zhilin Yang, Fan Yang, William W. Cohen, Ruslan R. Salakhutdinov
- 好的半监督学习需要一个坏的GAN

## 2017: transfer learning
### Hypothesis Transfer Learning via Transformation Functions
- Simon S. Du, Jayanth Koushik, Aarti Singh, Barnabas Poczos
- 使用变换函数的假设迁移学习
### Robust and Efficient Transfer Learning with Hidden Parameter Markov Decision Processes
- Taylor W. Killian, Samuel Daulton, George Konidaris, Finale Doshi-Velez
- 使用隐藏参数Markov决策过程的鲁棒性和有效的迁移学习

## 2017: online learning
### Online Learning of Optimal Bidding Strategy in Repeated Multi-Commodity Auctions
- M. Sevi Baltaoglu, Lang Tong, Qing Zhao
- 在重复多商品拍卖中优化投标策略的在线学习
### Online Learning for Multivariate Hawkes Processes
- Yingxiang Yang, Jalal Etesami, Niao He, Negar Kiyavash
- 多变量Hawkes过程的在线学习
### Stochastic and Adversarial Online Learning without Hyperparameters 
- Ashok Cutkosky, Kwabena A. Boahen
- 无超参的随机和对抗在线学习
### Online Learning with Transductive Regret
- Mehryar Mohri, Scott Yang
- 使用Transductive Regret的在线学习
### Online Learning with a Hint
- Ofer Dekel, arthur flajolet, Nika Haghtalab, Patrick Jaillet
- 带提示的在线学习
### Parameter-Free Online Learning via Model Selection
- Dylan J. Foster, Satyen Kale, Mehryar Mohri, Karthik Sridharan
- 使用模型选择的无参在线学习

# AAAI
## 2018: semi-supervised learning
### AAAI18 - Artificial Intelligence and the Web
#### Inferring Emotion from Conversational Voice Data: A Semi-Supervised Multi-Path Generative Neural Network Approach
- Suping Zhou,  Jia Jia,  Qi Wang,  Yufei Dong,  Yufeng Yin,  Kehua Lei
- 从语音会话数据中推理情感：一种半监督多路径生成神经网络方法
### AAAI18 - Human Computation and Crowd Sourcing
#### Semi-Supervised Learning From Crowds Using Deep Generative Models
- Kyohei Atarashi,  Satoshi Oyama,  Masahito Kurihara
- 使用深度生成模型从人群中进行半监督学习
### AAAI18 - Machine Learning Applications
#### DeepHeart: Semi-Supervised Sequence Learning for Cardiovascular Risk Prediction
- Brandon Ballinger,  Johnson Hsieh,  Avesh Singh,  Nimit Sohoni,  Jack Wang,  Geoffrey H. Tison,  Gregory M. Marcus,  Jose M. Sanchez,  Carol Maguire,  Jeffrey E. Olgin,  Mark J. Pletcher
- DeepHeart: 用于心血管风险预测的半监督序列学习
#### Semi-Supervised Biomedical Translation With Cycle Wasserstein Regression GANs  
- Matthew B. A. McDermott,  Tom Yan,  Tristan Naumann,  Nathan Hunt,  Harini Suresh,  Peter Szolovits,  Marzyeh Ghassemi
- 使用循环Wasserstein回归GAN的半监督生物医学翻译
### AAAI18 - Machine Learning Methods
#### ARC: Adversarial Robust Cuts for Semi-Supervised and Multi-Label Classification
- Sima Behpour,  Wei Xing,  Brian D. Ziebart
- ARC：用于半监督和多标签分类的对抗鲁棒性Cuts
#### Learning From Semi-Supervised Weak-Label Data  
- Hao-Chen Dong,  Yu-Feng Li,  Zhi-Hua Zhou
- 从半监督弱标注数据中学习
#### Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Learning  
- Qimai Li,  Zhichao Han,  Xiao-ming Wu
- 用于半监督学习的对图卷积网络的更深洞察
#### Adversarial Dropout for Supervised and Semi-Supervised Learning  
- Sungrae Park,  JunKeon Park,  Su-Jin Shin,  Il-Chul Moon
- 用于监督和半监督学习的对抗dropout
#### Interpretable Graph-Based Semi-Supervised Learning via Flows  
- Raif M. Rustamov,  James T. Klosowski
- 使用流的可解释的基于图的半监督学习
#### Semi-Supervised AUC Optimization Without Guessing Labels of Unlabeled Data  
- Zheng Xie,  Ming Li
- 不带未标注数据猜测标签的半监督AUC优化
### AAAI18 - Vision
#### SEE: Towards Semi-Supervised End-to-End Scene Text Recognition  
- Christian Bartz,  Haojin Yang,  Christoph Meinel
- SEE: 面向半监督端到端场景的文本识别
#### Semi-Supervised Bayesian Attribute Learning for Person Re-Identification  
- Wenhe Liu,  Xiaojun Chang,  Ling Chen,  Yi Yang
- 用于人的再次认证的半监督贝叶斯属性学习
#### Transferable Semi-Supervised Semantic Segmentation  
- Huaxin Xiao,  Yunchao Wei,  Yu Liu,  Maojun Zhang,  Jiashi Feng
- 可迁移的半监督语义分割
### Student Abstracts
#### Discriminative Semi-Supervised Feature Selection via Rescaled Least Squares Regression-Supplement  
- Guowen Yuan,  Xiaojun Chen,  Chen Wang,  Feiping Nie,  Liping Jing
- 使用Rescaled Least Squares Regression-Supplement的可区分的半监督学习特征选择
#### A Semi-Supervised Network Embedding Model for Protein Complexes Detection  
- Wei Zhao,  Jia Zhu,  Min Yang,  Danyang Xiao,  Gabriel Pui Cheong Fung,  Xiaojun Chen
- 用于蛋白质复合物检测的半监督网络嵌入模型

## 2018: transfer learning
### AAAI18 - Machine Learning Methods
#### Gaussian Process Decentralized Data Fusion Meets Transfer Learning in Large-Scale Distributed Cooperative Perception  
- Ruofei Ouyang,  Kian Hsiang Low  
- 高斯过程去中心化数据融合与大规模分布式协作感知中的迁移学习

### AAAI18 - NLP and Machine Learning
#### Few Shot Transfer Learning BetweenWord Relatedness and Similarity Tasks Using A Gated Recurrent Siamese Network  
- James O' Neill,  Paul Buitelaar  
- 在关联性和相似性任务之间使用Gated Recurrent Siamese网络Few Shot迁移学习

#### Dual Transfer Learning for Neural Machine Translation with Marginal Distribution Regularization  
- Yijun Wang,  Yingce Xia,  Li Zhao,  Jiang Bian,  Tao Qin,  Guiquan Liu,  Tie-Yan Liu
- 使用边距分布式正则的双迁移学习用于神经网络机器翻译

### Student Abstracts
#### Enhancing RNN Based OCR by Transductive Transfer Learning From Text to Images
- Yang He,  Jingling Yuan,  Lin Li  
- 增强基于OCR的RNN通过直推迁移学习实现从文本到图像的学习

#### Label Space Driven Heterogeneous Transfer Learning With Web Induced Alignment
- Sanatan Sukhija
- 使用web诱导对齐的标注空间驱动的异质迁移学习


## 2018: online learning
### AAAI18 - Machine Learning Methods
#### Online Learning for Structured Loss Spaces  
- Siddharth Barman,  Aditya Gopalan,  Aadirupa Saha
- 用于结构化损失空间的在线学习


## 2017: semi-supervised learning
### Machine Learning Applications
#### Semi-Supervised Multi-View Correlation Feature Learning with Application to Webpage Classification  
- Xiao-Yuan Jing,  Fei Wu,  Xiwei Dong,  Shiguang Shan,  Songcan Chen
- 随应用的半监督多view协同特征学习用于网页分类
#### Discriminative Semi-Supervised Dictionary Learning with Entropy Regularization for Pattern Classification  
- Meng Yang,  Lin Chen
- 使用熵正则化消除半监督字典学习用于模式分类
### Machine Learning Methods
#### Fast Generalized Distillation for Semi-Supervised Domain Adaptation  
- Shuang Ao,  Xiang Li,  Charles X. Ling
- 用于半监督域自适应的快速泛化提取
#### Semi-Supervised Adaptive Label Distribution Learning for Facial Age Estimation  
- Peng Hou,  Xin Geng,  Zeng-Wei Huo,  Jia-Qi Lv
- 用于面部年龄估计的半监督自适应标签分布式学习
#### Learning Safe Prediction for Semi-Supervised Regression  
- Yu-Feng Li,  Han-Wen Zha,  Zhi-Hua Zhou
- 用于半监督回归的安全预测学习
#### Semi-Supervised Classifications via Elastic and Robust Embedding  
- Yun Liu,  Yiming Guo,  Hua Wang,  Feiping Nie,  Heng Huang
- 使用弹性和鲁棒性嵌入的半监督分类
#### Multi-View Clustering and Semi-Supervised Classification with Adaptive Neighbours  
- Feiping Nie,  Guohao Cai,  Xuelong Li
- 使用自适应邻居的多面聚类和半监督分类
#### Fredholm Multiple Kernel Learning for Semi-Supervised Domain Adaptation  
- Wei Wang,  Hao Wang,  Chen Zhang,  Yang Gao
- 用于半监督域自适应的Fredholm多核学习
### Natural Language Processing and Machine Learning
#### A Unified Model for Cross-Domain and Semi-Supervised Named Entity Recognition in Chinese Social Media  
- Hangfeng He,  Xu Sun
- 在中国社交媒体中一种统一的模型用于跨域和半监督的实体识别
#### Variational Autoencoder for Semi-Supervised Text Classification  
- Weidi Xu,  Haoze Sun,  Chao Deng,  Ying Tan
- 用于半监督文字分类的可变自编码器
### Student Abstracts
#### Kernelized Evolutionary Distance Metric Learning for Semi-Supervised Clustering  
- Wasin Kalintha,  Satoshi Ono,  Masayuki Numao,  Ken-ichi Fukui
- 用于半监督聚类的核进化距离度量学习

## 2017: transfer learning
### Machine Learning Methods
#### Cross-Domain Kernel Induction for Transfer Learning  
- Wei-Cheng Chang,  Yuexin Wu,  Hanxiao Liu,  Yiming Yang
- 用于迁移学习的跨域核感应
#### Transfer Learning for Deep Learning on Graph-Structured Data  
- Jaekoo Lee,  Hyunjae Kim,  Jongsun Lee,  Sungroh Yoon
- 用于图结构数据上的深度学习的迁移学习
#### Sparse Deep Transfer Learning for Convolutional Neural Network  
- Jiaming Liu,  Yali Wang,  Yu Qiao
- 用于卷积神经网络的稀疏深度迁移学习
#### Distant Domain Transfer Learning  
- Ben Tan,  Yu Zhang,  Sinno Jialin Pan,  Qiang Yang
- 远域迁移学习，香港科技大学团队
### Student Abstracts
#### Robust and Efficient Transfer Learning with Hidden Parameter Markov Decision Processes  
- Taylor W. Killian,  George Konidaris,  Finale Doshi-Velez
- 使用隐藏参数Markov决策过程的鲁棒性和有效的迁移学习
### Doctoral Consortium
#### Accelerating Multiagent Reinforcement Learning through Transfer Learning  
- Felipe Leno da Silva,  Anna Helena Reali Costa
- 通过迁移学习加速多agent增强学习

## 2017: online learning
### Machine Learning Methods
#### A Framework of Online Learning with Imbalanced Streaming Data  
- Yan Yan,  Tianbao Yang,  Yi Yang,  Jianhui Chen
- 使用不平衡流式数据的在线学习框架
### Student Abstracts
#### Improving Performance of Analogue Readout Layers for Photonic Reservoir Computers with Online Learning  
- Piotr Antonik,  Marc Haelterman,  Serge Massar
- 提升模拟读出层性能用于带在线学习的光子水库计算机

# IJCAI
## 2018: semi-supervised learning
### Tri-net for Semi-Supervised Deep Learning
- Dong-Dong Chen, Wei Wang, Wei Gao, Zhi-Hua Zhou
- 用于半监督深度学习的Tri-net

### Teaching Semi-Supervised Classifier via Generalized Distillation
- Chen Gong, Xiaojun Chang, Meng Fang, Jian Yang
- 使用泛化提取来训练半监督分类器
### Self-weighted Multiple Kernel Learning for Graph-based Clustering and Semi-supervised Classification
- Zhao Kang, Xiao Lu, Jinfeng Yi, Zenglin Xu
- 用于基于图聚类和半监督分类的自权重多核学习

### Cutting the Software Building Efforts in Continuous Integration by Semi-Supervised Online AUC Optimization
- Zheng Xie, Ming Li
- 使用半监督在线AUC在连续集成中修剪软件构建影响

### Semi-Supervised Optimal Transport for Heterogeneous Domain Adaptation
- Yuguang Yan, Wen Li, Hanrui Wu, Huaqing Min, Mingkui Tan, Qingyao Wu
- 用于异质域自适应的半监督最佳传输
### Semi-Supervised Multi-Modal Learning with Incomplete Modalities
- Yang Yang, De-Chuan Zhan, Xiang-Rong Sheng, Yuan Jiang
- 使用不完全形式的半监督多模学习
### Semi-Supervised Optimal Margin Distribution Machines
- Teng Zhang, Zhi-Hua Zhou
- 半监督最佳边距分布机器
### Rademacher Complexity Bounds for a Penalized Multi-class Semi-supervised Algorithm (Extended Abstract)
- Yury Maximov, Massih-Reza Amini, Zaid Harchaoui
- Rademacher复杂边界作为惩罚项的一种多类半监督算法

## 2018: transfer learning
### Goal-Oriented Chatbot Dialog Management Bootstrapping with Transfer Learning
- Vladimir Ilievski, Claudiu Musat, Andreea Hossman, Michael Baeriswyl
- 面向目标的聊天机器人对话框管理集成迁移学习
## 2018: online learning
### Efficient Adaptive Online Learning via Frequent Directions
- Yuanyu Wan, Nan Wei, Lijun Zhang
- 使用频繁方向的有效自适应在线学习
### Bandit Online Learning on Graphs via Adaptive Optimization
- Peng Yang, Peilin Zhao, Xin Gao
- 使用自适应优化的图中的老虎机在线学习
## 2017: semi-supervised learning
### Using Graphs of Classifiers to Impose Declarative Constraints on Semi-supervised Learning
- Lidong Bing, William W. Cohen, Bhuwan Dhingra
- 使用分类器的图来消除半监督学习的声明式约束
### Semi-supervised Feature Selection via Rescaled Linear Regression
- Xiaojun Chen, Guowen Yuan, Feiping Nie, Joshua Zhexue Huang
- 使用再扩展线性回归的半监督特征选择

## 2017: transfer learning
### Transfer Learning in Multi-Armed Bandits: A Causal Approach
- Junzhe Zhang, Elias Bareinboim
- 多臂老虎机的迁移学习：一种因果方法
### Understanding How Feature Structure Transfers in Transfer Learning
- Tongliang Liu, Qiang Yang, Dacheng Tao
- 理解迁移学习中特征结构迁移是如何实现的，香港科技大学
### Completely Heterogeneous Transfer Learning with Attention - What And What Not To Transfer
- Seungwhan Moon, Jaime Carbonell
- 使用注意力机制进行完整的异质迁移学习-哪些能迁移以及哪些不能迁移
### Semi-Supervised Learning for Surface EMG-based Gesture Recognition
- Yu Du, Yongkang Wong, Wenguang Jin, Wentao Wei, Yu Hu, Mohan Kankanhalli, Weidong Geng
- 用于Surface基于EMG的手势识别的半监督学习
### Semi-supervised Max-margin Topic Model with Manifold Posterior Regularization
- Wenbo Hu, Jun Zhu, Hang Su, Jingwei Zhuo, Bo Zhang
- 带多种后向正则化项的半监督最大边距话题模型
### Semi-supervised Learning over Heterogeneous Information Networks by Ensemble of Meta-graph Guided Random Walks
- He Jiang, Yangqiu Song, Chenguang Wang, Ming Zhang, Yizhou Sun
- 通过组装元图导向的随机漫步法实现在异质信息网络上的半监督学习
### Semi-supervised Orthogonal Graph Embedding with Recursive Projections
- Hanyang Liu, Junwei Han, Feiping Nie
- 带递归规划的半监督正交图嵌入
### Adaptive Semi-Supervised Learning with Discriminative Least Squares Regression
- Minnan Luo, Lingling Zhang, Feiping Nie, Xiaojun Chang, Buyue Qian, Qinghua Zheng
- 带区别最小平方回归的自适应半监督学习
### SEVEN: Deep Semi-supervised Verification Networks
- Vahid Noroozi, Lei Zheng, Sara Bahaadini, Sihong Xie, Philip S. Yu
- SEVEN：深度半监督识别网络
### Linear Manifold Regularization with Adaptive Graph for Semi-supervised Dimensionality Reduction
- Kai Xiong, Feiping Nie, Junwei Han
- 用于半监督维度降低的带自适应图的线性多种正则化
### Semi-Supervised Deep Hashing with a Bipartite Graph
- Xinyu Yan, Lijun Zhang, Wu-Jun Li
- 带二部图的半监督深度哈希
### Adaptively Unified Semi-supervised Learning for Cross-Modal Retrieval
- Liang Zhang, Bingpeng Ma, Jianfeng He, Guorong Li, Qingming Huang, Qi Tian
- 用于跨模式检索的自适应统一半监督学习
### Robust Multilingual Named Entity Recognition with Shallow Semi-supervised Features (Extended Abstract)
- Rodrigo Agerri, German Rigau
- 带阴影半监督特征的鲁棒性多语言命名实体的识别

## 2017: online learning
暂无