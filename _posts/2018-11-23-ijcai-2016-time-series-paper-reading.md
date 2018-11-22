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
| Application | 目前该论文还是初级阶段，仅在数字识别、手势识别上具备实验 |
| Model/Method | SNN的4个变种方法 |
| Related works | 暂无 |
| dataset | 数字识别、手势识别 |
| baselines | SVM、LR、ENN |

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