---
layout: post
title:  "2016 IJCAI会议信息汇总"
categories: AI-Conference
tags: IJCAI
---

## 概述
官网：[http://ijcai-16.org/](http://ijcai-16.org/)  
Proceedings：[http://www.ijcai.org/Proceedings/2016](http://www.ijcai.org/Proceedings/2016)  
Awards：[http://ijcai-16.org/index.php/welcome/view/awards](http://ijcai-16.org/index.php/welcome/view/awards)  

会议亮点之一是google AlphaGo的演讲，主要为深度强化学习实现，未来目标在健康医疗。
两篇最佳论文，一篇为规划类论文，一篇为强化学习类论文。  

<!-- more -->
**Distinguished Paper Award**   
**(2902) Hierarchical Finite State Controllers for Generalized Planning**  
*Javier Segovia, Sergio Jimenez and Anders Jonsson*  
Abstract. Finite State Controllers (FSCs) are an efective way to represent sequential plans compactly. By imposing appropriate conditions on transitions, FSCs can also represent generalized plans that solve a range of planning problems from a given domain. Tis paper introduces the concept of hierarchical FSCs for planning by allowing controllers to call other controllers. It is shown that hierarchical FSCs can represent generalized plans more compactly than individual FSCs. Moreover, the call mechanism makes it possible to generate hierarchical FSCs in a modular fashion, or even to apply recursion. Te paper also introduces a compilation that enables a classical planner to generate hierarchical FSCs that solve challenging generalized planning problems. Te compilation takes as input a set of planning problems from a given domain and outputs a single classical planning problem, whose solution corresponds to a hierarchical FSC.  
有限状态控制器（FSC）是一种紧凑地表征顺序规划的有效方式。通过在过渡上施加适当的条件，FSC 也能表征解决给定领域内的一系列的规划问题。这篇论文介绍了分层 FSC的概念，它通过允许控制器调用其它控制器来进行规划。其中证明分层 FSC 可以比个体 FSC更紧凑地表征一般规划。此外，其调用机制允许以模块化的方式生成分层 FSC，甚至应用递归方式。论文还介绍了能让经典规划者生成分层 FSC 的汇编，这能解决很有挑战性的一般规划问题。该汇编以来自特定领域的规划问题集合作为输入，然后输出一个单一经典规划问题，这种解决方案对应一个分层 FSC。

**Distinguished Student Paper Award**  
**(1310) Using Task Features for Zero-Shot Knowledge Transfer in Lifelong Learning**  
*David Isele, Eric Eaton and Mohammad Rostami*  
Abstract. Knowledge transfer between tasks can improve the performance of learned models, but requires an accurate estimate of the inter-task relationships to identify the relevant knowledge to transfer. The inter-task relationships are typically estimated based on training data for each task, which is inefficient in lifelong learning settings where the goal is to learn each consecutive task rapidly from as little data as possible. To reduce this burden, the paper develops a lifelong reinforcement learning method based on coupled dictionary learning that incorporates highlevel task descriptors to model the inter-task relationships. It is shown that using task descriptors improves the performance of the learned task policies, providing both theoretical justification for the benefit and empirical demonstration of the improvement across a variety of dynamical control problems. Given only the descriptor for a new task, the lifelong learner is also able to accurately predict the task policy through zero-shot learning using the coupled dictionary, eliminating the need to pause to gather training data before addressing the task.  
任务间的知识迁移可以提升学习模型的表现，但是需要对任务间关系进行准确评估，从而识别迁移的相关知识。这些任务间的关系一般是基于每个任务的训练数据而进行评估的，对于从少量数据中快速学习每个连续任务为目标的终身学习来说，这个设定是效率低下的。为了减轻负担，我们基于耦合词典学习开发了一个终身强化学习方法，该耦合词典学习将高阶任务描述符合并到了任务间关系建模中。我们的结果表明，使用任务描述符能改善学习到的任务策略性能，既提供了我们方法有效的理论证明，又证明展示了在一系列动态控制问题上的进步。在只给描述符一个新任务的情况下，这一终身学习器也能够通过 zero-shot 学习使用耦合词典准确预测任务策略，不再需要在解决任务之前暂停收集训练数据了。

## Related Articles
[1] [人工智能下一步将会走向何方？|第25届国际联合会议IJCAI2016导览](https://www.leiphone.com/news/201607/MAdqVomAskkn3IUv.html)  
[2] [从吃豆人到星际争霸，人工智能在一些游戏上已经能玩得和顶尖人类玩家一样好|IJCAI2016前瞻](https://www.leiphone.com/news/201607/Q15IHkuaJvvomFBc.html)  
[3] [【IJCAI2016】主旨演讲及亮点速览，华人作者抢占半壁江山](https://yq.aliyun.com/articles/178401)  
[4] [分层有限状态控制器也能用于一般规划了|IJCAI2016杰出论文详解](https://www.leiphone.com/news/201607/Vx3yL7GwO1Gvivqw.html)  
[5] [AI的发展已经失去了方向？人工智能哲学学家Aaron Sloman IJCAI演讲](https://www.leiphone.com/news/201608/MIr9ziHXbIP0lcE4.html)  
[6] [没有黑科技 IJCAI 2016我们看什么？](https://www.leiphone.com/news/201607/aZoG0LuorhbZC1z9.html)  
[7] [在少量数据甚至无数据基础下也能进行终身学习|IJCAI2016杰出学生论文](https://www.leiphone.com/news/201607/NfF5j2y0DZ8clsvP.html)