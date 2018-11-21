---
layout: post
title:  "机器学习领域会议总结"
categories: paper-reading
tags: conference
---

> 会议部分参考：https://www.zhihu.com/question/31617024

### AAAI, The Association for the Advancement of Artificial Intelligence, 国际人工智能协会
主页：http://www.aaai.org/Conferences/conferences.php  

### NIPS, Neural Information Processing Systems, 神经信息处理系统大会
主页：https://nips.cc/  

### IJCAI, International Joint Conference on Artificial Intelligence, 国际人工智能联合会议
主页：https://www.ijcai-18.org/  
IJCAI始于1969年，最初每2年举行一次，从2015年开始改为每年一次。

### ICML, International Conference on Machine Learning, 国际机器学习大会
主页：https://2017.icml.cc/

### ICLR, International Conference on Learning Representations, 国际学习表征会议
主页：https://iclr.cc/  
2013年才刚刚成立了第一届。这个一年一度的会议虽然今年(2018)才办到第六届，但已经被学术研究者们广泛认可，被认为「深度学习的顶级会议」。  
这个会议的来头不小，由位列深度学习三大巨头之二的 Yoshua Bengio 和 Yann LeCun 牵头创办。  
Yoshua Bengio 是蒙特利尔大学教授，深度学习三巨头之一，他领导蒙特利尔大学的人工智能实验室（MILA）进行 AI 技术的学术研究。MILA 是世界上最大的人工智能研究中心之一，与谷歌也有着密切的合作。  
而 Yann LeCun 就自不用提，同为深度学习三巨头之一的他现任 Facebook 人工智能研究院（FAIR）院长、纽约大学教授。作为卷积神经网络之父，他为深度学习的发展和创新作出了重要贡献。  
至于创办 ICLR 的原因何在，雷锋网尝试从 Bengio 和 LeCun 于 ICLR 第一届官网所发布的公开信推测一二。  
「众所周知，数据的应用表征对于机器学习的性能有着重要影响。表征学习的迅猛发展也伴随着不少问题，比如我们如何更好地从数据中学习更具含义及有效的表征。我们对这个领域展开了探索，包括了深度学习、表征学习、度量学习、核学习、组合模型、非常线性结构预测及非凸优化等问题。尽管表征学习对于机器学习及包括视觉、语音、音频及 NLP领域起着至关重要的作用，目前还缺乏一个场所，能够让学者们交流分享该领域所关心的话题。ICLR 的宗旨正是填补这一鸿沟。」  
从两人的说法中，ICLR希望能为深度学习提供一个专业化的交流平台。但实际上 ICLR 不同于其它国际会议，得到好评的真正原因，并不只是他们二位所自带的名人光环，而在于它推行的 Open Review 评审制度。
> 作者：Amusi  
> 链接：https://www.zhihu.com/question/47940549/answer/329124002  
> 来源：知乎  
> 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。  
> 原文：http://finance.jrj.com.cn/tech/2017/04/20153122355479.shtml

### ICCV, International Conference on Computer Vision, 计算机视觉国际会议
主页：https://iccv2017.thecvf.com  
ICCV奇数年举办，ECCV偶数年举办
### CVPR, Conference on Computer Vision and Pattern Recognition, 计算机视觉与模式识别会议
主页：https://cvpr2017.thecvf.com
### ECCV, European Conference on Computer Vision, 欧洲计算机视觉会议
主页：https://eccv2018.org  
ICCV奇数年举办，ECCV偶数年举办
### 深度学习三巨头
#### Geoffrey E. Hinton
英国出生的计算机学家和心理学家，以其在神经网络方面的贡献闻名。Hinton是反向传播算法和对比散度算法的发明人之一，也是深度学习的积极推动者。目前担任多伦多大学计算机科学系教授。2013年3月加入Google，领导Google Brain项目。Hinton被人们称为“深度学习教父”，可以说是目前对深度学习领域影响最大的人。  
主页：http://www.cs.toronto.edu/~hinton/

#### Yann LeCun
法国出生的计算机科学家，他最著名的工作是光学字符识别和计算机视觉上使用卷积神经网络（CNN），他也被称为卷积网络之父。曾在多伦多大学跟随Geoffrey Hinton做博士后。1988年加入贝尔实验室，在贝尔实验室工作期间开发了一套能够识别手写数字的卷积神经网络系统，并把它命名为LeNet。这个系统能自动识别银行支票。2003年去了纽约大学担任教授，现在是纽约大学终身教授。2013年12月加入了Facebook，成为Facebook人工智能实验室的第一任主任。  
主页：http://yann.lecun.com/

#### Yoshua Bengio
毕业于麦吉尔大学，在MIT和贝尔实验室做过博士后研究员，自1993年之后就在蒙特利尔大学任教。在预训练问题，自动编码器降噪等领域做出重大贡献。Hinton、LeCun和Bengio被人们称为“深度学习三巨头”。这“三巨头”中的前两人早已投身工业界，而Bengio仍留在学术界教书，他曾说过：“我留在学术圈为全人类作贡献，而不是为某一公司赚钱”。2017年初Bengio选择加入微软成为战略顾问。他表示不希望有一家或者两家公司（他指的显然是Google和Facebook）成为人工智能变革中的唯一大玩家，这对研究社区没有好处，对人类也没有好处。  
主页：http://www.iro.umontreal.ca/~bengioy/yoshua_en/index.html  

bengio的主要贡献在于：
1. 他对rnn的一系列推动包括经典的neural language model，gradient vanishing 的细致讨论，word2vec的雏形，以及现在的machine translation；
2. 他是神经网络复兴的主要的三个发起人之一（这一点他们三个人都承认，之前他们一直在谋划大事，正是他们三个人的坚持才有了现在的神经网络复兴，这点最高票答案说的很对）包括了pre－training的问题，如何initialize参数的问题，以denoising atuencoder为代表的各种各样的autoencoder结构，generative model等等。
3. symbolic computional graph思想的theano。这个库启发了后来的多个库的开发（直接基于它的库就不提了比如keras），包括国内很火的MXnet，google的tensorflow以及berkeley的cgt等等，可以说这个工具以及所涵盖的思想可以算同类型库的鼻祖。
4. ICLR的推动者，个人认为ICLR是一种崭新的会议形式，包容开放，也正符合bengio本人的思想。
5. 其他paper。
> 作者：张赛峥
链接：https://www.zhihu.com/question/37922364/answer/74125553
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。