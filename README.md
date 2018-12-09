《Python机器学习及实践：从零开始通往Kaggle竞赛之路（第2版）》开源数据和代码
Datasets,tools and codes for the book: DIY_ML_Systems_with_Python_2nd_Edition
=====

第二版概要：
------

《Python机器学习实践（第二版）》一书适合所有对（深度）机器学习（Machine Learning）、数据挖掘（Data Mining），以及自然语言处理（Natural Language Processing）的技术实践感兴趣的初学者。

本书从零开始，以Python编程语言为基础，在不赘述大量数学模型与复杂编程知识的前提下，带领读者逐步熟悉并且掌握当下最流行的（深度）机器学习、数据挖掘，以及自然语言处理的开源工具库（包）：Scikit-learn、Google Tensorflow、Pandas、Matplotlib、NLTK、Gensim、XGBoost、OpenAI Gym等。从而使读者最终对Kaggle（国际最流行的机器学习竞赛平台）上的公开竞赛，乃至现实中的工程或者科研难题，都能够以Python语言快速上手；并结合本书推荐的开源工具，搭建行之有效的计算机程序来解决问题。全书分为五大章节，具体包括：

1.简介篇：介绍了机器学习的基本概念和要素、Python编程的基础知识，以及实践本书后续实例所需要的平台搭建步骤。

2.基础篇：讲述如何使用Scikit-learn作为基础工具，学习经典的（非深度/浅层）机器学习（监督、半监督、无监督、强化）模型，并从事相关的数据分析和预测任务。

3.进阶篇：涉及怎样借助Google TensorFlow平台实现深度（前馈、卷积、循环、对抗等）网络模型和技术，进一步提升现有机器学习系统的性能表现。

4.高级篇：阐述了对机器学习模型进一步优化选择和（超）参数调节的常用技巧；并补充了几项时下前沿的研究课题和一些处理实际应用所需要的其他工具（平台）。

5.实战篇：以Kaggle平台上的实际竞赛任务为例，帮助读者一步步使用本书介绍过的传统机器学习以及深度学习的模型和技巧，完成多项具有代表性的竞赛任务。

第二版目录

推荐序	9

第一版前言	13

第二版前言	15

1	简介篇	18

1.1	机器学习综述	18

1.1.1	任务（Task）	22

1.1.2	经验（Experience）	25

1.1.3	性能（Performance）	27
1.2	Python编程库	31
1.2.1	为什么要使用Python	32
1.2.2	Python机器学习的优势	33
1.2.3	NumPy & SciPy	34
1.2.4	Pandas	35
1.2.5	Matplotlib	35
1.2.6	Scikit-learn	36
1.2.7	Anaconda	36
1.2.8	Google TensorFlow	37
1.3	Python环境配置	39
1.3.1	Windows系统的环境搭建	40
1.3.2	Mac OS系统的环境搭建	40
1.3.3	Linux系统的环境搭建	51
1.4	Python编程基础	51
1.4.1	Python基本语法	51
1.4.2	Python数据类型	54
1.4.3	Python数据运算	57
1.4.4	Python流程控制	62
1.4.5	Python函数（模块）设计	65
1.4.6	Python编程库（包）导入	66
1.4.7	Python编程综合实践	67
1.5	章末小结	70
2	基础篇	71
2.1	经典的监督学习方法（Supervised Learning）	71
2.1.1	分类学习（Classification）	72
2.1.2	回归预测（Regression）	74
2.2	经典的半监督学习方法（Semi-supervised Learning）	75
2.2.1	自训练学习（Self-training）	75
2.2.2	协同训练学习（Co-training）	75
2.2.3	标签传播算法（Label Propagation）	75
2.3	经典的无监督学习方法（Unsupervised Learning）	75
2.3.1	聚类算法（Clustering）	75
2.3.2	特征降维与可视化（Dimensionality Reduction & Visualization）	76
2.3.3	概率密度估计（Density Estimation）	76
2.3.4	离群点检测（Novelty and Outlier Detection）	76
2.4	经典的强化学习方法（Reinforcement Learning）	76
2.4.1	策略梯度方法（Policy Gradient）	76
2.4.2	Q值法 （Q Learning）	76
2.5	章末小结	76
3	进阶篇	77
3.1	人工神经网络发展简史	77
3.2	前馈神经网络（Feedforward Neural Network）	78
3.3	卷积神经网络（Convolutional Neural Network）	79
3.4	残差神经网络（Residual Neural Network）	79
3.5	循环神经网络（Recurrent Neural Network）	79
3.6	自动编码器（AutoEncoder）	79
3.7	注意力机制（Attention Mechanism）	79
3.8	生成对抗网络（Generative Adversarial Network）	79
3.9	章末小结	79
4	高级篇	80
4.1	模型优化技巧	80
4.1.1	特征抽取（Feature Extraction）	80
4.1.2	特征选择（Feature Selection）	80
4.1.3	特征缩放（Feature Scaling）	80
4.1.4	模型过拟合与欠拟合（Overfitting & Underfitting）	81
4.1.5	模型正则化（Regularization）	81
4.1.6	随机失活（Dropout）	81
4.1.7	批标准化（Batch Normalization）	81
4.1.8	模型检验（Validation）	81
4.1.9	超参数搜索（Grid Search）	81
4.2	其他流行库实践	82
4.2.1	自然语言处理工具包（NLTK）	82
4.2.2	词向量技术（W2V, Glove & ELMo）	82
4.2.3	XGBoost模型	82
4.2.4	OpenAI Gym	82
4.3	前沿课题探究	83
4.3.1	深度强化学习（Deep Reinforcement Learning）	83
4.3.2	生成对抗网络辅助的半监督学习（Semi-supervised Learning with Generative Adversarial Networks）	83
4.3.3	对抗网络强化的半监督学习（Adversarial Networks for Reinforced Semi-supervised Learning）	83
4.4	章末小结	83
5	实战篇	84
5.1	Kaggle平台简介	84
5.2	Titanic罹难乘客预测	84
5.3	Ames房产价值评估	84
5.4	IMDB影评打分估计	84
5.5	CIFAR-10图像识别	84
5.6	章末小结	84
第一版后记	85
第二版后记	87




第二版代码下载地址
----

https://pan.baidu.com/s/122hrWsKnxjuHkz1rVYg6Kg (中国大陆地区的读者建议使用百度网盘)

https://github.com/godfanmiao/DIY_ML_Systems_with_Python_2nd_Edition (港澳台，以及海外地区的读者推荐使用Github)


第一版购买地址列表
----

https://books.google.com/books?id=6Ay-swEACAAJ (Google Books) 

https://www.amazon.cn/dp/B01M9813G3 (Amazon China) 

https://www.amazon.com/dp/B01M9813G3 (Amazon U.S.)

https://item.jd.com/11983227.html (JD.com) 


第一版代码下载地址：
----
https://pan.baidu.com/s/1bGp15G (中国大陆地区的读者建议使用百度网盘)

https://github.com/godfanmiao/DIY_ML_Systems_with_Python_1st_Edition (港澳台，以及海外地区的读者推荐使用Github)

