---
layout: post
title: My first Kaggle competition ENSP
tags: kaggle
math: true
date: 2023-1-10 20:56 +0800
---



# Novozymes Enzyme Stability Prediction

这算是我第一个真正参加的kaggle比赛，虽然因为public和private的偏差，导致我似乎错过了一块银牌。但是在这过程中我确实认识了很有意思的人，也在这中间确确实实学到了很多东西，从最开始的纸上谈兵到后面脚踏实地的敲代码。在这过程中我也天马行空地搓出一个似乎还行的模型，虽然很大可能是这个比赛本身的原因导致我那天马行空的模型反而可以取得一定的效果。

![](.\img\SJD}5[~JQ[7(PKN7JGB~)HL.png)

准备写下这篇文章的时候刚刚结束比赛，但是中间因为种种事情耽搁拖到了现在。直到插入这张图片我才发现，时间已经过去了三个月，这个比赛似乎应该是五个月前开始的，那时候的我只是单纯想找个东西糊弄一个机器学习大作业，没想到自己却越陷越深。

# 1，比赛简介

简单来说，比赛的目的是通过给定的氨基酸序列来预测酶的稳定性。很有意思的一点是比赛所给的测试集，是某种酶的野生型和它的三千种突变型组成。训练集中也包含了多种酶的野生型及其对应的突变型。所以比赛的目的，更加准确的说是，计算野生型酶在指定位置上的指定突变给酶稳定性带来的影响。

## 1.1，数据集介绍

比赛中提供的数据集如出去训练集测试集的CSV文件外只有一个很奇怪的PDB文件。正如前文所说，无论是训练集和测试集都是由突变型与野生型组成的。

![image-20230401153104926](.\img\image-20230401153104926.png)

上面就是这个比赛的数据集，只有这些，其中的test_label.csv是比赛结束后主办方提供的一个测评用的标签。

### 1.1.1，train.csv

![](.\img\~_UW[VGL`~BI0MBQ9[{QKXN.png)

训练集的数据中，给出的主要是

1 seq_id、pH、data_source，就是简单的id、pH信息，id按照顺序排序，似乎没有提供什么额外的信息。pH由于差异不大在后续的任务中，也没有什么很大的作用。最后一个更加可能就是为了注明数据来源。

2 protein_sequence，是需要测定的酶的氨基酸序列，这项数据本身可以通过分词器等操作提取出部分信息，同时也可以用过后续的特征提取获得更加有用的信息。

3 tm 本次比赛的目标，原本的意义是它的失活温度，同时由于更高的tm意味着更高的稳定点，同时由于比赛使用的是皮尔徐系数评价最后的成绩。同时 tm 热稳定性 ddg是呈正相关的，因此在后续的比赛中两者可以近似等价。

剩下的两个CSV文件没有什么好多介绍的，同训练集。

### 1.1.2，PDB文件介绍

PDB是protein data bank的简写，在生物学软件中，一般把蛋白质的三维结构信息用pdb文件保存。

完整的PDB文件提供了非常多的信息，包括作者，参考文献以及结构说明，如二硫键，螺旋，片层，活性位点。在使用PDB文件时请记住，一些建模软件可能不支持那些错误的输入格式。（这里主要说的就是一个python相关的三方库，和rosseta这个软件）

PDB格式以文本格式给出信息, 每一行信息称为一个 记录。一个PDB文件通常包括很多不同类型的记录, 它们以特定的顺序排列，用以描述结构。

这些说起来很复杂，其实我自己也没有仔细去研究，在我的大概理解中就是一个存储着三维结构信息的文件，读取解析这些文件的方法也是我在kaggle社区上现场学的。

### 1.1.3，额外数据集介绍

由于这次比赛中给的数据集较少，同时由于比赛的规则的允许，所以使用了大量的额外数据集。这些数据集主要用于thermonet网络的训练。

。。。。。

## 1.2，AlphaFold2

这是一个在比赛中多次提到的模型，也被认为是深度学习在自然科学界的一个重大的成果。

## 1.3，EDA

我的EDA做的并不是很好，而且也是借鉴大佬的EDA，所以不在这里多写，可以直接看大佬的，我后面直接摘下来几个比较重要的结论。

# 2，我的思路

刚刚接触比赛的时候，我还是一个新手，加上但是沉迷于XGBoost，所以我的第一个模型是通过XGBoost来完成的。嗯，然后，果不其然给了我当头一棒，极低的皮尔逊系数。虽然在后续调整的数据的处理方式，但是成长的有限。后面我开始查找相关的论文，这时候一个天马行空的想法出现在我脑海中。

（当然出现了一个小插曲，我在此以前报名了另一个kaggle比赛，然后随手提交了一个ensemble public的结果，但是好巧不巧，阴差阳错拿了铜牌，这个我受之有愧，所以我删号跑路了，换了一个新的号。因此导致我现在这个账号找不到我最开始的十几次提交）

那个天马行空的想法，其实很简单，

# 3，具体实现

blog中写的比较简略，具体的代码看后续的链接（我试图重构了它，但是我当初的代码实在写得太烂了）



# 4，参考

kaggle社区最大的魅力或许就是，前面的大佬总是乐于和他们分享自己的思路和见解，这些让我从中学到很多很多。

[XGBoost - 5000 Mutations 200 PDB Files [LB 0.410]](https://www.kaggle.com/code/cdeotte/xgboost-5000-mutations-200-pdb-files-lb-0-410)

[🧬 NESP: ThermoNet v2 🧬](https://www.kaggle.com/code/vslaykovsky/nesp-thermonet-v2)

[Surface area of the amino acids in the model structure](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/357899)

[1st place solution - Protein as a Graph](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/376371)


