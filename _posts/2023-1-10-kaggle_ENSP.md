---
layout: post
title: My first Kaggle competition ENSP
tags: kaggle
math: true
toc: true
date: 2023-1-10 20:56 +0800
---

纪念我第一个正式参加的kaggle比赛

# Novozymes Enzyme Stability Prediction

这算是我第一个真正参加的kaggle比赛，虽然因为public和private的偏差，导致我似乎错过了一块银牌。但是在这过程中我确实认识了很有意思的人，也在这中间确确实实学到了很多东西，从最开始的纸上谈兵到后面脚踏实地的敲代码。在这过程中我也天马行空地搓出一个似乎还行的模型，虽然很大可能是这个比赛本身的原因导致我那天马行空的模型反而可以取得一定的效果。

![](.\img\SJD}5[~JQ[7(PKN7JGB~)HL.png)

准备写下这篇文章的时候刚刚结束比赛，但是中间因为种种事情耽搁拖到了现在。直到插入这张图片我才发现，时间已经过去了三个月，这个比赛似乎应该是五个月前开始的，那时候的我只是单纯想找个东西糊弄一个机器学习大作业，没想到自己却越陷越深。

# 1，比赛简介

简单来说，比赛的目的是通过给定的氨基酸序列来预测酶的稳定性。很有意思的一点是比赛所给的测试集，是某种酶的野生型和它的三千种突变型组成。训练集中也包含了多种酶的野生型及其对应的突变型。所以比赛的目的，更加准确的说是，计算野生型酶在指定位置上的指定**单点突变**给酶稳定性带来的影响。

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

[14656 Unique Mutations+Voxel Features+PDBs](https://www.kaggle.com/code/vslaykovsky/14656-unique-mutations-voxel-features-pdbs)

这个数据集是截止比赛结束，最大的公开的体素特征数据集。它整合对齐了以下八个不同的数据集：

- [ThermoMutDB](http://biosig.unimelb.edu.au/thermomutdb)
- [iStable2.0](http://ncblab.nchu.edu.tw/iStable2/)
- [dt-xgboost-5000-mutations-200-pdb-files-lb-0-40](https://www.kaggle.com/code/cdeotte/xgboost-5000-mutations-200-pdb-files-lb-0-40)
- [S1626](https://aip.scitation.org/doi/suppl/10.1063/1.4947493)
- [S140](http://marid.bioc.cam.ac.uk/sdm2/data)
- [S2648](http://marid.bioc.cam.ac.uk/sdm2/data)
- [Q3214, Q1744](https://github.com/gersteinlab/ThermoNet/tree/master/data/datasets)
- [Q3421](https://github.com/gersteinlab/ThermoNet/tree/master/data/datasets)
- [FireProtDB (6713 mutations)](https://loschmidt.chemi.muni.cz/fireprotdb/)

## 1.2，DDG与DT



## 1.3，AlphaFold2

这是一个在比赛中多次提到的模型，也被认为是深度学习在自然科学界的一个重大的成果。

## 1.4，EDA

我的EDA做的并不是很好，而且也是借鉴大佬的EDA，所以不在这里多写，可以直接看大佬的，我后面直接摘下来几个比较重要的结论。

# 2，我的思路

刚刚接触比赛的时候，我还是一个新手，加上当时沉迷于XGBoost，所以我的第一个模型是通过XGBoost来完成的。嗯，然后，果不其然给了我当头一棒，极低的皮尔逊系数。虽然在后续调整的数据的处理方式，但是成长的有限。后面我开始查找相关的论文，这时候一个天马行空的想法出现在我脑海中。

（当然出现了一个小插曲，我在此以前报名了另一个kaggle比赛，然后随手提交了一个ensemble public的结果，但是好巧不巧，阴差阳错拿了铜牌，这个我受之有愧，所以我删号跑路了，换了一个新的号。因此导致我现在这个账号找不到我最开始的十几次提交）

那个天马行空的想法，其实很简单。提取出不同的编码，再将这些编码送入XGBoost模型中训练。

首先是基于结构信息的编码。先从蛋白质的三维结构中，提取出体素特征。利用体素特征和DDG训练thermonet，将thermonet最后的分类器拆解下来，提取出分类器的输入作为体素特征的编码。

之后是基于能量的特征，这里我选用的是全局和局部 Rosetta 能量分数。

基于序列的特征，我选用的是blosum100和demask替换分数。

至于为什么不直接使用一个深度网络模型完成全部过程，主要的原因如下。**因为蛋白质结果稳定性的预测，有很多domain knowledge，或者说算法应该有很多归纳偏置**，但是完全由深度网络去学习实现这些是很难而且成本很高的。同时由于该问题在生物学领域已经研究了很多年，有很多很有用的方法，如上文提到的**Rosetta 能量分数，blosum100和demask替换分数**等，使用单纯的深度网络很难结合这些。

最后的提交我得到了一个LB 0.592 本地CV 0.48的结果，当时由于后续大家更新了很多其他的方法。而LB持续走高，我也逐渐弃用了我这个天马行空的想法，最终这个天马行空的结果在private上的成绩是**0.52403 52/2482(top2%)**

![image-20230402205656234](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\image-20230402205656234.png)

![image-20230402205729362](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\image-20230402205729362.png)



# 3，具体实现

blog中写的比较简略，具体的代码看后续的链接（我试图重构了它，但是我当初的代码实在写得太烂了）

## 3.1，体素特征的提取

讲述thermonet前，最重要的一点是理清楚什么是 voxel feature（可以直接翻译成体素特征）。体素（voxel）是像素（pixel）和体积（volume）的组合词，我们可以简单地把体素理解成三维空间中的像素。就像像素是数字数据于二维空间分割上的最小单位，而体素则是数字数据于三维空间分割上的最小单位。

理清楚什么是体素特征以后，我们就要开始从蛋白质结构中提取体素特征了。首先，我的第一步是利用大分子建模套件Rosetta从野生型的结构构建突变体的结构模型。之后就是利用原子的生物物理特性对体素特征进行参数化。在这里，我们有一个重要的假设，就是可以通过对突变位点周围的3D生物物理环境进行建模来充分捕捉点突变的ΔΔG。因此我们将蛋白质视作一种3D图像处理。并对野生型及其突变点周围进行体素化来提取特征。

这里似乎很复杂，从它的过程上来看，就是我们利用某种预定的规则，将蛋白质的3D结构提取成一个[16, 16, 16]的体素特征，用来表征其相邻原子的生物物理性质。在这里我们选用了七套预定的规则（如下图），分别对突变型蛋白质及其野生型进行体素特征参数化，最后堆叠特征图成一个[16, 16, 16, 14]的张量。

![image-20230403161724336](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\image-20230403161724336.png)

在具体实现由三步构成，第一步生成一个松弛后的蛋白质文件

```
relax.static.linuxgccrelease -in:file:s XXXXX.pdb -relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false
```

第二步通过松弛后的蛋白质文件，生成指定突变型的蛋白质文件

```
rosetta_relax.py --rosetta-bin relax.static.linuxgccrelease -l VARIANT_LIST --base-dir /path/to/where/all/XXXXX_relaxed.pdb/is/stored
```

第三步就是根据前两部的结果，生成我们需要的特征（参数化后的体素特征）

```
gends.py -i VARIANT_LIST -o test_direct_stacked_16_1 -p /path/to/where/all/XXXXX_relaxed.pdb/is/stored --boxsize 16 --voxelsize 1
```

则三步均基于大分子建模套件Rosetta实现，且后两部封装于rosetta_relax.py 与 gends.py文件，详细见github。

## 3.2，构建Thermonet

由于提取出的体素特征无法直接作为编码，投入XGBoost中使用，因此我们需要用到Thermonet对体素特征进行特征提取将[16, 16, 16, 14]的体素特征张量，转化成一个[74, ]的向量。

这一步的大体思路是，以DDG和DT为目标训练一个Thermonet模型的改良版本。将这个深度学习模型视为特征提取器和分类器的模型，其中分类器为最后的全链接层，而特征提取器为剩余部分。当这个模型训练好以后，忽略分类器的输出，以特征提取器的输出作为体素特征的编码。

```python
class ThermoNet(th.nn.Module):
    def __init__(self, params):
        super().__init__()

        CONV_LAYER_SIZES = [14, 16, 24, 32, 48, 78, 128]
        FLATTEN_SIZES = [0, 5488, 5184, 4000, 3072, 2106, 1024]

        dropout_rate = params['dropout_rate']
        dropout_rate_dt = params['dropout_rate_dt']
        dense_layer_size = int(params['dense_layer_size'])
        layer_num = int(params['conv_layer_num'])
        silu = params['SiLU']

        self.params = params
        if silu:
            activation = nn.SiLU()
        else:
            activation = nn.ReLU()

        model = [
            th.nn.Sequential(
                *[th.nn.Sequential(
                    th.nn.Conv3d(in_channels=CONV_LAYER_SIZES[l], out_channels=CONV_LAYER_SIZES[l + 1], kernel_size=(3, 3, 3)),
                    activation
                ) for l in range(layer_num)]
            ),
            th.nn.MaxPool3d(kernel_size=(2,2,2)),
            th.nn.Flatten(),
        ]
        flatten_size = FLATTEN_SIZES[layer_num]
        if self.params['LayerNorm']:
            model.append(th.nn.LayerNorm(flatten_size))
        self.model = th.nn.Sequential(*model)

        self.ddG = th.nn.Sequential(
            th.nn.Dropout(p=dropout_rate),
            th.nn.Linear(in_features=flatten_size, out_features=dense_layer_size),
            activation,
            th.nn.Dropout(p=dropout_rate),
            th.nn.Linear(in_features=dense_layer_size, out_features=1)
        )
        self.dT = th.nn.Sequential(
            th.nn.Dropout(p=dropout_rate_dt),
            th.nn.Linear(in_features=flatten_size, out_features=dense_layer_size),
            activation,
            th.nn.Dropout(p=dropout_rate_dt),
            th.nn.Linear(in_features=dense_layer_size, out_features=1)
        )


    def forward(self, x):
        if self.params['diff_features']:
            x[:, 7:, ...] -= x[:, :7, ...]
        x = self.model(x)
        ddg = self.ddG(x)
        dt = self.dT(x)
        return ddg.squeeze(), dt.squeeze()
```

这是对Thermonet的一个修改版本，让它更加适应于这个比赛任务中的特征提取。主要改动有几点：

1，用pytorch重写了原本的代码，原本的代码为tf.keras，测试对比后和原本的性能没有太大差别。

2，由于使用了更大的数据集，我选择增加了一层卷积来提高模型的容量。

3，引入了一个辅助目标dt，dt和ddg的关系在前文中已经说明，由于数据集为多个数据集的合并，所有数据集部分数据中，给出了dt目标，所以我们引入了dt，并且加入另一个分类器。用来辅助我们训练特征提取器。同时修改损失函数成：
$$
L=(y_{ΔΔG}−\hat{y}_{ΔΔG})^2+C*(y_{ΔT}−\hat{y}_{ΔT})^2
$$


训练的过程没有很大的改动，只是由于引入了辅助目标，在计算损失函数时，需要将ddg和dt两者的损失stack在一起。

```python
for x, ddg, dt in tqdm(dl_val, desc='train', disable=True): 
    ddg_pred, dt_pred = model(x.to(DEVICE))
    ddg_preds.append(ddg_pred.cpu().numpy())
    dt_preds.append(dt_pred.cpu().numpy())
    ddg = ddg.to(DEVICE)
    dt = dt.to(DEVICE)
    not_nan_ddg = ~th.isnan(ddg)
    ddg_loss = criterion(ddg[not_nan_ddg], ddg_pred[not_nan_ddg])

    not_nan_dt = ~th.isnan(dt)
    dt_loss = criterion(dt[not_nan_dt], dt_pred[not_nan_dt])

    loss = th.stack([ddg_loss, dt_loss * params['C_dt_loss']])
    loss = loss[~th.isnan(loss)].sum()
```

在训练好模型以后，我测试了一下输出的性能，对于该比赛，皮尔逊系数从0.39，提升到0.49。我猜测这或许不是模型本身预测能力更强了，而是模型更加适应当前的任务了。

## 3.3，体素特征的编码

之后的任务就是利用已经训练好的特征提取器，对体素特征进行编码，我曾经的做法是拆解掉分类器，但最后我发现似乎有更加简单的做法：

```python
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.ddG[1].register_forward_hook(get_activation('ddG'))
```

这里的原理很简单，就是利用torch的hook机制，在每次推理的时候，截取model中的ddg子模块中第一个线性层的输出，并且将其存储于activation字典中，然后加入最终的特征编码中。

至此为止，我们提取出第一段编码。

## 3.4，基于Rosetta的能量分数

简单来说Rosetta就是一个大分子建模的套件，而pyrossetta是它的一个python接口，由于我们不是生物学专业的，我也只对它以及能量分数有个初步的了解。

```python

```



## 3.5，XGB回归

```python
def objective_regressor(X_train, y_train, X_val, y_val, target_value, trial):
    
    if ((target_value)):
        tree_methods = ['approx', 'hist', 'exact']
#         tree_methods = ['gpu_hist']
        boosting_lists = ['gbtree', 'gblinear']
        objective_list_reg = ['reg:squarederror']  # 'reg:gamma', 'reg:tweedie'
        boosting = trial.suggest_categorical('boosting', boosting_lists),
        tree_method = trial.suggest_categorical('tree_method', tree_methods),
        n_estimator = trial.suggest_int('n_estimators',20, 500, 10),
        max_depth = trial.suggest_int('max_depth', 10, 1000),
        reg_alpha = trial.suggest_int('reg_alpha', 1,10),
        reg_lambda = trial.suggest_int('reg_lambda', 1,10),
        min_child_weight = trial.suggest_int('min_child_weight', 1,5),
        gamma = trial.suggest_int('gamma', 1, 5),
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        objective = trial.suggest_categorical('objective', objective_list_reg),
        colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.8, 1, 0.05),
        colsample_bynode = trial.suggest_discrete_uniform('colsample_bynode', 0.8, 1, 0.05),
        colsample_bylevel = trial.suggest_discrete_uniform('colsample_bylevel', 0.8, 1, 0.05),
        subsample = trial.suggest_discrete_uniform('subsample', 0.7, 1, 0.05),
        nthread = -1
        
        
    xgboost_tune = xgb.XGBRegressor(
        tree_method=tree_method[0],
        boosting=boosting[0],
        reg_alpha=reg_alpha[0],
        reg_lambda=reg_lambda[0],
        gamma=gamma[0],
        objective=objective[0],
        colsample_bynode=colsample_bynode[0],
        colsample_bylevel=colsample_bylevel[0],
        n_estimators=n_estimator[0],
        max_depth=max_depth[0],
        min_child_weight=min_child_weight[0],
        learning_rate=learning_rate[0],
        subsample=subsample[0],
        colsample_bytree=colsample_bytree[0],
        eval_metric='rmsle',
        n_jobs=nthread,
        random_state=SEED)
    
    xgboost_tune.fit(X_train, y_train)
    pred_val = xgboost_tune.predict(X_val)
    
    return np.sqrt(mean_squared_error(y_val, pred_val))
```



# 4，参考

kaggle社区最大的魅力或许就是，前面的大佬总是乐于和他们分享自己的思路和见解，这些让我从中学到很多很多。

[XGBoost - 5000 Mutations 200 PDB Files [LB 0.410]](https://www.kaggle.com/code/cdeotte/xgboost-5000-mutations-200-pdb-files-lb-0-410)

[🧬 NESP: ThermoNet v2 🧬](https://www.kaggle.com/code/vslaykovsky/nesp-thermonet-v2)

[Surface area of the amino acids in the model structure](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/357899)

[1st place solution - Protein as a Graph](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/376371)

[NESP: relaxed rosetta scores](https://www.kaggle.com/code/shlomoron/nesp-relaxed-rosetta-scores)

下面是kaggle社区外参考的资料

[Predicting changes in protein thermodynamic stability upon point mutation with deep 3D convolutional neural networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008291)





