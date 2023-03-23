---
layout: post
title: 从transformer到VIT
tags: NLP CV Transformer VIT
math: true
date: 2023-3-23 20:56 +0800
---

[TOC]



# 一， Transformer

## 1， 前言介绍

Transformer由论文《Attention is All You Need》提出，现在是谷歌云TPU推荐的参考模型。论文相关的Tensorflow的代码可以从GitHub获取，其作为Tensor2Tensor包的一部分。哈佛的NLP团队也实现了一个基于PyTorch的版本，并注释该论文。[论文网址](https://arxiv.org/pdf/1706.03762.pdf)

在Transformer前，主流的LSTM,RNN,GRN等网络。通常沿着输入和输出序列的符号位置进行计算。在计算中将positions与steps对齐，它们生成一个隐藏状态h<sub>t</sub>序列，作为前面的隐藏状态h<sub>t-1</sub>和位置t的输入的函数。这种固有的顺序性质使得训练无法并行化，在较长的序列长度的处理中尤为严重，因为内存将会限制了跨样本的批量处理。

同时，自然语言处理任务中，数据文本往往有很强的前后依赖性。无论是通过循环神经网络处理这些关联，还是通过堆叠卷积层扩大感受野来处理这些依赖性，计算代价都是高昂的。在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数量随着位置之间的距离而增长，ConvS2S为线性增长，ByteNet为对数增长。学习遥远位置之间的依赖关系是十分困难的。但是在Transformer中，这被减少为一个常数数量的操作，尽管代价是由于平均注意力加权位置而降低了有效分辨率，他们用Multi-Head Attention抵消了这一影响，

Transformer，这是一种避免递归的模型架构，而是完全依赖于注意机制来绘制输入和输出之间的全局依赖关系。

## 2， 整体结构

首先介绍 Transformer 的整体结构，下图是 Transformer 用于中英文翻译的整体结构：

![image-20230318203631507](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\image-20230318203631507.png?raw=true)

和很多翻译任务的模型一样，transformer也是由编码解码模型组成。在自然语言处理的很多应用中，输入和输出都可以是不定长序列。以机器翻译为例，大多数时候，输入输出并不是像上图中一样等长。输入可以是一段不定长的英语文本序列，输出可以是一段不定长的法语文本序列，例如：
英语输入:“They” “are” “watching” “.”

法语输出:“Ils” “regardent” “.”

当输入和输出都是不定长序列时，可以使用编码器-解码器(encoder-decoder)架构或seq2seq模型。这两个都由两部分组成，分别叫做编码器和解码器。编码器用来分析输入序列，解码器用来生成输出序列。

简单来说，编码器就是将输入序列转化成一个中间表达特征，比如机器翻译任务中所谓的语义向量，或者说英文中包含的意思，而解码器则是将中间表达特征解码成输出，如这个意思所对应的法语。

因此我们可以看到 **Transformer** **由 Encoder 和 Decoder 两个部分组成**，Encoder 和 Decoder 都包含 6 个 block。在许多分类的任务中，Decoder往往会被删除。Transformer 的工作流程大体如下：

获取输入句子的每一个单词的表示向量 **X**，**X**由单词的 Embedding（Embedding就是从原始数据提取出来的Feature） 和单词位置的 Embedding 相加得到单词表示向量矩阵。由于这些序列中是有前后依赖性的，但是transformer不像RNN等循环网络，按时序输入，由神经元记忆之前的信息；而是同时全部输入，因此我们需要对他们加入位置编码，表示这些词or数据在原文（序列）中的顺序。

![image-20230318162702069](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\image-20230318162702069.png?raw=true)

将得到的单词表示向量矩阵 (如上图所示，每一行是一个单词的表示 **x**) 传入 Encoder 中，经过 6 个 Encoder block 后可以得到句子所有单词的编码信息矩阵 **C**单词向量矩阵用 **X**<sub>n*d</sub>表示， n 是句子中单词个数，d 是表示向量的维度 (论文中 d=512)。每一个 Encoder block 输出的矩阵维度与输入完全一致，同时这也是残差链接的需要。

类似与单向的循环网络，transform在解码的过程中，将要翻译的信息将会和前面的已经翻译有一定的关联。因此，transformer使用**MASK**操作实现这一点。将 Encoder 输出的编码信息矩阵 **C**传递到 Decoder 中，Decoder 依次会根据当前翻译过的单词 1~ i 翻译下一个单词 i+1。在使用的过程中，翻译到单词 i+1 的时候需要通过 **Mask **操作遮盖住 i+1 之后的单词。通常情况中，MASK操作是将那些被MASK的元素置成负无穷，这样在矩阵乘法以后再通过softmax以后将会成为零。

## 3，transformer的输入

Transformer 中单词的输入表示 x由单词 Embedding 和位置 Embedding （Positional Encoding）相加得到。

### 3.1 单词Embedding

Word-Embedding,直译是单词嵌入，在自然语言处理中用来给单词编码。

在提到编码时，我最先想到的是之前的真假新闻分类中的文本处理：删除停顿词、词干化、独热编码，但是这些方法在这里似乎不再适用。就像其中提到的独热编码，这种方法并没有什么实际意思，只是通过简单的0-1表示这个词是某词库里面的第几个词。

我们似乎可以继续用下面这种方式编码，大概是这么个样子,

banana:[0 1],

boy:[1 0],

girl:[-1 0],

orange[0 -1].

这种方式第一个维度代表是男是女，第二个维度表示是酸是甜，这样它赋予了每个维度它自己的意思，但是这种方式过于理想，现实中的词语背后的概念很多，我们在有限的维度中不能将它们完全表达出来。因此我们需要进一步改进。例如所以就有了根据语料来word-embedding。

在deepLearning.ai的sequence model中，吴恩达老师介绍了三种word-embedding方式

 第一种是构造出一个简单的language model，代表第i个词的embedding形式。用n个进行全连接，激活函数用softxmax，输出m个值，（m是词库的大小），代表这n个词后可能跟着某个词的概率（可能也是这n个词附近有某个词的概率，根据训练集的选取来）。用长度为n+1的window去从句子中选取输出与输出构成训练集。训练完后的所有就是所需要的embedding。word2vec是这种方法的超级简化版，其中n为1，设输入为x，输出为y，y为以x作为context的一个词，简单点来说，y是在x附近的词。word2vec方法有个叫“skip-gram”的思想，下面两种方法都会用到。

 第二种叫negative sampling，通过构造一个分类问题实现。给定输入c(context)，t(target)，输出t是否能以c为context。训练集的构成方式为对每一个单词都有，一组正确的，k组不正确的（t从语料中随机选）。有个细节是即使k组不正确的中有的确t是以c为context的，也算作不正确。我猜原因大概是这种情况几率小，且一个词可在多处重复出现，检查这个浪费性能。具体的模型为：代表c代表的权重，代表target的embedding形式，将两者内积用sigmoid输出，输出为t以c为context的概率。

 第三种叫作GloVe (Global vectors for word representation)，一种非监督学习，更加地粗暴。同样，令代表第i个词代表的权重，代表第j个词的embedding形式，代表语料中j以i为context的次数。然后

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-4f597eefc3b43129ad52582862aea633_1440w.jpeg?raw=true)

这里介绍的三种方式很粗略，因为我自己也不是很理解，建议去查看吴老师的视频。

### 3.2 位置 Embedding 

Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。**所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。位置 Embedding 用 **PE**表示，**PE** 的维度与单词 Embedding 是一样的。PE 可以通过训练得到，也可以使用某种公式计算得到。在 Transformer 中采用了后者。

想像一下如果我们要对一段序列（语句向量，骨架图数据）做位置编码，第一个想法是在每个时间tt时的开头或结尾拼上一个“token”，比如整型，1，2，3……  这样的坏处是，位置值会越来越大，而且不能适用于比训练时所用的序列更长的序列（这点多见于NLP），那我们缩放一下，变成 [0,1]，然而这样，不同长度序列的步长就不一致了。

 如果换一个思路，我们不用单个的值，而用一个和输入维度等长的向量相加，最自然的就是二进制编码（@计组）。由于一般的d<sub>model</sub>也比较大，所以2d<sub>model</sub>完全够用，这样的缺点是位置距离不连续，比如t=0时我们记token是0000，依次t=2是0001，t=3是0010。但是它毕竟不是二进制数，是一个向量，计算他们之间的距离会发现其并不连续。

所以一个很聪明的方案是利用有界，连续，简单的周期函数。计算公式如下：

![image-20230318185939152](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\image-20230318185939152.png?raw=true)

其中，pos 表示单词在句子中的位置，d 表示 PE的维度 (与词 Embedding 一样)，2i 表示偶数的维度，2i+1 表示奇数维度 (即 2i≤d, 2i+1≤d)。函数图像大致如下：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\transformer_3.jpg?raw=true)

使用这种公式计算 PE 有以下的好处：

使 PE 能够适应比训练集里面所有句子更长的句子，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。

可以让模型容易地计算出相对位置，对于固定长度的间距 k，**PE(pos+k)** 可以用 **PE(pos)** 计算得到。因为 Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)。

将单词的词 Embedding 和位置 Embedding 相加，就可以得到单词的表示向量 **x**，**x** 就是 Transformer 的输入。

```python
# 位置编码
def positional_encoding(length, embed_dim):
    dim = embed_dim // 2
    position = np.arange(length)[:, np.newaxis]  # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :] / dim  # (1, dim)

    angle = 1 / (10000 ** dim)  # (1, dim)
    angle = position * angle  # (pos, dim)

    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed
```



## 4，self-attention机制

### 4.1 attention机制介绍

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)注意力机制的开山论文。

文本翻译中目前主流采用的是编码器-解码器模型。这种结构的模型通常将输入序列编码成一个固定长度的向量表示，对于长度较短的输入序列而言，该模型能够学习出对应合理的向量表示。然而，这种模型存在的问题在于：当输入序列非常长时，模型难以学到合理的向量表示。这个问题限制了模型的性能，尤其当输入序列比较长时，模型的性能会变得很差。解决方法是将encoder的历史状态视作随机读取内存，这样不仅增加了源语言的维度，而且增加了记忆的持续时间（LSTM只是短时记忆）。

Attention机制的基本思想是，打破了传统编码器-解码器结构在编解码时都依赖于内部一个固定长度向量的限制。个人的理解类似于通过某种方法如下文中的相似度，对每个元素添加一个权重，告诉模型重点关注输入序列中的哪些部分。更为通俗的一种解释是，attention机制就是将encoder的每一个隐藏状态设定一个权重，根据权重的不同决定decoder输出更侧重于哪一个编码状态。

Attention机制的实现是通过保留编码器对输入序列的中间输出结果，然后训练一个模型来对这些输入进行选择性的学习并且在模型输出时将输出序列与之进行关联。

Attention机制的本质思想大抵如下图：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70.png?raw=true)

将source中的构成元素想像成一系列的（key，value）数据对构成，对于某个给定的Query，我们去计算Query和各个Key之间的相关性，得到每个key对应value的权重系数，然后对value进行加权求和，即最终的attention score。所以从上面分析可以看出，attention机制本质上就是一个加权求和的过程：
![\small Attention(Query, Source) = \sum_{i=1}^{L_{x}}Similarity(Query, Key_{i})* Value_{i}](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\sum_{i%3D1}^{L_{x}}Similarity(Query%2C Key_{i}) Value_{i}.gif?raw=true)

关于求Query和Key之间相关性的方法，文献中给出有几种：

![CGYSF{}TMD7VOP0XWH@G$0V](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\CGYSF{}TMD7VOP0XWH@G$0V.png?raw=true)

Attention的各种方法：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0thaXl1YW5fc2p0dQ==,size_16,color_FFFFFF,t_70.png?raw=true)



### 4.2 transformer中的self-attention机制

#### 4.2.1 self-attention机制的实现

自我注意，有时也称为内部注意，是一种将单个序列的不同位置联系起来的注意机制，以便计算序列的表示。

![image-20230318163257926](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\image-20230318163257926-1679316386272-1.png?raw=true)

上图是self-attention的计算公式，可以看到我们是先计算**Q**和**K**的内积，如上文提及的计算两者之间的相似度。更加具体的说法类似于计算每个词向量（即每一行或者说每一个样本）之间的相关性，相关性高的赋予一个较大的权重。同时由于在d<sub>k</sub>即矩阵的列数较大时，这个值计算出来将会很大，所以我们在下面加上除以d<sub>k</sub>开根。最后通过softmax得到V的权重，即注意力。

一个直观的理解：我们使用**Q**来查询各个键**K**，每个键有相应的值**V**，最终我们通过**Q**K<sup>T</sup>来计算注意力，这里的注意力即缩放后的点积注意力，来作为每个**V**的权重，最终输出结果。

![image-20230318163145867](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\image-20230318163145867.png?raw=true)



上图是论文中 Transformer 的内部结构图，左侧为 Encoder block，右侧为 Decoder block。红色圈中的部分为 **Multi-Head Attention**，是由多个 **Self-Attention**组成的，可以看到 Encoder block 包含一个 Multi-Head Attention，而 Decoder block 包含两个 Multi-Head Attention (其中有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。

因为 **Self-Attention**是 Transformer 的重点，所以我们重点关注 Multi-Head Attention 以及 Self-Attention，首先详细了解一下 Self-Attention 的内部逻辑。

下图左是 Self-Attention 的结构，下图右为Multi-Head Attention结构。其中Self-Attention在计算的时候需要用到矩阵**Q(查询),K(键值),V(值)**。在实际中，Self-Attention 接收的是输入(单词的表示向量x组成的矩阵X) 或者上一个 Encoder block 的输出。而**Q,K,V**正是通过 Self-Attention 的输入进行线性变换得到的。

![image-20230318163209360](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\image-20230318163209360.png?raw=true)

Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。**

得到矩阵 Q, K, V之后就可以计算出 Self-Attention 的输出了，计算的公式如下：

![image-20230318163257926](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\image-20230318163257926.png?raw=true)

公式中计算矩阵**Q**和**K**每一行向量的内积，为了防止内积过大，因此除以**d<sub>k</sub>**的平方根。**Q**乘以**K**的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为**Q**乘以 **K<sup>T</sup>** ，1234 表示的是句子中的单词。

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-9caab2c9a00f6872854fb89278f13ee1_1440w.webp?raw=true)

得到**QK<sup>T</sup>** 之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1。

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-96a3716cf7f112f7beabafb59e84f418_1440w.webp?raw=true)

上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 **Z<sub>1</sub>** 等于所有单词 i 的值 **V<sub>i</sub>** 根据 attention 系数的比例加在一起得到，如下图所示：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-27822b2292cd6c38357803093bea5d0e_r.jpg?raw=true)

本文中对自注意力机制解释的较为粗糙，更加的详细的过程可以参考[giegie的blog](https://zjwxdu.github.io/2023/01/04/Attention/)

#### 4.2.2 为什么选择self-attention机制

根据文中作者的说法，激发他们使用自注意力机制的动机有三个。

一个是每层的总计算复杂度。另一个是可以并行化的计算量，可以通过所需的最小顺序操作数量来衡量。

第三个是网络中长期依赖关系之间的路径长度。在许多序列转导任务中，学习长期依赖关系是一个关键的挑战。影响学习这种依赖关系能力的一个关键因素是向前和向后信号在网络中必须遍历的路径长度。输入和输出序列中任意位置组合之间的路径越短，就越容易学习长期依赖关系。

这些也对应着我们前文所说的，传统循环模型的缺陷和Attention机制的优势。

### 4.3 Multi-Head Attention

在上一步，我们已经知道怎么通过 Self-Attention 计算得到输出矩阵 Z，而 Multi-Head Attention 是由多个 Self-Attention 组合形成的，前文中以及给出 Multi-Head Attention 的结构图。

至于多头机制的作用，我的理解类似于卷积中的多通道，通过不同的通道学习不同的特征。而transformer则通过这个多头机制来构建与一个类似的多通道机制，不同的头来学习不同的特征。

Multi-Head Attention 包含多个 Self-Attention 层，首先将输入**X**分别传递到 h 个不同的 Self-Attention 中，计算得到 h 个输出矩阵**Z**。下图是 h=8 时候的情况，此时会得到 8 个输出矩阵**Z**。

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-6bdaf739fd6b827b2087b4e151c560f4_r.jpg?raw=true)

得到 8 个输出矩阵 **Z<sub>1</sub>** 到 **Z<sub>8</sub>**之后，Multi-Head Attention 将它们拼接在一起 **(Concat)**，然后传入一个**Linear**层，得到 Multi-Head Attention 最终的输出**Z**。

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-35d78d9aa9150ae4babd0ea6aa68d113_r.jpg?raw=true)

可以看到 Multi-Head Attention 输出的矩阵**Z**与其输入的矩阵**X**的维度是一样的。

```python
# use fix dimension
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_head,
                 batch_first,
                 ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )

    def forward(self, x):


        q = F.linear(x[:1], self.mha.in_proj_weight[:1024], self.mha.in_proj_bias[:1024])  # since we need only cls
        k = F.linear(x, self.mha.in_proj_weight[1024:2048], self.mha.in_proj_bias[1024:2048])
        v = F.linear(x, self.mha.in_proj_weight[2048:], self.mha.in_proj_bias[2048:])
        q = q.reshape(-1, 8, 128).permute(1, 0, 2)
        k = k.reshape(-1, 8, 128).permute(1, 2, 0)
        v = v.reshape(-1, 8, 128).permute(1, 0, 2)
        dot = torch.matmul(q, k) * (1 / 128 ** 0.5)  # H L L
        attn = F.softmax(dot, -1)  # L L
        out = torch.matmul(attn, v)  # L H dim
        out = out.permute(1, 0, 2).reshape(-1, 1024)
        out = F.linear(out, self.mha.out_proj.weight, self.mha.out_proj.bias)
        return out
```



## 5,  encoder介绍

如下图，transformer的encoder是由**N**=6个一样的layer堆叠而成，每个layer由两个sub-layer组成。第一个sub-layer为multi-head self-attention mechanism，即之前介绍的多头自注意模块。另一个原文描述中为a simple, position-wise fully connected feed-forward network，即简单的dense层（对应torch中的liner层）。同时，对两个sub-layer进行残差链接。每个sub-layer的输出可以表示为LayerNorm(x + Sublayer(x))。同时，残差链接需要输入输出大小相等，因此，模型中限定 **d<sub>model</sub>**=512。至此，模型将需要调整的超参降低为了两个，即 N 和 d<sub>model</sub>。

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-0203e83066913b53ec6f5482be092aa1_1440w.webp?raw=true)

综上所述，encoder中重要的模块主要是Multi-Head Attention, **Add & Norm, Feed Forward, Add & Norm** 组成的。刚刚已经了解了 Multi-Head Attention 的计算过程，现在了解一下 Add & Norm 和 Feed Forward 部分。

### 5.1 Add & Norm 层

Add & Norm 层由 Add 和 Norm 两部分组成，其计算公式如下：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-a4b35db50f882522ee52f61ddd411a5a_1440w.webp?raw=true)

Add &amp;amp;amp;amp;amp; Norm 公式

其中 **X**表示 Multi-Head Attention 或者 Feed Forward 的输入，MultiHeadAttention(**X**) 和 FeedForward(**X**) 表示输出 (输出与输入 **X** 维度是一样的，所以可以相加)。

**Add**指 **X**+MultiHeadAttention(**X**)，是一种残差连接，通常用于解决多层网络训练的问题，可以让网络只关注当前差异的部分，在 ResNet 中经常用到：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-4b3dde965124bd00f9893b05ebcaad0f_1440w.webp?raw=true)

残差连接

**Norm**常用的有Layer Normalization和Batch Normalization。transfomer中指的是 Layer Normalization，通常用于 RNN 结构，Layer Normalization 会将每一层神经元的输入都转成均值方差都一样的，这样可以加快收敛。Layer Normalization和Feature Normalization的区别主要在于，前者是对batch中的每一个样本单独进行norm，即layer norm。后者对于batch中的每一个样本的同一个特征值进行norm，也就是对其feature进行norm。layer norm和batch norm大多数时候是一样的。但是由于transformer中同一个batch中的每一个样本并不等长，使用batch norm，每次算出来的均值方差抖动较大，因此选用Layer Normalization。(输入不等长，后面全是0)这里讲的较为简略，具体可以参考：

[bilibili：Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE)

[CSDN：【深度学习】Layer Normalization](https://blog.csdn.net/weixin_31866177/article/details/121745858)

### 5.2 Feed Forward

正如我们前面所说，这是一个较为简单的网络层。

Feed Forward 层比较简单，是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-47b39ca4cc3cd0be157d6803c8c8e0a1_1440w.webp?raw=true)

Feed Forward

**X**是输入，Feed Forward 最终得到的输出矩阵的维度与**X**一致。

```python
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)
```



### 5.3 组成encoder

通过上面描述的 Multi-Head Attention, Feed Forward, Add & Norm 就可以构造出一个 Encoder block，Encoder block 接收输入矩阵 **X<sub>(n×d)</sub>** ，并输出一个矩阵 **O<sub>(n×d)</sub>** 。通过多个 Encoder block 叠加就可以组成 Encoder。

第一个 Encoder block 的输入为句子单词的表示向量矩阵，后续 Encoder block 的输入是前一个 Encoder block 的输出，最后一个 Encoder block 输出的矩阵就是**编码信息矩阵 C**，这一矩阵后续会用到 Decoder 中。

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-45db05405cb96248aff98ee07a565baa_1440w.webp?raw=true)

```python
class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_head,
                 out_dim,
                 batch_first=True,
                 ):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_head, batch_first)
        self.ffn = FeedForward(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x
```



## 6，decoder介绍

由于在除文本翻译的其他任务中，常常只采用transformer的encoder部分。因此，在本文中，对decoder部分的介绍并不如前面那么详细。(指照搬别人的)

### 6.1 Decoder 结构

Transformer 的 Decoder block 结构，与 Encoder block 相似，但是存在一些区别：

- 包含两个 Multi-Head Attention 层。
- 第一个 Multi-Head Attention 层采用了 Masked 操作。
- 第二个 Multi-Head Attention 层的**K, V**矩阵使用 Encoder 的**编码信息矩阵C**进行计算，而**Q**使用上一个 Decoder block 的输出计算。
- 最后有一个 Softmax 层计算下一个翻译单词的概率。

#### 6.1.1 第一个 Multi-Head Attention

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。下面以 "我有一只猫" 翻译成 "I have a cat" 为例，了解一下 Masked 操作。

下面的描述中使用了类似 Teacher Forcing 的概念，不熟悉 Teacher Forcing 的童鞋可以参考以下上一篇文章Seq2Seq 模型详解。在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译，如下图所示。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。

Decoder 可以在训练的过程中使用 Teacher Forcing 并且并行化训练，即将正确的单词序列 (<Begin> I have a cat) 和对应输出 (I have a cat <end>) 传递到 Decoder。那么在预测第 i 个输出时，就要将第 i+1 之后的单词掩盖住，**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "<Begin> I have a cat <end>"。**

#### 6.2.2 第二个Multi-Head Attention

Decoder block 第二个 Multi-Head Attention 变化不大， 主要的区别在于其中 Self-Attention 的 **K, V**矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用 **Encoder 的编码信息矩阵 C** 计算的。

根据 Encoder 的输出 **C**计算得到 **K, V**，根据上一个 Decoder block 的输出 **Z** 计算 **Q** (如果是第一个 Decoder block 则使用输入矩阵 **X** 进行计算)，后续的计算方法与之前描述的一致。

这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 **Mask**)。

#### 6.2.3 Softmax 预测输出单词

Decoder block 最后的部分是利用 Softmax 预测下一个单词，在之前的网络层我们可以得到一个最终的输出 Z，因为 Mask 的存在，使得单词 0 的输出 Z0 只包含单词 0 的信息，如下：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-335cfa1b345bdd5cf1e212903bb9b185_1440w-1679143408882-39.webp?raw=true)

Decoder Softmax 之前的 Z

Softmax 根据输出矩阵的每一行预测下一个单词：

![img](https://github.com/xiejingcheng/xiejingcheng.github.io/raw/main/_posts\img\v2-0938aa45a288b5d6bef6487efe53bd9d_1440w-1679143416785-41.webp)

Decoder Softmax 预测

这就是 Decoder block 的定义，与 Encoder 一样，Decoder 是由多个 Decoder block 组合而成。

## 7，transformer小结

这段本来打算写完，但是觉得自己现在对transformer的理解不是很深，等再过一段时间补上吧。

个人感觉，transformer最大的贡献就是打通了AI各个领域之间的壁垒。原先语言处理用RNN，图像处理用CNN，transformer的出现打通了这些壁垒，让模型可以跨领域迁移。

同时，transformer或许没有RNN和CNN那种强先验来辅助学习。但是transformer由于它强大的学习能力，没有明显的性能饱和上限，在大模型大数据的加持下，可以得到很高的效果。个人感觉颇有一种力大砖飞的意思。

下面是我从网上抄来的总结：

- Transformer 与 RNN 不同，可以比较好地并行训练。
- Transformer 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则 Transformer 就是一个词袋模型了。
- Transformer 的重点是 Self-Attention 结构，其中用到的 **Q, K, V**矩阵通过输出进行线性变换得到。
- Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score

# 四，参考

[bilibili：Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE)

[CSDN：【深度学习】Layer Normalization](https://blog.csdn.net/weixin_31866177/article/details/121745858)

[知乎：Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)

[CSDN: 理解Attention机制原理及模型](https://blog.csdn.net/Kaiyuan_sjtu/article/details/81806123)

[ZJWXDU: Understanding Self Attention](https://zjwxdu.github.io/2023/01/04/Attention/)

[编码器-解码器(seq2seq)](https://blog.csdn.net/tcn760/article/details/124432361)

[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

[DL笔记：Word-Embedding及几种方法](https://zhuanlan.zhihu.com/p/36155204)

[Encoder-decoder模型及Attention机制](https://blog.csdn.net/weixin_41744192/article/details/116430400)

