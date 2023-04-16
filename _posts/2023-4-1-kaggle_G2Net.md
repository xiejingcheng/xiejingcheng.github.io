---
layout: post
title: G2Net Detecting Continuous Gravitational Waves
tags: kaggle
math: true
toc: true
date: 2023-4-1 20:56 +0800
---

这篇blog来记录一下从Kaggle:G2Net Detecting Continuous Gravitational Waves中学到的一些知识，同时也是一个总结。

![image-20230415142933441](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\image-20230415142933441.png)

# 一，数据

不同于往年的G2Net，今年主办方提供了一个自主生成数据的库，使得我们可以扩充训练集，生成不同深度下的信号。实际上主办方给的数据，测试集远远大于训练集。。。

![img](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\G2Net_1.jpg)

上面是一个在近乎无噪声情况下的频谱图。理论上，引力波应该是一个简单的正弦波，那他的频谱图应该是一个简单的直线。但是实际上，由于地球的自转以及公转，我们接受到的引力波是经过所谓的“调制”的。所以我们看到的引力波频谱图，幅度和频率都是经过了“调制”。

详细地说，幅度调制是由于因为地球的旋转，探测器根据接收方向的不同有不同的敏感度。频率的调制一方面是其天体随着引力波的发射，其能量减少，发射的频率降低（这由一个参数F1F1描述），另一方面是由于地球公转而产生的多普勒频移。有时会发现竞赛所用的频谱图，引力波的频率向上漂变，这可能是一个“周期”中的上升部分，如果再将生成的时间设长一些可以看到：

![img](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\G2Net_2.jpg)

同时比赛中有个很重要的参数叫做信号深度$D$，严重影响信号的可见度:
$$
D=\frac{\sqrt{S_{n}}}{h_{0}}
$$


$D$越大，说明信号越不可见。在$D=20$左右就基本丧失了视觉特征，而比赛所用的样本的$D$广泛分布在10~100，是具有挑战性的。这里$h_0$在某种意义上代表了幅度（因为幅度还与cosιcos⁡ι有关），$S_n$是噪声的单边功率谱密度。

上面的似乎很容易看出“引力波”的存在，但是实际上，当$D=20$时，频率谱是这样的：

![img](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\G2Net_4.jpg)

同时除去幅度谱，我们还有一个功率谱，虽然两者之间只有一个根号的差别，但是我们可以先可视化一下：

![img](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\G2Net_6.jpg)

从直观的角度上来说，右边的概率谱更加稀疏，似乎可以避免我们去拟合一些无用的噪声。实际上通过我们的测试，功率谱似乎确实会更加好。

最后的最后，这里有一个小坑，float32最小能表达的能到1e-38，而数据的实部和虚部在1e-22。这说明当计算功率谱时，数量级会到1e-44。所以会产生严重的浮点失真。虽然python默认的是float64，但是某些库中为了某些原因可能使用的float32，这里就是我们曾经遇到过的。

或许我们不是物理学专业，不需要细究这些物理学原理。但是kaggle的本质是数据竞赛，要想取得好的成绩，最重要还是细究这些数据背后的意义。这点在这个比赛尤其重要，因为同时我们还需要生成数据集。

# 二，数据集生成

其实数据集的生成大致可以分为三步：

噪声的生成

信号的生成

伪影的注入

噪声其实最开始时，并没有想到可以直接生成。之前的处理方式都是，从举办方所给出的数据中提取出真实噪声，然后和信号进行组合，这种方式导致产生的噪声数据有限，并且容易拟合训练集中的噪声。

不记得是从哪位dalao开始，发现了测试集中噪声的规律，即噪声的实部虚部符合卡方分布，开始了一轮生成噪声的狂潮。

这种方法简单来说就三步

1，采样测试图像

```python
def extract_data_from_hdf5(path):
    data = {}
    with h5py.File(path, "r") as f:
        ID_key = list(f.keys())[0]
        # Retrieve the frequency data
        data['freq'] = np.array(f[ID_key]['frequency_Hz'])
        # Retrieve the Livingston decector data
        data['L1_SFTs_amplitudes'] = np.array(f[ID_key]['L1']['SFTs'])*1e22
        data['L1_ts'] = np.array(f[ID_key]['L1']['timestamps_GPS'])
        # Retrieve the Hanford decector data
        data['H1_SFTs_amplitudes'] = np.array(f[ID_key]['H1']['SFTs'])*1e22
        data['H1_ts'] = np.array(f[ID_key]['H1']['timestamps_GPS'])
    return data


def to_spectrogram(sfts):
    return sfts.real**2 + sfts.imag**2

sample_data = extract_data_from_hdf5('../input/g2net-detecting-continuous-gravitational-waves/test/00054c878.hdf5')
spec_h1 = to_spectrogram(sample_data['H1_SFTs_amplitudes'])

spec_h1.mean(), spec_h1.std()
```

这一步中统计测试图像的噪声平均值，标准差

2，根据其统计数据生成噪音

```python
def reconstruct_from_stat_complex(mean_arr):
    real = np.random.normal(size=(360, len(mean_arr)))
    imag = np.random.normal(size=(360, len(mean_arr)))
    for t, mean in enumerate(mean_arr):
        factor = mean / (real[:, t]**2+imag[:, t]**2).mean()
        real[:, t] *= np.sqrt(factor)
        imag[:, t] *= np.sqrt(factor)
    return (real + imag * 1j).astype(np.complex64)
```

3，对信号库中以相同频率生成的信号进行采样并注入

同时由于检测时的仪器问题，关机or仪器各种问题，图像中会产生伪影。对于一些伪影和仪器谱线，可以从测试集文件中简单地提取出来（检测sigma值）然后写入训练集中，来训练模型对其的鲁棒性。

```python
def extract_artifact(spec, n_sigma=8):
    if np.iscomplexobj(spec):
        spec = spec.real**2 + spec.imag**2
    spec_std = spec.std()
    spec_min = spec.min()
    amp_map = (spec - spec_min) / spec_std
    artifact_map = amp_map > n_sigma
    return artifact_map
```



通过这种方式，我们能够使用几乎无限的背景噪声模式来训练我们的模型。这防止了模型过度拟合背景噪声，并显着提高了模型性能



# 三，大卷积核

我在G2Net比赛里学到的另一个知识是：大卷积核。相比于小卷积核提取“材质”等细节特征，大卷积核更关注于形状。实际上，有一个很好的工作：Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs，在RepVGG的基础上提出了RepLKnet，证明大卷积核确实是可行的，是被“错杀”的设计元素。

但是那个工作是一个非常heavy的计算机视觉的工作，我基本无法复现。第一是昂贵的硬件要求，第二是从零开始训练大卷积核需要一些trick，例如重参数化等等，时间紧迫来不及了。然而在这次比赛里我得以体验了一把大卷积核，kaggle的一个用户laeyoung使用一层大卷积的layer，置于传统的backbone之前，训练出了一个很好的大卷积预训练模型。我们可以在它的基础上进行微调。

 注意，这里的大卷积核和RepLKnet的基本不同，少了大量的trick，而且也并不是一个“模型设计”，只是一个“主干层”。但仍然能学到很多知识：

```python
class LargeKernel_debias(nn.Conv2d):
    def forward(self, input: torch.Tensor):
        finput = input.flatten(0, 1)[:, None]
        target = abs(self.weight)
        target = target / target.sum((-1, -2), True)
        joined_kernel = torch.cat([self.weight, target], 0)
        reals = target.new_zeros(
            [1, 1] + [s + p * 2 for p, s in zip(self.padding, input.shape[-2:])]
        )
        reals[
            [slice(None)] * 2 + [slice(p, -p) if p != 0 else slice(None) for p in self.padding]
        ].fill_(1)
        output, power = torch.nn.functional.conv2d(
            finput, joined_kernel, padding=self.padding
        ).chunk(2, 1)
        ratio = torch.div(*torch.nn.functional.conv2d(reals, joined_kernel).chunk(2, 1))
        power = torch.mul(power, ratio)
        output = torch.sub(output, power)
        return output.unflatten(0, input.shape[:2]).flatten(1, 2)
```

real

![img](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\G2Net_10.jpg)

![img](D:\pypro\xiejingcheng.github.io\xiejingcheng.github.io\_posts\img\G2Net_11.jpg)

