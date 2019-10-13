# 2018

包含项目描述等内容。

---
- [项目计划](#项目计划)
- [工作总结](#工作总结) 
  - [第五六周(12.01~12.15)](#第五六周12011215)
  - [第三四周(11.16~11.30)](#第三四周11161130)


### 第五六周(12.01~12.15)

　　我首先对前两周阅读的论文(<a href="<https://arxiv.org/pdf/1608.08710.pdf>">Pruning Filters For Efficient Convnets</a>,  <https://arxiv.org/pdf/1608.08710.pdf>)中的网络剪枝算法在更深的网络`vgg`  和结构更复杂的网络 `resnet` 上进行代码重现。

　　此外，阅读了另一篇卷积神经网络剪枝论文(<a href="<https://arxiv.org/pdf/1608.08710.pdf>">Learning Efficient Convolutional Networks through Network Slimming</a>,  链接: http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)，研究了 https://github.com/Eric-mingjie/rethinking-network-pruning对论文的剪枝算法在 `vgg` 和 `resnet` 上重现。

#### 论文阅读

　　上述论文提出，在选择卷积层特征图`feature map`中减去的通道`channel`时，除了可以通过`L1_norm` 来评估通道的重要性，还可以通过给每一个通道`channel`一个可训练的参数`channel scaling factor`来衡量通道的重要性，即每个通道的输出值都会被乘以这个参数后传递到下一层，如果`scaling factor`很小，说明通道的重要性低，可以剪掉(下图中，橙色的通道的`scaling factor`接近`0`，可以被剪掉)。

　　值得注意的是，在`CNN`中，卷积层之后一般会伴随着一层批量归一化层`BatchNorm`层，而论文中 提到的`channel scaling factor`的功能恰好和`BatchNorm`层的参数`scaling factor`功能一致。

　　所以，可以直接用`BatchNorm`层自带的`scaling factor`(一般用$\gamma$ 表示) 来衡量通道`channel` 的重要性，这样做，不会引入新的参数。

<center><img src="images/channel_selection.png" style="zoom: 20%"></center>



　　训练时，损失函数`L`加上`scaling factor`( $\gamma $ )的损失值$\lambda \sum_{\gamma \in \Tau} g(\gamma)$，这里定义$g(\gamma) = |\gamma|$，即`scaling factor`( $\gamma $ )的绝对值。

<center><img src="images/loss_with_channel_select.png" style="zoom: 20%"></center>



#### 代码实现

```python
# 批量归一化BatchNorm层的梯度更新，需要 加上 lambda* sign(scaling factor)
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1
```

　　由于`vgg`不包含跳跃连接结构，在此只介绍较为复杂的`resnet`网络。

　　`resnet`的一个基本模块`BasicBlock` 如下，每个模块包含两个卷积核大小为`3*3` 的卷积层`conv`，由于`padding=2`, 经过一个`BasicBlock`后，图片的输入和输出的`height`、 `width`不变。

```python
    # BaicBlock的 前向传播
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x) # 对输入x和 经过BaicBlock的输入out进行维度匹配
            
        out = self.bn1(self.conv1(x))               
        out = self.relu(out)        
        out = self.bn2(self.conv2(out))        
       
        out = self.relu(out + residual) # 加上identity map项 residual
        return out
```



```python
# 一个输入16通道，输出16通道 的BasicBlock，conv1和conv2之间的通道数为12，对于模块外，只需关注输入输出的通道数
BasicBlock(
      (conv1): Conv2d(16, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (conv2): Conv2d(12, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
```

　　以`resnet14` 为例，每个`BaicBlock`表示成`(input_channels, output_channels)`的形式。可以看出，在同一个`layer`中, 相邻`BasicBlock`的通道数 一般保持不变。只在不同的层数`layer` 的间隙处，才进行通道数的加深(16到32， 32到64)。这样做是为了保证同一个`layer`中，在进行跳跃连接`skip` 时，通道数和长宽(在跨越不同`layer`时，一般会用`stride=2`减小长宽)保持一致。

```python
# resnet14 的结构： 3 layer， 2 Block per layer
ResNet(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  
  (layer1): Sequential(
    (0): BasicBlock(16,16)
    (1): BasicBlock(16,16)     
  )
  (layer2): Sequential(
    (0): BasicBlock(16,32)            
    (1): BasicBlock(32,32)    
  )
  (layer3): Sequential(
    (0): BasicBlock(32,64)     
    (1): BasicBlock(64, 64)     
  )
  (avgpool): AvgPool2d(kernel_size=8, stride=8, padding=0)
  (fc): Linear(in_features=64, out_features=10, bias=True)
)
```

　　`resnet`中，有另一种基本模块`Bottleneck`，利用`1x1` 的卷积核来改变通道数，用于减少参数数量。

<center><img src="images/bottleneck.png" style="zoom: 20%"> </center>



```python
# BottleNeck模块的前向传播
def forward(self, x):
     residual = x
	 if self.downsample is not None:
         residual = self.downsample(x)       
    out = self.bn1(x)
    out = self.select(out)  # 利用batchnorm层的scaling factor进行通道选择                  
    out = self.conv1(out)  # bn1 对 该层的输入进行剪枝  
    
    out = self.bn2(out)     
    out = self.conv2(out)  # bn2 对 该层的输入进行剪枝   
    
    out = self.bn3(out) 
    out = self.conv3(out)  # bn3对该层输入剪枝，该conv层输入的通道数减少, 输出通道不变      
    out += residual
    return out
```

　　一个`Bottleneck`举例如下：

```python
Bottleneck(
      (bn1): BatchNorm2d(16)
      (select): channel_selection() # 根据bn1的scaling fator 选择通道剪枝，剪 conv1的输入
      (conv1): Conv2d(8, 7, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn2): BatchNorm2d(7) # 剪去 conv1的输出
      (conv2): Conv2d(7, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn3): BatchNorm2d(10)
      (conv3): Conv2d(10, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (relu): ReLU(inplace)
      (downsample): Sequential( # 匹配残差模块的维度
        (0): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      )
```
### 第三四周(11.16~11.30)

　　第三四周期间，我阅读了一些神经网络剪枝(Neural Network Pruning)的文献。

　　并且，基于用`Python` 的`Pytorch`库，重现了论文中的剪枝算法(包括对卷积层`conv2d`的剪枝 和对全连接层`Linear`的剪枝)。

#### 论文阅读

　　对我影响最重大的一篇论文是研究卷积神经网络`CNN`的剪枝算法(<a href="<https://arxiv.org/pdf/1608.08710.pdf>">Pruning Filters For Efficient Convnets</a>,  链接是<https://arxiv.org/pdf/1608.08710.pdf>)。

　　论文的核心思想如下：

　　1. 从减少网络计算量(computation costs)、减少网络运行时间的角度，对卷积层`Conv2d`的剪枝比对全连接层`full connected`更重要，因为卷积操作更耗时。论文聚焦于对卷积层的剪枝(论文提出的剪枝方法对全连接层的剪枝也非常适应)。

　　2.为避免剪枝对网络结点间的密集连接`Dense`关系的破坏，论文指出对特征图`feature map` (即通道`channel`，也即对卷积核`kernel`剪枝)进行剪枝操作，剪枝操作会去掉$L1\_norm$ 较小的特征图。

　　下图中，$X_{i+1}$ 是当前卷积层的输入`input`，$X_{i+2}$ 是当前卷积层的输出`output`，也是下一层卷积层的输入`input`。$n_{i+1}$ 是当前卷积层的输入的通道数`channels`(也即输入的特征图`feature map` 的个数)，$n_{i+2}$ 是当前卷积层输出的通道数`channels`。

　　图中的$n_{i+1}*n_{i+2}$ 的矩阵是权重矩阵$W_{L+1}$，每一个小矩形都是一个$(kernel\_size, kernel\_size  )$的卷积核`kernel`。

<center><img src="images\卷积层剪枝.png" style="zoom: 20%"> </center>

  3. 每减去一张特征图，即减少了一个通道`channel`，会对当前卷积层的权重矩阵$W_L$和下一层卷积层的权重矩阵$W_{L+1}$ 减少一行或一列，这样使得剪枝后的网络还是处于密集连接的状态。

     　　

#### 代码重现

1. weight-pruning(全连接层剪枝)

   从实现对简单的三层全连接层网络`full connected`进行剪枝开始。选取的数据集是`mnist`数据集，输入图片是单通道是$(28,28)$ 图片，因此网络输入大小为$28*28=784$。 　

   ```python
   层数 类型
   1   MaskedLinear(in_features=784, out_features=200, bias=True)
   2   ReLU(inplace)
   3   MaskedLinear(in_features=200, out_features=200, bias=True)
   4   ReLU(inplace)
   5   MaskedLinear(in_features=200, out_features=10, bias=True)
   ```

   初始网络的每一层的输出结点个数为 cfg = [200, 200, 10 ]，而剪枝后的目标网络设置为 cfg = [50,100,10]。

    

2. filter-pruning(卷积层剪枝)

　　对有三层卷积层`Conv2d`的网络进行剪枝。选取的数据集还是`mnist`数据集，输入图片是单通道是$(28,28)$ 图片。 　

```python
MaskedConv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace)
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

MaskedConv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace)
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

MaskedConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
ReLU(inplace)
## 最后一层卷积层的输出 64张7*7的feature map，需要展平成 64*7*7=3136的一维输出
Linear(in_features=3136, out_features=10, bias=True)
```

　　初始网络的每一层卷积层的输出结点个数为 default_cfg = [32, 64, 64 ]，而剪枝后的目标网络卷积层的配置`config`为 cfg = [8,16,32]。

　　和前面对全连接层的剪枝的最大不同是，从卷积层的最后一层到第一层全连接层之间,有一个对卷积层进行`Flatten`的操作，需要特殊处理。

```python
ConvNet(
  (conv1): MaskedConv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu1): ReLU(inplace)
  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): MaskedConv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu2): ReLU(inplace)
  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): MaskedConv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu3): ReLU(inplace)
   ## 最后一层卷积层的输出 32张7*7的feature map，展平成 32*7*7=1568的一维输出
  (linear1): Linear(in_features=1568, out_features=10, bias=True)
)
```


