# 2018毕业设计

包含项目描述，参考文献，项目计划，会议总结等内容。

---

## 项目描述

　　此项目为基于增强拓扑的神经进化(NEAT)和剪枝(Pruning)来优化神经网络。

　　根据我所阅读的参考文献知，基于NEAT的方法具有全局优化(基于梯度的方法容易陷入局部最优解，而基于神经进化的方法属于全局优化，不容易陷入局部最优解)、减少参数规模(基于NEAT的网络结构是从简单到复杂演化的，参数规模比固定结构的网络更容易控制)，以及超参数自动演化等优点。

　　在我阅读的第二篇论文<a href="http://nn.cs.utexas.edu/keyword?stanley:ec02">Evolving Neural Networks through Augmenting Topologies</a>中，对于两个个体进行交叉(Crossover)的操作是通过神经网络连接(Connections)的对齐来实现的，对于两个个体都存在的连接，随机从一个父母中继承，对于不匹配的连接，则只保留适应度(Fitness)较高的父母的连接。

　　其实，这样的交叉操作不能保证产生的后代(Offspring)的适应度得分会更高。而且网络的结点数随着迭代的进行，只会越变越多，而且不是所有的结点都会发挥作用，我考虑将网络结点数量变少的可能性。在交叉时，考虑对父母(Parents)进行剪枝(Pruning)操作，去除父母中活性低的结点，再产生下一代，以此希望能控制整个网络的参数规模，并在保证总体性能稳定的情况下，加速神经网络的运行。	

## 项目关键技术

1. 基于神经进化(Neuro-Evolution)的深度学习结构搜索的方法

   参考文献：

   <a href="http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/108.pdf">Deep Learning Architecture Search by Neuro-Cell-based Evolution with Function-Preserving Mutations (Wistuba, 2018)</a>

   http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/108.pdf

2. 基于增强拓扑的神经进化(NEAT)

   参考文献：

   <a href="http://nn.cs.utexas.edu/keyword?stanley:ec02">Evolving Neural Networks through Augmenting Topologies</a>

   http://nn.cs.utexas.edu/keyword?stanley:ec02

## 项目计划

　　预计18年11月正式开始毕设，本学期最后一周（1月17日前）完成。

| 时间段          | 计划内容            |
| ------------ | --------------- |
| ~11月18日      | 神经进化、剪枝论文文献资料阅读 |
| 11月19日~12月2日 | 阅读的论文的源码理解、实现   |
| 12月3日~12月18日 | 基础编写代码、功能调试     |
| 12月19日~1月2日  | 代码优化、功能调试       |
| 1月3日~        | 开始着手论文撰写        |



## 会议总结

### 10.26第一次会议

　　在第一次会议上，我展示了此前阅读的两篇论文《<a href="http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/108.pdf">Deep Learning Architecture Search by Neuro-Cell-based Evolution with Function-Preserving Mutations (Wistuba, 2018)</a>》和《<a href="http://nn.cs.utexas.edu/keyword?stanley:ec02">Evolving Neural Networks through Augmenting Topologies</a>》。

　　其中，第一篇论文提出了一种基于神经进化(Neuro-Evolution)的深度学习结构搜索的方法，该方法基于遗传算法，使用一系列功能保持(Function-Preserving)的操作来改变神经网络的结构，例如插入一层可分离卷积层、扩宽某一层的结点数(Widening)、产生分支(Branching)、跳跃连接(Skip)等。功能保持，是指在个体发生变异(Mutations)后，对同一输入x，输出y不变，这是通过对突变处的权重进行设置来实现的。接着，会对新生成的网络进行训练，这时就能改变突变处的权重了。论文表明，基于功能保留操作的突变保证了比随机初始化更好的参数初始化，节省网络训练时间。

　　第二篇论文提出了一种叫做"增强拓扑的神经进化(NEAT)"，使得网络结构从简单向复杂进化。一开始，网络只包含输入层和输出层，之后，会逐渐变异、交叉出新的结点和连接。接着，通过自然选择筛选适应度评分(Fitness)神经网络高的网络，淘汰每个物种(Species)中得分偏低的个体(Individuals)。存活下来的个体将通过变异(Mutations)来增加连接(connections)或者增加结点(nodes)。　　

　　对于每个个体(由它所包含的连接(connections)的列表组成)，我们在对同一物种的两个个体进行交叉时，匹配的基因随机遗传。不匹配的基因从fitness更大的parent继承。

　　在会议中，导师指出了，其实这样的交叉操作不能保证产生的后代(Offspring)的适应度得分会更高。而且网络的结点数随着迭代的进行，只会越变越多，导师指出了网络结点是否有变少的可能性。

　　导师对我的展示给予了一些建议，教导我可以将思路放在将同一物种的两个神经网络交叉时，通过剪枝(Pruning)来实现确保交叉后的网络性能有所提高。

### 10.30第二次会议

　　本次会议，我认真吸取了导师的点评之后，确定了选题：基于增强拓扑的神经进化(NEAT)和剪枝(Pruning)来优化神经网络。在交叉时，考虑对父母(Parents)进行剪枝(Pruning)操作，剪去父母中活性低的结点，再交叉产生下一代，以此希望能控制整个网络的参数规模，并在保证网络性能稳定的情况下，加快网络的训练速度。

## 第一二周工作总结

　　在第一二周期间，我阅读了一些神经进化和网络剪枝的相关文献。

　　第一篇论文是微生物遗传算法(<a href="Harvey I. The microbial genetic algorithm[J].  2009, 5778:126-133.">The microbial genetic algorithm</a>,我也参考了相关博客对这篇论文的讲解https://blog.csdn.net/ZiTeng_Du/article/details/79613174)。论文中指出了，在产生新的一代时，一种常用的做法是，在利用当前的种群(current generation)产生下一代(next generation)后，新的种群就取代了原来的种群，这样做的缺点是，交叉变异并不能保证产生的子代的适应度(fitness)较父辈有所提高，有可能丢失了适应度(fitness)很高的个体，新的种群表现可能不如原来的种群。

　　因此，论文提出了另一种更为实用的做法(Steady State Version)，即在选择两个个体作为父母时，保留fitness较高的个体，fitness较差的个体被他们产生的下一代所取代(replace)，所以，我们只需对fitness较差的个体的染色体进行修改即可。这样做，也节省了一些中间变量，实现也较前一种方便。

　　第二篇论文是神经网络剪枝算法(<a href="<http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8192500&isnumber=8192462>">Scalpel: Customizing DNN pruning to the underlying hardware parallelism</a>,  链接是<http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8192500&isnumber=8192462>)。论文中指出了，权重剪枝(Weight Pruning)虽然去除了网络中冗余的结点，减小了网络的规模和计算量，但是并不一定能提高网络的运行速度和性能，因为剪枝可能破坏了网络层与层之间结点原本的密集连接关系，使得连接变得稀疏(sparse),而存储这些稀疏连接也需要额外的空间。这一点是值得我们警惕的，因为我们剪枝的目的在于减小网络参数规模，加快网络的运行，同时保证网络性能的相对稳定，如果网络过于稀疏无规律，剪枝的效果会打折扣，达不到我们要的目的。

　　论文提出了用于在并行度低的平台上效果较好的一种网络权重剪枝方法(SIMD-Aware Weight Pruning)，将网络的权重分成若干个大小相同的小组(weights grouping)，计算每个小组权重的均方根值(RMS)，若RMS小于设置好的阈值，则将整个小组去掉，否则保留整个小组。这样，既能保证去除冗余的权重，还能保证剪枝后的网络的结构仍然有规律性，不会像之前那样稀疏无规律。