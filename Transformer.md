## Transformer

面经：

- [Transformer大厂面试题汇总：应用开发者视角、Self-Attention、位置编码、三大架构选择与回答 | 卡码笔记｜程序员面试题库，Java、C++、Go、Agent、大模型八股文](https://notes.kamacoder.com/interview/llm/transformer_interview.html#_4-multi-head-attention-为什么要多个头)
- [(2 封私信 / 28 条消息) 大模型-Transformer 面试八股文，简单背一背 - 知乎](https://zhuanlan.zhihu.com/p/689965833)

Multi-Head Attention 上方还包括一个 Add & Norm 层：

- **Add** 表示残差连接 ([Residual Connection](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=Residual+Connection&zhida_source=entity)) 用于防止网络退化，通常用于解决多层网络训练的问题，加入残差块的目的是为了防止在深度神经网络的训练过程中发生退化的问题，退化的意思就是深度神经网络通过增加网络的层数，Loss逐渐减小，然后趋于稳定达到饱和，然后再继续增加网络层数，Loss反而增大。在 [ResNet](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=ResNet&zhida_source=entity) 中经常用到
- **Norm** 表示 [Layer Normalization](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=Layer+Normalization&zhida_source=entity)，通常用于 RNN 结构，用于对每一层的[激活值](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=激活值&zhida_source=entity)进行归一化。Layer Normalization 会将每一层[神经元](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=神经元&zhida_source=entity)的输入都转成[均值方差](https://zhida.zhihu.com/search?content_id=163422979&content_type=Article&match_order=1&q=均值方差&zhida_source=entity)都一样的，这样可以加快收敛，减少训练过程中的[协方差](https://zhida.zhihu.com/search?content_id=241456374&content_type=Article&match_order=1&q=协方差&zhida_source=entity)偏移。。
- **前馈神经网络**：全连接层是一个两层的神经网络，先线性变换，然后ReLU非线性，再线性变换。这两层网络就是为了将输入的Z映射到更加高维的空间中然后通过非线性函数ReLU进行筛选，筛选完后再变回原来的维度。为模型提供了[非线性处理](https://zhida.zhihu.com/search?content_id=241456374&content_type=Article&match_order=1&q=非线性处理&zhida_source=entity)能力，它在每个位置上独立地作用于其输入，有助于增加模型的复杂度和表达能力。

#### **为什么需要 Transformer？—— 传统模型的局限**

RNN/LSTM 的致命缺陷：

- **[长程依赖](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=长程依赖&zhida_source=entity)问题**：梯度消失/爆炸，难以捕捉[远距离依赖](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=远距离依赖&zhida_source=entity)
- **[串行计算](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=串行计算&zhida_source=entity)**：无法[并行化](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=并行化&zhida_source=entity)，训练速度慢
- **信息瓶颈**：固定维度的隐状态难以承载[长序列](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=长序列&zhida_source=entity)全部信息

##### Positional Encoding（[位置编码](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=位置编码&zhida_source=entity)）

 绝对位置编码（原始 Transformer）

<img src="C:\Users\CC\AppData\Roaming\Typora\typora-user-images\image-20260730111908532.png" alt="image-20260730111908532" style="zoom:73%;" />

- **优点**：可学习任意长度序列的位置关系（通过[正弦函数](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=正弦函数&zhida_source=entity)的周期性）
- **缺点**：无法显式建模相对位置

**旋转位置编码RoPE** 当前主流方案

> 🔥 **RoPE（Rotary Position Embedding）** 是 LLaMA、ChatGLM 等主流模型采用的位置[编码方案](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=编码方案&zhida_source=entity)，**用绝对位置编码实现相对位置感知** [[21]]

**核心思想**：

- 将 query/key [向量](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=4&q=向量&zhida_source=entity)在 2D 平面上旋转角度 （ 为位置）
- 旋转后[内积](https://zhida.zhihu.com/search?content_id=269389391&content_type=Article&match_order=1&q=内积&zhida_source=entity)天然包含相对位置信息：

**优势**：

1. 外推性好：训练时长 2K，推理可扩展至 32K+（配合 NTK-aware scaling）
2. 无参数：纯函数式变换，不增加可训练参数
3. 相对位置感知：天然支持相对位置建模

#### 经典面筋：

##### **为什么要除以dk**：

缩放点积注意力分数，从而避免注意力分数过大或过小，导致梯度不稳定或Softmax函数进入饱和区

##### **点积的大小问题**

- 当dk较大时，点积的值可能会变得非常大。这是因为点积是dk个维度求和，随着dk的增加，点积的值也会增加
- 如果点积的值过大，Softmax函数的输入会变得非常大，导致梯度非常小（梯度消失问题），从而影响模型的训练

##### **缩放点积的原因**

为了缓解上述问题，Transformer引入了缩放因子√dk，将点积的结果缩小 这样做可以：

1. 控制点积的方差：
   - 假设Q和K的每个元素是均值为0、方差为1的随机变量，那么点积QK^T的方差为dk
   - 通过除以dk，可以将点积的方差重新缩放为1，从而避免点积的值过大或过小
2. 稳定梯度：
   - Softmax函数对输入的尺度非常敏感。如果输入过大，Softmax的输出会接近一个one-hot向量，导致梯度非常小
   - 缩放点积可以确保Softmax的输入在一个合理的范围内，从而避免梯度消失问题
3. 提高模型性能：
   - 论文实验表明，缩放点积可以显著提高模型的训练稳定性和性能

#### **为什么用点积**：

点积能够有效衡量两个向量的相似性。在注意力机制中，通过计算查询向量（Query）和键向量（Key）的点积，可以评估它们之间的相关性，从而决定注意力权重。而且

- 点积在数学上更简单且易于优化，允许同时计算所有位置的注意力权重，便于并行计算
- 加性注意力需要额外的全连接层和非线性变换，计算复杂度高，且不好并行
- 点积可通过除以根号下dk进行缩放缓解梯度问题，加性注意力的不如其稳定 如果面的是搜广推或者多模态相关岗位，可能被问到点积和余弦相似度的比较

#### **为什么使用多头**

- 捕捉更多样的特征
  - 单头 只能从一个子空间计算注意力权重，可能无法充分捕捉输入序列中复杂的依赖关系
  - 多头 通过将输入映射到多个子空间，每个头可以关注不同的特征或模式
- 增强模型的表达能力
- 提高泛化能力
- 并行计算 多头注意力机制可以并行计算多个注意力头，充分利用GPU的并行计算能力

#### **为什么不用batch normalization而用layer normalization**

- BN对Batch Size敏感
- 在计算均值和方差时，BN会跨序列长度维度进行归一化。对于变长序列数据。BN的计算复杂且不稳定。LN对每个样本做归一化，不受序列长度影响，更适合处理变长序列数据
- BN在推理时带来了额外的复杂性。在训练时，BN会维护一个移动平均值（running mean 和 running variance），用于推理的归一化
- 论文中，Transformer的作者通过实验验证了LN比BN更合适

#### **Cross-Attention**

在Decoder的 交叉注意力机制（Cross-Attention） 中，Q、K、V 的来源不同：

- Query (Q)：来自 Decoder 的输入（目标序列）
- Key(K)和Value(V)：来自 Encoder 的输出（源序列的上下文表示） 这种设计使得 Decoder 能够根据 Encoder 的输出生成目标序列

#### **Decoder为什么要做Mask**

使用Mask主要是为了防止信息泄露

- 防止信息泄露
  - 自回归生成：在生成任务中，Decoder需要逐个生成输出序列的每个元素。Mask确保在生成第t个元素时，只能看到前t−1个元素，防止模型利用未来信息
  - 训练一致性：训练时，Decoder需要模拟生成过程，确保每个时间步只能依赖已生成的部分，保持训练与推理的一致性
- 处理变长序列
  - 填充部分屏蔽：对于变长序列，填充部分需要被Mask掉，避免模型关注无效信息
  

#### **什么是KV cache？ 什么是prompt cache**

我理解 KV Cache 和 Prompt Caching 是同一个机制在两个时间尺度上的应用。

**KV Cache** 是「**单次推理内**」的优化。自回归生成时，每次生成新 token 都要让模型重新对前面所有 token 算 attention。如果每次都从零开始算，N 个 token 的总计算量是 O(N³)，根本不可接受。KV Cache 把前面所有 token 的 K 和 V 矩阵缓存在 GPU 显存里，每次新 token 只算自己的 Q、K、V，然后跟缓存的 K/V 做 attention，把总计算量从 O(N³) 降到 O(N²)。

**Prompt Caching** 是「**跨请求**」的优化。把上面 KV Cache 的概念从「单次生成内」扩展到「不同请求之间」。如果两个请求的 Prompt 前缀完全相同（比如都用同样的 System Prompt），第一个请求算完的 KV Cache 在 API 服务器上保留下来，第二个请求遇到相同前缀直接跳过计算、复用已有 KV Cache，只算新增的部分。

价值上的区别：

- **KV Cache** 解决的是「**让自回归生成可行**」，是 Transformer 推理的基本盘
- **Prompt Caching** 解决的是「**降低 API 成本和延迟**」，是工程层面的 ROI 优化。不同厂商的计费规则不一样，比如 Claude 的缓存读取价格可以低到普通输入 token 的 10%，OpenAI 等平台也有自己的缓存折扣；延迟收益也和前缀长度、命中率、服务端负载有关，不能死记一个固定比例

最关键的认知是，**Prompt Caching 不是新发明，是 KV Cache 这个底层机制的工程级延伸**。理解了 KV Cache，Prompt Caching 几乎是自然推论。

实际工程使用 Prompt Caching 的核心要点是：**固定内容在前、动态内容在后**，前缀只要差一个字符就缓存 miss

#### 多头注意力有哪些局限？MQA，GQA，Flash Attention 如何解决？

我理解 MHA 有三个核心痛点。

第一是「**显存爆炸**」。推理时每个 head 都要为序列里所有 token 保存自己的 K 和 V 矩阵，这就是 KV Cache。头数越多、序列越长，显存占用越夸张。一个 7B 模型跑 32K 上下文，光 KV Cache 就能吃掉十几 GB。

第二是「**访存慢**」。Attention 计算里 softmax 那步要把整个 N×N 的注意力矩阵搬来搬去，频繁读写 GPU 显存，瓶颈不在算力而在「内存带宽」。

第三是「**N² 复杂度**」。注意力分数矩阵是 N×N 的，序列翻倍计算量翻 4 倍，长上下文极其昂贵。

工业界对应了三类优化。MQA 让所有 head 共享一份 K/V，显存压到 1/H，但表达力损失明显。GQA 是折中方案：把 H 个 head 分成 G 组，每组共享一份 K/V，效果接近 MHA 但显存接近 MQA，Llama 2 70B、Llama 3、Qwen 2/3 的不少主力模型都用这个思路。Flash Attention 是另一条思路，不改变 MHA 的结构，而是从计算实现层面把 N×N 的注意力矩阵切成小块、用 GPU 片上缓存做在线 softmax，避免反复读写大矩阵，显存从 O(N²) 降到 O(N)，速度还更快。

最关键的认知是：MQA/GQA 是「结构层」的优化，Flash Attention 是「实现层」的优化，两者是叠加关系不冲突，现在的主流模型基本上都是 GQA + Flash Attention 一起用

#### 什么是MOE

我理解 MoE（Mixture of Experts，混合专家模型）的核心思想是把传统 Transformer 中的 FFN（前馈网络）层替换成 N 个并行的「专家网络」，再加一个 Router 来决定每个 token 进哪个专家。

核心设计哲学是「**总参数大，但激活参数小**」。比如 DeepSeek V3 总参数 671B，但每个 token 推理时只激活 37B（约 1/18）。这样能用「总参数 671B 的知识量」+「激活参数 37B 的推理成本」，达到 Dense 模型做不到的「学得多 + 跑得快」。

具体看 MoE 三个核心组件。

**1. 多个专家（Experts）**：把 Transformer 每层的 FFN 复制 N 份（典型 N=8、64、256），每份就是一个独立的「专家」，在训练中各自学到不同的「擅长方向」（语言、代码、数学、知识等）

**2. Router（路由器）**：每个 token 进到 MoE 层时，Router 算一个「专家偏好分数」，决定这个 token 该去哪个专家。最常见的是 Top-K 路由（K=1 或 K=2），DeepSeek V3 是 Top-8 + 1 个共享专家

**3. 负载均衡**：训练时要加辅助损失防止「专家不平衡」（Router 偏爱某几个专家，其他专家没被训过），保证所有专家都在学

为什么 DeepSeek V3、Mixtral、部分 Qwen 模型都在用 MoE？

- **训练性价比高**：同样算力下训出来的 MoE 模型，效果接近一个大 Dense 模型，但参数总量是 Dense 的 5-20 倍
- **推理成本可控**：每个 token 只用一小部分参数，推理速度和小 Dense 模型相当
- **可扩展性强**：要增加模型容量，加专家数比加层数容易

但 MoE 也有挑战：训练难度高（专家不平衡、Router 训不稳、并行化复杂）；显存占用高（虽然激活只用 37B，但所有专家的参数都要加载到显存，671B 全量）；推理时通信开销（分布式部署时专家分散在多张 GPU，token 路由有跨卡通信）。

MoE 是 2024-2026 年大模型最重要的架构方向之一，DeepSeek V3、DeepSeek R1、Mixtral、Grok、部分 Qwen MoE 模型都用了这条路线。但它不是唯一答案，很多主力 Dense 模型依然在生产里很常见，尤其是中小规模和部署稳定性优先的场景