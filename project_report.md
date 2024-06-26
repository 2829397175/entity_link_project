## 实体链接Project Report

### 1 引言

实体链接是自然语言处理（NLP）中的一项关键任务，旨在将文本中的提及（mentions）与知识库中的相应实体进行关联。在大规模的文本数据中，实体链接可以帮助我们识别和链接出现的具体实体，从而提高文本理解和信息检索的准确性。

### 2 数据集和任务描述

数据集：Knowledge_base中的Test-Zero Shot，包含Mentions4715，Entities4046，Domain里有News, Social Media, E-books等。

任务：选取**Zero-Shot**任务，将上下文(Context)中的待链接词(Mention)与知识库中的相应实体(Entity)相对应。

### 3 方法与实验结果

我们共尝试了三种方法来完成实体链接任务，分别是**基于MuVER的方法**，**基于大语言模型的方法**和**基于BLINK的方法**。

#### 3.1 基于MuVER的方法

**方法。**我们将实体链接视为匹配问题，使用两个独立的编码器，实体编码器f和提及编码器g，采用BERT作为文本输入的编码结构，表示为：
$$
f(t, d) = T_1([CLS]t[ENT]d[SEP])
$$

$$
g(m) = T_2([CLS]\text{ct}x_l[M_s]m[M_e]\text{ct}x_r[SEP])
$$

我们通过将提及与实体进行比较来匹配。由于不同上下文中的提及对应于描述中的不同部分，我们将描述分割成几个句子，每个句子称为一个视角v，它包含部分信息，以形成实体e的视角集合V。

给定实体e的视角集合$V = {v_1, v_2, ..., v_k}$，确定提及m是否与实体e匹配可以用以下度量空间匹配：
$$
d(m, V) = || g(m) - f(t, [v_1, v_2, ..., v_k, v_i ∈ V])
$$
为了找到提及m的最优实体，我们选择具有最小匹配距离的实体。我们引入NCE损失来建立度量空间，估计视角和提及之间的相似性：
$$
L_{NCE} = E[\log \frac{(\exp d(m, Q^∗(m, e)))}{∑_{e_i∈E'} \exp(d(m, Q^∗(m, e_i)))}]
$$
然而，由于方程中的非可微分子集操作，$Q^∗(m, e)$​无法计算。此外，通过穷举检查所有子集来获得最优视角是耗时的。因此我们考虑只包含一个视角的子集来近似最优视角。

考虑集合$Q_1 ⊂ V$和$Q_2 ⊂ V$以及距离度量$d(Q_1, Q_2) = || f(t, Q_1) - f(t, Q_2) ||$，对于每次迭代，我们搜索前k个最远的视角对$(Q_1, Q_2)$，形成一个新的视角集合$Q_0 = Q_1 ∪ Q_2$，并将$Q_0$扩展到V，通过$f(t, Q_0)$对合并后的$Q_0$进行编码，为所涉及的实体生成一个新的表示。搜索和合并迭代进行，直到$|V|$达到最大允许值或迭代次数达到预设值。

**实验设置。**

训练参数：

+ epoch 30
+ train_batch_size 128
+ learning_rate 1e-5

评估参数：

+ max_cand_len 40
+ max_seq_len 128
+ eval_batch_size 16

**结果分析。**
|   Task    |  Acc     |  R@2   |  R@4   |  R@8   |  R@16  |  R@32  |  R@50  |  R@64  |
|------------|-------- |--------|--------|--------|--------|--------|--------|--------|
| Entity Link | 0.6157 | 0.7578 | 0.7877 | 0.7909 | 0.8231 | 0.8231 | 0.8231 | 0.8231 |



#### 3.2 基于ChatGPT模型的方法

我们使用Close-Track的OpenAI模型来完成任务，选取了gpt-3.5-turbo和gpt-4-turbo。

**实验设置。**

+ 模型：gpt-3.5-turbo/gpt-4-turbo

+ 温度：0.7
+ 最大token：100

**Prompt。**

```
你是一个中文实体链接的代理，你需要将上下文(Context)中的待链接词(Mention)与知识库中的相应实体(Entity)相对应。
上下文:3月21日（周四）下午，湖南大学岳麓书院特邀澳门大学中文系邓国光教授与香港中文大学国学中心主任**邓立光**教授讲学。主题分别为“朱子提出‘继天立极’的道统大义”与“陆象山论心即理与公私
义利之辨”，讲座采用会讲形式，二位主讲人将分别从朱熹和陆九渊的思想核心入手，探讨中国文化整体精神之精义与活力。届时凤凰网国学频道将进行现场图文直播，敬请关注。
待链接词:邓立光
来源:https://edu.rednet.cn/content/2019/03/19/5239180.html
领域:news
你的任务是给出实体在 Wikidata 的对应 QID ，实体在上下文中用**标注，Wikidata 的 url 为 https://www.wikidata.org/wiki/。
```

**结果分析。**

返回的JSON结果如下，其中null为GPT没有按照指定格式返回（即没有以Q开头并跟一串数字）。

```
[
    "Q7110140",
    null,
    null,
    null,
    null,
    null,
    null,
    "Q21809621",
    null,
    null,
    ...
]
```

经过统计，在4714个测试用例中，null值为3840，也就是说GPT返回了18.5%的有效数据，而在这18.5%的数据中，只有7个返回正确。因此实验表明，GPT并不能完成任务，一些可能的原因如下：

+ 模型能力和训练数据限制：尽管GPT是一个非常强大的语言模型，但它可能仍然存在一些限制。GPT-3.5是在2021年之前的数据上进行训练的，因此它可能没有接触到一些最新的实体和链接信息。
+ 我们使用的是Zero-shot任务，模型对于实体链接的知识只在预训练的时候得到。如果在训练GPT模型时，实体链接任务的样本数量相对较少，或者实体链接的正确答案在数据集中占比较小，模型可能没有足够的机会学习到正确的链接模式和特征。
+ 实体链接任务可能涉及到多义性和歧义性的情况，即一个实体可能有多个可能的链接目标。如果模型无法准确理解上下文并进行正确的消歧，它可能会选择错误的链接目标。

#### 3.3 基于BLINK的方法

**方法。**我们使用的双编码器用两个独立的BERT transformers将模型上下文/提及和实体编码为密集向量。我们构建每个提及示例的输入如下：
$$
[CLS] \text{ctxt}_l [M_s] \text{mention} [M_e] \text{ctxt}_r [SEP]
$$
我们的实体模型的输入如下：
$$
[CLS] \text{title} [ENT] \text{description} [SEP]
$$
评分实体候选项$e_i$的得分由点积给出。优化网络的训练目标是最大化正确实体相对于同一批次（随机抽样的）实体的得分，损失函数如下：
$$
L(m_i, e_i) = −s(m_i, e_i) + \log \sum^B_{j=1}\exp(s(m_i, e_j))
$$
由双编码器检索到的候选项随后传递给交叉编码器进行排序。交叉编码器使用一个转换器对上下文/提及和实体进行编码，并应用额外的线性层来计算每对的最终得分。

**实验设置。**

训练Biencoder model。

+ learning_rate 1e-05
+ num_train_epochs 5
+ max_context_length 128
+ max_cand_length 128
+ train_batch_size 128
+ eval_batch_size 64

训练和评估cross-encoder model。

+ learning_rate 2e-05
+ num_train_epochs 5
+ max_context_length 128
+ max_cand_length 128
+ train_batch_size 2
+ eval_batch_size 2

**结果分析。**
