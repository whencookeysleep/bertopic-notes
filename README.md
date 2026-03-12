# BERTopic 学习笔记

## 1. 什么是 BERTopic

BERTopic 是一种基于 **Transformer 语义表示 + 聚类 + c-TF-IDF 关键词提取** 的主题建模方法。  
它的核心目标是：**既利用深度语义表示能力发现文本中的潜在主题，又尽可能保持主题结果的可解释性**。

和传统 LDA 这类概率主题模型相比，BERTopic 更强调：

- 使用预训练语言模型获得更好的语义向量
- 用聚类方法发现主题，而不是完全依赖词袋共现
- 用 c-TF-IDF 为每个主题提取代表词，提高可解释性
- 整个流程模块化，可替换 embedding、降维、聚类、表示方法

---

## 2. 标准 BERTopic 的基本流程

一个标准的 BERTopic 流程通常可以概括为：

1. **文本向量化（Embedding）**  
   使用 Sentence-Transformers 或其他 embedding 模型，将每篇文档转换为语义向量。

2. **降维（Dimensionality Reduction）**  
   常用 UMAP 对高维语义向量降维，便于后续聚类。

3. **聚类（Clustering）**  
   常用 HDBSCAN 自动发现文本簇，每个簇可看作一个潜在主题。

4. **主题表示（Topic Representation）**  
   将同一簇的文档合并为“类文档”，再通过 c-TF-IDF 提取每个主题的代表词。

5. **主题优化与解释（Optional）**  
   可以进一步使用 KeyBERT、LLM、自定义标签等方式改进主题名称和描述。

---

## 3. 如何理解 BERTopic 支持的多种类型

BERTopic 官网列出的这些“类型”并不都在做同一件事。  
它们大致可以分为三类：

### A. 控制“主题如何被发现”
- Guided
- Supervised
- Semi-supervised
- Manual
- Zero-shot

### B. 控制“主题如何被表示”
- Seed Words
- Multi-aspect
- Text Generation / LLM

### C. 控制“主题如何被分析或扩展”
- Multi-topic distributions
- Hierarchical
- Class-based
- Dynamic
- Online / Incremental
- Merge Models
- Multimodal

也就是说，有些方法是在影响“怎么分主题”，有些是在影响“怎么给主题命名和解释”，还有一些是在主题建好之后进一步做结构分析、时间分析或扩展分析。

---

# 4. 各类型详细解读

---

## 4.1 Guided Topic Modeling

### 含义
Guided Topic Modeling 指的是：  
**用户提前提供一些种子主题词（seed topic words），引导模型更容易形成与这些先验语义相关的主题。**

### 核心思想
它不是完全无监督，因为你给了模型方向；  
但它也不是完全监督，因为你没有给每篇文档明确标签。

它更像是一种：

> **带有软约束的主题发现方式**

### 适合场景
- 已经知道语料中大概率会出现哪些主题
- 希望模型围绕这些主题方向组织文档
- 但仍保留一定的自动发现能力

### 举例
如果你在分析科研论文，知道里面大概率有：

- Topic Modeling
- Clustering
- Embedding
- LLM

那么可以提前给一些种子词，引导 BERTopic 更容易聚出这些方向的主题。

---

## 4.2 Supervised Topic Modeling

### 含义
Supervised Topic Modeling 指的是：  
**你已经有文档标签，BERTopic 不再完全依赖无监督聚类，而是直接利用这些标签构建可解释主题。**

### 核心思想
普通文本分类只告诉你“属于哪一类”；  
Supervised BERTopic 除了利用标签，还能进一步输出：

- 每个类别对应的关键词
- 每个类别的代表文档
- 每个类别更清晰的主题解释

### 适合场景
- 已有完整标签体系
- 想让分类结果具备更强可解释性
- 想知道“这个类别到底由哪些关键词定义”

### 本质理解
它本质上是：

> **把“类别标签”转化为“可解释主题表示”**

---

## 4.3 Semi-supervised Topic Modeling

### 含义
Semi-supervised Topic Modeling 指的是：  
**只有部分文档有标签，或者标签信息不完整，模型利用这些部分标签去引导主题发现。**

### 核心思想
它介于无监督和监督之间：

- 比纯无监督多了一点先验
- 比纯监督又保留了探索未知结构的能力

通常可以理解为：

> **给模型一些提示，让它在更合理的语义空间中聚类**

### 适合场景
- 只有少量人工标注
- 不希望完全放弃无监督探索能力
- 语料中既有已知主题，也可能有未知主题

---

## 4.4 Manual Topic Modeling

### 含义
Manual Topic Modeling 指的是：  
**文档分组已经由人工或其他方法确定，BERTopic 不负责“发现主题”，只负责“解释主题”。**

### 核心思想
你已经知道哪些文档是一组，只是不知道怎么总结每组的主题词和代表语义。  
这时可以直接把已有分组交给 BERTopic，让它帮你生成：

- 关键词
- 标签
- topic representation

### 适合场景
- 已有人工标签
- 已有外部聚类结果
- 只想借助 BERTopic 的表示能力，不需要它重新聚类

### 一句话理解
> **你先分组，BERTopic 帮你解释**

---

## 4.5 Zero-shot Topic Modeling

### 含义
Zero-shot Topic Modeling 指的是：  
**你先给定一组预定义主题名称，模型先判断哪些文档与这些主题语义最接近，剩余无法归类的文档再交给普通 BERTopic 继续聚类。**

### 核心思想
它不是把所有文档硬塞进预设主题中，  
而是采用“先匹配、再聚类”的混合策略：

1. 用预设主题名和文档做语义相似度匹配
2. 能匹配上的文档直接归入相应主题
3. 其余文档再通过普通 BERTopic 自动发现新主题

### 适合场景
- 已经有一套预定义主题框架
- 但又不想只做封闭分类
- 希望系统既能识别已知主题，也能发现新主题

### 一句话理解
> **先按已知主题吸收一部分文档，再让模型探索剩余未知部分**

---

## 4.6 Seed Words

### 含义
Seed Words 与 Guided 很像，但其实不同。  
它的作用不是提前定义主题，而是：

> **提高某些词在主题表示中的重要性**

### 核心思想
有些领域术语、缩写词在普通语义模型里不一定很突出，  
但在你的场景中非常重要，比如：

- GIS
- CRS
- LLM
- SAR
- c-TF-IDF

这时可以通过 Seed Words 告诉 BERTopic：

> “这些词在生成主题描述时要更受重视。”

### 适合场景
- 专业领域语料
- 缩写较多
- 需要更专业的主题关键词表示

### 一句话理解
> **它影响的是“主题怎么描述”，不是“主题怎么形成”**

---

## 4.7 Multi-topic Distributions

### 含义
标准 BERTopic 往往默认一篇文档对应一个主主题。  
但现实中，一篇文档往往可能同时涉及多个主题。  
Multi-topic distributions 就是为了解决这个问题。

### 核心思想
它不只问：

> “这篇文档属于哪个主题？”

而是进一步问：

> “这篇文档分别有多大程度属于哪些主题？”

### 适合场景
- 长文本
- 综述类文章
- 社交媒体长帖
- 同时涵盖多个议题的文档

### 一句话理解
> **从“单标签主题归属”变成“多主题分布表示”**

---

## 4.8 Hierarchical Topic Modeling

### 含义
Hierarchical Topic Modeling 指的是：  
**在已有主题基础上进一步建立层级结构，分析哪些主题之间更接近，哪些主题可以归并为更高层主题。**

### 核心思想
平铺的 topic list 往往太散，层级主题可以帮助理解：

- 哪些主题是大类
- 哪些主题是子类
- 哪些主题之间关系紧密

### 举例
比如最终主题可能形成这样的结构：

- Remote Sensing
  - Hyperspectral
  - SAR
  - Entity Alignment
- NLP
  - Topic Modeling
  - LLM
  - Information Extraction

### 适合场景
- 主题较多
- 需要从粗到细理解主题空间
- 需要搭建主题树状结构

### 一句话理解
> **把离散主题组织成层级主题体系**

---

## 4.9 Class-based Topic Modeling

### 含义
Class-based 更准确地说，可以理解为：

> **分析不同类别或群体中，同一主题是如何被表达的**

### 核心思想
不是重新为每个类独立建模，而是：

- 先得到全局主题
- 再分析这些主题在不同类别中的差异表达

### 适合场景
- 比较不同人群、地区、学科、时间段
- 看“同一个主题”在不同群体中的不同说法
- 想做 subgroup analysis

### 一句话理解
> **主题是全局的，表达是分组变化的**

---

## 4.10 Dynamic Topic Modeling

### 含义
Dynamic Topic Modeling 指的是：

> **分析主题随时间如何演化**

### 核心思想
不是每个时间点重新训练一套全新的主题模型，  
而是在全局主题基础上，观察每个主题在不同时间窗口中的表示变化。

换句话说，它更关注：

- 某个主题在不同时间点是否持续存在
- 它的关键词是否变化
- 它的关注重点是否转移

### 适合场景
- 新闻舆情分析
- 学术研究趋势分析
- 社交媒体事件追踪
- 政策文本演化分析

### 一句话理解
> **关注主题的“时间演化”，而不是静态主题本身**

---

## 4.11 Online / Incremental Topic Modeling

### 含义
Online / Incremental Topic Modeling 指的是：

> **数据不是一次性全部到齐，而是分批到来，模型能够逐步更新主题结果**

### 核心思想
传统 BERTopic 更适合离线一次性训练；  
而在线/增量版本适合在以下情况下使用：

- 数据量太大
- 数据流式到达
- 希望不断更新已有主题模型

### 适合场景
- 实时评论流
- 持续增长的论文库
- 每天新增文档的数据平台
- 内存无法容纳全部文档

### 一句话理解
> **不是一次性建模，而是边来数据边更新主题**

---

## 4.12 Multimodal Topic Modeling

### 含义
Multimodal Topic Modeling 指的是：

> **同时利用多种模态信息建模主题，例如文本 + 图像**

### 核心思想
有些主题仅看文本不够，仅看图像也不够。  
将两者结合后，可以得到更完整的主题表示。

### 适合场景
- 社交媒体图文内容
- 商品图片 + 商品描述
- 医学图像 + 报告文本
- 遥感图块 + 配套说明文本

### 一句话理解
> **利用多模态信息共同定义主题**

---

## 4.13 Multi-aspect Topic Modeling

### 含义
Multi-aspect Topic Modeling 指的是：

> **为同一个主题生成多个表示视角**

### 核心思想
一个主题不一定只用一组关键词表示。  
它还可以同时拥有：

- 关键词表示
- 短语表示
- 摘要表示
- 自定义标签表示
- LLM 生成描述

### 适合场景
- 论文写作
- 结果展示
- 产品界面说明
- 需要机器可读 + 人类可读双重表示

### 一句话理解
> **同一个 topic，可以有不止一种表达方式**

---

## 4.14 Text Generation / LLM

### 含义
Text Generation / LLM 指的是：

> **借助大语言模型对 BERTopic 生成的主题进行更自然的标签生成、摘要生成或解释增强**

### 核心思想
传统 topic 表示可能只是：

- topic
- model
- data
- learning
- results

虽然可解释，但不够自然。  
借助 LLM，可以把它改写为：

- “Large Language Model Fine-tuning Methods”
- “Remote Sensing Entity Alignment Approaches”
- “Customer Login and Authentication Issues”

### 适合场景
- 需要自然语言标签
- 需要生成主题摘要
- 需要更适合展示或报告的结果形式

### 一句话理解
> **用 LLM 提升主题标签与摘要的自然性和可读性**

---

## 4.15 Merge Models

### 含义
Merge Models 是较新的扩展能力，指的是：

> **将多个独立训练的 BERTopic 模型进行合并**

### 核心思想
如果数据来自不同批次、不同来源，或者你分阶段训练了多个模型，  
可以通过 merge 的方式把这些模型整合起来，而不是重新全量训练。

### 适合场景
- 增量训练
- 多批次语料分别建模
- 多个子语料分别训练后再整合
- 联邦式或分布式主题分析

### 一句话理解
> **不是重新训练一个大模型，而是把多个已有主题模型拼接整合**

---

# 5. 最容易混淆的几组概念

---

## 5.1 Guided vs Zero-shot vs Seed Words

### Guided
给“种子主题词组”，影响主题形成方向。

### Zero-shot
给“预定义主题标签”，先分流文档，再聚类剩余文档。

### Seed Words
提高某些词在主题表示中的权重，优化主题描述。

### 简单区分
- **Guided**：引导主题生成
- **Zero-shot**：先按已知主题分流文档
- **Seed Words**：优化主题词表表达

---

## 5.2 Supervised vs Semi-supervised vs Manual

### Supervised
标签较完整，模型利用标签进行主题构建。

### Semi-supervised
标签不完整，标签只用于引导聚类方向。

### Manual
分组已经完全确定，BERTopic 只负责主题解释。

### 简单区分
- **Supervised**：我教你学
- **Semi-supervised**：我给你提示，你自己聚
- **Manual**：我已经分好了，你帮我解释

---

# 6. 实际使用时如何选择

## 如果你只是想探索语料里的潜在主题
优先使用：

- 标准 BERTopic

适合纯无监督探索。

---

## 如果你知道一部分主题，但不想完全人工分类
优先考虑：

- Guided
- Zero-shot
- Seed Words

适合带先验知识但仍想保留自动发现能力的场景。

---

## 如果你已经有标签体系
优先考虑：

- Supervised
- Manual

前者适合“标签学习 + 解释”，  
后者适合“已有分组 + 解释”。

---

## 如果你只有少量标签
优先考虑：

- Semi-supervised

适合部分标注场景。

---

## 如果文档经常包含多个主题
优先考虑：

- Multi-topic distributions

适合多主题混合文档。

---

## 如果你关心主题之间的关系结构
优先考虑：

- Hierarchical

适合建立主题层级。

---

## 如果你关心时间上的演化
优先考虑：

- Dynamic

适合分析主题随时间的变化。

---

## 如果你关心不同群体如何谈同一主题
优先考虑：

- Class-based

适合 group comparison。

---

## 如果数据是分批到来的
优先考虑：

- Online / Incremental
- Merge Models

适合大规模或流式数据场景。

---

## 如果数据不止文本
优先考虑：

- Multimodal

适合图文联合建模。

---

## 如果你更关心展示效果与可读性
优先考虑：

- Multi-aspect
- Text Generation / LLM

适合生成更自然、更丰富的主题表示。

---

# 7. 总结

BERTopic 的强大之处不只是“能做主题建模”，更在于它提供了一套**高度模块化、可扩展的主题分析框架**。

它支持的各种类型，本质上分别对应不同问题：

- 有些在解决 **如何发现主题**
- 有些在解决 **如何表示主题**
- 有些在解决 **如何分析主题结构、变化与扩展场景**

所以理解 BERTopic 时，不要把这些类型看成一堆并列功能按钮，而应该把它们理解为：

> **围绕主题发现、主题表示、主题分析三大环节的一组扩展机制**

---

# 8. 一句话总表

| 类型 | 核心作用 | 一句话理解 |
|---|---|---|
| Guided | 引导主题发现 | 给主题形成方向 |
| Supervised | 利用完整标签建模 | 用标签生成可解释主题 |
| Semi-supervised | 利用部分标签引导 | 给一点提示再聚类 |
| Manual | 解释已有分组 | 你分组，我解释 |
| Zero-shot | 预定义主题匹配 | 先按已知主题分流 |
| Seed Words | 优化主题表示 | 强化关键词权重 |
| Multi-topic distributions | 多主题归属 | 一篇文档可对应多个主题 |
| Hierarchical | 层级主题分析 | 建立主题树结构 |
| Class-based | 分组对比 | 比较不同群体中的主题表达 |
| Dynamic | 时间演化分析 | 观察主题随时间变化 |
| Online / Incremental | 增量更新 | 边来数据边建模 |
| Multimodal | 多模态建模 | 文本和图像一起建主题 |
| Multi-aspect | 多视角表示 | 一个主题多种表达 |
| Text Generation / LLM | 自然语言增强 | 用 LLM 生成更自然的标签和摘要 |
| Merge Models | 模型合并 | 多个主题模型整合 |

---

# 如果是初学 BERTopic，建议先按下面顺序理解：

1. 先掌握标准 BERTopic 流程  
2. 再理解 Guided / Zero-shot / Semi-supervised 这些“如何引导主题发现”的方法  
3. 然后理解 Hierarchical / Dynamic / Multi-topic distributions 这些“如何分析主题结构”的方法  
4. 最后再看 LLM / Multi-aspect / Multimodal / Merge Models 这些高级扩展

这样会比一上来把所有类型混在一起更容易建立清晰框架。

---

# 10. 后续可扩展内容

后面还可以继续补充：

- BERTopic 与 LDA、Top2Vec 的区别
- BERTopic 的核心参数解释
- BERTopic 的常见踩坑
- BERTopic 在科研论文中的写法
- BERTopic 在中文语料上的注意事项

---

## 参考说明

本笔记是基于对 BERTopic 方法框架与各类扩展功能的系统整理，重点强调概念理解、适用场景与不同类型之间的关系辨析，适合作为入门到进阶的理解性笔记使用。
