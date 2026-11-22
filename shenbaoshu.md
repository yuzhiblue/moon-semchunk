# moon-semchunk 项目申报书

## 基本信息

- 项目名称：moon-semchunk：面向 RAG 的 MoonBit 语义分块库
- 参赛者：yuzhiblue
- 联系方式：630110598@qq.com
- GitHub 仓库链接：https://github.com/yuzhiblue/moon-semchunk
- 项目方向：MoonBit RAG 生态基础库 / 文本处理基础设施
- 是否为移植项目：是（算法移植与 MoonBit 原生重写）

## 项目简介

RAG 流程中，文档切分质量直接决定检索效果。固定长度或递归切分会在语义中途断开，把同一话题的句子拆散、把不同话题的句子拼在一起，导致检索命中率下降。语义分块（Semantic Chunking）按句子语义边界切分，把话题相关的句子聚成块，是提升检索质量的关键一环。

MoonBit 生态现状：`moonrag` 提供 Recursive / Token / Markdown / HTML 切分器，`MoonRetrieve`、`moonnlp` 也具备文档切分能力，但**全部是规则驱动或结构驱动，没有任何基于 embedding 相似度的语义分块实现**。这是一个真实且明确的生态缺口——RAG 开发者在 MoonBit 中只能选择"切得整齐"但"切得不准"的方案。moon-semchunk 补齐这个空缺，作为独立、可复用的基础件，与现有 RAG 库形成互补。

技术上，本项目选择"语义相似度"路径而非按长度或按结构切分：后两者在生态中已有成熟实现，第三条路径才是缺口。核心链路为：分句 → 调用方注入的 embedder 回调生成向量 → 相邻句余弦差异 → 统计阈值检测断点 → 合并为 Chunk。embedder 设计为 `(String) -> Array[Double]` 纯回调，库本身零模型依赖、零 FFI、零文件 I/O，整条链路在 native / wasm / js 上均可确定性地测试和复现。这一设计让库保持极小的依赖面和极高的可组合性。

## 核心功能范围

- 句子切分（中英文标点），并处理小数点（`3.14`）、英文缩写（`Dr. Smith`、`e.g.`）与连续标点（`...`、`?!`）的上下文判断，不产生纯标点垃圾句；
- 余弦相似度与相邻句差异计算；
- 三种断点策略：Percentile / StandardDeviation / Interquartile，参数做范围钳制；
- 句子合并为 `Chunk`，提供一站式 `chunk()` API；
- 确定性对照基准：三种已知主题边界的合成语料、固定长度切分基线，以边界 F1 与主题纯度量化语义切分收益；
- 差异测试、golden 数值回归、黑盒与白盒测试，覆盖核心路径与边界条件；
- README 示例覆盖基本用法、策略选择与自定义 embedder 接入。

## 预期验收产物

- `yuzhiblue/moon-semchunk` 库，已发布到 mooncakes.io（当前版本 0.1.2），可通过 `moon add yuzhiblue/moon-semchunk` 直接引入；
- 42 个测试全部通过，CI 覆盖 check / build / test / run example / run benchmark；
- 可运行示例 `cmd/main` 与基准报告 `cmd/bench`，确定性输出、可复现；
- 基准数据证明：三组语料上语义分块边界 F1 与主题纯度均为 1.0，固定长度基线显著更低；
- 完整 README：项目定位、使用场景、API 说明、边界行为、移植来源与许可证。

明确不做：不内置 embedding 模型、不调用 LLM 服务、不做向量存储与检索、不做全局钩子与 FFI。

## 移植或参考说明

- 原项目名称：LangChain `SemanticChunker`
- 原项目链接：https://github.com/langchain-ai/langchain-text-splitters
- 原项目许可证：MIT

- 原项目名称：LlamaIndex `SemanticSplitterNodeParser`
- 原项目链接：https://github.com/run-llama/llama_index
- 原项目许可证：MIT

- 本项目许可证：MIT

本项目是算法移植与 MoonBit 原生重写，非源码复制。移植范围：统计阈值计算、相似度转差异值、相邻句断点检测的整体思路。重新设计部分：MoonBit 纯函数 API、embedder 回调注入、`Chunk` 数据结构、分句边界处理与确定性对照基准。未移植：LangChain 的 buffer 窗口与多模型适配层、LlamaIndex 的 token 预算与层级节点结构。与 `moonrag` 等项目的关系是互补：它们做规则切分与检索，本项目做语义切分，接口设计便于接入现有 RAG 流程。