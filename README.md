# moon-semchunk

语义分块（Semantic Chunking）库，用纯 MoonBit 实现，面向 RAG / 文档检索场景。

## 项目价值与生态定位

RAG 流程里，切分（chunking）质量直接决定检索效果。固定长度或递归切分会在语义中途断开，把相关句子拆散。语义分块按句子语义边界切分，把话题相关的句子聚在一起，是检索质量提升的关键一环。

MoonBit 生态现状：`moonrag` 提供 Recursive / Token / Markdown / HTML 切分器，`MoonRetrieve`、`moonnlp` 也有文档切分能力，但**全部是规则驱动或结构驱动，没有任何基于 embedding 相似度的语义分块**。`moon-semchunk` 补齐这个空缺，可作为 RAG 生态的独立基础件。

## 预期使用场景

1. **企业知识库入库切分**：将产品手册、FAQ、内部制度文档导入 RAG 系统时，用 `chunk()` 替换固定长度切分。调用方在 `embedder` 回调中接入实际的 embedding 服务，库按句子语义边界成块，避免把一个问题的题干和答案切到不同块里，直接提升检索命中率。
2. **长篇文档与论文的检索增强**：对用户上传的长篇技术文档或论文做索引时，用 `StandardDeviation` / `Interquartile` 策略识别章节主题切换点，使每个 chunk 围绕单一主题，减少跨主题噪声进入检索结果。
3. **中英混排语料处理**：`split_sentences` 同时识别中英文句末标点，可对双语客服对话、中英混排的技术笔记直接切分，无需维护两套规则；`embedder` 由调用方注入，可随时换用任意多语言向量模型。

## 交付范围与工程边界

交付：

- 句子切分（中英文标点）
- 余弦相似度与相邻句差异计算
- 三种断点策略：percentile / standard_deviation / interquartile
- 句子合并为 `Chunk`
- 一站式 `chunk()` API（embedder 以回调注入）
- 接口设计便于接入 `moonrag` / `MoonRetrieve` 等现有 RAG 流程（不依赖其类型或运行时）

明确不做：

- 不内置 embedding 模型，不调用任何 LLM 服务。embedder 由调用方注入，库保持纯函数、零模型依赖。
- 不做向量存储与检索（由 `moonrag` / `vcdb` 等负责）。
- 不做全局钩子、文件 I/O 或 FFI。

## 实现路径与技术理解

语义分块的常见路径有三类：按长度、按结构（标题/段落）、按语义相似度。本项目选择第三条，因为前两条在生态中已有成熟实现，而第三条是真实缺口。

核心链路：`split_sentences` → `embedder` 回调 → `cosine` 相似度 → `1 - similarity` 差异序列 → 断点检测 → 合并。关键点在于 embedder 是 `(String) -> Array[Double]` 的纯回调：这让整条链路在任意后端（native / wasm / js）可测、可复现，也避免了把模型依赖绑进库里。

## 与已有项目的关系

算法参考 LangChain 的 `SemanticChunker`（MIT，<https://github.com/langchain-ai/langchain-text-splitters>）和 LlamaIndex 的 `SemanticSplitterNodeParser`（MIT，<https://github.com/run-llama/llama_index>），是**算法移植与 MoonBit 原生重写**，非源码复制。与 `moonrag` 等的关系是互补：它们做规则切分与检索，本项目做语义切分，接口设计便于协同。

移植范围：统计阈值计算（percentile / standard_deviation / interquartile）、相似度转差异值、相邻句断点检测的整体思路。未移植：LangChain 的 buffer_size 窗口、sentence_grouper 与多模型适配层，以及 LlamaIndex 的 token 预算、层级节点结构和 LLM 改写逻辑。重写部分：MoonBit 纯函数 API、embedder 回调注入和 `Chunk` 数据结构。

## 快速开始

```
moon add yuzhiblue/moon-semchunk
```

```moonbit
fn embed(_s : String) -> Array[Double] { [1.0, 0.0] }
let chunks = @moon_semchunk.chunk(text, embed, @moon_semchunk.Percentile(95.0))
```

运行示例：`moon run cmd/main --target wasm`

## 测试与验证

`moon test --deny-warn --target wasm`：32 个测试全部通过（黑盒 17 + 白盒 10 + golden 5），覆盖分句、相似度、三种断点策略、合并、一站式 API 与数值回归。CI 覆盖 check / build / test / run example。

## 许可证

MIT。算法参考 LangChain `SemanticChunker`（MIT，<https://github.com/langchain-ai/langchain-text-splitters>）与 LlamaIndex `SemanticSplitterNodeParser`（MIT，<https://github.com/run-llama/llama_index>）；本项目未复制其源码，详见"与已有项目的关系"。
