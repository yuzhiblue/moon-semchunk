# moon-semchunk

Semantic chunking for RAG in pure MoonBit.

Splits documents at semantic boundaries instead of fixed character counts.
Sentences are embedded via a caller-supplied callback, adjacent cosine
differences are computed, and statistical breakpoints decide where one chunk
ends and the next begins.

## Why

Naive fixed-length or recursive splitting cuts mid-thought and harms retrieval
quality. Semantic chunking keeps topically related sentences together. Existing
MoonBit RAG libraries provide recursive / token / markdown / HTML splitters, but
none implement similarity-based semantic boundaries. `moon-semchunk` fills that
gap as a standalone, reusable library.

Design follows LangChain's `SemanticChunker` (MIT,
<https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/text_splitter.py>) and LlamaIndex's
`SemanticSplitterNodeParser` (MIT, <https://github.com/run-llama/llama_index>),
reimplemented natively in MoonBit. It is a port of the algorithm, not a copy of
the source. Ported: statistical threshold computation, similarity-to-difference
conversion and adjacent-sentence breakpoint detection. Not ported: buffer
windows, token budgets, hierarchical node structures and LLM-based rewrites.

## Install

```
moon add yuzhiblue/moon-semchunk
```

## Usage

```moonbit nocheck
///|
fn embed(text : String) -> Array[Double] {
  // Replace with a call to your embedding provider.
  [1.0, 0.0]
}

///|
fn main {
  let text = "The cat sleeps. The cat plays. The ocean is deep. Ships cross it."
  let chunks = @moon_semchunk.chunk(
    text,
    embed,
    @moon_semchunk.Percentile(95.0),
  )
  for chunk in chunks {
    println(chunk.text)
  }
}
```

The library never calls a model itself. `embedder : (String) -> Array[Double]`
is a pure callback, so the whole pipeline is deterministic and testable on
every backend (native / wasm / wasm-gc / js).

## API

- `chunk(text, embedder, strategy)` — one-shot semantic splitting.
- `chunk_with_options(text, embedder, options)` — configurable splitting with context windows and size budgets.
- `ChunkOptions` — configuration for buffer size, max chunk size and overlap.
- `embed_with_context(sentences, embedder, buffer_size)` — context-aware sentence embeddings.
- `chunk_by_budget(text, max_size, overlap)` — rule-only size-capped splitting.
- `chunk_stats(chunks)` — count, min/max/mean length and sentence count.
- `add_sentence_overlap(chunks, count)` — post-split sentence overlap.
- `split_sentences(text)` — sentence segmentation (`.!?。！？`) with decimal-point, abbreviation and repeated-punctuation handling.
- `cosine(a, b)` — cosine similarity.
- `adjacent_differences(embeddings)` — `1 - cosine` between neighbours.
- `detect_breakpoints(differences, strategy)` — boolean breakpoint mask.
- `merge_sentences(sentences, breakpoints)` — assemble `Chunk` values.

`Threshold` strategies:

- `Percentile(p)` — break above the p-th percentile of differences (default 95).
- `StandardDeviation(k)` — break above `mean + k * stddev` (default 3).
- `Interquartile(k)` — break above `Q3 + k * IQR` (default 1.5).

## License

MIT. Algorithm references: LangChain `SemanticChunker` (MIT, <https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/text_splitter.py>) and LlamaIndex `SemanticSplitterNodeParser` (MIT, <https://github.com/run-llama/llama_index>). Source code was not copied.
