// Learn more about moon.mod configuration:
// https://docs.moonbitlang.com/en/latest/toolchain/moon/module.html
//
// To add a dependency, run this command in your terminal:
//   moon add moonbitlang/x
//
// Or manually declare it in `import`, for example:
// import {
//   "moonbitlang/x@0.4.6",
// }

name = "yuzhiblue/moon-semchunk"

version = "0.2.0"

readme = "README.mbt.md"

repository = "https://github.com/yuzhiblue/moon-semchunk"

license = "MIT"

keywords = [ "rag", "semantic-chunking", "nlp", "text" ]

preferred_target = "wasm"

description = "Semantic chunking for RAG: embedding-similarity-based document splitting with a pluggable Embedder contract."
