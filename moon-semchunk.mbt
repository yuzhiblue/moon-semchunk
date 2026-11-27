// moon-semchunk: semantic chunking for RAG.
//
// Splits a document at semantic boundaries instead of fixed character counts.
// The design follows LangChain's SemanticChunker (MIT), but is reimplemented
// natively in MoonBit with a pluggable Embedder contract: the library never
// calls a model itself, it only consumes vectors supplied by the caller.

///|
/// A chunk produced by semantic splitting.
pub(all) struct Chunk {
  text : String
  index : Int32
}

///|
/// Strategy for deciding where a semantic boundary lies.
pub(all) enum Threshold {
  /// Keep the largest `amount`-percent of differences as breakpoints.
  Percentile(Double)
  /// Break where difference > mean + `amount` * stddev.
  StandardDeviation(Double)
  /// Break where difference > Q3 + `amount` * IQR.
  Interquartile(Double)
}

// ---------------------------------------------------------------------------
// Sentence splitting
// ---------------------------------------------------------------------------

///|
/// Break text into sentences at sentence-ending punctuation.
///
/// Supported terminators: `.`, `!`, `?`, `。`, `！`, `？`. The terminator is
/// kept at the end of the sentence. Empty sentences are dropped.
///
/// Context rules avoid the three common false positives of naive splitters:
/// a dot between digits is a decimal point (`3.14`), a dot after a known
/// abbreviation before a capital is not a boundary (`Dr. Smith`), and a run
/// of punctuation (`...`, `!!?`) is kept as one terminator.
pub fn split_sentences(text : String) -> Array[String] {
  let out : Array[String] = []
  let chars = text.to_array()
  let n = chars.length()
  let mut buf = StringBuilder()
  let mut last_word = StringBuilder()
  let mut i = 0
  while i < n {
    let ch = chars[i]
    if is_sentence_end(ch) {
      let prev_digit = i > 0 && is_ascii_digit(chars[i - 1])
      let next_alpha = i + 1 < n && is_ascii_alpha(chars[i + 1])
      let next_digit = i + 1 < n && is_ascii_digit(chars[i + 1])
      let next_space = i + 1 < n && chars[i + 1] == ' '
      let word = last_word.to_string()
      let is_decimal = ch == '.' && prev_digit && next_digit
      let is_inner_dot = ch == '.' &&
        next_alpha &&
        is_abbreviation_context(word)
      let is_abbreviation = ch == '.' &&
        next_space &&
        is_known_abbreviation(word)
      if !is_decimal && !is_inner_dot && !is_abbreviation {
        buf.write_char(ch)
        while i + 1 < n && is_sentence_end(chars[i + 1]) {
          i += 1
          buf.write_char(chars[i])
        }
        push_sentence(out, buf)
        buf = StringBuilder()
        last_word = StringBuilder()
        i += 1
        continue
      }
    }
    buf.write_char(ch)
    if ch == ' ' {
      last_word = StringBuilder()
    } else if is_ascii_alpha(ch) || is_ascii_digit(ch) || ch == '.' {
      last_word.write_char(ch)
    }
    i += 1
  }
  push_sentence(out, buf)
  out
}

///|
fn is_sentence_end(ch : Char) -> Bool {
  match ch {
    '.' | '!' | '?' | '。' | '！' | '？' => true
    _ => false
  }
}

///|
fn is_ascii_digit(ch : Char) -> Bool {
  ch >= '0' && ch <= '9'
}

///|
fn is_ascii_alpha(ch : Char) -> Bool {
  (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
}

///|
fn is_known_abbreviation(word : String) -> Bool {
  match word {
    "Dr"
    | "Mr"
    | "Mrs"
    | "Ms"
    | "Prof"
    | "Sr"
    | "Jr"
    | "St"
    | "vs"
    | "etc"
    | "e.g"
    | "i.e"
    | "No"
    | "Fig"
    | "Inc"
    | "Ltd"
    | "Co"
    | "approx"
    | "dept"
    | "est"
    | "min"
    | "max" => true
    _ => false
  }
}

///|
/// A dot followed by a letter is an inner dot only inside (or directly
/// after) a known abbreviation, e.g. `e.g` or `Dr.Smith`. Ordinary words
/// such as `One.Two` stay splittable.
fn is_abbreviation_context(word : String) -> Bool {
  if is_known_abbreviation(word) {
    true
  } else {
    match word {
      "e" | "i" => true
      _ => false
    }
  }
}

///|
fn push_sentence(out : Array[String], buf : StringBuilder) -> Unit {
  let s = buf.to_string().trim().to_owned()
  if s.length() > 0 {
    out.push(s)
  }
}

// ---------------------------------------------------------------------------
// Vector math
// ---------------------------------------------------------------------------

///|
/// Cosine similarity between two vectors.
/// Returns 0.0 when either vector has zero magnitude or lengths differ.
pub fn cosine(a : Array[Double], b : Array[Double]) -> Double {
  let n = a.length()
  if n != b.length() {
    return 0.0
  }
  let mut dot = 0.0
  let mut na = 0.0
  let mut nb = 0.0
  for i in 0..<n {
    let av = a[i]
    let bv = b[i]
    dot += av * bv
    na += av * av
    nb += bv * bv
  }
  if na == 0.0 || nb == 0.0 {
    return 0.0
  }
  dot / (na.sqrt() * nb.sqrt())
}

///|
/// Convert similarity into a distance-like difference value.
fn diff_of(similarity : Double) -> Double {
  1.0 - similarity
}

///|
/// Adjacent sentence differences from a list of embeddings.
pub fn adjacent_differences(embeddings : Array[Array[Double]]) -> Array[Double] {
  let out : Array[Double] = []
  let n = embeddings.length()
  for i in 1..<n {
    out.push(diff_of(cosine(embeddings[i - 1], embeddings[i])))
  }
  out
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

///|
fn mean(xs : Array[Double]) -> Double {
  let n = xs.length()
  if n == 0 {
    return 0.0
  }
  let mut sum = 0.0
  for x in xs {
    sum += x
  }
  sum / n.to_double()
}

///|
fn stddev(xs : Array[Double]) -> Double {
  let n = xs.length()
  if n == 0 {
    return 0.0
  }
  let m = mean(xs)
  let mut sum = 0.0
  for x in xs {
    sum += (x - m) * (x - m)
  }
  (sum / n.to_double()).sqrt()
}

///|
fn sorted_copy(xs : Array[Double]) -> Array[Double] {
  let out : Array[Double] = []
  for x in xs {
    out.push(x)
  }
  out.sort()
  out
}

///|
/// Linear-interpolated percentile of a sorted array (p in 0..100).
fn percentile(sorted : Array[Double], p : Double) -> Double {
  let n = sorted.length()
  if n == 0 {
    return 0.0
  }
  if n == 1 {
    return sorted[0]
  }
  let pct = if p < 0.0 { 0.0 } else if p > 100.0 { 100.0 } else { p }
  let rank = (n - 1).to_double() * (pct / 100.0)
  let lo = rank.to_int()
  let hi = (lo + 1).min(n - 1)
  let frac = rank - lo.to_double()
  sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

///|
/// Decide which adjacent positions are semantic breakpoints.
/// Returns a boolean array of length `differences.length()`.
pub fn detect_breakpoints(
  differences : Array[Double],
  strategy : Threshold,
) -> Array[Bool] {
  let cut = threshold_value(differences, strategy)
  let out : Array[Bool] = []
  for d in differences {
    out.push(d > cut)
  }
  out
}

///|
fn threshold_value(xs : Array[Double], strategy : Threshold) -> Double {
  match strategy {
    Percentile(amount) => percentile(sorted_copy(xs), amount)
    StandardDeviation(amount) => mean(xs) + amount * stddev(xs)
    Interquartile(amount) => {
      let s = sorted_copy(xs)
      let q1 = percentile(s, 25.0)
      let q3 = percentile(s, 75.0)
      q3 + amount * (q3 - q1)
    }
  }
}

// ---------------------------------------------------------------------------
// Chunk assembly
// ---------------------------------------------------------------------------

///|
/// Merge sentences into chunks at breakpoint positions.
///
/// `breakpoints[i]` is true when a boundary sits between sentence `i` and
/// sentence `i + 1`. Sentences are joined with a single space.
pub fn merge_sentences(
  sentences : Array[String],
  breakpoints : Array[Bool],
) -> Array[Chunk] {
  let out : Array[Chunk] = []
  let n = sentences.length()
  if n == 0 {
    return out
  }
  let parts : Array[String] = []
  for i in 0..<n {
    parts.push(sentences[i])
    if i < breakpoints.length() && breakpoints[i] {
      out.push({ text: join_parts(parts), index: out.length(), })
      parts.clear()
    }
  }
  if parts.length() > 0 {
    out.push({ text: join_parts(parts), index: out.length(), })
  }
  out
}

///|
fn join_parts(parts : Array[String]) -> String {
  let buf = StringBuilder()
  let n = parts.length()
  for i in 0..<n {
    if i > 0 {
      buf.write_char(' ')
    }
    buf.write_string(parts[i])
  }
  buf.to_string()
}

// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

///|
/// Split text at semantic boundaries using a caller-supplied embedder.
///
/// This is the one-shot entry point: it splits into sentences, obtains a
/// vector for each sentence via `embedder`, computes adjacent differences,
/// detects breakpoints with `strategy`, and merges sentences into chunks.
///
/// The library never calls a model itself; `embedder` is a pure callback so
/// the whole pipeline stays deterministic and testable on every backend.
pub fn chunk(
  text : String,
  embedder : (String) -> Array[Double],
  strategy : Threshold,
) -> Array[Chunk] {
  let sentences = split_sentences(text)
  if sentences.length() < 2 {
    if sentences.length() == 1 {
      return [{ text: sentences[0], index: 0, }]
    }
    return []
  }
  let embeddings : Array[Array[Double]] = []
  for s in sentences {
    embeddings.push(embedder(s))
  }
  let differences = adjacent_differences(embeddings)
  let breakpoints = detect_breakpoints(differences, strategy)
  merge_sentences(sentences, breakpoints)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

///|
/// Options for the configurable chunking entry point.
///
/// `buffer_size` controls context-aware embedding: when greater than zero,
/// sentence `i` is embedded together with `buffer_size` neighbours on each
/// side, so its vector carries local context. This mirrors LangChain's
/// `buffer_size` but stays a pure callback contract.
///
/// `max_chunk_size` caps the character length of a chunk. A chunk that is
/// still too long after semantic splitting is split again on sentence
/// boundaries. `overlap` repeats that many trailing sentences of one piece
/// at the head of the next, preserving context across the artificial cut.
pub(all) struct ChunkOptions {
  /// Threshold strategy used to decide breakpoints.
  strategy : Threshold
  /// Number of neighbouring sentences included per side when embedding.
  buffer_size : Int32
  /// Maximum characters per chunk; 0 disables the size budget.
  max_chunk_size : Int32
  /// Number of overlapping sentences when a chunk is split for size.
  overlap : Int32
}

///|
/// Build options with defaults: no context window and no size budget.
pub fn default_options(strategy : Threshold) -> ChunkOptions {
  { strategy, buffer_size: 0, max_chunk_size: 0, overlap: 0, }
}

///|
/// Clamp negative option values to zero so the pipeline stays total.
fn normalise_options(options : ChunkOptions) -> ChunkOptions {
  {
    strategy: options.strategy,
    buffer_size: if options.buffer_size < 0 {
      0
    } else {
      options.buffer_size
    },
    max_chunk_size: if options.max_chunk_size < 0 {
      0
    } else {
      options.max_chunk_size
    },
    overlap: if options.overlap < 0 {
      0
    } else {
      options.overlap
    },
  }
}

// ---------------------------------------------------------------------------
// Context-aware embedding
// ---------------------------------------------------------------------------

///|
/// Build the text fed to the embedder for sentence `index`.
///
/// With `radius = 0` this is the sentence itself. Otherwise the returned
/// string concatenates sentences from `max(0, index - radius)` through
/// `min(n - 1, index + radius)` with single spaces, keeping sentence order.
fn context_window(
  sentences : Array[String],
  index : Int32,
  radius : Int32,
) -> String {
  let n = sentences.length()
  let start = if index - radius < 0 { 0 } else { index - radius }
  let stop = if index + radius >= n { n - 1 } else { index + radius }
  let buf = StringBuilder()
  for i in start..<=stop {
    if i > start {
      buf.write_char(' ')
    }
    buf.write_string(sentences[i])
  }
  buf.to_string()
}

///|
/// Embed every sentence with its context window.
///
/// The result has the same length as `sentences`. `buffer_size` is clamped to
/// zero for negative input, making this function total for any caller value.
pub fn embed_with_context(
  sentences : Array[String],
  embedder : (String) -> Array[Double],
  buffer_size : Int32,
) -> Array[Array[Double]] {
  let radius = if buffer_size < 0 { 0 } else { buffer_size }
  let out : Array[Array[Double]] = []
  for i in 0..<sentences.length() {
    out.push(embedder(context_window(sentences, i, radius)))
  }
  out
}

///|
/// Compute adjacent differences from context-aware embeddings.
///
/// This is the context-aware counterpart of `adjacent_differences`, kept as
/// a separate entry so callers can compose their own pipelines.
pub fn contextual_differences(
  sentences : Array[String],
  embedder : (String) -> Array[Double],
  buffer_size : Int32,
) -> Array[Double] {
  adjacent_differences(embed_with_context(sentences, embedder, buffer_size))
}

// ---------------------------------------------------------------------------
// Size budget and overlap
// ---------------------------------------------------------------------------

///|
/// Greedily group sentences so each group stays within `max_size` characters.
///
/// A single sentence longer than `max_size` is still emitted alone; the
/// budget is a soft cap that only splits between sentence boundaries.
fn group_by_budget(
  sentences : Array[String],
  max_size : Int32,
  overlap : Int32,
) -> Array[Array[String]] {
  let out : Array[Array[String]] = []
  if sentences.length() == 0 {
    return out
  }
  let mut current : Array[String] = []
  let mut length = 0
  for sentence in sentences {
    let extra = if current.length() == 0 { 0 } else { 1 }
    if current.length() > 0 && length + extra + sentence.length() > max_size {
      out.push(current)
      // Only carry over the trailing sentences that still fit next to the
      // new one. Otherwise overlap alone can push a chunk past the budget.
      let room = max_size - sentence.length() - 1
      let tail = overlap.min(current.length())
      let stop = current.length() - tail
      let mut j = current.length() - 1
      let mut chosen_start = current.length()
      let mut kept = 0
      while j >= stop {
        let added = if j == current.length() - 1 {
          current[j].length()
        } else {
          1 + current[j].length()
        }
        if kept + added > room {
          break
        }
        kept += added
        chosen_start = j
        j -= 1
      }
      let next : Array[String] = []
      for k in chosen_start..<current.length() {
        next.push(current[k])
      }
      current = next
      length = kept
    }
    if current.length() > 0 {
      length += 1
    }
    current.push(sentence)
    length += sentence.length()
  }
  if current.length() > 0 {
    out.push(current)
  }
  out
}

///|
/// Apply the size budget to an existing chunk list and reindex the result.
///
/// Chunks already within the budget are kept as-is. Oversized chunks are
/// regrouped from their sentences, with optional overlap between pieces.
fn apply_budget(
  chunks : Array[Chunk],
  max_size : Int32,
  overlap : Int32,
) -> Array[Chunk] {
  if max_size <= 0 {
    return chunks
  }
  let out : Array[Chunk] = []
  for chunk in chunks {
    if chunk.text.length() <= max_size {
      out.push({ text: chunk.text, index: out.length(), })
    } else {
      let sentences = split_sentences(chunk.text)
      let groups = group_by_budget(sentences, max_size, overlap)
      for group in groups {
        out.push({ text: join_parts(group), index: out.length(), })
      }
    }
  }
  out
}

///|
/// Split text into budgeted pieces without semantic analysis.
///
/// Useful as a rule-based baseline or when callers already know the document
/// has a single topic and only need a size cap with overlap.
pub fn chunk_by_budget(
  text : String,
  max_size : Int32,
  overlap : Int32,
) -> Array[Chunk] {
  let sentences = split_sentences(text)
  let groups = group_by_budget(
    sentences,
    if max_size < 0 {
      0
    } else {
      max_size
    },
    overlap,
  )
  let out : Array[Chunk] = []
  for group in groups {
    out.push({ text: join_parts(group), index: out.length(), })
  }
  out
}

// ---------------------------------------------------------------------------
// Configurable entry point
// ---------------------------------------------------------------------------

///|
/// Split text with full control over context, strategy and size budget.
///
/// Pipeline: split into sentences, embed each with its context window,
/// compute adjacent differences, detect breakpoints, merge into chunks,
/// then enforce the size budget with overlap. The simpler `chunk()` function
/// is equivalent to calling this with `default_options(strategy)`.
pub fn chunk_with_options(
  text : String,
  embedder : (String) -> Array[Double],
  options : ChunkOptions,
) -> Array[Chunk] {
  let opts = normalise_options(options)
  let sentences = split_sentences(text)
  if sentences.length() == 0 {
    return []
  }
  if sentences.length() == 1 {
    return [{ text: sentences[0], index: 0, }]
  }
  let embeddings = embed_with_context(sentences, embedder, opts.buffer_size)
  let differences = adjacent_differences(embeddings)
  let breakpoints = detect_breakpoints(differences, opts.strategy)
  let merged = merge_sentences(sentences, breakpoints)
  apply_budget(merged, opts.max_chunk_size, opts.overlap)
}

// ---------------------------------------------------------------------------
// Quality statistics
// ---------------------------------------------------------------------------

///|
/// Summary statistics for a chunk list, useful for tuning a RAG pipeline.
pub(all) struct ChunkStats {
  /// Number of chunks.
  count : Int32
  /// Shortest chunk length in characters; 0 when there are no chunks.
  min_length : Int32
  /// Longest chunk length in characters; 0 when there are no chunks.
  max_length : Int32
  /// Mean chunk length in characters; 0 when there are no chunks.
  mean_length : Double
  /// Total number of sentences across all chunks.
  sentence_count : Int32
}

///|
/// Compute quality statistics from a chunk list.
///
/// Does not re-split text; it inspects the chunks as given, so it can be
/// applied to output from any entry point in this module.
pub fn chunk_stats(chunks : Array[Chunk]) -> ChunkStats {
  if chunks.length() == 0 {
    return {
      count: 0,
      min_length: 0,
      max_length: 0,
      mean_length: 0.0,
      sentence_count: 0,
    }
  }
  let mut smallest = chunks[0].text.length()
  let mut largest = smallest
  let mut total = 0
  let mut sentences = 0
  for chunk in chunks {
    let len = chunk.text.length()
    if len < smallest {
      smallest = len
    }
    if len > largest {
      largest = len
    }
    total += len
    sentences += split_sentences(chunk.text).length()
  }
  {
    count: chunks.length(),
    min_length: smallest,
    max_length: largest,
    mean_length: total.to_double() / chunks.length().to_double(),
    sentence_count: sentences,
  }
}

///|
/// Add sentence-level overlap to an existing chunk list.
///
/// Unlike the budget overlap in `ChunkOptions`, this operates after all
/// splitting is done: each chunk (except the first) is prefixed with the
/// last `count` sentences of its predecessor. Indices are reassigned.
pub fn add_sentence_overlap(
  chunks : Array[Chunk],
  count : Int32,
) -> Array[Chunk] {
  if chunks.length() < 2 || count <= 0 {
    return chunks
  }
  let out : Array[Chunk] = []
  let mut previous : Array[String] = []
  for chunk in chunks {
    let mut text = chunk.text
    if previous.length() > 0 {
      let prefix = join_parts(previous)
      text = prefix + " " + text
    }
    out.push({ text, index: out.length(), })
    previous = split_sentences(chunk.text)
    if previous.length() > count {
      let tail : Array[String] = []
      for i in (previous.length() - count)..<previous.length() {
        tail.push(previous[i])
      }
      previous = tail
    }
  }
  out
}

///|
/// Concatenate all chunk texts with a separator, preserving order.
pub fn flatten_chunks(chunks : Array[Chunk], separator : String) -> String {
  let buf = StringBuilder()
  for i in 0..<chunks.length() {
    if i > 0 {
      buf.write_string(separator)
    }
    buf.write_string(chunks[i].text)
  }
  buf.to_string()
}

///|
/// Filter chunks by predicate while keeping indices contiguous.
///
/// The `index` field of the returned chunks is reassigned from zero so the
/// result remains a well-formed chunk list.
pub fn filter_chunks(
  chunks : Array[Chunk],
  predicate : (Chunk) -> Bool,
) -> Array[Chunk] {
  let out : Array[Chunk] = []
  for chunk in chunks {
    if predicate(chunk) {
      out.push({ text: chunk.text, index: out.length(), })
    }
  }
  out
}

///|
/// Drop chunks shorter than `min_length` characters.
///
/// Very short chunks are often noise from stray punctuation or headings;
/// removing them before embedding can improve retrieval quality. Indices are
/// reassigned so the result stays well-formed.
pub fn drop_short_chunks(
  chunks : Array[Chunk],
  min_length : Int32,
) -> Array[Chunk] {
  filter_chunks(chunks, fn(chunk) { chunk.text.length() >= min_length })
}

///|
/// Drop chunks longer than `max_length` characters.
///
/// Complements `drop_short_chunks` for pipelines that prefer to discard
/// oversized material rather than split it further.
pub fn drop_long_chunks(
  chunks : Array[Chunk],
  max_length : Int32,
) -> Array[Chunk] {
  filter_chunks(chunks, fn(chunk) { chunk.text.length() <= max_length })
}

///|
/// Create a ChunkOptions with a context window and size budget configured.
///
/// This is a convenience constructor for the common case where callers want
/// context-aware embedding and a hard size cap, leaving overlap at zero.
pub fn budget_options(
  strategy : Threshold,
  buffer_size : Int32,
  max_chunk_size : Int32,
) -> ChunkOptions {
  { strategy, buffer_size, max_chunk_size, overlap: 0, }
}

///|
/// Create a ChunkOptions with all fields specified.
pub fn make_options(
  strategy : Threshold,
  buffer_size : Int32,
  max_chunk_size : Int32,
  overlap : Int32,
) -> ChunkOptions {
  { strategy, buffer_size, max_chunk_size, overlap, }
}

///|
/// Convert a chunk list into plain text strings in order.
pub fn chunk_texts(chunks : Array[Chunk]) -> Array[String] {
  let out : Array[String] = []
  for chunk in chunks {
    out.push(chunk.text)
  }
  out
}

///|
/// Count the total characters across all chunks.
pub fn total_chars(chunks : Array[Chunk]) -> Int32 {
  let mut total = 0
  for chunk in chunks {
    total += chunk.text.length()
  }
  total
}

///|
/// Check that chunk indices are contiguous starting from zero.
///
/// Useful as an invariant check after composing several chunk operations.
pub fn chunks_well_formed(chunks : Array[Chunk]) -> Bool {
  for i in 0..<chunks.length() {
    if chunks[i].index != i {
      return false
    }
  }
  true
}

///|
/// Split a long string into equal-width windows for batching.
///
/// This is not semantic chunking; it is a helper for feeding many documents
/// through an embedder in batches. The final window may be shorter.
pub fn window_text(text : String, width : Int32) -> Array[String] {
  if width <= 0 {
    return []
  }
  let out : Array[String] = []
  let chars = text.to_array()
  let mut start = 0
  while start < chars.length() {
    let stop = (start + width).min(chars.length())
    let buf = StringBuilder()
    for i in start..<stop {
      buf.write_char(chars[i])
    }
    out.push(buf.to_string())
    start = stop
  }
  out
}
