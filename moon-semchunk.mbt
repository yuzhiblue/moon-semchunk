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
      let is_decimal = ch == '.' && prev_digit && next_digit
      let is_inner_dot = ch == '.' && next_alpha
      let is_abbreviation = ch == '.' &&
        next_space &&
        is_known_abbreviation(last_word.to_string())
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
