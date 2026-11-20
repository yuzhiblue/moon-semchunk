///| Deterministic benchmark: semantic chunking vs fixed-size baselines.
///
/// The corpus is generated in-process with known topic blocks. The embedder
/// maps each sentence to a one-hot topic vector, so this is an idealised
/// setup: it validates that the algorithm recovers known boundaries, not
/// retrieval quality under real embeddings.

///|
/// A sentence with its ground-truth topic label.
pub(all) struct LabelledSentence {
  text : String
  topic : Int32
}

///|
/// A synthetic document made of contiguous topic blocks.
pub(all) struct BenchCase {
  name : String
  sentences : Array[LabelledSentence]
  topics : Int32
}

///|
/// Quality of one split strategy on one case.
pub(all) struct SplitMetrics {
  chunks : Int32
  boundary_f1 : Double
  purity : Double
}

///|
/// One row of the benchmark report.
pub(all) struct BenchRow {
  case_name : String
  label : String
  metrics : SplitMetrics
  api_chunks : Int32
}

///| Build a case with `topics` blocks, each containing `per_topic` sentences.

///|
/// Supports up to six topics.
pub fn make_case(name : String, topics : Int32, per_topic : Int32) -> BenchCase {
  let themes = ["cat", "ocean", "forest", "desert", "city", "space"]
  let sentences : Array[LabelledSentence] = []
  for t in 0..<topics {
    for s in 0..<per_topic {
      let text = "The \{themes[t]} topic, sentence \{s}."
      sentences.push({ text, topic: t, })
    }
  }
  { name, sentences, topics, }
}

///|
fn one_hot(topic : Int32, n : Int32) -> Array[Double] {
  let v : Array[Double] = []
  for i in 0..<n {
    if i == topic {
      v.push(1.0)
    } else {
      v.push(0.0)
    }
  }
  v
}

///|
/// Deterministic embedder: recovers the topic label from the sentence text.
pub fn case_embedder(c : BenchCase) -> (String) -> Array[Double] {
  fn(text : String) -> Array[Double] {
    let mut topic = 0
    for s in c.sentences {
      if s.text == text {
        topic = s.topic
      }
    }
    one_hot(topic, c.topics)
  }
}

///|
/// Boundaries where the ground-truth topic changes.
pub fn true_breakpoints(c : BenchCase) -> Array[Bool] {
  let out : Array[Bool] = []
  let n = c.sentences.length()
  if n < 2 {
    return out
  }
  for i in 0..<(n - 1) {
    out.push(c.sentences[i].topic != c.sentences[i + 1].topic)
  }
  out
}

///|
fn fixed_breakpoints(n : Int32, size : Int32) -> Array[Bool] {
  let out : Array[Bool] = []
  if n < 2 {
    return out
  }
  for i in 0..<(n - 1) {
    out.push((i + 1) % size == 0)
  }
  out
}

///|
fn group_by_breakpoints(
  c : BenchCase,
  bps : Array[Bool],
) -> Array[Array[Int32]] {
  let groups : Array[Array[Int32]] = []
  let mut cur : Array[Int32] = []
  for i in 0..<c.sentences.length() {
    cur.push(c.sentences[i].topic)
    if i < bps.length() && bps[i] {
      groups.push(cur)
      cur = []
    }
  }
  if cur.length() > 0 {
    groups.push(cur)
  }
  groups
}

///|
fn boundary_f1(pred : Array[Bool], actual : Array[Bool]) -> Double {
  let mut hit = 0
  let mut false_positive = 0
  let mut missed = 0
  for i in 0..<pred.length() {
    if pred[i] && actual[i] {
      hit += 1
    } else if pred[i] {
      false_positive += 1
    } else if actual[i] {
      missed += 1
    }
  }
  let precision = if hit + false_positive == 0 {
    0.0
  } else {
    hit.to_double() / (hit + false_positive).to_double()
  }
  let recall = if hit + missed == 0 {
    0.0
  } else {
    hit.to_double() / (hit + missed).to_double()
  }
  if precision + recall == 0.0 {
    0.0
  } else {
    2.0 * precision * recall / (precision + recall)
  }
}

///|
fn group_purity(groups : Array[Array[Int32]]) -> Double {
  let mut correct = 0
  let mut total = 0
  for g in groups {
    total += g.length()
    let mut best = 0
    for t in g {
      let mut count = 0
      for u in g {
        if u == t {
          count += 1
        }
      }
      if count > best {
        best = count
      }
    }
    correct += best
  }
  if total == 0 {
    0.0
  } else {
    correct.to_double() / total.to_double()
  }
}

///|
fn metrics_of(c : BenchCase, bps : Array[Bool]) -> SplitMetrics {
  let groups = group_by_breakpoints(c, bps)
  {
    chunks: groups.length(),
    boundary_f1: boundary_f1(bps, true_breakpoints(c)),
    purity: group_purity(groups),
  }
}

///|
fn join_text(sentences : Array[LabelledSentence]) -> String {
  let buf = StringBuilder()
  let mut first = true
  for s in sentences {
    if !first {
      buf.write_char(' ')
    }
    buf.write_string(s.text)
    first = false
  }
  buf.to_string()
}

///|
/// Run all cases; the semantic row is checked against the end-to-end API.
///
/// p70 is deliberate: with one-hot embeddings adjacent differences are
/// exactly 0 inside a topic and 1 at boundaries. Higher percentiles (p80/p95)
/// can land on the maximum itself when boundaries are frequent, so `d > cut`
/// finds no breakpoints. p70 stays below the boundary mass in all three
/// geometries and keeps the comparison deterministic.
pub fn run_benchmark() -> Array[BenchRow] {
  let cases = [
    make_case("4 topics x 5 sentences", 4, 5),
    make_case("3 topics x 7 sentences", 3, 7),
    make_case("6 topics x 3 sentences", 6, 3),
  ]
  let rows : Array[BenchRow] = []
  for c in cases {
    let embedder = case_embedder(c)
    let embeddings : Array[Array[Double]] = []
    for s in c.sentences {
      embeddings.push(embedder(s.text))
    }
    let diffs = @semchunk.adjacent_differences(embeddings)
    let semantic_bps = @semchunk.detect_breakpoints(
      diffs,
      @semchunk.Percentile(70.0),
    )
    let chunks = @semchunk.chunk(
      join_text(c.sentences),
      embedder,
      @semchunk.Percentile(70.0),
    )
    rows.push({
      case_name: c.name,
      label: "semantic",
      metrics: metrics_of(c, semantic_bps),
      api_chunks: chunks.length(),
    })
    rows.push({
      case_name: c.name,
      label: "fixed-6",
      metrics: metrics_of(c, fixed_breakpoints(c.sentences.length(), 6)),
      api_chunks: 0,
    })
    rows.push({
      case_name: c.name,
      label: "fixed-4",
      metrics: metrics_of(c, fixed_breakpoints(c.sentences.length(), 4)),
      api_chunks: 0,
    })
  }
  rows
}
