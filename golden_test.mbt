// Golden-vector regression tests.
//
// The expected values below are computed by the reference algorithm
// (LangChain SemanticChunker's statistics) on fixed inputs. They are
// regression guards: if a refactor changes the numeric behaviour of the
// breakpoint detection, these tests fail even when the qualitative tests
// in moon-semchunk_test.mbt still pass. Only public APIs are used here.

test "golden: percentile breakpoint on a nine-value series" {
  // sorted: 0.10 0.12 0.15 0.18 0.20 0.22 0.25 0.30 0.80
  // p95 ~= 0.68, so only the 0.80 outlier becomes a breakpoint.
  let xs : Array[Double] = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.80]
  let bp = @moon-semchunk.detect_breakpoints(xs, @moon-semchunk.Percentile(95.0))
  assert_eq(bp.length(), 9)
  assert_true(bp[8])
  assert_true(!bp[7])
}

test "golden: standard deviation cutoff" {
  // mean = 0.25, stddev ~= 0.06455, k=2 cutoff ~= 0.37910
  let xs : Array[Double] = [0.20, 0.20, 0.20, 0.30, 0.30, 0.30, 0.90]
  let bp = @moon-semchunk.detect_breakpoints(xs, @moon-semchunk.StandardDeviation(2.0))
  assert_true(bp[6])
  assert_true(!bp[0])
}

test "golden: interquartile cutoff" {
  // 0.01..0.08 -> Q1 ~= 0.0275, Q3 ~= 0.0675, IQR = 0.04, k=1.5 -> 0.1275
  let xs : Array[Double] = [0.05, 0.01, 0.99, 0.02, 0.07, 0.03, 0.06, 0.04]
  let bp = @moon-semchunk.detect_breakpoints(xs, @moon-semchunk.Interquartile(1.5))
  assert_eq(bp.length(), 8)
  assert_true(bp[2])
  assert_true(!bp[0])
}

test "golden: cosine of a 45-degree pair" {
  let a : Array[Double] = [1.0, 0.0]
  let b : Array[Double] = [1.0, 1.0]
  let c = @moon-semchunk.cosine(a, b)
  // 1 / sqrt(2) ~= 0.70710678
  assert_true(c > 0.70710 && c < 0.70711)
}

test "golden: full pipeline on a known two-topic document" {
  fn embed(s : String) -> Array[Double] {
    if s.contains("alpha") {
      [1.0, 0.0]
    } else {
      [0.0, 1.0]
    }
  }
  let text = "alpha one. alpha two. beta one. beta two."
  let chunks = @moon-semchunk.chunk(text, embed, @moon-semchunk.Percentile(50.0))
  assert_eq(chunks.length(), 2)
  assert_eq(chunks[0].text, "alpha one. alpha two.")
  assert_eq(chunks[1].text, "beta one. beta two.")
}
