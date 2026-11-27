// Whitebox tests for internal helpers and invariants.

///|
test "mean: empty and single element" {
  assert_eq(mean([]), 0.0)
  let one : Array[Double] = [2.5]
  assert_eq(mean(one), 2.5)
}

///|
test "mean: typical values" {
  let xs : Array[Double] = [1.0, 2.0, 3.0, 4.0]
  assert_eq(mean(xs), 2.5)
}

///|
test "stddev: constant series has zero spread" {
  let xs : Array[Double] = [3.0, 3.0, 3.0]
  assert_true(stddev(xs) < 0.000001)
}

///|
test "stddev: empty series" {
  assert_eq(stddev([]), 0.0)
}

///|
test "percentile: interpolated median" {
  let s : Array[Double] = [1.0, 2.0, 3.0, 4.0]
  let p = percentile(s, 50.0)
  assert_true(p > 2.0 && p < 3.0)
}

///|
test "percentile: edges" {
  let s : Array[Double] = [1.0, 2.0, 3.0]
  assert_eq(percentile(s, 0.0), 1.0)
  assert_eq(percentile(s, 100.0), 3.0)
}

///|
test "threshold_value: all-equal series yields no breakpoints" {
  let xs : Array[Double] = [0.5, 0.5, 0.5]
  // A threshold equal to the value means no diff is strictly greater,
  // so every strategy correctly produces an empty breakpoint mask.
  let bp1 = detect_breakpoints(xs, Percentile(95.0))
  let bp2 = detect_breakpoints(xs, StandardDeviation(3.0))
  let bp3 = detect_breakpoints(xs, Interquartile(1.5))
  for i in 0..<bp1.length() {
    assert_true(!bp1[i])
    assert_true(!bp2[i])
    assert_true(!bp3[i])
  }
}

///|
test "diff_of: maps similarity to distance" {
  assert_eq(diff_of(1.0), 0.0)
  assert_true(diff_of(0.0) > 0.999999)
}

///|
test "is_sentence_end: recognizes all terminators" {
  assert_true(is_sentence_end('.'))
  assert_true(is_sentence_end('!'))
  assert_true(is_sentence_end('?'))
  assert_true(is_sentence_end('。'))
  assert_true(is_sentence_end('！'))
  assert_true(is_sentence_end('？'))
  assert_true(!is_sentence_end('a'))
}

///|
test "join_parts: inserts single spaces" {
  let parts : Array[String] = ["A.", "B.", "C."]
  assert_eq(join_parts(parts), "A. B. C.")
}

///|
test "is_abbreviation_context: only known abbreviations count" {
  assert_true(is_abbreviation_context("e"))
  assert_true(is_abbreviation_context("i"))
  assert_true(is_abbreviation_context("Dr"))
  assert_true(!is_abbreviation_context("One"))
}
