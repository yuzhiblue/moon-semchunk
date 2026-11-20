// Blackbox tests for the deterministic benchmark harness.

///|
test "semantic chunking recovers all known topic boundaries" {
  for row in @bench.run_benchmark() {
    if row.label == "semantic" {
      assert_true(row.metrics.boundary_f1 > 0.95)
      assert_true(row.metrics.purity > 0.95)
    }
  }
}

///|
test "fixed-size baselines are clearly worse on every case" {
  for row in @bench.run_benchmark() {
    if row.label != "semantic" {
      assert_true(row.metrics.boundary_f1 < 0.7)
      assert_true(row.metrics.purity < 0.9)
    }
  }
}

///|
test "end-to-end chunk() agrees with breakpoint grouping" {
  for row in @bench.run_benchmark() {
    if row.label == "semantic" {
      assert_eq(row.api_chunks, row.metrics.chunks)
    }
  }
}

///|
test "cases cover different topic geometries" {
  let case = @bench.make_case("probe", 4, 5)
  assert_eq(case.topics, 4)
  assert_eq(case.sentences.length(), 20)
  let bps = @bench.true_breakpoints(case)
  assert_eq(bps.length(), 19)
  let mut count = 0
  for b in bps {
    if b {
      count += 1
    }
  }
  assert_eq(count, 3)
}
