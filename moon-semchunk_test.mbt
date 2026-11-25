// Blackbox tests for the public API of moon-semchunk.
// Run with `moon test` (or `moon test --deny-warn`).

///|
test "split_sentences: English punctuation" {
  let got = @moon-semchunk.split_sentences("Hello world. How are you? Fine!")
  assert_eq(got.length(), 3)
  assert_eq(got[0], "Hello world.")
  assert_eq(got[1], "How are you?")
  assert_eq(got[2], "Fine!")
}

///|
test "split_sentences: Chinese punctuation" {
  let got = @moon-semchunk.split_sentences("你好。世界！真的吗？")
  assert_eq(got.length(), 3)
  assert_eq(got[0], "你好。")
  assert_eq(got[1], "世界！")
  assert_eq(got[2], "真的吗？")
}

///|
test "split_sentences: drops empty and trailing whitespace" {
  let got = @moon-semchunk.split_sentences("  First.   Second.  ")
  assert_eq(got.length(), 2)
  assert_eq(got[0], "First.")
  assert_eq(got[1], "Second.")
}

///|
test "split_sentences: no punctuation yields one sentence" {
  let got = @moon-semchunk.split_sentences("a single sentence")
  assert_eq(got.length(), 1)
}

///|
test "cosine: identical vectors" {
  let a : Array[Double] = [1.0, 0.0, 0.0]
  let b : Array[Double] = [1.0, 0.0, 0.0]
  let c = @moon-semchunk.cosine(a, b)
  assert_true(c > 0.999999)
}

///|
test "cosine: orthogonal vectors" {
  let a : Array[Double] = [1.0, 0.0]
  let b : Array[Double] = [0.0, 1.0]
  let c = @moon-semchunk.cosine(a, b)
  assert_true(c < 0.000001)
}

///|
test "cosine: opposite vectors" {
  let a : Array[Double] = [1.0, 0.0]
  let b : Array[Double] = [-1.0, 0.0]
  let c = @moon-semchunk.cosine(a, b)
  assert_true(c < -0.999999)
}

///|
test "cosine: length mismatch and zero vector" {
  let a : Array[Double] = [1.0, 0.0]
  let b : Array[Double] = [1.0, 0.0, 0.0]
  assert_eq(@moon-semchunk.cosine(a, b), 0.0)
  let z : Array[Double] = [0.0, 0.0]
  assert_eq(@moon-semchunk.cosine(a, z), 0.0)
}

///|
test "adjacent_differences: identical embeddings give zero" {
  let e : Array[Array[Double]] = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
  let d = @moon-semchunk.adjacent_differences(e)
  assert_eq(d.length(), 2)
  assert_true(d[0] < 0.000001)
  assert_true(d[1] < 0.000001)
}

///|
test "detect_breakpoints: percentile catches the outlier" {
  // Three small diffs and one big jump; the 50th percentile sits near the
  // middle, so only the largest diff becomes a breakpoint.
  let diffs : Array[Double] = [0.01, 0.02, 0.90, 0.03]
  let bp = @moon-semchunk.detect_breakpoints(
    diffs,
    @moon-semchunk.Percentile(50.0),
  )
  assert_eq(bp.length(), 4)
  assert_true(bp[2])
  assert_true(!bp[0])
}

///|
test "detect_breakpoints: standard deviation" {
  let diffs : Array[Double] = [0.1, 0.1, 0.1, 0.9]
  let bp = @moon-semchunk.detect_breakpoints(
    diffs,
    @moon-semchunk.StandardDeviation(0.5),
  )
  // mean=0.3, stddev=0.346; cutoff = 0.473; only 0.9 exceeds it.
  assert_true(bp[3])
  assert_true(!bp[0])
}

///|
test "detect_breakpoints: interquartile" {
  let diffs : Array[Double] = [0.01, 0.02, 0.03, 0.04, 0.99]
  let bp = @moon-semchunk.detect_breakpoints(
    diffs,
    @moon-semchunk.Interquartile(1.5),
  )
  assert_eq(bp.length(), 5)
  assert_true(bp[4])
  assert_true(!bp[0])
}

///|
test "merge_sentences: splits at a single boundary" {
  let sentences : Array[String] = ["A.", "B.", "C."]
  let bp : Array[Bool] = [false, true]
  let chunks = @moon-semchunk.merge_sentences(sentences, bp)
  assert_eq(chunks.length(), 2)
  assert_eq(chunks[0].text, "A. B.")
  assert_eq(chunks[0].index, 0)
  assert_eq(chunks[1].text, "C.")
  assert_eq(chunks[1].index, 1)
}

///|
test "merge_sentences: empty input" {
  let sentences : Array[String] = []
  let bp : Array[Bool] = []
  let chunks = @moon-semchunk.merge_sentences(sentences, bp)
  assert_eq(chunks.length(), 0)
}

///|
test "chunk: single sentence returns one chunk" {
  fn embed(_s : String) -> Array[Double] {
    [1.0, 0.0]
  }
  let chunks = @moon-semchunk.chunk(
    "Only one.",
    embed,
    @moon-semchunk.Percentile(95.0),
  )
  assert_eq(chunks.length(), 1)
  assert_eq(chunks[0].text, "Only one.")
}

///|
test "chunk: empty text returns no chunks" {
  fn embed(_s : String) -> Array[Double] {
    [1.0, 0.0]
  }
  let chunks = @moon-semchunk.chunk("", embed, @moon-semchunk.Percentile(95.0))
  assert_eq(chunks.length(), 0)
}

///|
test "chunk: topic shift produces multiple chunks" {
  // Two clusters of sentences on different axes; the shift should be the
  // largest difference and become a breakpoint under percentile strategy.
  fn embed(s : String) -> Array[Double] {
    if s.contains("cat") {
      [1.0, 0.0]
    } else {
      [0.0, 1.0]
    }
  }
  let text = "The cat sleeps. The cat eats. The ocean is deep. The ocean is wide."
  let chunks = @moon-semchunk.chunk(
    text,
    embed,
    @moon-semchunk.Percentile(70.0),
  )
  assert_eq(chunks.length(), 2)
  assert_true(chunks[0].text.contains("cat"))
  assert_true(chunks[1].text.contains("ocean"))
}

///|
test "split_sentences: decimal point is not a boundary" {
  let got = @moon-semchunk.split_sentences("Pi is 3.14 today.")
  assert_eq(got.length(), 1)
  assert_eq(got[0], "Pi is 3.14 today.")
}

///|
test "split_sentences: known abbreviation is not a boundary" {
  let got = @moon-semchunk.split_sentences("Dr. Smith went home.")
  assert_eq(got.length(), 1)
  assert_eq(got[0], "Dr. Smith went home.")
}

///|
test "split_sentences: inner dot of e.g. is not a boundary" {
  let got = @moon-semchunk.split_sentences("Use a tool, e.g. a hammer.")
  assert_eq(got.length(), 1)
}

///|
test "split_sentences: consecutive punctuation stays in one sentence" {
  let got = @moon-semchunk.split_sentences("Wait... what happened?")
  assert_eq(got.length(), 2)
  assert_eq(got[0], "Wait...")
  assert_eq(got[1], "what happened?")
}

///|
test "split_sentences: mixed terminators in a run are one boundary" {
  let got = @moon-semchunk.split_sentences("Really?! Yes.")
  assert_eq(got.length(), 2)
  assert_eq(got[0], "Really?!")
  assert_eq(got[1], "Yes.")
}

///|
test "detect_breakpoints: percentile out of range is clamped" {
  let diffs : Array[Double] = [0.1, 0.2, 0.9]
  let low = @moon-semchunk.detect_breakpoints(
    diffs,
    @moon-semchunk.Percentile(-50.0),
  )
  let high = @moon-semchunk.detect_breakpoints(
    diffs,
    @moon-semchunk.Percentile(200.0),
  )
  assert_eq(low.length(), 3)
  assert_eq(high.length(), 3)
  assert_eq(
    low,
    @moon-semchunk.detect_breakpoints(diffs, @moon-semchunk.Percentile(0.0)),
  )
  assert_eq(
    high,
    @moon-semchunk.detect_breakpoints(diffs, @moon-semchunk.Percentile(100.0)),
  )
}

///|
test "context_window: radius zero returns the sentence itself" {
  let sentences : Array[String] = ["A.", "B.", "C."]
  fn embed(s : String) -> Array[Double] {
    [s.length().to_double(), 0.0]
  }
  let vectors = @moon-semchunk.embed_with_context(sentences, embed, 0)
  assert_eq(vectors.length(), 3)
  assert_eq(vectors[1][0], 2.0)
}

///|
test "embed_with_context: neighbours are included" {
  let sentences : Array[String] = ["A.", "B.", "C."]
  fn embed(s : String) -> Array[Double] {
    [s.length().to_double(), 0.0]
  }
  let vectors = @moon-semchunk.embed_with_context(sentences, embed, 1)
  assert_eq(vectors.length(), 3)
  assert_true(vectors[1][0] > 2.0)
}

///|
test "chunk_by_budget: splits long text and reindexes" {
  let chunks = @moon-semchunk.chunk_by_budget(
    "One. Two. Three. Four. Five.", 10, 0,
  )
  assert_true(chunks.length() >= 2)
  assert_true(@moon-semchunk.chunks_well_formed(chunks))
  for chunk in chunks {
    assert_true(chunk.text.length() <= 10)
  }
}

///|
test "chunk_with_options: negative values are normalised" {
  fn embed(_s : String) -> Array[Double] {
    [1.0, 0.0]
  }
  let opts = @moon-semchunk.make_options(
    @moon-semchunk.Percentile(70.0),
    -1,
    -1,
    -1,
  )
  let chunks = @moon-semchunk.chunk_with_options("A. B. C.", embed, opts)
  assert_eq(chunks.length(), 1)
}

///|
test "chunk_stats: empty and non-empty" {
  let empty = @moon-semchunk.chunk_stats([])
  assert_eq(empty.count, 0)
  assert_eq(empty.mean_length, 0.0)
  let chunks : Array[@moon-semchunk.Chunk] = [
    { text: "A.", index: 0, },
    { text: "Hello world.", index: 1, },
  ]
  let stats = @moon-semchunk.chunk_stats(chunks)
  assert_eq(stats.count, 2)
  assert_eq(stats.min_length, 2)
  assert_eq(stats.max_length, 12)
  assert_true(stats.mean_length > 6.0)
  assert_eq(stats.sentence_count, 2)
}

///|
test "add_sentence_overlap: prefixes previous tail" {
  let chunks : Array[@moon-semchunk.Chunk] = [
    { text: "A. B.", index: 0, },
    { text: "C. D.", index: 1, },
  ]
  let overlapped = @moon-semchunk.add_sentence_overlap(chunks, 1)
  assert_eq(overlapped.length(), 2)
  assert_true(overlapped[1].text.contains("B."))
  assert_true(@moon-semchunk.chunks_well_formed(overlapped))
}

///|
test "filter helpers and flatten preserve order" {
  let chunks : Array[@moon-semchunk.Chunk] = [
    { text: "A.", index: 0, },
    { text: "Long chunk here.", index: 1, },
  ]
  let kept = @moon-semchunk.drop_short_chunks(chunks, 5)
  assert_eq(kept.length(), 1)
  assert_eq(kept[0].text, "Long chunk here.")
  assert_eq(@moon-semchunk.total_chars(chunks), 18)
  assert_eq(@moon-semchunk.flatten_chunks(chunks, "|"), "A.|Long chunk here.")
}

///|
test "window_text: batches long strings" {
  let windows = @moon-semchunk.window_text("abcdefg", 3)
  assert_eq(windows.length(), 3)
  assert_eq(windows[0], "abc")
  assert_eq(windows[2], "g")
  assert_eq(@moon-semchunk.window_text("abc", 0).length(), 0)
}
