///|
/// Runnable example for moon-semchunk.
///
/// This demo uses a tiny deterministic "embedder" so it needs no model or
/// network. In a real RAG pipeline, replace it with a call to your embedding
/// provider. Run with: moon run cmd/main --target wasm
fn demo_embedder(text : String) -> Array[Double] {
  // A crude 2D embedding: axis 0 tracks "cat", axis 1 tracks "ocean".
  // This makes the semantic shift visible without an external model.
  let lower = text.to_lower()
  let cat = if lower.contains("cat") { 1.0 } else { 0.0 }
  let ocean = if lower.contains("ocean") { 1.0 } else { 0.0 }
  if cat == 0.0 && ocean == 0.0 {
    [0.5, 0.5]
  } else {
    [cat, ocean]
  }
}

fn main {
  let text = "The cat sleeps on the mat. The cat loves to play. The ocean is deep and blue. Ships cross the ocean daily."
  let chunks = @semchunk.chunk(text, demo_embedder, @semchunk.Percentile(70.0))
  println("Semantic chunks:")
  for i in 0..<chunks.length() {
    println("[\{i}] \{chunks[i].text}")
  }
}
