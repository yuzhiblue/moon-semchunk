///| Prints a deterministic benchmark report to stdout.

///|
/// Run with: moon run cmd/bench --target wasm
fn main {
  println("case | method | chunks | boundary_F1 | purity")
  for row in @bmark.run_benchmark() {
    let m = row.metrics
    println(
      "\{row.case_name} | \{row.label} | \{m.chunks} | \{m.boundary_f1} | \{m.purity}",
    )
  }
}
