# Demo corpus: "LumenQueue" docs (fictional)

**Everything in `corpus/` is invented.** LumenQueue is not a real product. Every price, limit
and date in it was written to test retrieval. Don't read anything in it as a fact about a
real service.

Why a fictional corpus?

- **The CI eval must be reproducible.** The golden set (`../../eval/golden.jsonl`) only works
  against docs that never change underneath it.
- **It contains deliberate distractors.** v1 vs v2 rate limits, Standard vs Enterprise
  retention, two near-identical error codes. Pure vector search tends to get these wrong, and
  hybrid retrieval plus reranking should get them right.
- **It's small enough to index in about a second on a CI runner.**

To use your own domain, point `askdocs ingest` at a folder of `.md`, `.txt`, `.html` or `.pdf`
files and write a golden set for it. See "Bring your own docs" in the main README.
