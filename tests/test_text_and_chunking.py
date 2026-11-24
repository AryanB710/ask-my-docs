from askdocs.config import ChunkingConfig
from askdocs.ingest.chunker import chunk_document
from askdocs.ingest.loaders import Document, parse_markdown
from askdocs.text import split_answer_sentences, tokenize


def test_tokenize_keeps_identifiers_and_normalizes_numbers():
    toks = tokenize("Error LQ-1033 on ap-south: 3,000 requests, retention_seconds")
    assert "lq-1033" in toks and "ap-south" in toks and "retention_seconds" in toks
    assert "3000" in toks
    assert "request" in toks  # light stemming


def test_answer_sentence_split_keeps_citations_attached():
    parts = split_answer_sentences("The limit is 500 [1]. Bursts are not allowed [2][3].")
    assert parts == ["The limit is 500 [1].", "Bursts are not allowed [2][3]."]


def _doc(md: str) -> Document:
    title, sections = parse_markdown(md, "Fallback")
    return Document("doc.md", title, sections, "doc.md")


def test_chunks_never_cross_headings_and_carry_section_path():
    md = "# Title\n\n## A\n\nAlpha text.\n\n## B\n\n### B1\n\nBeta text."
    chunks = chunk_document(_doc(md), ChunkingConfig(max_words=100))
    assert [c.section for c in chunks] == ["A", "B > B1"]
    assert chunks[1].header == "Title > B > B1"


def test_table_rows_are_not_split():
    md = "# T\n\n## Codes\n\n| a | b |\n|---|---|\n| LQ-1 | first meaning |\n| LQ-2 | second meaning |"
    (chunk,) = chunk_document(_doc(md), ChunkingConfig(max_words=100))
    assert "| LQ-2 | second meaning |" in chunk.text


def test_long_section_split_with_overlap_and_size_bound():
    sentences = " ".join(f"Sentence number {i} has five words." for i in range(60))
    chunks = chunk_document(_doc(f"# T\n\n## S\n\n{sentences}"), ChunkingConfig(max_words=50, overlap_words=12))
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 50 + 12 for c in chunks)
    # overlap: the tail of chunk i is repeated at the start of chunk i+1
    last = chunks[0].text.splitlines()[-1]
    assert last in chunks[1].text.splitlines()[:3]


def test_code_fence_headings_are_not_headings():
    md = "# T\n\n## S\n\n```\n# not a heading\n```\n\nafter"
    chunks = chunk_document(_doc(md), ChunkingConfig())
    assert len(chunks) == 1 and "# not a heading" in chunks[0].text
