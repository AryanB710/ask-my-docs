"""Turn files into (title, [(section_path, markdown_text)]) so the chunker has a
single format to work on. HTML and PDF are converted to markdown-ish text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


@dataclass
class Section:
    path: list[str]
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    sections: list[Section]
    source_path: str


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def parse_markdown(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    title = fallback_title
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section([h for _, h in stack] or [title], body))
        buf.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue
        m = None if in_code else _HEADING_RE.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and title == fallback_title:
                title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()
    return title, sections


def _html_to_markdown(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "tr"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name.startswith("h"):
            lines += ["", "#" * int(el.name[1]) + " " + txt, ""]
        elif el.name == "li":
            lines.append("- " + txt)
        elif el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["td", "th"])]
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines += [txt, ""]
    return "\n".join(lines)


def load_file(path: Path, root: Path) -> Document:
    doc_id = str(path.relative_to(root))
    fallback = path.stem.replace("-", " ").replace("_", " ").title()
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        title, sections = parse_markdown(path.read_text(encoding="utf-8", errors="replace"), fallback)
    elif suffix in {".html", ".htm"}:
        title, sections = parse_markdown(_html_to_markdown(path.read_text(encoding="utf-8", errors="replace")), fallback)
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) or fallback
        sections = [
            Section([title, f"Page {i + 1}"], txt.strip())
            for i, page in enumerate(reader.pages)
            if (txt := page.extract_text() or "").strip()
        ]
    else:
        raise ValueError(f"unsupported file type: {path}")
    return Document(doc_id=doc_id, title=title, sections=sections, source_path=str(path))


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus dir not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"no supported files ({', '.join(sorted(SUPPORTED))}) in {root}")
    return [load_file(p, root) for p in files]
