from __future__ import annotations

from askdocs.types import Retrieved

ABSTAIN_TOKEN = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are a documentation assistant. Answer ONLY from the numbered sources.

Rules:
1. Every sentence in your answer MUST end with one or more citations like [1] or [2][3].
2. Only cite a source if it directly states the fact in that sentence.
3. Do not use outside knowledge. Do not guess numbers, dates or limits.
4. If the sources do not contain the answer, reply with exactly: {ABSTAIN_TOKEN}
5. Be concise: 1-3 sentences unless the question asks for steps."""

STRICT_SUFFIX = """

Your previous answer broke the citation rules. These sentences were missing a citation or
were not supported by the source they cited:
{problems}
Rewrite the answer. Keep only claims that the sources state directly, and cite each one."""


def format_sources(contexts: list[Retrieved]) -> str:
    return "\n\n".join(f"[{i}] ({r.chunk.header})\n{r.chunk.text}" for i, r in enumerate(contexts, start=1))


def build_messages(question: str, contexts: list[Retrieved], problems: list[str] | None = None) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT
    if problems:
        system += STRICT_SUFFIX.format(problems="\n".join(f"- {p}" for p in problems))
    user = f"Sources:\n\n{format_sources(contexts)}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
