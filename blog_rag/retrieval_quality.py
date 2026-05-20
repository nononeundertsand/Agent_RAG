import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document

from .config import RETRIEVAL_MIN_DOCS, RETRIEVAL_MIN_QUERY_COVERAGE, RETRIEVAL_MIN_SCORE


_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass
class RetrievalQuality:
    doc_count: int
    max_score: float
    query_coverage: float
    source_count: int
    should_expand: bool
    reason: str


def tokenize(text: str) -> set[str]:
    tokens = set()
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if re.search(r"[\u4e00-\u9fff]", token):
            tokens.update(char for char in token if "\u4e00" <= char <= "\u9fff")
        elif len(token) > 1:
            tokens.add(token)
    return tokens


def _coverage(query: str, docs: Iterable[Document]) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    doc_tokens: set[str] = set()
    for doc in docs:
        doc_tokens.update(tokenize(doc.page_content[:2000]))
    return len(query_tokens & doc_tokens) / len(query_tokens)


def assess_retrieval_quality(query: str, scored_docs: list[tuple[float, str, Document]]) -> RetrievalQuality:
    docs = [doc for _, _, doc in scored_docs]
    doc_count = len(docs)
    max_score = max((score for score, _, _ in scored_docs), default=0.0)
    query_coverage = _coverage(query, docs)
    source_count = len({doc.metadata.get("source", "") for doc in docs if doc.metadata.get("source")})

    failures = []
    if doc_count < RETRIEVAL_MIN_DOCS:
        failures.append("too_few_docs")
    if max_score < RETRIEVAL_MIN_SCORE:
        failures.append("low_vector_score")
    if query_coverage < RETRIEVAL_MIN_QUERY_COVERAGE:
        failures.append("low_query_coverage")

    return RetrievalQuality(
        doc_count=doc_count,
        max_score=max_score,
        query_coverage=query_coverage,
        source_count=source_count,
        should_expand=bool(failures),
        reason=",".join(failures) if failures else "pass",
    )
