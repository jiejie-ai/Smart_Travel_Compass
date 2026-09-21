"""Multi-query expansion for RAGFlow retrieval."""

EXPANSION_ASPECTS = ["景点", "美食", "交通", "住宿", "天气", "攻略", "门票", "文化"]


def expand_query(query: str) -> list[str]:
    expanded = [query]
    for aspect in EXPANSION_ASPECTS:
        expanded.append(f"{query} {aspect}")
    return expanded[:5]
