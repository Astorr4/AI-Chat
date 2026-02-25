import re
from config import HYBRID_VECTOR_WEIGHT, HYBRID_KEYWORD_WEIGHT
# ------------------------------
# Список стоп-слов (минимальный)
# ------------------------------
STOPWORDS = {
    "как", "что", "если", "при",
    "для", "это", "в", "на",
    "и", "или", "а", "но",
    "по", "с", "из", "к"
}


def keyword_score(query, text):

    query_tokens = re.findall(r"\w+", query.lower())
    text_lower = text.lower()

    score = 0
    weighted_matches = 0

    for token in query_tokens:

        # Игнорируем короткие слова
        if len(token) <= 3:
            continue

        # Игнорируем стоп-слова
        if token in STOPWORDS:
            continue

        if token in text_lower:

            weight = 1.0

            # Усиливаем цифры (401, 503, 0.05)
            if re.search(r"\d", token):
                weight += 0.5

            # Усиливаем слова с подчёркиванием (error_rate)
            if "_" in token:
                weight += 0.3

            # Усиливаем uppercase (CRITICAL)
            if token.isupper():
                weight += 0.4

            score += weight
            weighted_matches += 1

    if weighted_matches == 0:
        return 0.0

    # Нормализация
    return score / weighted_matches


class Reranker:

    def rerank(self, query, documents, top_k=3):

        if not documents:
            return []

        scored_docs = []

        for doc in documents:

            vector_score = doc.get("score", 0)
            k_score = keyword_score(query, doc.get("text", ""))

            # 🔥 Сбалансированный hybrid
            final_score = (
                    HYBRID_VECTOR_WEIGHT * vector_score +
                    HYBRID_KEYWORD_WEIGHT * k_score
            )

            doc["hybrid_score"] = final_score
            scored_docs.append(doc)

        scored_docs.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return scored_docs[:top_k]
