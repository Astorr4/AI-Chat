import numpy as np
import re
from core.reranker import Reranker, keyword_score
import difflib
from config import (
    SIMILARITY_STRONG,
    SIMILARITY_WEAK,
    BASE_RETRIEVAL_THRESHOLD,
    FUZZY_MATCH_CUTOFF,
    ADAPTIVE_THRESHOLD_MIN,
    ADAPTIVE_THRESHOLD_MAX
)

class RAG:
    def __init__(self, embed_model, vector_store, llm):
        self.embed = embed_model
        self.store = vector_store
        self.llm = llm
        self.reranker = Reranker()
        self.last_query_vector = None
        self.topic_shift_count = 0

    # ----------------------------------------------------
    # Query expansion (улучшено)
    # ----------------------------------------------------
    def expand_query(self, question, chat_memory, topic_mode=None):

        question_clean = question.strip()

        # Если памяти нет — ничего не расширяем
        if not chat_memory:
            return question_clean

        # Если новая тема — не расширяем
        if topic_mode == "new":
            return question_clean

        # Находим последнее пользовательское сообщение
        last_user_message = None
        for msg in reversed(chat_memory):
            if msg["role"] == "user":
                if msg["content"].strip() != question_clean:
                    last_user_message = msg["content"].strip()
                    break

        if not last_user_message:
            return question_clean

        word_count = len(question_clean.split())

        # -----------------------------------------
        # 1️⃣ Очень короткий вопрос (1–2 слова)
        # -----------------------------------------
        if word_count <= 2:
            # если strong — расширяем агрессивнее
            if topic_mode == "strong":
                return f"{question_clean} в контексте {last_user_message}"

            # если weak — мягкое расширение
            if topic_mode == "weak":
                return f"{question_clean} связано с {last_user_message}"

            return question_clean

        # -----------------------------------------
        # 2️⃣ Короткий вопрос (<=4 слова)
        # -----------------------------------------
        if word_count <= 4:

            if topic_mode == "strong":
                return f"{question_clean} про {last_user_message}"

            if topic_mode == "weak":
                return f"{question_clean} в рамках {last_user_message}"

            return question_clean

        # -----------------------------------------
        # 3️⃣ Триггеры уточнения
        # -----------------------------------------
        clarification_triggers = [
            "подробнее",
            "распиши",
            "объясни",
            "почему",
            "а если",
            "а что если",
            "а почему",
            "как это",
            "что насчет"
        ]

        if any(t in question_clean.lower() for t in clarification_triggers):

            if topic_mode in ["strong", "weak"]:
                return f"{question_clean} в контексте {last_user_message}"

        # -----------------------------------------
        # 4️⃣ В остальных случаях не расширяем
        # -----------------------------------------

        return question_clean

    def recalibrate_confidence(
            self,
            max_score,
            avg_score,
            coverage,
            similarity,
            topic_mode
    ):

        base = max_score

        # penalize if weak average
        if avg_score and avg_score < max_score * 0.6:
            base *= 0.9

        # penalize weak coverage
        if coverage < 0.5:
            base *= 0.85

        # small bonus for strong topic continuity
        if topic_mode == "strong" and similarity and similarity > 0.8:
            base *= 1.05

        return round(min(base, 0.92), 3)
    # ----------------------------------------------------
    # Multi-query split
    # ----------------------------------------------------
    def split_into_subqueries(self, question):
        separators = [" и ", " также ", ","]

        for sep in separators:
            if sep in question.lower():
                parts = question.split(sep)
                return [
                    p.strip()
                    for p in parts
                    if len(p.strip()) > 5
                ]

        return [question]

    # ----------------------------------------------------
    # MMR (Max Marginal Relevance)
    # ----------------------------------------------------
    def mmr(self, query_embedding, candidates, top_k=3, lambda_param=0.7):

        candidates = [
            c for c in candidates
            if c.get("vector") is not None
        ]

        if not candidates:
            return []

        selected = []
        selected_indices = []

        candidate_vectors = np.array(
            [c["vector"] for c in candidates]
        )
        candidate_scores = np.array(
            [c["score"] for c in candidates]
        )
        query_embedding = np.array(query_embedding)

        for _ in range(min(top_k, len(candidates))):

            if len(selected) == 0:
                idx = np.argmax(candidate_scores)
                selected.append(candidates[idx])
                selected_indices.append(idx)
                continue

            mmr_scores = []

            for i, candidate in enumerate(candidates):

                if i in selected_indices:
                    mmr_scores.append(-np.inf)
                    continue

                similarity_to_query = candidate_scores[i]

                similarity_to_selected = max(
                    np.dot(candidate_vectors[i], candidate_vectors[j])
                    for j in selected_indices
                )

                score = (
                        lambda_param * similarity_to_query
                        - (1 - lambda_param) * similarity_to_selected
                )

                mmr_scores.append(score)

            idx = np.argmax(mmr_scores)
            selected.append(candidates[idx])
            selected_indices.append(idx)

        return selected

    def lexical_search(self, query, top_k=5):

        # Получаем больше кандидатов из Qdrant
        results = self.store.search(
            self.embed.encode(query),
            limit=30
        )

        candidates = []

        query_tokens = re.findall(r"\w+", query.lower())

        for r in results:
            text = r.payload.get("text", "")
            text_lower = text.lower()

            score = 0

            for token in query_tokens:

                # точное совпадение
                if token in text_lower:
                    score += 1
                    continue

                # fuzzy совпадение
                words = re.findall(r"\w+", text_lower)
                if difflib.get_close_matches(token, words, n=1, cutoff=FUZZY_MATCH_CUTOFF):
                    score += 1

            if score > 0:
                candidates.append({
                    "text": text,
                    "source": r.payload.get("source", ""),
                    "doc_type": r.payload.get("doc_type"),
                    "score": score,
                    "vector": r.vector
                })

        # сортируем по lexical score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return {
            "documents": candidates[:top_k],
            "similarity": None,
            "topic_mode": "strong"
        }
    # ----------------------------------------------------
    # Основной поиск (улучшен)
    # ----------------------------------------------------
    def search(self, query, top_k=3):
        if len(query.split()) <= 2:
            return self.lexical_search(query, top_k)

        # ==============================
        # 1. Encode query
        # ==============================

        query_vector = np.array(self.embed.encode(query))
        query_vector /= (np.linalg.norm(query_vector) + 1e-9)

        # ==============================
        # 2. Semantic topic detection
        # ==============================

        similarity = None
        topic_mode = "new"

        if self.last_query_vector is not None:
            similarity = float(np.dot(query_vector, self.last_query_vector))

            if similarity > SIMILARITY_STRONG:
                topic_mode = "strong"
            elif similarity > SIMILARITY_WEAK:
                topic_mode = "weak"
            else:
                topic_mode = "new"

        # сохраняем текущий вектор
        self.last_query_vector = query_vector

        # ==============================
        # 3. Detect query type
        # ==============================

        query_type = self.detect_query_type(query)

        # ==============================
        # 4. Metadata-aware search
        # ==============================

        if query_type in ["numeric", "excel"]:
            results = self.store.search(
                query_vector,
                limit=10,
                doc_type="excel_row"
            )

        elif query_type == "text":
            results = self.store.search(
                query_vector,
                limit=10,
                doc_type="text_chunk"
            )

        else:
            results = self.store.search(
                query_vector,
                limit=10
            )

        if not results:

            # fallback keyword search
            keyword_candidates = []

            all_docs = self.store.search(query_vector, limit=20)

            for r in all_docs:
                text = r.payload.get("text", "")
                score = keyword_score(query, text)

                if score > 0:
                    keyword_candidates.append({
                        "text": text,
                        "source": r.payload.get("source", ""),
                        "doc_type": r.payload.get("doc_type"),
                        "score": score,
                        "vector": r.vector,
                    })

            if keyword_candidates:
                return {
                    "documents": keyword_candidates[:top_k],
                    "similarity": similarity,
                    "topic_mode": topic_mode
                }

            return {
                "documents": [],
                "similarity": similarity,
                "topic_mode": topic_mode
            }

        # ==============================
        # 5. Prepare candidates
        # ==============================

        candidates = []

        for r in results:
            candidates.append({
                "text": r.payload.get("text", ""),
                "source": r.payload.get("source", ""),
                "doc_type": r.payload.get("doc_type"),
                "score": r.score,
                "vector": r.vector,
            })

        # ==============================
        # 6. Rerank
        # ==============================

        reranked = self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )

        # ==============================
        # 7. Filter weak matches
        # ==============================

        threshold = 0.3

        if len(query.split()) <= 2:
            threshold = 0.15

        filtered = [
            doc for doc in reranked
            if doc.get("hybrid_score", 0) > threshold
        ]

        return {
            "documents": filtered,
            "similarity": similarity,
            "topic_mode": topic_mode
        }

    def detect_query_type(self, query: str):

        query_lower = query.lower()

        # если есть цифры → вероятно Excel
        if re.search(r"\d", query):
            return "numeric"

        excel_triggers = [
            "таблица", "лист", "строка",
            "значение", "показатель",
            "id", "код", "номер"
        ]

        if any(t in query_lower for t in excel_triggers):
            return "excel"

        text_triggers = [
            "как работает",
            "опиши",
            "регламент",
            "процесс",
            "политика"
        ]

        if any(t in query_lower for t in text_triggers):
            return "text"

        return "general"

    def retrieval_self_check(
            self,
            query,
            documents,
            topic_mode=None,
            similarity=None
    ):
        """
        Улучшенная версия:
        - fuzzy coverage
        - numeric consistency
        - адаптивный threshold
        """

        if not documents:
            return False, "no_documents"

        combined_text = " ".join(
            doc.get("text", "")
            for doc in documents
        ).lower()

        context_words = re.findall(r"\w+", combined_text)

        query_tokens = [
            t for t in re.findall(r"\w+", query.lower())
            if len(t) > 4 or re.search(r"\d", t)
        ]

        if not query_tokens:
            return True, 1.0

        # -----------------------------
        # Fuzzy coverage
        # -----------------------------

        matches = 0

        for token in query_tokens:

            # точное совпадение
            if token in context_words:
                matches += 1
                continue

            # fuzzy совпадение
            fuzzy_matches = difflib.get_close_matches(
                token,
                context_words,
                n=1,
                cutoff=FUZZY_MATCH_CUTOFF
            )

            if fuzzy_matches:
                matches += 1

        coverage = matches / len(query_tokens)

        # -----------------------------
        # Numeric consistency
        # -----------------------------

        numeric_tokens = re.findall(r"\d+", query)

        for num in numeric_tokens:
            if num not in combined_text:
                return False, "numeric_mismatch"

        # -----------------------------
        # Adaptive threshold
        # -----------------------------

        base_threshold = BASE_RETRIEVAL_THRESHOLD

        # короткий follow-up
        if len(query_tokens) <= 2 and topic_mode in ["strong", "weak"]:
            base_threshold = 0.1

        # новая тема — строже
        if topic_mode == "new":
            base_threshold += 0.05

        # высокая similarity — мягче
        if similarity and similarity > 0.75:
            base_threshold -= 0.05

        if coverage < 0.4:
            return False, "low_keyword_coverage"

        return True, coverage
    def dynamic_top_k(self, query, subqueries_count, topic_mode):

        word_count = len(query.split())
        has_numbers = bool(re.search(r"\d", query))

        top_k = 3

        if word_count <= 4:
            top_k = 2

        if word_count > 10:
            top_k = 5

        if has_numbers:
            top_k = max(top_k, 4)

        if subqueries_count > 1:
            top_k = max(top_k, 6)

        if topic_mode == "strong":
            top_k += 1

        return min(top_k, 8)

    def adaptive_threshold(self, base_threshold, topic_mode, similarity):

        threshold = base_threshold

        if topic_mode == "new":
            threshold += 0.05

        if topic_mode == "strong":
            threshold -= 0.05

        if similarity and similarity > 0.8:
            threshold -= 0.03

        return max(ADAPTIVE_THRESHOLD_MIN, min(threshold, ADAPTIVE_THRESHOLD_MAX))

    def verify_answer(self, question, context, answer):

        messages = [
            {
                "role": "system",
                "content": (
                    "Ты проверяешь корректность ответа. "
                    "Ответь одним словом: VALID, PARTIAL или INVALID. "
                    "Если ответ содержит информацию, отсутствующую в контексте — INVALID. "
                    "Если ответ частично выходит за контекст — PARTIAL. "
                    "Если полностью основан на контексте — VALID."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Контекст:\n{context}\n\n"
                    f"Вопрос:\n{question}\n\n"
                    f"Ответ:\n{answer}"
                )
            }
        ]

        result = self.llm.generate(messages)

        result = result.strip().upper()

        if "INVALID" in result:
            return "INVALID"
        if "PARTIAL" in result:
            return "PARTIAL"

        return "VALID"

    def normalize_query(self, query: str):

        stopwords = {
            "слушай", "можешь", "пожалуйста",
            "вообще", "простыми", "словами",
            "а", "ну", "подскажи",
            "расскажи", "объясни",
            "что", "такое", "зачем"
        }

        tokens = re.findall(r"\w+", query.lower())
        filtered = [
            t for t in tokens
            if t not in stopwords
        ]

        return " ".join(filtered)



