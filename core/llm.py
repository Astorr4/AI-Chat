import json
import random
import time
from urllib.parse import urlparse
import requests
from config import (
    LLM_URL,
    LLM_LOCAL_ONLY,
    TEMPERATURE,
    MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BASE_DELAY,
    LLM_RETRY_MAX_DELAY,
    LLM_CIRCUIT_BREAKER_THRESHOLD,
    LLM_CIRCUIT_BREAKER_RESET_SECONDS,
)


class LLM:
    def __init__(self):
        self.url = LLM_URL
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

        if LLM_LOCAL_ONLY:
            parsed = urlparse(self.url)
            host = (parsed.hostname or "").lower()
            if host not in {"127.0.0.1", "localhost", "::1"}:
                raise RuntimeError(
                    "LLM_LOCAL_ONLY=True, но LLM_URL указывает не на локальный endpoint"
                )

    def _is_circuit_open(self):
        return time.time() < self._circuit_open_until

    def _record_success(self):
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= LLM_CIRCUIT_BREAKER_THRESHOLD:
            self._circuit_open_until = time.time() + LLM_CIRCUIT_BREAKER_RESET_SECONDS

    @staticmethod
    def _retry_delay(attempt_index):
        base = LLM_RETRY_BASE_DELAY * (2 ** attempt_index)
        base = min(base, LLM_RETRY_MAX_DELAY)
        jitter = random.uniform(0, 0.25 * base)
        return base + jitter

    # ==============================
    # Обычная генерация (без стрима)
    # ==============================
    def generate(self, messages):
        if self._is_circuit_open():
            return "[LLM ERROR] circuit_open"

        last_error = None

        for attempt in range(LLM_RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    self.url,
                    json={
                        "messages": messages,
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                    },
                    timeout=LLM_TIMEOUT_SECONDS,
                )

                response.raise_for_status()
                data = response.json()
                self._record_success()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                last_error = e
                self._record_failure()
                if attempt < LLM_RETRY_ATTEMPTS - 1:
                    time.sleep(self._retry_delay(attempt))

        return f"[LLM ERROR] {str(last_error)}"

    def generate_title(self, question, answer):

        messages = [
            {
                "role": "system",
                "content": "Сгенерируй короткий заголовок (до 6 слов) для этого диалога. Без пояснений."
            },
            {
                "role": "user",
                "content": f"Вопрос: {question}\nОтвет: {answer}"
            }
        ]

        title = ""

        for token in self.stream(messages):
            title += token

        return title.strip().replace('"', '')

    # ==============================
    # Стриминг токенов
    # ==============================
    def stream(self, messages):
        """
        Генератор токенов из llama-server (OpenAI-compatible stream)
        """
        if self._is_circuit_open():
            yield "\n\n[STREAM ERROR] circuit_open"
            return

        last_error = None

        for attempt in range(LLM_RETRY_ATTEMPTS):
            try:
                response = requests.post(
                    self.url,
                    json={
                        "messages": messages,
                        "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "stream": True,
                    },
                    stream=True,
                    timeout=LLM_TIMEOUT_SECONDS,
                )

                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    decoded_line = line.decode("utf-8").strip()

                    if not decoded_line.startswith("data:"):
                        continue

                    data_str = decoded_line.replace("data: ", "")

                    if data_str == "[DONE]":
                        self._record_success()
                        return

                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content

                    except json.JSONDecodeError:
                        continue

                self._record_success()
                return

            except Exception as e:
                last_error = e
                self._record_failure()
                if attempt < LLM_RETRY_ATTEMPTS - 1:
                    time.sleep(self._retry_delay(attempt))

        yield f"\n\n[STREAM ERROR] {str(last_error)}"
