import json
import requests
from config import LLM_URL, TEMPERATURE, MAX_TOKENS


class LLM:
    def __init__(self):
        self.url = LLM_URL

    # ==============================
    # Обычная генерация (без стрима)
    # ==============================
    def generate(self, messages):
        try:
            response = requests.post(
                self.url,
                json={
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=300
            )

            response.raise_for_status()
            data = response.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            return f"[LLM ERROR] {str(e)}"

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
        try:
            response = requests.post(
                self.url,
                json={
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "stream": True
                },
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                decoded_line = line.decode("utf-8").strip()

                # llama-server отправляет строки вида:
                # data: {...json...}
                if not decoded_line.startswith("data:"):
                    continue

                data_str = decoded_line.replace("data: ", "")

                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)

                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    content = delta.get("content")
                    if content:
                        yield content

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            yield f"\n\n[STREAM ERROR] {str(e)}"
