import requests
import time
import uuid
import random
import json

BASE_URL = "http://localhost:8000/chat-sync"

# ======== ТЕСТОВЫЕ ВОПРОСЫ ========

QUESTIONS = [
    # Elastic
    "Что такое Elastic?",
    "Что означает ошибка 401?",
    "Что означает ошибка 503?",
    "Как проверить состояние кластера?",
    "Что означает статус green?",
    "Как перезапустить ноду Elastic?",

    # Monitoring
    "Что такое мониторинг?",
    "Какие параметры отслеживает мониторинг?",
    "Хранит ли мониторинг логи?",
    "Что делать при деградации сервиса?",

    # Payment
    "Какие события считаются критическими?",
    "Как рассчитывается error_rate?",
    "Что делать если error_rate больше 0.05?",
    "Что делать при CRITICAL событии?",

    # Power BI
    "Когда формируется ежемесячный отчёт?",
    "Что делать если отчёт не обновляется?",

    # DevOps
    "Что такое DevOps?",
    "Какие принципы DevOps?",

    # Resume
    "Где работал Сорокин Андрей?",
    "Есть ли опыт Jenkins?",

    # Noise
    "Как варить кофе?",
    "Что такое Kubernetes?"
]

TOTAL_REQUESTS = 50
FOLLOWUP_RATIO = 0.3

def send_request(question, session_id):
    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": session_id
    }

    start = time.time()

    response = requests.post(
        BASE_URL,
        headers=headers,
        json={"question": question},
        timeout=120   # важно!
    )

    duration = time.time() - start

    if response.status_code != 200:
        return {
            "confidence": 0,
            "duration": duration,
            "rejected": True
        }

    data = response.json()

    confidence = data.get("confidence", 0)
    rejected = data.get("rejected", False)

    return {
        "confidence": confidence,
        "duration": duration,
        "rejected": rejected
    }


def main():
    results = []

    current_session = str(uuid.uuid4())

    for i in range(TOTAL_REQUESTS):

        # 70% — новый чат
        if random.random() > FOLLOWUP_RATIO:
            current_session = str(uuid.uuid4())

        question = random.choice(QUESTIONS)

        print(f"[{i+1}/{TOTAL_REQUESTS}] → {question}")

        result = send_request(question, current_session)
        results.append(result)

    # ======= АНАЛИТИКА =======

    total = len(results)
    rejected = sum(1 for r in results if r["rejected"])
    avg_conf = sum(
        r["confidence"] for r in results
        if r["confidence"] is not None
    ) / total

    avg_time = sum(r["duration"] for r in results) / total

    print("\n====== LOAD TEST RESULT ======")
    print(f"Total requests: {total}")
    print(f"Rejection rate: {round(rejected/total*100,2)}%")
    print(f"Average confidence: {round(avg_conf,3)}")
    print(f"Average response time: {round(avg_time,2)} sec")


if __name__ == "__main__":
    main()