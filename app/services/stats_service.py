import sqlite3

from core.database import DB_PATH


def build_rag_stats(from_date: str = None, to_date: str = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    conditions = []
    params = []

    if from_date and to_date:
        conditions.append("date(created_at) BETWEEN ? AND ?")
        params.extend([from_date, to_date])

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    cursor.execute(f"SELECT COUNT(*) FROM rag_metrics {where_clause}", params)
    total_requests = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT COUNT(*) FROM rag_metrics
        {where_clause}
        {"AND" if where_clause else "WHERE"} rejected_reason IS NULL
        """,
        params,
    )
    success_count = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT rejected_reason, COUNT(*)
        FROM rag_metrics
        {where_clause}
        {"AND" if where_clause else "WHERE"} rejected_reason IS NOT NULL
        GROUP BY rejected_reason
        """,
        params,
    )
    rejection_types = dict(cursor.fetchall())

    rejection_count = sum(v for k, v in rejection_types.items() if k is not None)

    rejection_rate = (rejection_count / total_requests if total_requests else 0)

    cursor.execute(f"SELECT AVG(confidence) FROM rag_metrics {where_clause}", params)
    avg_confidence = cursor.fetchone()[0] or 0

    cursor.execute(f"SELECT AVG(total_time) FROM rag_metrics {where_clause}", params)
    avg_total_time = cursor.fetchone()[0] or 0

    cursor.execute(f"SELECT AVG(llm_time) FROM rag_metrics {where_clause}", params)
    avg_llm_time = cursor.fetchone()[0] or 0

    cursor.execute(f"SELECT AVG(context_chars) FROM rag_metrics {where_clause}", params)
    avg_context = cursor.fetchone()[0] or 0

    scatter_conditions = conditions.copy()
    scatter_conditions.append("rejected_reason IS NULL")

    scatter_where = ""
    if scatter_conditions:
        scatter_where = "WHERE " + " AND ".join(scatter_conditions)

    cursor.execute(
        f"""
        SELECT context_chars, llm_time
        FROM rag_metrics
        {scatter_where}
        """,
        params,
    )

    scatter_data = [
        {"x": row[0], "y": row[1]}
        for row in cursor.fetchall()
        if row[0] and row[1]
    ]

    confidence_buckets = {
        "0-0.4": 0,
        "0.4-0.5": 0,
        "0.5-0.6": 0,
        "0.6-0.7": 0,
        "0.7-0.8": 0,
        "0.8-1.0": 0,
    }

    cursor.execute(f"SELECT confidence FROM rag_metrics {where_clause}", params)

    for (conf,) in cursor.fetchall():
        if conf is None:
            continue
        if conf < 0.4:
            confidence_buckets["0-0.4"] += 1
        elif conf < 0.5:
            confidence_buckets["0.4-0.5"] += 1
        elif conf < 0.6:
            confidence_buckets["0.5-0.6"] += 1
        elif conf < 0.7:
            confidence_buckets["0.6-0.7"] += 1
        elif conf < 0.8:
            confidence_buckets["0.7-0.8"] += 1
        else:
            confidence_buckets["0.8-1.0"] += 1

    latency_buckets = {
        "0-5": 0,
        "5-10": 0,
        "10-20": 0,
        "20-30": 0,
        "30-60": 0,
        "60+": 0,
    }

    cursor.execute(f"SELECT total_time FROM rag_metrics {where_clause}", params)

    for (t,) in cursor.fetchall():
        if t < 5:
            latency_buckets["0-5"] += 1
        elif t < 10:
            latency_buckets["5-10"] += 1
        elif t < 20:
            latency_buckets["10-20"] += 1
        elif t < 30:
            latency_buckets["20-30"] += 1
        elif t < 60:
            latency_buckets["30-60"] += 1
        else:
            latency_buckets["60+"] += 1

    score_buckets = {
        "0-0.3": 0,
        "0.3-0.5": 0,
        "0.5-0.7": 0,
        "0.7-1.0": 0,
    }

    cursor.execute(f"SELECT max_score FROM rag_metrics {where_clause}", params)

    for (score,) in cursor.fetchall():
        if score is None:
            continue
        if score < 0.3:
            score_buckets["0-0.3"] += 1
        elif score < 0.5:
            score_buckets["0.3-0.5"] += 1
        elif score < 0.7:
            score_buckets["0.5-0.7"] += 1
        else:
            score_buckets["0.7-1.0"] += 1

    health = []

    if avg_confidence > 0.7:
        health.append("Retrieval работает стабильно (confidence высокий).")
    elif avg_confidence > 0.6:
        health.append("Retrieval удовлетворительный, возможна точечная оптимизация.")
    else:
        health.append("Confidence низкий — требуется улучшение retrieval.")

    if avg_total_time > 25:
        health.append("Время ответа высокое — узкое место LLM.")
    elif avg_total_time > 15:
        health.append("Время ответа среднее.")
    else:
        health.append("Скорость ответа хорошая.")

    if rejection_rate > 0.25:
        health.append("Процент отказов высокий — возможно threshold слишком строгий.")
    elif rejection_rate > 0.15:
        health.append("Процент отказов в пределах нормы.")
    else:
        health.append("Система отвечает почти на все запросы.")

    trend_conditions = []
    trend_params = []

    if from_date and to_date:
        trend_conditions.append("date(created_at) BETWEEN ? AND ?")
        trend_params.extend([from_date, to_date])

    trend_conditions.append("confidence IS NOT NULL")

    trend_where = ""
    if trend_conditions:
        trend_where = "WHERE " + " AND ".join(trend_conditions)

    cursor.execute(
        f"""
    SELECT date(created_at), AVG(confidence)
    FROM rag_metrics
    {trend_where}
    GROUP BY date(created_at)
    ORDER BY date(created_at)
    """,
        trend_params,
    )

    rows = cursor.fetchall()

    confidence_trend = [
        {"date": row[0], "value": round(row[1], 3)}
        for row in rows
        if row[1] is not None
    ]

    cursor.execute(
        f"""
    SELECT topic_mode, COUNT(*)
    FROM rag_metrics
    {where_clause}
    GROUP BY topic_mode
    """,
        params,
    )

    topic_distribution = dict(cursor.fetchall())

    cursor.execute(
        f"""
    SELECT SUM(is_followup), COUNT(*)
    FROM rag_metrics
    {where_clause}
    """,
        params,
    )

    followups, total = cursor.fetchone()

    followup_rate = (followups / total) if total else 0

    cursor.execute(
        f"""
    SELECT AVG(retrieval_time), AVG(llm_time)
    FROM rag_metrics
    {where_clause}
    """,
        params,
    )

    avg_retrieval_time, avg_llm_time = cursor.fetchone()

    cursor.execute(
        f"""
    SELECT
        AVG(max_score),
        AVG(coverage),
        AVG(threshold)
    FROM rag_metrics
    {where_clause}
    """,
        params,
    )

    row = cursor.fetchone()

    avg_max_score = row[0] if row and row[0] else 0
    avg_coverage = row[1] if row and row[1] else 0
    avg_threshold = row[2] if row and row[2] else 0
    conn.close()

    return {
        "aggregates": {
            "total_requests": total_requests,
            "success_count": success_count,
            "rejection_count": rejection_count,
            "rejection_rate": rejection_rate,
            "avg_confidence": avg_confidence,
            "avg_total_time": avg_total_time,
            "avg_llm_time": avg_llm_time,
            "avg_context_chars": avg_context,
        },
        "rejection_types": rejection_types,
        "confidence_distribution": confidence_buckets,
        "latency_distribution": latency_buckets,
        "scatter_data": scatter_data,
        "confidence_trend": confidence_trend,
        "topic_distribution": topic_distribution,
        "followup_rate": followup_rate,
        "avg_retrieval_time": avg_retrieval_time,
        "avg_llm_time": avg_llm_time,
        "health_summary": health,
        "score_distribution": score_buckets,
        "advanced_metrics": {
            "avg_max_score": avg_max_score,
            "avg_coverage": avg_coverage,
            "avg_threshold": avg_threshold,
        },
    }
