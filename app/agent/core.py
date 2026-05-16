import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from app.agent.tools.sql_tool import get_schema, generate_sql, run_query, interpret
from app.agent.tools.ml_tool import predict_demand
from sqlalchemy import text
from utils.db import get_engine

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
engine = get_engine()


def route_question(question):
    prompt = f"""You are an intelligent router. Decide which tool should answer the user's question.

Tool 1: sql_query — for querying historical data, e.g. "which category sells best", "past sales trends", "customer distribution"
Tool 2: demand_forecast — for predicting future sales, e.g. "how much should I stock next month", "future demand forecast", "inventory optimization"

User question: {question}

Reply with only the tool name: sql_query or demand_forecast. Nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def fetch_real_lag_features(category: str, target_year: int, target_month: int):
    """
    从数据库查真实历史销量，计算 lag_1、lag_2、rolling_mean_3。
    比"默认填500"更准确，也让这个 Agent 真正做到 chained tool use：
    预测问题 → 先查 SQL → 再跑 ML。
    """
    query = text("""
        SELECT
            DATE_TRUNC('month', o.order_purchase_timestamp::timestamp) AS month,
            COUNT(oi.order_item_id) AS sales_count
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE p.product_category_name = :category
          AND o.order_purchase_timestamp IS NOT NULL
          AND DATE_TRUNC('month', o.order_purchase_timestamp::timestamp)
              < DATE_TRUNC('month', MAKE_DATE(:year, :month, 1)::timestamp)
        GROUP BY month
        ORDER BY month DESC
        LIMIT 3
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"category": category, "year": target_year, "month": target_month})
        rows = result.fetchall()

    if len(rows) < 1:
        return None, f"数据库中找不到品类 '{category}' 的历史数据，请检查品类名称（需要葡萄牙语，如 cama_mesa_banho）"

    # rows 按时间倒序：rows[0] = 上个月, rows[1] = 上上个月, rows[2] = 3个月前
    sales = [int(r[1]) for r in rows]

    lag_1 = sales[0]
    lag_2 = sales[1] if len(sales) >= 2 else sales[0]
    rolling_mean_3 = round(sum(sales) / len(sales), 1)

    return {
        "lag_1": lag_1,
        "lag_2": lag_2,
        "rolling_mean_3": rolling_mean_3,
        "history_used": [{"month": str(r[0])[:7], "sales": int(r[1])} for r in rows]
    }, None


def extract_forecast_params(question: str):
    prompt = f"""User question: {question}

Extract the forecast parameters from this question and return JSON:
{{
    "category": "category name in Portuguese (e.g. cama_mesa_banho)",
    "month": month as integer,
    "year": year as integer
}}
Return only the JSON, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    params_str = response.choices[0].message.content.strip()
    params_str = params_str.replace("```json", "").replace("```", "").strip()
    return json.loads(params_str)


def handle_forecast(question: str):
    # Step 1: LLM 提取品类 + 时间
    params = extract_forecast_params(question)
    category = params['category']
    month = params['month']
    year = params['year']

    # Step 2: 自动查数据库获取真实 lag 特征（chained tool use）
    lag_features, error = fetch_real_lag_features(category, year, month)
    if error:
        return None, None, error

    # Step 3: XGBoost 预测
    prediction, feature_impact = predict_demand(
        category=category,
        month=month,
        year=year,
        lag_1=lag_features['lag_1'],
        lag_2=lag_features['lag_2'],
        rolling_mean_3=lag_features['rolling_mean_3'],
    )

    if prediction is None:
        return None, None, feature_impact  # feature_impact 此时是 error message

    # Step 4: LLM 把技术结果翻译成业务语言
    history_str = " | ".join([f"{h['month']}: {h['sales']}件" for h in lag_features['history_used']])
    top_factor = max(feature_impact, key=lambda k: abs(feature_impact[k]))

    raw_result = f"""
品类：{category}
预测月份：{year}年{month}月
预测销量：{prediction} 件

历史参考数据（来自数据库）：{history_str}
最重要的预测因子：{top_factor}（重要性：{feature_impact[top_factor]:.3f}）
"""

    prompt = f"""You are a data analyst. Explain the following forecast result to an e-commerce seller in plain English.
    Do not use technical jargon. Give a practical inventory recommendation in 2-3 sentences.

    {raw_result}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    business_insight = response.choices[0].message.content.strip()

    return prediction, feature_impact, business_insight


def ask(question: str):
    result = {
        "question": question,
        "tool": None,
        "sql": None,
        "columns": None,
        "rows": None,
        "prediction": None,
        "feature_impact": None,
        "insight": None,
        "error": None,
    }

    tool = route_question(question)
    result["tool"] = tool

    if tool == "demand_forecast":
        prediction, feature_impact, insight = handle_forecast(question)
        result["prediction"] = prediction
        result["feature_impact"] = feature_impact
        result["insight"] = insight
        if prediction is None:
            result["error"] = insight
    else:
        schema = get_schema()
        sql = generate_sql(question, schema)
        columns, rows = run_query(sql)
        insight = interpret(question, columns, rows)
        result["sql"] = sql
        result["columns"] = columns
        result["rows"] = [list(row) for row in rows[:10]]
        result["insight"] = insight

    return result