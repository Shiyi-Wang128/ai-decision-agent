import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from app.agent.tools.sql_tool import get_schema, generate_sql, run_query, interpret
from app.agent.tools.ml_tool import predict_demand

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def route_question(question):
    prompt = f"""你是一个智能路由器，判断用户的问题应该用哪个工具回答。

工具一：sql_query - 用于查询历史数据，比如"哪个品类卖得最好"、"过去销售趋势"、"客户分布"
工具二：demand_forecast - 用于预测未来销量，比如"下个月应该备多少货"、"未来需求预测"、"库存优化"

用户问题：{question}

只回答工具名称，sql_query 或者 demand_forecast，不要其他任何内容。"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def handle_forecast(question):
    prompt = f"""用户问题：{question}

请从这个问题里提取预测参数，返回JSON格式：
{{
    "category": "品类名（用葡萄牙语，比如cama_mesa_banho）",
    "month": 月份数字,
    "year": 年份数字,
    "lag_1": 上个月预估销量（如果没提到就用500）,
    "lag_2": 上上个月预估销量（如果没提到就用500）,
    "rolling_mean_3": 近三个月平均销量（如果没提到就用500）
}}
只返回JSON，不要其他内容。"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    params_str = response.choices[0].message.content.strip()
    params_str = params_str.replace("```json", "").replace("```", "").strip()
    params = json.loads(params_str)

    prediction, shap = predict_demand(
        category=params['category'],
        month=params['month'],
        year=params['year'],
        lag_1=params['lag_1'],
        lag_2=params['lag_2'],
        rolling_mean_3=params['rolling_mean_3']
    )

    if prediction is None:
        return None, None, f"无法预测：{shap}"

    raw_result = f"""
预测结果：{params['category']} 在 {params['year']}年{params['month']}月 预计销量为 {prediction} 件

SHAP解释：
{chr(10).join([f"  {k}: {v:+.1f}" for k, v in shap.items()])}
主要影响因素：{max(shap, key=lambda k: abs(shap[k]))}
"""

    prompt2 = f"""你是一个数据分析师，用简单易懂的中文向业务人员解释以下预测结果，不要提SHAP这个词，把技术术语翻译成业务语言，2-3句话：

{raw_result}"""

    response2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt2}]
    )
    business_insight = response2.choices[0].message.content.strip()

    return prediction, shap, business_insight

def ask(question):
    result = {
        "question": question,
        "tool": None,
        "sql": None,
        "columns": None,
        "rows": None,
        "prediction": None,
        "shap": None,
        "insight": None
    }

    tool = route_question(question)
    result["tool"] = tool

    if tool == "demand_forecast":
        prediction, shap, insight = handle_forecast(question)
        result["prediction"] = prediction
        result["shap"] = shap
        result["insight"] = insight
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