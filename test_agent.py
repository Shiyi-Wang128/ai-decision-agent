import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from utils.db import get_engine
from app.agent.tools.ml_tool import predict_demand

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
engine = get_engine()


def get_schema():
    schema = {}
    with engine.connect() as conn:
        tables = ['customers', 'orders', 'order_items', 'products', 'sellers']
        for table in tables:
            result = conn.execute(text(
                f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table}'"
            ))
            schema[table] = [(row[0], row[1]) for row in result]
    return schema


def route_question(question):
    """让LLM判断这个问题该用哪个工具"""
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


def generate_sql(question, schema):
    schema_str = ""
    for table, columns in schema.items():
        cols = ", ".join([f"{col} ({dtype})" for col, dtype in columns])
        schema_str += f"Table {table}: {cols}\n"

    prompt = f"""You are a SQL expert. Given the following database schema:
{schema_str}
Write a PostgreSQL query to answer this question: {question}
Return ONLY the SQL query, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def run_query(sql):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = result.fetchall()
        columns = result.keys()
        return list(columns), rows


def interpret(question, columns, rows):
    data_str = ", ".join(columns) + "\n"
    for row in rows[:10]:
        data_str += str(row) + "\n"

    prompt = f"""你是一个数据分析师。用户问了这个问题：{question}

查询结果如下：
{data_str}

请用2-3句话总结这个结果，给出业务洞察和建议。用中文回答。"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def handle_forecast(question):
    """处理预测类问题"""
    # 让LLM从问题里提取参数
    prompt = f"""用户问题：{question}

请从这个问题里提取预测参数，返回JSON格式：
{{
    "category": "品类名（用葡萄牙语，比如cama_mesa_banho）",
    "month": 月份数字,
    "year": 年份数字,
    "lag_1": 上个月预估销量（如果没提到就用500），
    "lag_2": 上上个月预估销量（如果没提到就用500），
    "rolling_mean_3": 近三个月平均销量（如果没提到就用500）
}}
只返回JSON，不要其他内容。"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    import json
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
        return f"无法预测：{shap}"

    # 找出影响最大的特征
    top_factor = max(shap, key=lambda k: abs(shap[k]))

    result = f"""
预测结果：{params['category']} 在 {params['year']}年{params['month']}月 预计销量为 {prediction} 件

SHAP解释（各特征对预测的影响）：
{chr(10).join([f"  {k}: {v:+.1f}" for k, v in shap.items()])}

主要影响因素：{top_factor}（影响值：{shap[top_factor]:+.1f}）
"""
    return result


def ask(question):
    print(f"\n问题: {question}")

    # 路由决策
    tool = route_question(question)
    print(f"使用工具: {tool}")

    if tool == "demand_forecast":
        result = handle_forecast(question)
        print(result)
        # 用自然语言解释预测结果
        prompt = f"""你是一个数据分析师，用简单易懂的中文向业务人员解释以下预测结果，不要提SHAP这个词，把技术术语翻译成业务语言，2-3句话：

    {result}"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        print("业务解读：")
        print(response.choices[0].message.content.strip())
    else:
        schema = get_schema()
        sql = generate_sql(question, schema)
        print(f"生成的SQL:\n{sql}")
        columns, rows = run_query(sql)
        print(f"\n原始数据:")
        print(", ".join(columns))
        for row in rows[:10]:
            print(row)
        insight = interpret(question, columns, rows)
        print(f"\n分析结论:\n{insight}")


if __name__ == '__main__':
    while True:
        question = input("\n请输入你的问题（输入q退出）：")
        if question.lower() == 'q':
            break
        ask(question)