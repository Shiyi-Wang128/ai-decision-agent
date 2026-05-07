import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import text
from utils.db import get_engine

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
        columns = list(result.keys())
        return columns, rows

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