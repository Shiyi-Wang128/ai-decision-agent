"""
Text-to-SQL Evaluation Suite
测试 Agent 的 SQL 生成能力，分两个层次：
  - Execution Accuracy: 生成的 SQL 能否成功执行（不报错）
  - Result Accuracy:    执行结果是否符合预期（有已知答案的题目）
"""

import os
import time
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from app.agent.tools.sql_tool import get_schema, generate_sql, run_query

load_dotenv()

# ── 测试集 ──────────────────────────────────────────────────────────────────
# check_fn: 接收 (columns, rows)，返回 (passed: bool, detail: str)
# check_fn=None 表示只测 execution accuracy，不验证结果

TEST_CASES = [
    # ── 品类分析 ────────────────────────────────────────────────────────────
    {
        "id": "Q01",
        "question": "哪个品类的订单数量最多？",
        "check_fn": lambda cols, rows: (
            len(rows) > 0,
            f"应返回至少1行，实际为空") if rows else (False, "结果为空"),
    },
    {
        "id": "Q02",
        "question": "列出销量前5的产品品类及其订单数量",
        "check_fn": lambda cols, rows: (
            len(rows) >= 5,
            f"应返回至少5行，实际返回 {len(rows)} 行"
        ),
    },
    {
        "id": "Q03",
        "question": "每个品类的平均订单金额是多少？按平均金额降序排列",
        "check_fn": lambda cols, rows: (
            len(rows) > 0 and len(cols) == 2,
            f"应返回2列（品类、均价），实际列数：{len(cols)}"
        ),
    },
    {
        "id": "Q04",
        "question": "哪些品类的订单量超过1000？",
        "check_fn": None,
    },
    {
        "id": "Q05",
        "question": "health_beauty 品类一共有多少个订单？",
        "check_fn": lambda cols, rows: (
            rows[0][0] > 5000,
            f"health_beauty 订单数应大于5000，实际：{rows[0][0]}"
        ) if rows else (False, "结果为空"),
    },

    # ── 地理分析 ────────────────────────────────────────────────────────────
    {
        "id": "Q06",
        "question": "哪个城市的客户下单数量最多？",
        "check_fn": lambda cols, rows: (
            "sao paulo" in str(rows[0][0]).lower(),
            f"第一名应为 São Paulo，实际得到 {rows[0][0]}"
        ) if rows else (False, "结果为空"),
    },
    {
        "id": "Q07",
        "question": "每个州的客户数量是多少？按数量降序排列前10",
        "check_fn": lambda cols, rows: (
            len(rows) >= 5,
            f"应返回至少5行，实际 {len(rows)} 行"
        ),
    },
    {
        "id": "Q08",
        "question": "卖家主要分布在哪些州？",
        "check_fn": None,
    },

    # ── 时间趋势 ────────────────────────────────────────────────────────────
    {
        "id": "Q09",
        "question": "每个月的订单量是多少？按时间排序",
        "check_fn": lambda cols, rows: (
            len(rows) > 12,
            f"数据跨越2年应超过12个月，实际 {len(rows)} 行"
        ),
    },
    {
        "id": "Q10",
        "question": "2017年11月的订单量比10月多多少？",
        "check_fn": None,  # 验证 Black Friday 效应存在即可，不做精确断言
    },
    {
        "id": "Q11",
        "question": "哪一年的总销售额最高？",
        "check_fn": lambda cols, rows: (
            str(rows[0][0]) in ["2018", "2017"],
            f"最高年份应为2017或2018，实际：{rows[0][0]}"
        ) if rows else (False, "结果为空"),
    },
    {
        "id": "Q12",
        "question": "每周哪一天的订单量最多？",
        "check_fn": None,
    },

    # ── 支付方式 ────────────────────────────────────────────────────────────
    {
        "id": "Q13",
        "question": "最常用的支付方式是什么？",
        "check_fn": lambda cols, rows: (
            "credit_card" in str(rows[0][0]).lower(),
            f"最常用支付方式应为 credit_card，实际：{rows[0][0]}"
        ) if rows else (False, "结果为空"),
    },
    {
        "id": "Q14",
        "question": "信用卡和boleto支付的平均订单金额分别是多少？",
        "check_fn": lambda cols, rows: (
            len(rows) >= 2,
            f"应返回至少2行，实际 {len(rows)} 行"
        ),
    },

    # ── 卖家分析 ────────────────────────────────────────────────────────────
    {
        "id": "Q15",
        "question": "销售额最高的前5个卖家是谁？",
        "check_fn": lambda cols, rows: (
            len(rows) >= 5,
            f"应返回5行，实际 {len(rows)} 行"
        ),
    },
    {
        "id": "Q16",
        "question": "平均每个卖家销售了多少个订单？",
        "check_fn": lambda cols, rows: (
            rows[0][0] > 0,
            f"均值应大于0，实际：{rows[0][0]}"
        ) if rows else (False, "结果为空"),
    },

    # ── 评分分析 ────────────────────────────────────────────────────────────
    {
        "id": "Q17",
        "question": "平均评分最低的品类是哪些？列出后5名",
        "check_fn": lambda cols, rows: (
            len(rows) >= 5,
            f"应返回5行，实际 {len(rows)} 行"
        ),
    },
    {
        "id": "Q18",
        "question": "评分为1分的订单占总订单的比例是多少？",
        "check_fn": None,
    },

    # ── 多表JOIN ────────────────────────────────────────────────────────────
    {
        "id": "Q19",
        "question": "每个品类的平均配送天数是多少？",
        "check_fn": lambda cols, rows: (
            len(rows) > 10,
            f"应返回多个品类，实际 {len(rows)} 行"
        ),
    },
    {
        "id": "Q20",
        "question": "哪些品类的货运费用最高？列出前5名及平均运费",
        "check_fn": lambda cols, rows: (
            len(rows) >= 5,
            f"应返回5行，实际 {len(rows)} 行"
        ),
    },
]


# ── 执行评估 ─────────────────────────────────────────────────────────────────

def run_evaluation():
    schema = get_schema()
    results = []

    print("=" * 65)
    print("Text-to-SQL Evaluation Suite")
    print(f"Total questions: {len(TEST_CASES)}")
    print("=" * 65)

    for tc in TEST_CASES:
        qid = tc["id"]
        question = tc["question"]
        check_fn = tc["check_fn"]

        result = {
            "id": qid,
            "question": question,
            "sql": None,
            "execution_pass": False,
            "result_pass": None,   # None = not checked
            "detail": "",
            "error": None,
        }

        try:
            sql = generate_sql(question, schema)
            result["sql"] = sql

            columns, rows = run_query(sql)
            result["execution_pass"] = True

            if check_fn is not None:
                passed, detail = check_fn(columns, rows)
                result["result_pass"] = passed
                result["detail"] = detail
            else:
                result["detail"] = f"返回 {len(rows)} 行，{len(columns)} 列"

        except SQLAlchemyError as e:
            result["error"] = f"SQL Error: {str(e)[:120]}"
        except Exception as e:
            result["error"] = f"Error: {str(e)[:120]}"

        # 打印单条结果
        exec_icon = "✅" if result["execution_pass"] else "❌"
        if result["result_pass"] is True:
            result_icon = "✅"
        elif result["result_pass"] is False:
            result_icon = "⚠️"
        else:
            result_icon = "—"

        print(f"\n{qid}: {question}")
        print(f"  Execution: {exec_icon}  |  Result Check: {result_icon}")
        if result["detail"]:
            print(f"  {result['detail']}")
        if result["error"]:
            print(f"  ERROR: {result['error']}")
        if result["sql"]:
            # 只打印 SQL 前两行，避免输出过长
            sql_preview = result["sql"].split("\n")[0][:100]
            print(f"  SQL: {sql_preview}...")

        results.append(result)
        time.sleep(0.5)  # 避免 API rate limit

    # ── 汇总报告 ──────────────────────────────────────────────────────────
    total = len(results)
    exec_pass = sum(1 for r in results if r["execution_pass"])
    checked = [r for r in results if r["result_pass"] is not None]
    result_pass = sum(1 for r in checked if r["result_pass"])

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Execution Accuracy: {exec_pass}/{total} = {exec_pass/total*100:.1f}%")
    if checked:
        print(f"Result Accuracy:    {result_pass}/{len(checked)} = {result_pass/len(checked)*100:.1f}%  (on {len(checked)} verified questions)")

    # 失败的题目
    failed = [r for r in results if not r["execution_pass"]]
    if failed:
        print(f"\nFailed questions ({len(failed)}):")
        for r in failed:
            print(f"  {r['id']}: {r['question']}")
            print(f"    {r['error']}")

    wrong = [r for r in checked if not r["result_pass"]]
    if wrong:
        print(f"\nWrong results ({len(wrong)}):")
        for r in wrong:
            print(f"  {r['id']}: {r['question']}")
            print(f"    {r['detail']}")

    print("=" * 65)
    return results


if __name__ == "__main__":
    run_evaluation()