import streamlit as st
import sys
import os

# 把项目根目录加到路径里
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from app.agent.core import ask

st.set_page_config(page_title="AI 决策助手", page_icon="🤖", layout="wide")

st.title("🤖 AI 决策助手")
st.caption("基于真实电商数据，用自然语言提问，获取数据分析和销量预测")

# 示例问题
st.markdown("**示例问题：**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("哪些品类销量最高？"):
        st.session_state.question = "哪些品类销量最高？"
with col2:
    if st.button("哪个城市客户最多？"):
        st.session_state.question = "哪个城市客户最多？"
with col3:
    if st.button("预测cama_mesa_banho下个月销量"):
        st.session_state.question = "预测cama_mesa_banho这个品类2024年1月的销量"

# 输入框
question = st.text_input(
    "输入你的问题：",
    value=st.session_state.get("question", ""),
    placeholder="例如：哪些品类的退款率最高？"
)

if st.button("分析", type="primary") and question:
    with st.spinner("Agent 正在思考..."):
        result = ask(question)

    st.divider()

    # 显示使用了哪个工具
    tool_label = "📊 历史数据查询" if result["tool"] == "sql_query" else "🔮 销量预测"
    st.markdown(f"**使用工具：** {tool_label}")

    if result["tool"] == "sql_query":
        # 显示SQL
        if result["sql"]:
            with st.expander("查看生成的 SQL"):
                st.code(result["sql"], language="sql")

        # 显示数据表格
        if result["rows"]:
            import pandas as pd
            df = pd.DataFrame(result["rows"], columns=result["columns"])
            st.subheader("查询结果")
            st.dataframe(df, use_container_width=True)

    else:
        # 显示预测结果
        if result["prediction"]:
            st.metric(label="预测销量", value=f"{result['prediction']} 件")

        # 显示SHAP
        if result["shap"]:
            with st.expander("查看特征影响分析"):
                import pandas as pd
                shap_df = pd.DataFrame([
                    {"特征": k, "影响值": v}
                    for k, v in result["shap"].items()
                ]).sort_values("影响值", ascending=False)
                st.dataframe(shap_df, use_container_width=True)

    # 显示业务结论
    if result["insight"]:
        st.subheader("💡 业务结论")
        st.info(result["insight"])