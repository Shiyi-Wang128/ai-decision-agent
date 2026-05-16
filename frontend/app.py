import streamlit as st
import sys
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from app.agent.core import ask

st.set_page_config(page_title="E-Commerce Decision Agent", page_icon="📦", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📦 E-Commerce Decision Agent")
st.caption("Built on the Olist Brazilian E-Commerce Dataset (100K+ orders, 71 categories) · Ask questions in plain language — the agent picks the right tool automatically")

st.divider()

# ── Example questions ─────────────────────────────────────────────────────────
st.markdown("**Quick questions:**")
col1, col2, col3, col4 = st.columns(4)

EXAMPLES = [
    ("📊 Top Categories", "List the top 10 product categories by order volume"),
    ("🌎 Regional Sales", "Which state has the most customers? Show top 5"),
    ("📈 Sales Trend", "What was the monthly order volume in 2017?"),
    ("🔮 Demand Forecast", "Forecast demand for bed & bath products (cama_mesa_banho) in September 2018"),
]

for col, (label, q) in zip([col1, col2, col3, col4], EXAMPLES):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.question = q

# ── Input ─────────────────────────────────────────────────────────────────────
question = st.text_input(
    "Or type your own question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. Which product categories have the lowest average review score?",
)

run = st.button("Analyze", type="primary")

# ── Run & display ─────────────────────────────────────────────────────────────
if run and question:
    with st.spinner("Agent is thinking..."):
        result = ask(question)

    st.divider()

    if result["tool"] == "sql_query":
        st.markdown("### 📊 Historical Data Query")
    else:
        st.markdown("### 🔮 Demand Forecast")

    if result.get("error"):
        st.error(f"Something went wrong: {result['error']}")
        st.stop()

    # ── SQL results ───────────────────────────────────────────────────────
    if result["tool"] == "sql_query":

        if result.get("sql"):
            with st.expander("🔍 View generated SQL"):
                st.code(result["sql"], language="sql")

        if result.get("rows") and result.get("columns"):
            df = pd.DataFrame(result["rows"], columns=result["columns"])

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.markdown("**Query Results**")
                st.dataframe(df, use_container_width=True)

            with col_right:
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                text_cols = df.select_dtypes(exclude="number").columns.tolist()

                if numeric_cols and text_cols and len(df) <= 30:
                    st.markdown("**Visualization**")
                    chart_df = df.set_index(text_cols[0])[numeric_cols[0]]
                    st.bar_chart(chart_df)

    # ── Forecast results ──────────────────────────────────────────────────
    else:
        if result.get("prediction") is not None:

            col_metric, _ = st.columns([1, 2])
            with col_metric:
                st.metric(
                    label="Predicted Sales Volume",
                    value=f"{result['prediction']} units",
                )

            feature_data = result.get("feature_impact") or result.get("shap")
            if feature_data:
                with st.expander("📌 View feature importance"):
                    fi_df = (
                        pd.DataFrame([
                            {"Feature": k, "Importance": round(v, 3)}
                            for k, v in feature_data.items()
                        ])
                        .sort_values("Importance", ascending=False)
                        .reset_index(drop=True)
                    )
                    col_tbl, col_chart = st.columns([1, 1])
                    with col_tbl:
                        st.dataframe(fi_df, use_container_width=True)
                    with col_chart:
                        st.bar_chart(fi_df.set_index("Feature")["Importance"])

    # ── Business insight ──────────────────────────────────────────────────
    if result.get("insight"):
        st.markdown("---")
        st.markdown("### 💡 Business Insight")
        st.info(result["insight"])