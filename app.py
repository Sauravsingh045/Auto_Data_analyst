from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
from pipeline import analyze, clean, llm_summary, load_sheets

st.set_page_config(page_title="Auto Data Analyst", layout="wide")
st.title("Auto Data Analyst")
st.caption("Upload a raw Excel/CSV file. Get cleaning, simple EDA and a written conclusion.")

file = st.file_uploader("Upload file", type=["xlsx", "xls", "csv"])
if not file:
    st.stop()

sheets = load_sheets(file)
sheet = st.selectbox("Sheet", list(sheets)) if len(sheets) > 1 else list(sheets)[0]
raw = sheets[sheet]

with st.spinner("Cleaning and analysing..."):
    df, log = clean(raw)
    facts = analyze(df)
    summary = llm_summary(facts, log)
kinds = facts["column_types"]


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def grid(items, fn, per_row=2):
    """Lay out one chart per item, per_row charts per row."""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, item in zip(cols, items[i:i + per_row]):
            with col:
                fn(item)


st.subheader("Conclusion")
st.markdown(summary)

t_over, t_cat, t_num, t_rel, t_clean, t_data = st.tabs(
    ["Overview", "Categories", "Numbers", "Relationships", "Cleaning log", "Data"])

with t_over:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells (before cleaning)", f"{int(raw.isna().sum().sum()):,}")
    st.write("**How each column was treated**")
    st.table(pd.DataFrame({"type": [k for k, v in kinds.items() for _ in v],
                           "column": [c for v in kinds.values() for c in v]}))
    if kinds["id"]:
        st.info(f"ID columns {kinds['id']} are identifiers, so they are excluded from stats and charts.")
    if kinds["numeric"]:
        st.write("**Summary statistics**")
        st.dataframe(df[kinds["numeric"]].describe().T.round(2), use_container_width=True)

with t_cat:
    if kinds["categorical"]:
        grid(kinds["categorical"], lambda c: show(charts.countplot(df, c)))
    else:
        st.write("No categorical columns found.")

with t_num:
    if kinds["numeric"]:
        grid(kinds["numeric"], lambda c: show(charts.distribution(df, c)))
    else:
        st.write("No numeric columns found.")

with t_rel:
    if len(kinds["numeric"]) >= 2:
        show(charts.heatmap(df, kinds["numeric"]))
    pairs = [(g, m) for g in kinds["categorical"][:2] for m in kinds["numeric"][:2]]
    grid(pairs, lambda p: show(charts.box_by_group(df, *p)))
    if "trend" in facts:
        show(charts.monthly_trend(df, facts["trend"]["date"], facts["trend"]["metric"]))
    if not pairs and len(kinds["numeric"]) < 2 and "trend" not in facts:
        st.write("Not enough column types for relationship charts.")

with t_clean:
    for line in log:
        st.write("•", line)

with t_data:
    st.dataframe(df, use_container_width=True)

buf = BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    df.to_excel(xw, sheet_name="cleaned", index=False)
    pd.DataFrame({"summary": summary.split("\n")}).to_excel(xw, sheet_name="summary", index=False)
    pd.DataFrame({"cleaning_log": log}).to_excel(xw, sheet_name="cleaning_log", index=False)
st.download_button("Download cleaned data + summary (.xlsx)", buf.getvalue(), file_name="analysis_output.xlsx")
