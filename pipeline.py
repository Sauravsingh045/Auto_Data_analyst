"""Core pipeline: load -> clean -> analyze -> summarize. No UI code here."""
import os
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?"}


# ---------------------------------------------------------------- loading
def load_sheets(file) -> dict:
    """Return {sheet_name: DataFrame}. Works for .xlsx/.xls/.csv."""
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        last_err = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                file.seek(0)
                return {"data": pd.read_csv(file, encoding=enc, sep=None, engine="python")}
            except UnicodeDecodeError as err:
                last_err = err
        raise last_err
    return pd.read_excel(file, sheet_name=None)


# --------------------------------------------------------------- cleaning
def _unique(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def clean(df: pd.DataFrame, max_missing: float = 0.6):
    """Return (clean_df, log). Every change is recorded in log."""
    log = []
    df = df.copy()
    start_shape = df.shape

    # 1. column names
    cols = [re.sub(r"\W+", "_", str(c).strip()).strip("_") or f"col_{i}"
            for i, c in enumerate(df.columns)]
    df.columns = _unique(cols)

    # 2. empty rows / columns
    before = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape != before:
        log.append(f"Removed {before[0]-df.shape[0]} empty rows and "
                   f"{before[1]-df.shape[1]} empty columns.")

    # 3. whitespace + fake-null tokens in text columns
    for c in df.select_dtypes(include="object").columns:
        s = df[c].astype("string").str.strip()
        df[c] = s.mask(s.str.lower().isin(NULL_TOKENS))
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)

    # 4. text that is really numbers (e.g. "$1,200", "45%")
    for c in df.select_dtypes(include="object").columns:
        raw = df[c].dropna().astype(str)
        if raw.empty:
            continue
        stripped = raw.str.replace(r"[,\s$€£₹%]", "", regex=True)
        nums = pd.to_numeric(stripped, errors="coerce")
        if nums.notna().mean() >= 0.9:
            full = df[c].astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True)
            df[c] = pd.to_numeric(full, errors="coerce")
            log.append(f"Converted '{c}' from text to numeric.")

    # 5. text that is really dates
    for c in df.select_dtypes(include="object").columns:
        sample = df[c].dropna().astype(str)
        if sample.empty or sample.str.len().median() < 6:
            continue
        best, best_ok = None, 0.0
        for dayfirst in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst)
            if parsed.notna().mean() > best_ok:
                best, best_ok = dayfirst, parsed.notna().mean()
        if best_ok >= 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=best)
            log.append(f"Converted '{c}' to datetime.")

    # 6. duplicates
    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates()
        log.append(f"Dropped {dups} duplicate rows.")

    # 7. missing values
    for c in list(df.columns):
        n_miss = int(df[c].isna().sum())
        miss = n_miss / max(len(df), 1)
        if n_miss == 0:
            continue
        if miss > max_missing:
            df = df.drop(columns=c)
            log.append(f"Dropped '{c}' ({miss:.0%} missing).")
        elif pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
            log.append(f"Filled {n_miss} missing values in '{c}' with median.")
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            log.append(f"'{c}' has {n_miss} missing dates (left as is).")
        else:
            df[c] = df[c].fillna("Unknown")
            log.append(f"Filled {n_miss} missing values in '{c}' with 'Unknown'.")

    df = df.reset_index(drop=True)
    log.insert(0, f"Shape: {start_shape} -> {df.shape}")
    return df, log


# --------------------------------------------------------------- analysis
ID_NAME = re.compile(r"(^|_)(id|uuid|guid|index|idx|key)$|^(id|index)(_|$)", re.I)
METRIC_NAME = re.compile(r"sales|amount|revenue|purchase|price|total|profit|cost|spend|income|quantity|qty", re.I)


def classify_columns(df: pd.DataFrame) -> dict:
    """Split columns into: id, numeric, categorical, date, text, constant.
    ID/text/constant columns are kept in the data but excluded from stats and charts."""
    n = max(len(df), 1)
    kinds = {"id": [], "numeric": [], "categorical": [], "date": [], "text": [], "constant": []}
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        ratio = nun / n
        if pd.api.types.is_datetime64_any_dtype(s):
            kinds["date"].append(c)
        elif nun <= 1:
            kinds["constant"].append(c)
        elif pd.api.types.is_numeric_dtype(s):
            is_int = bool((s.dropna() % 1 == 0).all())
            if is_int and (ratio > 0.95 or (ID_NAME.search(c) and ratio > 0.5)):
                kinds["id"].append(c)
            elif is_int and nun <= 8:
                kinds["categorical"].append(c)      # flags, ratings, small codes
            else:
                kinds["numeric"].append(c)
        else:
            if ratio > 0.9 and ID_NAME.search(c):
                kinds["id"].append(c)
            elif nun <= 20:
                kinds["categorical"].append(c)
            else:
                kinds["text"].append(c)              # names, emails, free text
    return kinds


def analyze(df: pd.DataFrame) -> dict:
    """Simple EDA facts, driven by column types."""
    k = classify_columns(df)
    num, cat, dt = k["numeric"], k["categorical"], k["date"]
    facts = {"rows": len(df), "columns": df.shape[1], "column_types": k}

    if cat:
        facts["category_counts"] = {
            c: {str(v): {"count": int(n), "pct": round(n / len(df) * 100, 1)}
                for v, n in df[c].value_counts().head(5).items()}
            for c in cat}

    if num:
        facts["numeric_summary"] = (
            df[num].describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_dict("index"))
        out = {}
        for c in num:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            cnt = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
            if cnt:
                out[c] = cnt
        facts["outliers"] = out

    if len(num) >= 2:
        full = df[num].corr()
        pairs = sorted(((a, b, float(full.loc[a, b])) for a, b in combinations(num, 2)
                        if pd.notna(full.loc[a, b])), key=lambda t: abs(t[2]), reverse=True)
        facts["top_correlations"] = [{"a": a, "b": b, "r": round(r, 2)}
                                     for a, b, r in pairs[:5] if abs(r) >= 0.3]

    if cat and num:
        groups = []
        for c in cat[:3]:
            sizes = df[c].value_counts()
            valid = sizes[sizes >= max(5, 0.02 * len(df))].index   # ignore tiny groups
            for m in num[:3]:
                g = df[df[c].isin(valid)].groupby(c)[m].mean().sort_values(ascending=False)
                overall = abs(df[m].mean()) or 1
                if len(g) > 1 and (g.iloc[0] - g.iloc[-1]) / overall >= 0.10:   # skip trivial gaps
                    groups.append({"group": c, "metric": m,
                                   "best": str(g.index[0]), "best_val": round(float(g.iloc[0]), 2),
                                   "worst": str(g.index[-1]), "worst_val": round(float(g.iloc[-1]), 2)})
        facts["group_insights"] = groups

    if dt:
        d = dt[0]
        metric = next((c for c in num if METRIC_NAME.search(c)), None)   # None -> count records
        g = df.dropna(subset=[d]).set_index(d)
        ts = (g[metric].resample("MS").sum() if metric else g.iloc[:, 0].resample("MS").size())
        if len(ts) >= 4 and ts.iloc[0] != 0:
            ts = ts.iloc[:-1]                       # last month is usually partial
            facts["trend"] = {"date": d, "metric": metric or "record count",
                              "first": round(float(ts.iloc[0]), 2), "last": round(float(ts.iloc[-1]), 2),
                              "change_pct": round(float((ts.iloc[-1] / ts.iloc[0] - 1) * 100), 1),
                              "peak_month": str(ts.idxmax().date())}
    return facts


# ---------------------------------------------------------------- summary
def rule_based_summary(facts: dict, log: list) -> str:
    k = facts["column_types"]
    lines = [f"The dataset has {facts['rows']:,} rows and {facts['columns']} columns: "
             f"{len(k['numeric'])} numeric, {len(k['categorical'])} categorical, {len(k['date'])} date, "
             f"{len(k['id'])} ID and {len(k['text'])} free-text (ID/text columns are not analysed)."]
    for col, vals in list(facts.get("category_counts", {}).items())[:4]:
        top = list(vals.items())
        desc = ", ".join(f"{v} {d['pct']}% ({d['count']:,})" for v, d in top[:3])
        lines.append(f"- {col}: {desc}.")
    for c in facts.get("top_correlations", [])[:3]:
        kind = "positive" if c["r"] > 0 else "negative"
        lines.append(f"- '{c['a']}' and '{c['b']}' have a {kind} correlation (r={c['r']}).")
    for g in facts.get("group_insights", [])[:3]:
        lines.append(f"- Average {g['metric']} is highest for {g['group']}={g['best']} ({g['best_val']}) "
                     f"and lowest for {g['worst']} ({g['worst_val']}).")
    if "trend" in facts:
        t = facts["trend"]
        lines.append(f"- Monthly {t['metric']} changed {t['change_pct']}% from first to last full month; "
                     f"peak month was {t['peak_month']}.")
    if facts.get("outliers"):
        lines.append("- Possible outliers in: " + ", ".join(f"{a} ({b})" for a, b in facts["outliers"].items()) + ".")
    return "\n".join(lines)


def llm_summary(facts: dict, log: list, model: str = None) -> str:
    """Plain-English conclusion from Claude. Only aggregated stats are sent, never raw rows.
    Falls back to the rule-based summary on any failure."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return rule_based_summary(facts, log)
    try:
        import json
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "You are a data analyst. Below are computed statistics from a cleaned dataset. "
            "Write a concise summary: 1) what the data is about, 2) 3-5 key findings with numbers, "
            "3) anomalies or data-quality caveats, 4) 2-3 suggested next steps. "
            "Only use the numbers given; do not invent any. ID columns carry no meaning; ignore them.\n\n"
            f"CLEANING LOG:\n{json.dumps(log)}\n\nSTATS:\n{json.dumps(facts, default=str)}")
        resp = client.messages.create(model=model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                                      max_tokens=1000, messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text
    except Exception as e:
        return rule_based_summary(facts, log) + f"\n\n(AI summary unavailable: {e})"
