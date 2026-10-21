"""Seaborn/matplotlib charts. Each function returns a matplotlib Figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def countplot(df, col, top=12):
    """Labelled count bars, same style as sns.countplot + ax.bar_label."""
    order = df[col].value_counts().index[:top]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=col, data=df, order=order, ax=ax)
    for bars in ax.containers:
        ax.bar_label(bars)
    ax.set_title(f"Count of {col}")
    if max(len(str(v)) for v in order) > 6 or len(order) > 6:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def distribution(df, col):
    """Histogram (with density curve) on top, boxplot below."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    sns.histplot(df[col], kde=True, ax=a1)
    a1.set_title(f"Distribution of {col}")
    a1.set_xlabel("")
    sns.boxplot(x=df[col], ax=a2)
    fig.tight_layout()
    return fig


def box_by_group(df, group, metric, top=10):
    order = df[group].value_counts().index[:top]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=group, y=metric, data=df, order=order, ax=ax)
    ax.set_title(f"{metric} by {group}")
    if len(order) > 6:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def heatmap(df, cols):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig


def monthly_trend(df, date_col, metric):
    """metric may be 'record count' to plot rows per month."""
    g = df.dropna(subset=[date_col]).set_index(date_col)
    if metric == "record count":
        ts, title = g.iloc[:, 0].resample("MS").size(), "Records per month"
    else:
        ts, title = g[metric].resample("MS").sum(), f"Monthly total of {metric}"
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=ts.index, y=ts.values, marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(date_col)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig
