# One-way ANOVA, the route from BIT 5534 week 2 lecture 2.
#
# Run instructions:
# 1. Install the packages listed below.
# 2. Save this file anywhere.
# 3. Double-click it, then choose a .csv, .xlsx or .xls file.
#
# Installation, Windows:  python -m pip install pandas scipy statsmodels openpyxl
# Installation, macOS:    python3 -m pip install pandas scipy statsmodels openpyxl
#
# This is the fallback tool. It does the same job as the one you specified in
# class and it needs no extra package for Games-Howell, which it computes from
# the studentized range distribution in scipy. Use it if your own script will
# not run, so that you can still do the reading. Keep your own prompt in the
# thread either way, because the prompt is the assessed work.
#
# The file must have a header row. The first column is the continuous response
# and the second column is the categorical factor. Nothing is named in the
# script, so it runs on any file of that shape.

import math
import sys
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA = 0.05


def fmt_p(value):
    if not np.isfinite(value):
        return "not available"
    return "< 0.0001" if value < 0.0001 else f"{value:.4f}"


def load_data(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Choose a .csv, .xlsx or .xls file.")


def clean_data(raw):
    if raw.shape[1] < 2:
        raise ValueError("The selected file must have at least two columns.")
    response_name, factor_name = str(raw.columns[0]), str(raw.columns[1])
    response = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    factor = raw.iloc[:, 1].astype("string").str.strip()
    keep = response.notna() & factor.notna() & factor.ne("")
    clean = pd.DataFrame({"response": response[keep].astype(float).to_numpy(),
                          "factor": factor[keep].astype(str).to_numpy()})
    if clean.empty:
        raise ValueError("No usable rows remain after removing invalid data.")
    counts = {"response_name": response_name, "factor_name": factor_name,
              "original_rows": len(raw), "removed_rows": int(len(raw) - len(clean)),
              "usable_rows": len(clean)}
    return clean, counts


def group_summary(data):
    summary = (data.groupby("factor", sort=False)["response"]
               .agg(n="size", mean="mean", sd=lambda x: x.std(ddof=1)).reset_index())
    if len(summary) < 2:
        raise ValueError("At least two factor groups are required.")
    small = summary.loc[summary["n"] < 2, "factor"].tolist()
    if small:
        raise ValueError("Every factor group needs at least two observations. "
                         "Groups with fewer than two: " + ", ".join(map(str, small)))
    return summary


def welch_anova(groups):
    k = len(groups)
    n = np.array([len(x) for x in groups], dtype=float)
    means = np.array([x.mean() for x in groups], dtype=float)
    variances = np.array([x.var(ddof=1) for x in groups], dtype=float)
    if np.any(variances <= 0) or not np.all(np.isfinite(variances)):
        raise ValueError("Welch's ANOVA cannot be computed because at least one "
                         "group has zero or invalid variance.")
    weights = n / variances
    total = weights.sum()
    weighted_mean = np.sum(weights * means) / total
    correction = np.sum(((1.0 - weights / total) ** 2) / (n - 1.0))
    statistic = (np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)) / (
        1.0 + (2.0 * (k - 2.0) / (k ** 2 - 1.0)) * correction)
    df1 = float(k - 1)
    df2 = float((k ** 2 - 1.0) / (3.0 * correction))
    return float(statistic), df1, df2, float(stats.f.sf(statistic, df1, df2))


def games_howell(labels, groups):
    """Games-Howell adjusted p-values, one per pair.

    Each pair gets its own variance estimate and its own Welch-Satterthwaite
    degrees of freedom, which is why this holds when the groups have different
    spreads and why it has less power than Tukey when they do not.
    """
    k = len(labels)
    out = {}
    for i, j in combinations(range(k), 2):
        x, y = groups[i], groups[j]
        ni, nj = len(x), len(y)
        vi, vj = x.var(ddof=1), y.var(ddof=1)
        se = math.sqrt((vi / ni + vj / nj) / 2.0)
        diff = float(x.mean() - y.mean())
        if se == 0:
            p = 1.0 if diff == 0 else 0.0
        else:
            q = abs(diff) / se
            df = (vi / ni + vj / nj) ** 2 / (
                (vi / ni) ** 2 / (ni - 1) + (vj / nj) ** 2 / (nj - 1))
            p = float(stats.studentized_range.sf(q, k, df))
        out[frozenset((labels[i], labels[j]))] = (diff, p)
    return out


def tukey(labels, data):
    result = pairwise_tukeyhsd(endog=data["response"], groups=data["factor"], alpha=ALPHA)
    table = pd.DataFrame(result._results_table.data[1:],
                         columns=result._results_table.data[0])
    out = {}
    for _, row in table.iterrows():
        out[frozenset((str(row["group1"]), str(row["group2"])))] = (
            float(row["meandiff"]), float(row["p-adj"]))
    return out


def connecting_letters(labels, pairs, means):
    """Letters from the maximal cliques of the not-separated graph.

    Every pair is keyed by a frozenset, so a pair is found whatever order its two
    names arrive in. Keying by an ordered tuple is the bug that makes a letter
    report crash or come out wrong the moment the level names are not in
    alphabetical order, and it is invisible on data where they happen to be.
    """
    order = sorted(labels, key=lambda x: (-means[x], str(x)))
    adjacency = {v: set() for v in order}
    for key, (_diff, p) in pairs.items():
        if p >= ALPHA:
            a, b = tuple(key)
            adjacency[a].add(b)
            adjacency[b].add(a)

    cliques = []

    def bron_kerbosch(current, candidates, excluded):
        if not candidates and not excluded:
            cliques.append(frozenset(current))
            return
        pool = candidates | excluded
        pivot = max(pool, key=lambda v: len(adjacency[v] & candidates))
        for v in list(candidates - adjacency[pivot]):
            bron_kerbosch(current | {v}, candidates & adjacency[v], excluded & adjacency[v])
            candidates = candidates - {v}
            excluded = excluded | {v}

    bron_kerbosch(set(), set(order), set())
    cliques.sort(key=lambda c: (min(order.index(x) for x in c), -len(c)))
    alphabet = [chr(97 + i) for i in range(26)]
    assigned = {label: "" for label in order}
    for letter, clique in zip(alphabet, cliques):
        for label in clique:
            assigned[label] += letter
    return order, assigned


def analyze(path):
    data, counts = clean_data(load_data(path))
    summary = group_summary(data)
    labels = [str(x) for x in summary["factor"]]
    groups = [data.loc[data["factor"] == label, "response"].to_numpy() for label in labels]
    means = dict(zip(labels, summary["mean"].astype(float)))

    lines = ["ONE-WAY ANOVA REPORT", "=" * 78,
             f"Input file: {Path(path).name}",
             f"Response column: {counts['response_name']}",
             f"Factor column: {counts['factor_name']}",
             f"Significance level alpha: {ALPHA}", "",
             f"Rows read: {counts['original_rows']}   "
             f"removed: {counts['removed_rows']}   usable: {counts['usable_rows']}", "",
             "Group summaries:", "Group\tn\tMean\tSD"]
    for _, row in summary.iterrows():
        lines.append(f"{row['factor']}\t{int(row['n'])}\t{row['mean']:.4f}\t{row['sd']:.4f}")

    f_stat, f_p = stats.f_oneway(*groups)
    lines += ["", "Standard one-way ANOVA F test:",
              f"F({len(groups) - 1}, {len(data) - len(groups)}) = {float(f_stat):.4f}",
              f"p-value = {fmt_p(float(f_p))}"]
    if f_p >= ALPHA:
        lines += ["Decision: the F test is not significant at alpha = 0.05.",
                  "The route stops here. No multiple comparison was performed."]
        return "\n".join(lines) + "\n"

    lines += ["Decision: the F test is significant at alpha = 0.05.",
              "Proceeding to Levene's test.", ""]
    lev_stat, lev_p = stats.levene(*groups, center="median")
    lines += ["Levene's test for equality of variances:",
              f"W = {float(lev_stat):.4f}", f"p-value = {fmt_p(float(lev_p))}"]

    if lev_p >= ALPHA:
        route = "Ordinary one-way ANOVA, then Tukey HSD"
        lines += ["Decision: variances are treated as alike.",
                  f"Route selected: {route}", "", "Pairwise Tukey HSD comparisons:"]
        pairs = tukey(labels, data)
        method = "Tukey HSD"
    else:
        lines += ["Decision: variances are treated as unequal.",
                  "Running Welch's one-way ANOVA.", ""]
        w_stat, w_df1, w_df2, w_p = welch_anova(groups)
        lines += ["Welch's one-way ANOVA:",
                  f"F({w_df1:.0f}, {w_df2:.2f}) = {w_stat:.4f}",
                  f"p-value = {fmt_p(w_p)}"]
        if w_p >= ALPHA:
            lines += ["Decision: Welch's ANOVA is not significant at alpha = 0.05.",
                      "The route stops here. No multiple comparison was performed."]
            return "\n".join(lines) + "\n"
        route = "Welch's ANOVA, then Games-Howell"
        lines += ["Decision: Welch's ANOVA is significant at alpha = 0.05.",
                  f"Route selected: {route}", "", "Pairwise Games-Howell comparisons:"]
        pairs = games_howell(labels, groups)
        method = "Games-Howell"

    lines.append("Group 1\tGroup 2\tMean difference\tAdjusted p\tSeparated")
    for a, b in combinations(labels, 2):
        diff, p = pairs[frozenset((a, b))]
        if means[a] < means[b]:
            diff = -abs(diff)
        else:
            diff = abs(diff)
        lines.append(f"{a}\t{b}\t{diff:.4f}\t{fmt_p(p)}\t{'yes' if p < ALPHA else 'no'}")

    order, assigned = connecting_letters(labels, pairs, means)
    lines += ["", f"Connecting letters from {method}:",
              "Groups sharing a letter are not separated at alpha = 0.05.",
              "Group\tMean\tSD\tConnecting letters"]
    sds = dict(zip(summary["factor"].astype(str), summary["sd"].astype(float)))
    for label in order:
        lines.append(f"{label}\t{means[label]:.4f}\t{sds[label]:.4f}\t{assigned[label]}")
    return "\n".join(lines) + "\n"


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            title="Choose a CSV or Excel data file",
            filetypes=[("Supported data files", "*.csv *.xlsx *.xls"),
                       ("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"),
                       ("All files", "*.*")])
        if not selected:
            return
        report = analyze(selected)
        report_path = Path(selected).with_name(f"{Path(selected).stem}_anova_report.txt")
        report_path.write_text(report, encoding="utf-8")
        window = tk.Toplevel(root)
        window.title("One-way ANOVA results")
        window.geometry("1000x700")
        box = scrolledtext.ScrolledText(window, wrap=tk.NONE, font=("Courier New", 10))
        box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        box.insert(tk.END, report)
        box.configure(state=tk.DISABLED)
        tk.Button(window, text="Close", command=window.destroy).pack(pady=(0, 8))
        messagebox.showinfo("Report saved", f"The report was also saved to:\n{report_path}",
                            parent=window)
        window.wait_window()
    except Exception as exc:
        messagebox.showerror("The analysis could not be completed",
                             f"{exc}\n\nTechnical details:\n{traceback.format_exc(limit=3)}",
                             parent=root)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(analyze(sys.argv[1]))
    else:
        main()
