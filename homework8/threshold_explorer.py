# -*- coding: utf-8 -*-
"""threshold_explorer.py, BIT 5534 week 9, with the largest F1 on the Curves tab for homework 8.

Reads a file that holds an actual category column and a saved probability of the
positive class, such as the files Save probabilities writes in the four course
tools. Every row whose probability is at or above the threshold is predicted
positive, and every row below it is predicted negative.

The left side lists the rows from the highest probability to the lowest, with the
actual label, the predicted label and the type of each row: TP, FP, FN or TN.
The Higher and Lower buttons (or Page Up and Page Down) move the threshold to the next larger or smaller
probability that occurs in the column, so each press moves at least one row
across the line. The Tables tab shows the confusion matrix, the matrix divided by
its row sums (TPR, FNR, FPR, TNR), the matrix divided by its column sums
(precision and the other three), F1, accuracy and misclassification. The Curves
tab draws recall, precision, F1, accuracy and misclassification across every
threshold, each with a checkbox, with a red line at the current threshold and
the five values at that line printed above the plot. Under the five values it
prints the largest F1 in the column and the threshold that gives it, marked on the
plot by a dotted purple line. A threshold typed into the box that prints like a
probability in the column, to four decimals, becomes that probability, so typing a
threshold the tool printed gives the tool's counts. The ROC tab draws TPR against
FPR over every threshold with the AUC, the optimal threshold (the point furthest
from the diagonal, largest TPR - FPR) and a red point at the current threshold. The
Lift tab draws lift against PP%, the share of rows at or above the threshold, with
a red point at the current threshold.

    python threshold_explorer.py                      opens the window
    python threshold_explorer.py FILE                 opens the window on FILE
    python threshold_explorer.py --selftest FILE ACTUAL PROB POSITIVE [ROLECOL ROLEVALUE] [THRESHOLD]
                                                      prints the numbers, opens no window

Needs pandas, numpy and matplotlib. Tkinter comes with Python.
"""
import math
import sys

import numpy as np
import pandas as pd

TYPE_COLORS = {"TP": "#d3f0d8", "FN": "#f8d0d0", "FP": "#fde3c4", "TN": "#dbe6f3"}
ALL_ROWS = "all rows"
ABOVE = math.inf          # a threshold above every probability: no row is predicted positive


# ------------------------------------------------------------------ the arithmetic
def read_table(path):
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def select_rows(df, role_col=None, role_value=None):
    """The rows the matrix is computed on. With a role column, only the rows holding
    role_value, such as the test rows of a fold column."""
    if role_col and role_col != ALL_ROWS and role_value not in (None, ""):
        return df[df[role_col].astype(str) == str(role_value)]
    return df


def prepare(df, actual, prob, positive):
    """Drop rows with no actual label or no probability, and sort from the highest
    probability to the lowest. Ties keep the order they have in the file."""
    d = df[[actual, prob]].copy()
    d = d[d[actual].notna() & pd.to_numeric(d[prob], errors="coerce").notna()]
    d[prob] = pd.to_numeric(d[prob])
    d["_row"] = d.index
    d = d.sort_values(prob, ascending=False, kind="mergesort")
    is_pos = (d[actual].astype(str) == str(positive)).to_numpy()
    return d, d[prob].to_numpy(float), is_pos


def thresholds(p):
    """Every probability that occurs in the column, smallest first."""
    return np.unique(p)


def step(t, levels, direction):
    """The next larger (direction +1) or smaller (-1) threshold. Above the largest
    probability the threshold is ABOVE, where no row is predicted positive."""
    if direction > 0:
        up = levels[levels > t]
        return float(up[0]) if len(up) else ABOVE
    if t == ABOVE:
        return float(levels[-1])
    down = levels[levels < t]
    return float(down[-1]) if len(down) else float(t)


def snap(t, levels):
    """A typed threshold that prints like a probability in the column, to four decimals,
    becomes that probability, so typing a threshold the tool printed gives the tool's counts.
    When several probabilities print alike, the smallest of them is used."""
    if t == ABOVE or levels is None:
        return t
    same = [float(v) for v in levels if f"{v:.4f}" == f"{t:.4f}"]
    return min(same) if same else t


def classify(p, is_pos, t):
    pred = p >= t
    kinds = np.where(pred & is_pos, "TP", np.where(pred & ~is_pos, "FP",
                     np.where(~pred & is_pos, "FN", "TN")))
    return pred, kinds


def max_f1(p, is_pos):
    """The largest F1 over every threshold that occurs in the column, and the threshold
    that gives it. On a tie the higher threshold is reported."""
    levels = thresholds(p)[::-1]
    best = None
    for t in levels:
        pred = p >= t
        TP = int((pred & is_pos).sum()); FP = int((pred & ~is_pos).sum()); FN = int((~pred & is_pos).sum())
        if TP == 0:
            continue
        pr, rc = TP / (TP + FP), TP / (TP + FN)
        f1 = 2 * pr * rc / (pr + rc)
        if best is None or f1 > best["F1"] + 1e-12:
            best = {"threshold": float(t), "F1": f1, "precision": pr, "recall": rc, "TP": TP, "FP": FP, "FN": FN}
    return best


def ratio(a, b):
    return a / b if b else float("nan")


def measures(kinds):
    TP = int((kinds == "TP").sum()); FP = int((kinds == "FP").sum())
    FN = int((kinds == "FN").sum()); TN = int((kinds == "TN").sum())
    n = TP + FP + FN + TN
    m = {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "n": n,
         "actual positive": TP + FN, "actual negative": FP + TN,
         "predicted positive": TP + FP, "predicted negative": FN + TN,
         "TPR": ratio(TP, TP + FN), "FNR": ratio(FN, TP + FN),
         "FPR": ratio(FP, FP + TN), "TNR": ratio(TN, FP + TN),
         "precision": ratio(TP, TP + FP), "FDR": ratio(FP, TP + FP),
         "FOR": ratio(FN, FN + TN), "NPV": ratio(TN, FN + TN),
         "accuracy": ratio(TP + TN, n), "misclassification": ratio(FP + FN, n)}
    pr, rc = m["precision"], m["TPR"]
    m["F1"] = 2 * pr * rc / (pr + rc) if (pr == pr and rc == rc and pr + rc > 0) else float("nan")
    return m


def roc_lift(p, is_pos):
    """ROC and lift over every threshold, highest first.

    The ROC starts at (0, 0), a threshold above every row, and each step down to the
    next probability in the column adds the rows at that probability, so tied rows
    move together. AUC is the trapezoid sum of 0.5 (TPR_i + TPR_i-1)(FPR_i - FPR_i-1).
    The optimal threshold is the one whose point lies furthest from the diagonal,
    which is the largest TPR - FPR. Lift at a threshold is TP / (PP x AP / N), the
    precision over the share of positives in the rows, against PP% = PP / N."""
    levels = thresholds(p)[::-1]
    n, ap = len(p), int(is_pos.sum())
    an = n - ap
    tpr, fpr, ppc, lift = [0.0], [0.0], [], []
    for t in levels:
        pred = p >= t
        tp = int((pred & is_pos).sum()); fp = int((pred & ~is_pos).sum()); pp = tp + fp
        tpr.append(ratio(tp, ap)); fpr.append(ratio(fp, an))
        ppc.append(pp / n); lift.append(ratio(tp, pp * ap / n) if ap else float("nan"))
    tpr, fpr = np.array(tpr), np.array(fpr)
    auc = float(np.sum(0.5 * (tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])))
    j = tpr[1:] - fpr[1:]
    k = int(np.nanargmax(j))
    best = {"threshold": float(levels[k]), "FPR": float(fpr[k + 1]), "TPR": float(tpr[k + 1]), "J": float(j[k])}
    return {"levels": levels, "tpr": tpr, "fpr": fpr, "auc": auc, "best": best,
            "pp_pct": np.array(ppc), "lift": np.array(lift), "base": ap / n if n else float("nan")}


def lift_at(m, base):
    """Lift and PP% at the current threshold, from its measures."""
    pp, n = m["predicted positive"], m["n"]
    if pp == 0 or not n:
        return float("nan"), 0.0
    return ratio(m["precision"], base), pp / n


def fmt(x, digits=3):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return "n/a" if x != x else f"{x:.{digits}f}"


def tstr(t):
    return "above every row" if t == ABOVE else f"{t:.4f}"


def report(m, t, positive, negative):
    """The results as plain text, for Copy results and for --selftest."""
    L = [f"Threshold {tstr(t)}: a row is predicted {positive} when its probability is at or above it.",
         f"Rows {m['n']}, actual {positive} {m['actual positive']}, actual {negative} {m['actual negative']}.",
         "",
         f"Confusion matrix            predicted {positive}   predicted {negative}",
         f"  actual {positive:<20} TP {m['TP']:<14} FN {m['FN']}",
         f"  actual {negative:<20} FP {m['FP']:<14} TN {m['TN']}",
         "",
         "Divided by the row sums (the actual class)",
         f"  TPR, recall, sensitivity      {fmt(m['TPR'])}",
         f"  FNR, miss rate                {fmt(m['FNR'])}",
         f"  FPR, fall out                 {fmt(m['FPR'])}",
         f"  TNR, specificity              {fmt(m['TNR'])}",
         "",
         "Divided by the column sums (the predicted class)",
         f"  precision, PPV                {fmt(m['precision'])}",
         f"  false discovery rate, FDR     {fmt(m['FDR'])}",
         f"  false omission rate, FOR      {fmt(m['FOR'])}",
         f"  negative predictive value     {fmt(m['NPV'])}",
         "",
         f"F1 {fmt(m['F1'])}   accuracy {fmt(m['accuracy'])}   misclassification {fmt(m['misclassification'])}"]
    return "\n".join(L)


def selftest(args):
    path, actual, prob, positive = args[:4]
    rest = args[4:]
    role_col = role_value = None
    t = 0.5
    if len(rest) >= 2:
        role_col, role_value = rest[0], rest[1]
        rest = rest[2:]
    if rest:
        t = float(rest[0])
    df = select_rows(read_table(path), role_col, role_value)
    d, p, is_pos = prepare(df, actual, prob, positive)
    t = snap(t, thresholds(p))
    _, kinds = classify(p, is_pos, t)
    neg = [str(v) for v in pd.unique(d[actual].astype(str)) if str(v) != str(positive)]
    m = measures(kinds)
    print(report(m, t, positive, neg[0] if len(neg) == 1 else "other"))
    R = roc_lift(p, is_pos)
    b = R["best"]
    lf, pc = lift_at(m, R["base"])
    print(f"\nAUC {R['auc']:.3f}   optimal threshold {b['threshold']:.4f}: FPR {b['FPR']:.3f}, TPR {b['TPR']:.3f}, TPR - FPR {b['J']:.3f}")
    print(f"Lift at this threshold {fmt(lf)} with {pc:.1%} of the rows predicted {positive}")
    F = max_f1(p, is_pos)
    if F:
        print(f"Largest F1 {F['F1']:.3f} at threshold {F['threshold']:.4f}: precision {F['precision']:.3f}, recall {F['recall']:.3f}")


# ------------------------------------------------------------------ the window
def main(path=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    root = tk.Tk()
    root.title("Threshold explorer")
    root.geometry("1420x840")
    st = {"df": None, "path": None, "levels": None, "t": 0.5}

    top = ttk.Frame(root, padding=6); top.pack(fill="x")
    ttk.Button(top, text="Open file", command=lambda: open_file()).grid(row=0, column=0, padx=4)
    file_lbl = ttk.Label(top, text="no file"); file_lbl.grid(row=0, column=1, columnspan=5, sticky="w")

    def combo(label, col, width=22):
        ttk.Label(top, text=label).grid(row=1, column=col, sticky="e", padx=(8, 2))
        c = ttk.Combobox(top, state="readonly", width=width)
        c.grid(row=1, column=col + 1, sticky="w")
        return c
    actual_cb = combo("Actual", 0)
    prob_cb = combo("Probability", 2)
    pos_cb = combo("Positive class", 4, 12)
    role_cb = combo("Rows", 6, 14)
    rval_cb = combo("equal to", 8, 14)

    bar = ttk.Frame(root, padding=(6, 2)); bar.pack(fill="x")
    ttk.Label(bar, text="Threshold").pack(side="left")
    t_var = tk.StringVar(value="0.5000")
    t_entry = ttk.Entry(bar, textvariable=t_var, width=16); t_entry.pack(side="left", padx=4)
    ttk.Button(bar, text="Higher ▲", command=lambda: move(+1)).pack(side="left", padx=2)
    ttk.Button(bar, text="Lower ▼", command=lambda: move(-1)).pack(side="left", padx=2)
    ttk.Button(bar, text="Copy results", command=lambda: copy()).pack(side="left", padx=12)
    status = ttk.Label(bar, text=""); status.pack(side="left", padx=8)

    body = ttk.Frame(root, padding=6); body.pack(fill="both", expand=True)
    left = ttk.Frame(body); left.pack(side="left", fill="both", expand=True)
    cols = ("rank", "prob", "actual", "pred", "type")
    tree = ttk.Treeview(left, columns=cols, show="headings", height=28)
    for c, h, w in zip(cols, ("Rank", "Probability", "Actual", "Predicted", "Type"), (60, 110, 110, 110, 70)):
        tree.heading(c, text=h); tree.column(c, width=w, anchor="center")
    for k, colr in TYPE_COLORS.items():
        tree.tag_configure(k, background=colr)
    tree.tag_configure("line", background="#212529", foreground="white")
    sb = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True); sb.pack(side="left", fill="y")

    nb = ttk.Notebook(body); nb.pack(side="left", fill="both", expand=True)
    right = tk.Frame(nb, padx=14); nb.add(right, text="  Tables  ")
    curves = tk.Frame(nb, padx=8, pady=6); nb.add(curves, text="  Curves  ")
    roc_tab = tk.Frame(nb, padx=8, pady=6); nb.add(roc_tab, text="  ROC  ")
    lift_tab = tk.Frame(nb, padx=8, pady=6); nb.add(lift_tab, text="  Lift  ")
    BIG = ("Segoe UI", 12); HEAD = ("Segoe UI", 12, "bold"); SMALL = ("Segoe UI", 10)

    def cell(parent, r, c, text="", bg="white", font=BIG, w=16):
        l = tk.Label(parent, text=text, bg=bg, font=font, width=w, relief="solid", bd=1, pady=6)
        l.grid(row=r, column=c, sticky="nsew")
        return l

    def section(title):
        tk.Label(right, text=title, font=HEAD, anchor="w").pack(fill="x", pady=(10, 2))
        f = tk.Frame(right); f.pack(anchor="w")
        return f

    g1 = section("Confusion matrix")
    cm = {}
    cell(g1, 0, 0, "", "#f1f3f5"); hp = cell(g1, 0, 1, "predicted +", "#f1f3f5", HEAD)
    hn = cell(g1, 0, 2, "predicted −", "#f1f3f5", HEAD); cell(g1, 0, 3, "row sum", "#f1f3f5", HEAD, 10)
    ap = cell(g1, 1, 0, "actual +", "#f1f3f5", HEAD); an = cell(g1, 2, 0, "actual −", "#f1f3f5", HEAD)
    cm["TP"] = cell(g1, 1, 1, "", TYPE_COLORS["TP"]); cm["FN"] = cell(g1, 1, 2, "", TYPE_COLORS["FN"])
    cm["FP"] = cell(g1, 2, 1, "", TYPE_COLORS["FP"]); cm["TN"] = cell(g1, 2, 2, "", TYPE_COLORS["TN"])
    cm["rp"] = cell(g1, 1, 3, "", w=10); cm["rn"] = cell(g1, 2, 3, "", w=10)
    cell(g1, 3, 0, "column sum", "#f1f3f5", HEAD); cm["cp"] = cell(g1, 3, 1); cm["cn"] = cell(g1, 3, 2)
    cm["n"] = cell(g1, 3, 3, "", w=10)

    g2 = section("Divided by the row sums, the actual class")
    rn = {}
    cell(g2, 0, 0, "", "#f1f3f5"); heads = [cell(g2, 0, 1, "", "#f1f3f5", HEAD), cell(g2, 0, 2, "", "#f1f3f5", HEAD),
                                           cell(g2, 1, 0, "", "#f1f3f5", HEAD), cell(g2, 2, 0, "", "#f1f3f5", HEAD)]
    rn["TPR"] = cell(g2, 1, 1, "", TYPE_COLORS["TP"]); rn["FNR"] = cell(g2, 1, 2, "", TYPE_COLORS["FN"])
    rn["FPR"] = cell(g2, 2, 1, "", TYPE_COLORS["FP"]); rn["TNR"] = cell(g2, 2, 2, "", TYPE_COLORS["TN"])
    tk.Label(right, font=SMALL, justify="left", anchor="w", text=(
        "TPR, true positive rate: recall, sensitivity        FNR, false negative rate: miss rate\n"
        "FPR, false positive rate: fall out                     TNR, true negative rate: specificity")).pack(fill="x")

    g3 = section("Divided by the column sums, the predicted class")
    cn = {}
    cell(g3, 0, 0, "", "#f1f3f5"); heads += [cell(g3, 0, 1, "", "#f1f3f5", HEAD), cell(g3, 0, 2, "", "#f1f3f5", HEAD),
                                            cell(g3, 1, 0, "", "#f1f3f5", HEAD), cell(g3, 2, 0, "", "#f1f3f5", HEAD)]
    cn["precision"] = cell(g3, 1, 1, "", TYPE_COLORS["TP"]); cn["FOR"] = cell(g3, 1, 2, "", TYPE_COLORS["FN"])
    cn["FDR"] = cell(g3, 2, 1, "", TYPE_COLORS["FP"]); cn["NPV"] = cell(g3, 2, 2, "", TYPE_COLORS["TN"])
    tk.Label(right, font=SMALL, justify="left", anchor="w", text=(
        "precision, positive predictive value: TP / (TP + FP)      false omission rate: FN / (FN + TN)\n"
        "false discovery rate: FP / (TP + FP)                       negative predictive value: TN / (FN + TN)")).pack(fill="x")

    g4 = section("One number each")
    tail = tk.Label(right, font=("Segoe UI", 13, "bold"), anchor="w", justify="left"); tail.pack(fill="x")
    tk.Label(right, font=SMALL, anchor="w", justify="left",
             text="F1 = 2 × precision × recall / (precision + recall), the harmonic mean of the two").pack(fill="x")

    # ---------------------------------------------------------- the Curves tab
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    CURVES = [("TPR", "recall", "#2f9e44"), ("precision", "precision", "#f08c00"), ("F1", "F1", "#7048e8"),
              ("accuracy", "accuracy", "#1c7ed6"), ("misclassification", "misclassification", "#212529")]
    readout = tk.Frame(curves); readout.pack(fill="x", pady=(0, 4))
    t_read = tk.Label(readout, font=("Segoe UI", 13, "bold"), fg="#c92a2a", width=18, anchor="w")
    t_read.pack(side="left")
    show, vals = {}, {}
    for key, lab, col in CURVES:
        f = tk.Frame(readout); f.pack(side="left", padx=5)
        show[key] = tk.BooleanVar(value=True)
        tk.Checkbutton(f, text=lab, variable=show[key], fg=col, font=("Segoe UI", 11, "bold"),
                       command=lambda: draw_curves()).pack(anchor="w")
        vals[key] = tk.Label(f, font=("Segoe UI", 14, "bold"), fg=col); vals[key].pack(anchor="w")
    f1_best = tk.Label(curves, font=("Segoe UI", 12, "bold"), fg="#7048e8", anchor="w", justify="left")
    f1_best.pack(fill="x", pady=(0, 4))
    fig = Figure(figsize=(7.4, 5.2), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=curves)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    cur = {"line": None, "sweep": None, "roc": None, "rocpt": None, "liftpt": None}

    def head(parent):
        f = tk.Frame(parent); f.pack(fill="x", pady=(0, 4))
        dyn = tk.Label(f, font=("Segoe UI", 14, "bold"), fg="#c92a2a", anchor="w", justify="left")
        dyn.pack(fill="x")
        sta = tk.Label(f, font=("Segoe UI", 12), fg="#343a40", anchor="w", justify="left")
        sta.pack(fill="x")
        return dyn, sta
    roc_dyn, roc_sta = head(roc_tab)
    rfig = Figure(figsize=(6.4, 5.6), dpi=100); rax = rfig.add_subplot(111)
    rcanvas = FigureCanvasTkAgg(rfig, master=roc_tab); rcanvas.get_tk_widget().pack(fill="both", expand=True)
    lift_dyn, lift_sta = head(lift_tab)
    lfig = Figure(figsize=(7.4, 5.2), dpi=100); lax = lfig.add_subplot(111)
    lcanvas = FigureCanvasTkAgg(lfig, master=lift_tab); lcanvas.get_tk_widget().pack(fill="both", expand=True)

    def draw_roc_lift():
        R = cur["roc"]
        rax.clear(); lax.clear()
        if R is None:
            rcanvas.draw_idle(); lcanvas.draw_idle(); return
        rax.plot([0, 1], [0, 1], color="#adb5bd", lw=1.2, ls="--", label="random, AUC 0.5")
        rax.plot(R["fpr"], R["tpr"], color="#1971c2", lw=2.2, label=f"ROC, AUC {R['auc']:.3f}")
        b = R["best"]
        rax.plot([b["FPR"], b["FPR"]], [b["FPR"], b["TPR"]], color="#2f9e44", lw=1.5, ls=":")
        rax.plot([b["FPR"]], [b["TPR"]], marker="s", ms=9, color="#2f9e44", label="optimal, furthest from the diagonal")
        rax.set_xlim(-0.01, 1.01); rax.set_ylim(-0.01, 1.01); rax.set_aspect("equal")
        rax.set_xlabel("FPR, false positive rate"); rax.set_ylabel("TPR, true positive rate, recall")
        rax.grid(alpha=0.25); rax.legend(loc="lower right", frameon=False, fontsize=9)
        cur["rocpt"], = rax.plot([0], [0], marker="o", ms=13, color="#e03131", zorder=5)
        roc_sta.config(text=f"AUC {R['auc']:.3f}      optimal threshold {b['threshold']:.4f}:  "
                            f"FPR {b['FPR']:.3f}, TPR {b['TPR']:.3f}")
        lax.axhline(1.0, color="#adb5bd", lw=1.2, ls="--", label="baseline, lift 1")
        lax.plot(R["pp_pct"] * 100, R["lift"], color="#7048e8", lw=2.0, label="lift")
        lax.set_xlim(0, 100); top = np.nanmax(R["lift"]) if len(R["lift"]) else 1
        lax.set_ylim(0, max(1.2, top * 1.1))
        lax.set_xlabel("PP%, the share of rows at or above the threshold, from the top of the table")
        lax.set_ylabel("lift, precision divided by AP / N")
        lax.grid(alpha=0.25); lax.legend(loc="upper right", frameon=False, fontsize=9)
        cur["liftpt"], = lax.plot([np.nan], [np.nan], marker="o", ms=13, color="#e03131", zorder=5)
        lift_sta.config(text=f"baseline AP / N = {R['base']:.3f}: picking rows at random finds positives at this rate, lift 1")
        place_points()

    def place_points():
        R = cur["roc"]
        if R is None or "m" not in st or cur["rocpt"] is None:
            return
        m, t = st["m"], st["t"]
        cur["rocpt"].set_data([m["FPR"]], [m["TPR"]])
        roc_dyn.config(text=f"threshold {tstr(t)}      FPR {fmt(m['FPR'])}      TPR {fmt(m['TPR'])}")
        lf, pc = lift_at(m, R["base"])
        if lf == lf:
            cur["liftpt"].set_data([pc * 100], [lf])
            lift_dyn.config(text=f"threshold {tstr(t)}      PP% {pc:.1%}      lift {lf:.3f}")
        else:
            cur["liftpt"].set_data([np.nan], [np.nan])
            lift_dyn.config(text=f"threshold {tstr(t)}      no row is predicted positive yet, so lift is undefined")
        rcanvas.draw_idle(); lcanvas.draw_idle()

    def build_sweep(p, is_pos):
        levels = thresholds(p)[::-1]
        rows = [measures(classify(p, is_pos, t)[1]) for t in levels]
        return levels, {k: np.array([r[k] for r in rows], float) for k, _, _ in CURVES}

    def draw_curves():
        ax.clear()
        if cur["sweep"] is None:
            canvas.draw_idle(); return
        levels, ys = cur["sweep"]
        for key, lab, col in CURVES:
            if show[key].get():
                ax.step(levels, ys[key], where="post", color=col, lw=2, label=lab)
        ax.set_xlim(1.0, 0.0); ax.set_ylim(0, 1.02)
        ax.set_xlabel("threshold, falling from left to right")
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        F = cur.get("f1best")
        if F:
            ax.axvline(F["threshold"], color="#7048e8", lw=1.5, ls=":")
            f1_best.config(text=f"largest F1 {F['F1']:.3f} at threshold {F['threshold']:.4f}      "
                                f"precision {F['precision']:.3f}, recall {F['recall']:.3f}      (dotted purple line)")
        else:
            f1_best.config(text="")
        cur["line"] = ax.axvline(1.0, color="#e03131", lw=2)
        place_line()

    def place_line():
        if cur["line"] is None or "m" not in st:
            return
        t = st["t"]
        xt = 1.0 if t == ABOVE else min(max(t, 0.0), 1.0)
        cur["line"].set_xdata([xt, xt])
        canvas.draw_idle()
        m = st["m"]
        t_read.config(text=f"threshold {tstr(t)}")
        for key, _, _ in CURVES:
            vals[key].config(text=fmt(m[key]))

    def numeric_cols(df):
        return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    def open_file(p=None):
        p = p or filedialog.askopenfilename(filetypes=[("Data", "*.csv *.xlsx *.xls"), ("All", "*.*")])
        if not p:
            return
        try:
            df = read_table(p)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Open file", str(e)); return
        st["df"], st["path"] = df, p
        file_lbl.config(text=f"{p}   {len(df)} rows")
        cats = [c for c in df.columns if df[c].nunique(dropna=True) <= 20]
        actual_cb["values"] = cats
        prob_cb["values"] = [c for c in numeric_cols(df) if df[c].between(0, 1).all() or "prob" in c.lower()]
        role_cb["values"] = [ALL_ROWS] + [c for c in df.columns if df[c].nunique(dropna=True) <= 12]
        actual_cb.set(next((c for c in cats if "binary" in c.lower()), cats[0] if cats else ""))
        probs = list(prob_cb["values"])
        prob_cb.set(probs[0] if probs else "")
        roles = [c for c in role_cb["values"] if c != ALL_ROWS and "test" in set(df[c].astype(str))]
        role_cb.set(roles[0] if roles else ALL_ROWS)
        fill_levels(); fill_roles(); recompute()

    def fill_levels(*_):
        df = st["df"]
        if df is None or not actual_cb.get():
            return
        counts = df[actual_cb.get()].astype(str).value_counts()
        pos_cb["values"] = list(counts.index)
        prob = prob_cb.get()
        guess = next((lv for lv in counts.index if prob.endswith(" " + lv)), counts.index[-1])
        pos_cb.set(guess)

    def fill_roles(*_):
        df = st["df"]
        rc = role_cb.get()
        if df is None or rc in ("", ALL_ROWS):
            rval_cb["values"] = []; rval_cb.set(""); return
        vals = sorted(df[rc].dropna().astype(str).unique())
        rval_cb["values"] = vals
        rval_cb.set("test" if "test" in vals else vals[0])

    def current():
        df = select_rows(st["df"], role_cb.get(), rval_cb.get())
        return prepare(df, actual_cb.get(), prob_cb.get(), pos_cb.get())

    def typed(*_):
        """A threshold typed into the box and confirmed with Enter."""
        txt = t_var.get().strip().lower()
        try:
            st["t"] = ABOVE if txt.startswith("above") else snap(float(txt), st.get("levels"))
        except ValueError:
            status.config(text="Type a number such as 0.35, then press Enter."); return
        recompute()

    def recompute():
        if st["df"] is None or not prob_cb.get() or not actual_cb.get():
            return
        try:
            d, p, is_pos = current()
        except Exception as e:  # noqa: BLE001
            status.config(text=str(e)); return
        st["levels"] = thresholds(p)
        t = st["t"]
        t_var.set(tstr(t))
        pred, kinds = classify(p, is_pos, t)
        pos = pos_cb.get()
        others = [v for v in pos_cb["values"] if v != pos]
        neg = others[0] if len(others) == 1 else "not " + pos
        tree.delete(*tree.get_children())
        line_at = int(pred.sum())
        for i, (pv, a, pr, k) in enumerate(zip(p, d[actual_cb.get()].astype(str), pred, kinds), 1):
            if i == line_at + 1:
                tree.insert("", "end", values=("", tstr(t), "threshold", "", ""), tags=("line",))
            tree.insert("", "end", values=(i, f"{pv:.4f}", a, pos if pr else neg, k), tags=(k,))
        if line_at == len(p):
            tree.insert("", "end", values=("", tstr(t), "threshold", "", ""), tags=("line",))
        n_items = len(p) + 1
        root.update_idletasks()
        tree.yview_moveto(max(0.0, (line_at - 13) / n_items))
        m = measures(kinds)
        st["m"], st["neg"] = m, neg
        for k in ("TP", "FN", "FP", "TN"):
            cm[k].config(text=f"{k}  {m[k]}")
        cm["rp"].config(text=m["actual positive"]); cm["rn"].config(text=m["actual negative"])
        cm["cp"].config(text=m["predicted positive"]); cm["cn"].config(text=m["predicted negative"])
        cm["n"].config(text=m["n"])
        for k in rn:
            rn[k].config(text=f"{k}  {fmt(m[k])}")
        for k, lab in (("precision", "precision"), ("FOR", "FOR"), ("FDR", "FDR"), ("NPV", "NPV")):
            cn[k].config(text=f"{lab}  {fmt(m[k])}")
        names = (f"predicted {pos}", f"predicted {neg}", f"actual {pos}", f"actual {neg}")
        for w, txt in zip([hp, hn, ap, an] + heads, names * 3):
            w.config(text=txt)
        tail.config(text=f"F1 {fmt(m['F1'])}     accuracy {fmt(m['accuracy'])}     "
                         f"misclassification {fmt(m['misclassification'])}")
        key = (id(st["df"]), actual_cb.get(), prob_cb.get(), pos_cb.get(), role_cb.get(), rval_cb.get())
        if cur.get("key") != key:
            cur["key"] = key
            cur["sweep"] = build_sweep(p, is_pos)
            cur["f1best"] = max_f1(p, is_pos)
            cur["roc"] = roc_lift(p, is_pos)
            draw_curves()
            draw_roc_lift()
        else:
            place_line()
            place_points()
        status.config(text=f"{m['n']} rows, {len(st['levels'])} distinct probabilities. "
                           f"{m['predicted positive']} rows at or above the threshold are predicted {pos}.")

    def move(direction):
        if st["levels"] is None:
            return
        st["t"] = step(st["t"], st["levels"], direction)
        recompute()

    def copy():
        if "m" not in st:
            return
        txt = report(st["m"], st["t"], pos_cb.get(), st["neg"])
        R = cur.get("roc")
        if R is not None:
            b = R["best"]; lf, pc = lift_at(st["m"], R["base"])
            txt += (f"\n\nAUC {R['auc']:.3f}   optimal threshold {b['threshold']:.4f}: FPR {b['FPR']:.3f}, TPR {b['TPR']:.3f}"
                    f"\nLift at this threshold {fmt(lf)} with {pc:.1%} of the rows predicted {pos_cb.get()}")
        F = cur.get("f1best")
        if F:
            txt += f"\nLargest F1 {F['F1']:.3f} at threshold {F['threshold']:.4f}: precision {F['precision']:.3f}, recall {F['recall']:.3f}"
        root.clipboard_clear(); root.clipboard_append(txt)
        status.config(text="Results copied to the clipboard.")

    actual_cb.bind("<<ComboboxSelected>>", lambda e: (fill_levels(), recompute()))
    prob_cb.bind("<<ComboboxSelected>>", lambda e: (fill_levels(), recompute()))
    pos_cb.bind("<<ComboboxSelected>>", lambda e: recompute())
    role_cb.bind("<<ComboboxSelected>>", lambda e: (fill_roles(), recompute()))
    rval_cb.bind("<<ComboboxSelected>>", lambda e: recompute())
    t_entry.bind("<Return>", typed)
    root.bind_all("<Prior>", lambda e: move(+1)); root.bind_all("<Next>", lambda e: move(-1))
    if path:
        root.after(100, lambda: open_file(path))
    root.mainloop()


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--selftest":
        selftest(a[1:])
    else:
        main(a[0] if a else None)
