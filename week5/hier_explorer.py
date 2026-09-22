# -*- coding: utf-8 -*-
"""Hierarchical clustering explorer for BIT 5534 week 5.

The companion tool. It does what the starter specification asks for, and it is
here for anybody whose own build did not finish. A tool you did not write still
teaches the reading.

    python hier_explorer.py                 pick the file from the dialog
    python hier_explorer.py data.csv        open that file straight away
    python hier_explorer.py data.csv --selftest    no window, prints the numbers

Needs pandas, numpy, matplotlib, scipy and tkinter. On Windows, if a library is
missing, run: py -m pip install pandas numpy matplotlib scipy openpyxl
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

LINKAGES = ["ward", "single", "complete", "average", "centroid"]
PALETTE = ["#d23b2e", "#2f9e44", "#2c6fbb", "#e8890c", "#7048e8",
           "#0ca678", "#e64980", "#845ef7", "#f08c00", "#495057"]


# ----------------------------------------------------------------- the analysis
def prepare(frame, columns, standardize=True):
    """Return the matrix the tree is built on, and the rows that survived.

    A row holding a missing value in any chosen column cannot be given a
    distance to anything, so it is dropped and the count of dropped rows is
    reported rather than absorbed.
    """
    sub = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    keep = sub.notna().all(axis=1).values
    A = sub.values[keep].astype(float)
    if standardize:
        sd = A.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        A = (A - A.mean(axis=0)) / sd
    return A, keep


def build(A, method="ward"):
    return linkage(A, method=method)


def cut(Z, k):
    return fcluster(Z, k, criterion="maxclust")


def heights_by_k(Z, most=12):
    """The height of each merge against the number of clusters it produced."""
    h = Z[:, 2]
    n = min(most, len(h))
    ks = np.arange(1, n + 1)
    return ks, h[::-1][:n]


def profile(frame, columns, labels, standardize_table=True):
    """One row per cluster: how many rows it holds and where its mean sits."""
    sub = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    A = sub.values.astype(float)
    if standardize_table:
        sd = A.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        A = (A - A.mean(axis=0)) / sd
    out = []
    for c in sorted(set(labels)):
        m = labels == c
        row = [int(m.sum())] + list(np.round(A[m].mean(axis=0), 2))
        out.append([c] + row)
    cols = ["cluster", "rows"] + list(columns)
    return pd.DataFrame(out, columns=cols)


# ----------------------------------------------------------------- the drawing
def cluster_link_colors(Z, labels):
    """Colour every link by the cluster it sits inside, grey above the cut.

    scipy colours a dendrogram in leaf order with a palette of its own, which
    has no relation to the numbers fcluster returns. Left alone, the orange
    branch of the tree is not the orange cluster in the table, and the failure
    is quiet because both pictures are individually correct.
    """
    n = len(labels)
    colors = {}
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        ca = colors.get(a, -1) if a >= n else int(labels[a])
        cb = colors.get(b, -1) if b >= n else int(labels[b])
        colors[i + n] = ca if (ca == cb and ca > 0) else -1
    return {i + n: (PALETTE[(colors[i + n] - 1) % len(PALETTE)]
                    if colors[i + n] > 0 else "#adb5bd")
            for i in range(len(Z))}


def draw_dendrogram(axd, frame, label_col, Z, labels, k, method):
    h = Z[:, 2]
    threshold = (h[-k] + h[-k + 1]) / 2.0 if 1 < k < len(h) else h[-1] * 1.1
    leaf = None
    if label_col:
        leaf = [str(v) for v in frame[label_col].values]
    link_color = cluster_link_colors(Z, labels)
    dendrogram(Z, ax=axd, labels=leaf, no_labels=leaf is None, leaf_font_size=7,
               link_color_func=lambda i: link_color[i])
    if leaf is not None:
        axd.tick_params(axis="x", labelrotation=90)
    axd.axhline(threshold, color="#d23b2e", linewidth=1.4, linestyle=(0, (5, 4)))
    axd.set_title(f"Dendrogram, {method} linkage, cut at {k}", fontsize=10)
    axd.set_ylabel("merge height")


def draw_heights(axs, Z, k):
    ks, hts = heights_by_k(Z)
    axs.plot(ks, hts, marker="o", color="#212529", markersize=5, linewidth=1.4)
    if k <= len(hts):
        axs.scatter([k], [hts[k - 1]], s=130, facecolor="none",
                    edgecolor="#d23b2e", linewidth=2.0, zorder=4)
    axs.invert_xaxis()
    axs.set_xticks(ks)
    axs.set_xlabel("number of clusters the merge produced")
    axs.set_ylabel("height of that merge")
    axs.set_title("Where the heights start rising", fontsize=10)
    axs.grid(True, color="#dee2e6", linewidth=0.7)
    axs.set_axisbelow(True)


def draw_parallel(axp, frame, columns, labels):
    sub = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    A = sub.values.astype(float)
    sd = A.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    A = (A - A.mean(axis=0)) / sd
    xs = np.arange(len(columns))
    for i in range(A.shape[0]):
        axp.plot(xs, A[i], color=PALETTE[(labels[i] - 1) % len(PALETTE)],
                 linewidth=0.9, alpha=0.30)
    for c in sorted(set(labels)):
        axp.plot(xs, A[labels == c].mean(axis=0),
                 color=PALETTE[(c - 1) % len(PALETTE)], linewidth=2.6,
                 marker="o", markersize=4.5, label=f"cluster {c}", zorder=4)
    axp.legend(frameon=False, fontsize=8, ncol=min(5, len(set(labels))))
    axp.set_xticks(xs)
    axp.set_xticklabels(list(columns), rotation=30, ha="right", fontsize=8)
    axp.set_ylabel("standard deviations from the column mean")
    axp.set_title("Parallel coordinates, faint lines are rows and solid lines are cluster means",
                  fontsize=10)
    axp.grid(True, axis="y", color="#dee2e6", linewidth=0.7)
    axp.set_axisbelow(True)


def draw_table(axt, frame, columns, labels, standardized):
    axt.axis("off")
    prof = profile(frame, columns, labels)
    cell = [[str(int(row[0])), str(int(row[1]))] + [f"{v:.2f}" for v in row[2:]]
            for row in prof.values]
    t = axt.table(cellText=cell, colLabels=list(prof.columns), loc="center",
                  cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1.0, 1.35)
    for j, c in enumerate(prof["cluster"].values):
        t[(j + 1, 0)].set_facecolor(PALETTE[(int(c) - 1) % len(PALETTE)])
        t[(j + 1, 0)].set_text_props(color="white")
    note = "analysis on standardized columns" if standardized else "analysis on the columns as they are"
    axt.set_title(f"Cluster profiles in standard deviations, {note}", fontsize=10)


# ----------------------------------------------------------------- the window
def run_gui(initial_path=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    state = {"frame": None, "path": None, "Z": None, "labels": None, "columns": None}

    root = tk.Tk()
    root.title("Hierarchical clustering explorer, BIT 5534")
    root.geometry("1450x900")

    side = ttk.Frame(root, padding=10)
    side.pack(side="left", fill="y")
    right = ttk.Frame(root)
    right.pack(side="right", fill="both", expand=True)

    # Four tabs, one panel each. Three sampled runs of the specification put all
    # four panels in one pane and every one of them came out squeezed, so the
    # specification now asks for tabs and this tool has to match what it hands out.
    tabs = ttk.Notebook(right)
    tabs.pack(fill="both", expand=True)
    figs, canvases = {}, {}
    for key, title in (("dendrogram", "Dendrogram"),
                       ("heights", "Merge heights"),
                       ("parallel", "Parallel coordinates"),
                       ("table", "Cluster table")):
        page = ttk.Frame(tabs)
        tabs.add(page, text=title)
        figs[key] = Figure(figsize=(11.5, 7.6), dpi=100)
        canvases[key] = FigureCanvasTkAgg(figs[key], master=page)
        canvases[key].get_tk_widget().pack(fill="both", expand=True)

    path_var = tk.StringVar(value="no file open")
    std_var = tk.BooleanVar(value=True)
    link_var = tk.StringVar(value="ward")
    k_var = tk.IntVar(value=5)
    status = tk.StringVar(value="Open a file, choose the columns, then press Run.")
    k_range = tk.StringVar(value="")

    ttk.Label(side, textvariable=path_var, wraplength=250).pack(anchor="w", pady=(0, 6))
    ttk.Button(side, text="Open data file", command=lambda: open_file()).pack(fill="x")

    ttk.Label(side, text="Analysis columns").pack(anchor="w", pady=(12, 2))
    cols_box = tk.Listbox(side, selectmode="extended", height=12, exportselection=False)
    cols_box.pack(fill="x")

    ttk.Label(side, text="Label column, optional").pack(anchor="w", pady=(12, 2))
    label_box = ttk.Combobox(side, state="readonly", values=["none"])
    label_box.set("none")
    label_box.pack(fill="x")

    ttk.Checkbutton(side, text="Standardize the chosen columns",
                    variable=std_var).pack(anchor="w", pady=(12, 2))

    ttk.Label(side, text="Linkage").pack(anchor="w", pady=(12, 2))
    link_box = ttk.Combobox(side, state="readonly", values=LINKAGES, textvariable=link_var)
    link_box.pack(fill="x")

    ttk.Label(side, text="Number of clusters").pack(anchor="w", pady=(12, 2))
    k_spin = ttk.Spinbox(side, from_=2, to=2, increment=1, textvariable=k_var, width=8)
    k_spin.pack(anchor="w")
    ttk.Label(side, textvariable=k_range, foreground="#868e96").pack(anchor="w")

    ttk.Button(side, text="Run", command=lambda: run()).pack(fill="x", pady=(16, 4))
    ttk.Button(side, text="Save cluster numbers", command=lambda: save()).pack(fill="x")
    ttk.Label(side, textvariable=status, wraplength=250,
              foreground="#495057").pack(anchor="w", pady=(14, 0))

    def load(path):
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                frame = pd.read_excel(path)
            else:
                frame = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Could not read the file", str(exc))
            return
        state["frame"] = frame
        state["path"] = path
        path_var.set(os.path.basename(path) + f"   {len(frame)} rows")
        numeric = [c for c in frame.columns
                   if pd.to_numeric(frame[c], errors="coerce").notna().sum() > 0
                   and pd.api.types.is_numeric_dtype(pd.to_numeric(frame[c], errors="coerce"))
                   and pd.to_numeric(frame[c], errors="coerce").notna().all()]
        cols_box.delete(0, "end")
        for c in numeric:
            cols_box.insert("end", c)
        cols_box.selection_set(0, "end")
        label_box["values"] = ["none"] + [str(c) for c in frame.columns]
        label_box.set("none")
        k_spin.configure(to=max(2, len(frame)))
        k_range.set(f"2 to {max(2, len(frame))}")
        if int(k_var.get()) > len(frame):
            k_var.set(min(5, len(frame)))
        status.set(f"{len(numeric)} numeric columns offered. Choose and press Run.")

    def open_file():
        p = filedialog.askopenfilename(
            title="Choose a data file",
            filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*")])
        if p:
            load(p)

    def run():
        if state["frame"] is None:
            status.set("Open a data file first.")
            return
        chosen = [cols_box.get(i) for i in cols_box.curselection()]
        if len(chosen) < 2:
            status.set("Choose at least two analysis columns.")
            return
        k = int(k_var.get())
        A, keep = prepare(state["frame"], chosen, std_var.get())
        if len(A) < k:
            status.set("Fewer usable rows than clusters asked for.")
            return
        Z = build(A, link_var.get())
        labels = cut(Z, k)
        frame = state["frame"].loc[keep].reset_index(drop=True)
        lab_col = None if label_box.get() == "none" else label_box.get()
        for key in figs:
            figs[key].clear()
        draw_dendrogram(figs["dendrogram"].add_subplot(111), frame, lab_col, Z, labels,
                        k, link_var.get())
        draw_heights(figs["heights"].add_subplot(111), Z, k)
        draw_parallel(figs["parallel"].add_subplot(111), frame, chosen, labels)
        draw_table(figs["table"].add_subplot(111), frame, chosen, labels, std_var.get())
        for key in figs:
            figs[key].tight_layout()
            canvases[key].draw()
        state.update({"Z": Z, "labels": labels, "columns": chosen, "kept": frame})
        sizes = ", ".join(str(int((labels == c).sum())) for c in sorted(set(labels)))
        dropped = int((~keep).sum())
        extra = f"  {dropped} rows dropped for missing values." if dropped else ""
        status.set(f"{link_var.get()} linkage, {k} clusters, sizes {sizes}."
                   f"  First merge height {Z[0, 2]:.4f}.{extra}")

    def save():
        if state.get("labels") is None:
            status.set("Run the analysis first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         initialfile="clusters.csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        out = state["kept"].copy()
        out["Cluster"] = state["labels"]
        out.to_csv(p, index=False)
        status.set(f"Wrote {os.path.basename(p)}.")

    if initial_path and os.path.exists(initial_path):
        load(initial_path)
    root.mainloop()


def selftest(path):
    frame = pd.read_csv(path)
    cols = [c for c in frame.columns
            if pd.to_numeric(frame[c], errors="coerce").notna().all()]
    cols = [c for c in cols if c.lower() != "cluster"]
    print(f"{os.path.basename(path)}: {len(frame)} rows, analysis columns {cols}")
    for method in LINKAGES:
        A, keep = prepare(frame, cols, True)
        Z = build(A, method)
        lab = cut(Z, 5)
        sizes = sorted((int((lab == c).sum()) for c in set(lab)), reverse=True)
        print(f"  {method:9s} first merge {Z[0, 2]:.4f}  cut at five {Z[:, 2][-5]:.4f}"
              f"  root {Z[-1, 2]:.4f}  sizes {sizes}")
    A, keep = prepare(frame, cols, True)
    lab = cut(build(A, "ward"), 5)
    print(profile(frame.loc[keep], cols, lab).to_string(index=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--selftest" in sys.argv:
        selftest(args[0])
    else:
        run_gui(args[0] if args else None)
