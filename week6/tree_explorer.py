#!/usr/bin/env python3
"""
Single-window decision tree explorer.

BIT 5534 lec6_1 demonstration and backup tool, written 2026-09-28 against the
same sixteen bullet specification the students send, so that what the room sees
on the projector is what their own tools should show. Splits are scored by the
likelihood ratio chi-square, G squared, unadjusted. JMP adjusts its logworth for
the number of cuts a column offers, so JMP's logworths differ from the ones shown
here while the tree it grows on freshmen.csv is the same.

    python tree_explorer.py                  opens a file dialog
    python tree_explorer.py freshmen.csv     opens that file
    python tree_explorer.py --selftest FILE  prints the numbers, opens no window

Required packages: pandas, numpy, matplotlib, scipy (and openpyxl for .xlsx).
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

SELFTEST = "--selftest" in sys.argv
if not SELFTEST:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

CLASS_COLORS = ["#c0392b", "#2c6fbb", "#27ae60", "#d68910", "#8e44ad", "#16a085"]
MAX_LEVELS = 12          # 2**11 - 1 = 2047 partitions, the most a column may offer


# ----------------------------------------------------------------- the model
def g_squared(table):
    """Likelihood ratio chi-square of a two way table of counts."""
    t = np.asarray(table, dtype=float)
    n = t.sum()
    if n == 0:
        return 0.0
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    mask = t > 0
    return float(2.0 * (t[mask] * np.log(t[mask] / expected[mask])).sum())


def logworth(g2, n_classes):
    p = chi2.sf(g2, max(n_classes - 1, 1))
    return float(-np.log10(max(p, 1e-300)))


def class_counts(y, classes):
    return np.array([(y == c).sum() for c in classes], dtype=int)


def best_cut_for_column(x, y, classes, numeric):
    """Return (cut, g2, left_mask) for the best cut on one column, or None.

    A numeric cut is the point halfway between two neighboring distinct values,
    and the left branch is every row below it. A non-numeric cut is a division of
    the levels into two groups, and the left branch is the group holding the first
    level in sorted order, so each division is tried once and not twice.
    """
    best = None
    if numeric:
        values = np.sort(np.unique(x))
        for a, b in zip(values[:-1], values[1:]):
            cut = (a + b) / 2.0
            left = x < cut
            table = [class_counts(y[left], classes), class_counts(y[~left], classes)]
            g2 = g_squared(table)
            if best is None or g2 > best[1] + 1e-12:
                best = (cut, g2, left)
    else:
        levels = sorted(pd.unique(x).tolist(), key=str)
        if len(levels) < 2 or len(levels) > MAX_LEVELS:
            return None
        first, rest = levels[0], levels[1:]
        for r in range(0, len(rest)):
            for combo in itertools.combinations(rest, r):
                group = (first,) + combo
                left = np.isin(x, group)
                table = [class_counts(y[left], classes), class_counts(y[~left], classes)]
                g2 = g_squared(table)
                if best is None or g2 > best[1] + 1e-12:
                    best = (tuple(group), g2, left)
    return best


def candidates(df, idx, inputs, target, classes, numeric_cols):
    """One row per input column: its best cut, G squared and logworth at this node."""
    sub = df.loc[idx]
    y = sub[target].to_numpy()
    rows = []
    for col in inputs:
        res = best_cut_for_column(sub[col].to_numpy(), y, classes, col in numeric_cols)
        if res is None:
            continue
        cut, g2, left = res
        rows.append({"column": col, "cut": cut, "g2": g2,
                     "logworth": logworth(g2, len(classes)),
                     "left": sub.index[left], "right": sub.index[~left]})
    rows.sort(key=lambda r: -r["g2"])
    return rows


def describe_cut(col, cut, left, numeric):
    if numeric:
        return (f"{col} < {cut:.4g}") if left else (f"{col} >= {cut:.4g}")
    text = ", ".join(str(v) for v in cut)
    return (f"{col} in {{{text}}}") if left else (f"{col} not in {{{text}}}")


class Node:
    _next = 0

    def __init__(self, idx, rule, parent=None):
        self.idx = idx
        self.rule = rule            # the condition that leads here from the parent
        self.parent = parent
        self.children = []
        self.split = None           # (column, cut, g2) once split
        self.id = Node._next
        Node._next += 1

    def path(self):
        parts, node = [], self
        while node is not None and node.rule:
            parts.append(node.rule)
            node = node.parent
        return list(reversed(parts)) or ["(all rows)"]


def grow(df, inputs, target, classes, numeric_cols, n_splits):
    """Grow greedily, one split at a time, always at the leaf whose best cut has
    the largest G squared. Returns the root, the leaves in order and how many
    splits were actually made."""
    Node._next = 0
    root = Node(df.index, "")
    leaves = [root]
    cache = {}
    made = 0
    while made < n_splits:
        best = None
        for leaf in leaves:
            if len(np.unique(df.loc[leaf.idx, target])) < 2:
                continue
            if leaf.id not in cache:
                c = candidates(df, leaf.idx, inputs, target, classes, numeric_cols)
                cache[leaf.id] = c[0] if c else None
            top = cache[leaf.id]
            if top is None or top["g2"] <= 1e-12:
                continue
            if best is None or top["g2"] > best[1]["g2"] + 1e-12:
                best = (leaf, top)
        if best is None:
            break
        leaf, top = best
        num = top["column"] in numeric_cols
        left = Node(top["left"], describe_cut(top["column"], top["cut"], True, num), leaf)
        right = Node(top["right"], describe_cut(top["column"], top["cut"], False, num), leaf)
        leaf.children = [left, right]
        leaf.split = (top["column"], top["cut"], top["g2"])
        pos = leaves.index(leaf)
        leaves[pos:pos + 1] = [left, right]
        made += 1
    return root, leaves, made


def log_likelihood(df, leaves, target, classes):
    ll = 0.0
    for leaf in leaves:
        counts = class_counts(df.loc[leaf.idx, target].to_numpy(), classes)
        n = counts.sum()
        for c in counts:
            if c > 0:
                ll += c * np.log(c / n)
    return ll


def entropy_r2(df, leaves, target, classes):
    ll0 = log_likelihood(df, [Node(df.index, "")], target, classes)
    if ll0 == 0:
        return 0.0
    return 1.0 - log_likelihood(df, leaves, target, classes) / ll0


def prepare(df, target, inputs):
    """Drop rows with a missing value in a used column; classify the inputs."""
    used = [target] + list(inputs)
    before = len(df)
    clean = df.dropna(subset=used).copy()
    dropped = before - len(clean)
    clean[target] = clean[target].astype(str)
    numeric_cols = {c for c in inputs if pd.api.types.is_numeric_dtype(clean[c])}
    for c in inputs:
        if c not in numeric_cols:
            clean[c] = clean[c].astype(str)
    classes = sorted(clean[target].unique().tolist())
    return clean.reset_index(drop=True), dropped, numeric_cols, classes


def predictions(df, leaves, target, classes):
    pred = pd.Series(index=df.index, dtype=object)
    probs = pd.DataFrame(index=df.index, columns=classes, dtype=float)
    for leaf in leaves:
        counts = class_counts(df.loc[leaf.idx, target].to_numpy(), classes)
        share = counts / counts.sum()
        pred.loc[leaf.idx] = classes[int(np.argmax(counts))]
        for c, s in zip(classes, share):
            probs.loc[leaf.idx, c] = s
    return pred, probs


def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


# ----------------------------------------------------------------- self test
def selftest(path):
    df = read_table(path)
    target = "return" if "return" in df.columns else df.columns[0]
    inputs = [c for c in df.columns if c != target]
    data, dropped, numeric_cols, classes = prepare(df, target, inputs)
    print(f"file {path}: {len(data)} rows, {dropped} dropped, classes {classes}")
    print("root candidates")
    for r in candidates(data, data.index, inputs, target, classes, numeric_cols):
        cut = f"{r['cut']:.4f}" if r["column"] in numeric_cols else str(r["cut"])
        print(f"  {r['column']:22} {cut:52} G2 {r['g2']:7.3f}  logworth {r['logworth']:6.3f}"
              f"  left {len(r['left'])} right {len(r['right'])}")
    print("split history")
    for n in range(0, 9):
        root, leaves, made = grow(data, inputs, target, classes, numeric_cols, n)
        pred, _ = predictions(data, leaves, target, classes)
        acc = (pred == data[target]).mean()
        print(f"  {made} splits  entropy R2 {entropy_r2(data, leaves, target, classes):.3f}"
              f"  accuracy {acc:.2f}  leaves {len(leaves)}")
    root, leaves, made = grow(data, inputs, target, classes, numeric_cols, 4)
    print("leaf report at 4 splits")
    for leaf in leaves:
        counts = class_counts(data.loc[leaf.idx, target].to_numpy(), classes)
        print(f"  {dict(zip(classes, counts.tolist()))}  " + " AND ".join(leaf.path()))


# ----------------------------------------------------------------- the window
if not SELFTEST:
    class App(tk.Tk):
        def __init__(self, path=None):
            super().__init__()
            self.title("Decision tree explorer")
            self.geometry("1400x860")
            self.raw = None
            self.data = None
            self.leaves = []
            self.root_node = None
            self.classes = []
            self.numeric_cols = set()
            self._build()
            if path:
                self.load(path)

        # layout
        def _build(self):
            side = ttk.Frame(self, padding=8)
            side.pack(side="left", fill="y")
            ttk.Button(side, text="Open file...", command=self.open_file).pack(fill="x")
            self.file_label = ttk.Label(side, text="no file", wraplength=250)
            self.file_label.pack(fill="x", pady=(4, 10))

            ttk.Label(side, text="Target column").pack(anchor="w")
            self.target_var = tk.StringVar()
            self.target_box = ttk.Combobox(side, textvariable=self.target_var, state="readonly")
            self.target_box.pack(fill="x")
            self.target_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_inputs())

            ttk.Label(side, text="Input columns").pack(anchor="w", pady=(10, 0))
            self.input_list = tk.Listbox(side, selectmode="multiple", height=12, exportselection=False)
            self.input_list.pack(fill="x")

            ttk.Label(side, text="Number of splits").pack(anchor="w", pady=(10, 0))
            self.splits_var = tk.IntVar(value=1)
            self.splits_box = ttk.Spinbox(side, from_=0, to=50, increment=1, width=6,
                                          textvariable=self.splits_var, command=self.run)
            self.splits_box.pack(anchor="w")
            self.splits_box.bind("<Return>", lambda e: self.run())

            ttk.Button(side, text="Grow tree", command=self.run).pack(fill="x", pady=(10, 0))

            ttk.Label(side, text="Leaf for the candidates table").pack(anchor="w", pady=(14, 0))
            self.leaf_var = tk.StringVar()
            self.leaf_box = ttk.Combobox(side, textvariable=self.leaf_var, state="readonly", width=30)
            self.leaf_box.pack(fill="x")
            self.leaf_box.bind("<<ComboboxSelected>>", lambda e: self.show_candidates())

            ttk.Button(side, text="Save predictions...", command=self.save).pack(fill="x", pady=(14, 0))
            self.status = ttk.Label(side, text="", wraplength=250, foreground="#444")
            self.status.pack(fill="x", pady=(14, 0))

            self.tabs = ttk.Notebook(self)
            self.tabs.pack(side="left", fill="both", expand=True)
            self.fig_tree, self.canvas_tree = self._figure_tab("Tree")
            self.cand_text = self._text_tab("Candidates")
            self.fig_hist, self.canvas_hist = self._figure_tab("Split history")
            self.leaf_text = self._text_tab("Leaf report")
            self.conf_text = self._text_tab("Confusion table")

        def _figure_tab(self, name):
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=name)
            fig = Figure(figsize=(10, 7), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            return fig, canvas

        def _text_tab(self, name):
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=name)
            text = tk.Text(frame, font=("Consolas", 11), wrap="none")
            text.pack(fill="both", expand=True)
            text.tag_configure("head", font=("Consolas", 11, "bold"))
            for i, col in enumerate(CLASS_COLORS):
                text.tag_configure(f"c{i}", foreground=col)
            return text

        # data
        def open_file(self):
            path = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.xlsx"), ("All", "*.*")])
            if path:
                self.load(path)

        def load(self, path):
            try:
                self.raw = read_table(path)
            except Exception as exc:
                messagebox.showerror("Could not read the file", str(exc))
                return
            self.file_label.config(text=os.path.basename(path))
            cols = list(self.raw.columns)
            self.target_box["values"] = cols
            self.target_var.set("return" if "return" in cols else cols[0])
            self.refresh_inputs(select_all=True)

        def refresh_inputs(self, select_all=True):
            target = self.target_var.get()
            self.input_list.delete(0, "end")
            for c in self.raw.columns:
                if c != target:
                    self.input_list.insert("end", c)
            if select_all:
                self.input_list.select_set(0, "end")

        def run(self):
            if self.raw is None:
                return
            target = self.target_var.get()
            inputs = [self.input_list.get(i) for i in self.input_list.curselection()]
            if not inputs:
                self.status.config(text="Choose at least one input column.")
                return
            self.data, dropped, self.numeric_cols, self.classes = prepare(self.raw, target, inputs)
            self.target, self.inputs = target, inputs
            asked = int(self.splits_var.get())
            self.root_node, self.leaves, made = grow(self.data, inputs, target, self.classes,
                                                     self.numeric_cols, asked)
            note = f"{len(self.data)} rows used, {dropped} dropped for missing values."
            if made < asked:
                note += (f" You asked for {asked} splits and the tree stopped at {made}, "
                         f"because no leaf has a cut left that separates the classes. "
                         f"Your setting is unchanged.")
            self.status.config(text=note)
            self.made = made
            self.leaf_box["values"] = [f"leaf {i + 1}  (n={len(l.idx)})" for i, l in enumerate(self.leaves)]
            self.leaf_box.current(0)
            self.draw_tree()
            self.show_candidates()
            self.draw_history()
            self.show_leaves()
            self.show_confusion()

        # outputs
        def draw_tree(self):
            fig = self.fig_tree
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            positions, depth = {}, {}
            leaf_order = {id(l): i for i, l in enumerate(self.leaves)}

            def place(node, d):
                depth[id(node)] = d
                if not node.children:
                    positions[id(node)] = leaf_order[id(node)]
                    return positions[id(node)]
                xs = [place(ch, d + 1) for ch in node.children]
                positions[id(node)] = sum(xs) / len(xs)
                return positions[id(node)]

            place(self.root_node, 0)
            max_d = max(depth.values()) if depth else 0
            n_leaves = max(len(self.leaves), 1)

            def xy(node):
                return (positions[id(node)] + 0.5) / n_leaves, 1 - (depth[id(node)] + 0.5) / (max_d + 1)

            def draw(node):
                x, y = xy(node)
                for ch in node.children:
                    cx, cy = xy(ch)
                    ax.plot([x, cx], [y, cy], color="#999", lw=1, zorder=1)
                    rule = ch.rule
                    if len(rule) > 28:
                        rule = rule.replace(" {", "\n{").replace(", ", ",\n")
                    ax.text(x + 0.62 * (cx - x), y + 0.62 * (cy - y), rule, fontsize=8, ha="center",
                            va="center", bbox=dict(fc="white", ec="none", pad=1), zorder=2)
                    draw(ch)
                counts = class_counts(self.data.loc[node.idx, self.target].to_numpy(), self.classes)
                label = "\n".join(f"{c}: {n}" for c, n in zip(self.classes, counts))
                colour = CLASS_COLORS[int(np.argmax(counts)) % len(CLASS_COLORS)]
                ax.text(x, y, f"n={counts.sum()}\n{label}", fontsize=9, ha="center", va="center",
                        bbox=dict(fc="white", ec=colour, lw=2, boxstyle="round,pad=0.4"), zorder=3)

            draw(self.root_node)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{self.made} splits, {len(self.leaves)} leaves", fontsize=11)
            self.canvas_tree.draw()

        def show_candidates(self):
            if not self.leaves:
                return
            i = self.leaf_box.current()
            leaf = self.leaves[max(i, 0)]
            t = self.cand_text
            t.delete("1.0", "end")
            t.insert("end", f"Leaf {i + 1}: " + " AND ".join(leaf.path()) + "\n", "head")
            counts = class_counts(self.data.loc[leaf.idx, self.target].to_numpy(), self.classes)
            t.insert("end", "Counts  " + "   ".join(f"{c}: {n}" for c, n in zip(self.classes, counts)) + "\n\n")
            rows = candidates(self.data, leaf.idx, self.inputs, self.target, self.classes, self.numeric_cols)
            t.insert("end", f"{'column':24}{'best cut':46}{'G squared':>12}{'logworth':>11}\n", "head")
            for r in rows:
                if r["column"] in self.numeric_cols:
                    cut = f"< {r['cut']:.4g}"
                else:
                    cut = "{" + ", ".join(str(v) for v in r["cut"]) + "}"
                t.insert("end", f"{r['column']:24}{cut[:44]:46}{r['g2']:12.3f}{r['logworth']:11.3f}\n")
            if not rows:
                t.insert("end", "No column offers a cut at this leaf.\n")

        def draw_history(self):
            fig = self.fig_hist
            fig.clear()
            ax = fig.add_subplot(111)
            xs, ys = [], []
            n = 0
            while True:
                _, leaves, made = grow(self.data, self.inputs, self.target, self.classes,
                                       self.numeric_cols, n)
                if made < n:
                    break
                xs.append(made)
                ys.append(entropy_r2(self.data, leaves, self.target, self.classes))
                n += 1
                if n > 40:
                    break
            ax.plot(xs, ys, marker="o", color="#2c6fbb")
            ax.axvline(self.made, color="black", lw=1.5)
            ax.set_xlabel("Number of splits")
            ax.set_ylabel("Entropy R squared")
            ax.set_ylim(0, 1.02)
            ax.set_xticks(xs)
            ax.grid(alpha=0.3)
            cur = ys[xs.index(self.made)] if self.made in xs else None
            if cur is not None:
                ax.set_title(f"Entropy R squared {cur:.3f} at {self.made} splits", fontsize=11)
            self.canvas_hist.draw()

        def show_leaves(self):
            t = self.leaf_text
            t.delete("1.0", "end")
            header = f"{'leaf':6}" + "".join(f"{c:>8}" for c in self.classes) + f"{'predicted':>12}   rule\n"
            t.insert("end", header, "head")
            for i, leaf in enumerate(self.leaves):
                counts = class_counts(self.data.loc[leaf.idx, self.target].to_numpy(), self.classes)
                k = int(np.argmax(counts))
                t.insert("end", f"{i + 1:<6}" + "".join(f"{n:>8}" for n in counts))
                t.insert("end", f"{self.classes[k]:>12}", f"c{k % len(CLASS_COLORS)}")
                t.insert("end", "   " + " AND ".join(leaf.path()) + "\n")

        def show_confusion(self):
            pred, _ = predictions(self.data, self.leaves, self.target, self.classes)
            actual = self.data[self.target]
            t = self.conf_text
            t.delete("1.0", "end")
            t.insert("end", "Rows are the actual class, columns the predicted class.\n\n", "head")
            t.insert("end", f"{'actual':12}" + "".join(f"{c:>20}" for c in self.classes) + "\n", "head")
            for a in self.classes:
                row = actual == a
                total = int(row.sum())
                t.insert("end", f"{a:12}")
                for p in self.classes:
                    n = int(((pred == p) & row).sum())
                    pct = 100.0 * n / total if total else 0.0
                    t.insert("end", f"{n:>10} ({pct:5.1f}%)  ")
                t.insert("end", "\n")
            acc = (pred == actual).mean()
            t.insert("end", f"\nOverall accuracy {acc:.3f}, {int((pred == actual).sum())} of {len(actual)} rows\n")

        def save(self):
            if not self.leaves:
                return
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            pred, probs = predictions(self.data, self.leaves, self.target, self.classes)
            out = self.data.copy()
            out["predicted"] = pred
            for c in self.classes:
                out[f"prob_{c}"] = probs[c].round(4)
            out.to_csv(path, index=False)
            self.status.config(text=f"Saved {len(out)} rows to {os.path.basename(path)}.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    if SELFTEST:
        selftest(args[0] if args else "freshmen.csv")
    else:
        App(args[0] if args else None).mainloop()
