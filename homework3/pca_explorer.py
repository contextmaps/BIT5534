"""
Principal component analysis in one window.

BIT 5534 week 4. This is the tool the lecture asks you to build. It is here so
that a build that did not finish in class does not cost you the rest of the
session. It came out of the specification on the page and it does what that
specification asked for.

    python pca_explorer.py
    python pca_explorer.py plant_monitors.csv

Double clicking works too. It needs pandas, numpy, matplotlib and tkinter.
On Windows:  py -m pip install pandas numpy matplotlib openpyxl
On a Mac:    python3 -m pip install pandas numpy matplotlib openpyxl
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

PALETTE = ["#d23b2e", "#2c6fbb", "#2f9e44", "#e8890c", "#7048e8",
           "#0ca678", "#c2255c", "#495057", "#a61e4d", "#1864ab",
           "#5c940d", "#d9480f"]


# ----------------------------------------------------------------- the maths


def principal_components(frame):
    """Correlation matrix PCA on a frame of numeric columns.

    Returns the eigenvalues largest first, the eigenvectors in the same
    order, the scores, and the loadings. Loadings are the eigenvector
    components scaled by the square root of their eigenvalue, which is what
    puts every variable inside the unit circle and makes a loading readable
    as the correlation between that variable and that component.
    """
    values = frame.to_numpy(dtype=float)
    means = values.mean(axis=0)
    spreads = values.std(axis=0, ddof=1)
    if np.any(spreads == 0):
        dead = [c for c, s in zip(frame.columns, spreads) if s == 0]
        raise ValueError("these columns never change, so they cannot be "
                         "standardized: " + ", ".join(map(str, dead)))

    standardized = (values - means) / spreads
    correlation = np.corrcoef(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    scores = standardized @ eigenvectors
    loadings = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))
    return eigenvalues, eigenvectors, scores, loadings


def numeric_columns(frame):
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def read_table(path):
    lowered = str(path).lower()
    if lowered.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


# ------------------------------------------------------------------ drawing


def draw_eigenvalues(ax_bars, ax_table, eigenvalues):
    total = eigenvalues.sum()
    percent = 100.0 * eigenvalues / total
    cumulative = np.cumsum(percent)
    rows = np.arange(1, len(eigenvalues) + 1)
    size = 9 if len(rows) <= 22 else 7

    ax_bars.barh(rows, eigenvalues, color="#2c6fbb", edgecolor="black",
                 height=0.7, linewidth=0.4)
    ax_bars.set_ylim(len(rows) + 0.6, -0.15)
    ax_bars.set_yticks(rows)
    ax_bars.set_yticklabels([f"PC{r}" for r in rows], fontsize=size)
    ax_bars.set_xlabel("eigenvalue")
    ax_bars.set_title("Eigenvalues, with the cumulative percent over them",
                      fontsize=11, pad=34)
    ax_bars.grid(axis="x", linestyle="--", alpha=0.35)

    over = ax_bars.twiny()
    over.plot(cumulative, rows, color="#e8890c", marker="o", markersize=3.5,
              linewidth=1.6)
    over.set_xlim(0, 105)
    over.set_ylim(ax_bars.get_ylim())
    over.set_xlabel("cumulative percent", color="#e8890c", fontsize=9,
                    labelpad=1)
    over.tick_params(axis="x", colors="#e8890c", labelsize=8)

    ax_table.set_ylim(ax_bars.get_ylim())
    ax_table.set_xlim(0, 1)
    ax_table.axis("off")
    headers = ("Eigenvalue", "Percent", "Cum percent")
    for x, head in zip((0.18, 0.55, 0.92), headers):
        ax_table.text(x, 0.3, head, ha="right", va="center",
                      fontsize=size, fontweight="bold")
    for row, value, pct, cum in zip(rows, eigenvalues, percent, cumulative):
        ax_table.text(0.18, row, f"{value:.4f}", ha="right", va="center",
                      fontsize=size)
        ax_table.text(0.55, row, f"{pct:.2f}", ha="right", va="center",
                      fontsize=size)
        ax_table.text(0.92, row, f"{cum:.2f}", ha="right", va="center",
                      fontsize=size)
    return percent, cumulative


def draw_scores(ax, scores, percent, first, second, labels, label_name):
    ax.axhline(0, color="#adb5bd", linewidth=0.8)
    ax.axvline(0, color="#adb5bd", linewidth=0.8)
    x = scores[:, first]
    y = scores[:, second]

    if labels is None:
        ax.scatter(x, y, s=9, c="#495057", alpha=0.7, linewidths=0)
    else:
        groups = pd.Series(labels).astype("object")
        distinct = list(pd.unique(groups.dropna()))
        if len(distinct) <= len(PALETTE):
            for i, name in enumerate(distinct):
                mask = (groups == name).to_numpy()
                ax.scatter(x[mask], y[mask], s=9, c=PALETTE[i % len(PALETTE)],
                           alpha=0.75, linewidths=0, label=str(name))
            ax.legend(title=label_name, fontsize=8, title_fontsize=8,
                      loc="best", framealpha=0.85)
        else:
            shade = pd.to_numeric(groups, errors="coerce").to_numpy(dtype=float)
            ax.scatter(x, y, s=9, c=shade, cmap="viridis", alpha=0.8,
                       linewidths=0)

    ax.set_xlabel(f"PC{first + 1}  ({percent[first]:.1f}%)")
    ax.set_ylabel(f"PC{second + 1}  ({percent[second]:.1f}%)")
    ax.set_title("Score plot, one point per row", fontsize=11)


def draw_loadings(ax, loadings, names, percent, first, second):
    circle = plt_circle(1.0)
    ax.plot(circle[0], circle[1], color="#ced4da", linewidth=1.0)
    ax.axhline(0, color="#adb5bd", linewidth=0.8)
    ax.axvline(0, color="#adb5bd", linewidth=0.8)

    for name, row in zip(names, loadings):
        ax.annotate("", xy=(row[first], row[second]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#2f9e44",
                                    linewidth=1.2))
        ax.text(row[first] * 1.08, row[second] * 1.08, str(name),
                fontsize=8, ha="center", va="center")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel(f"PC{first + 1}  ({percent[first]:.1f}%)")
    ax.set_ylabel(f"PC{second + 1}  ({percent[second]:.1f}%)")
    ax.set_title("Loadings plot, one arrow per variable", fontsize=11)


def plt_circle(radius, steps=200):
    angle = np.linspace(0, 2 * np.pi, steps)
    return radius * np.cos(angle), radius * np.sin(angle)


# ------------------------------------------------------------------ the app


class Explorer(tk.Tk):
    def __init__(self, initial_path=None):
        super().__init__()
        self.title("Principal component analysis")
        self.geometry("1380x860")

        self.frame = None
        self.result = None
        self.path = None

        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_top()
        self._build_side()
        self._build_canvas()

        if initial_path:
            self.load(initial_path)

    # ---- widgets

    def _build_top(self):
        bar = ttk.Frame(self, padding=(10, 8))
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.columnconfigure(1, weight=1)
        ttk.Button(bar, text="Choose file...", command=self.choose
                   ).grid(row=0, column=0)
        self.path_label = ttk.Label(bar, text="no file chosen",
                                    foreground="#666666")
        self.path_label.grid(row=0, column=1, sticky="w", padx=10)

    def _build_side(self):
        side = ttk.Frame(self, padding=(10, 0, 10, 10))
        side.grid(row=1, column=0, sticky="ns")
        side.rowconfigure(1, weight=1)

        ttk.Label(side, text="Variables in the analysis",
                  font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0,
                                                           sticky="w")
        holder = ttk.Frame(side)
        holder.grid(row=1, column=0, sticky="ns", pady=(4, 8))
        holder.rowconfigure(0, weight=1)
        self.var_list = tk.Listbox(holder, selectmode="extended", width=26,
                                   exportselection=False, activestyle="none")
        self.var_list.grid(row=0, column=0, sticky="ns")
        bar = ttk.Scrollbar(holder, orient="vertical",
                            command=self.var_list.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.var_list.configure(yscrollcommand=bar.set)
        ttk.Label(side, text="Only numeric columns are listed.",
                  foreground="#666666").grid(row=2, column=0, sticky="w")

        picks = ttk.Frame(side)
        picks.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        ttk.Button(picks, text="All", width=6,
                   command=lambda: self.var_list.selection_set(0, "end")
                   ).grid(row=0, column=0)
        ttk.Button(picks, text="None", width=6,
                   command=lambda: self.var_list.selection_clear(0, "end")
                   ).grid(row=0, column=1, padx=4)

        ttk.Label(side, text="Label column, optional",
                  font=("TkDefaultFont", 10, "bold")).grid(row=4, column=0,
                                                           sticky="w")
        self.label_choice = ttk.Combobox(side, state="readonly", width=24)
        self.label_choice.grid(row=5, column=0, sticky="ew", pady=(4, 2))
        ttk.Label(side, text="It colors the points and never enters\n"
                             "the analysis.",
                  foreground="#666666").grid(row=6, column=0, sticky="w",
                                             pady=(0, 10))

        ttk.Button(side, text="Run the analysis", command=self.run
                   ).grid(row=7, column=0, sticky="ew", pady=(0, 12))

        ttk.Label(side, text="Score plot axes",
                  font=("TkDefaultFont", 10, "bold")).grid(row=8, column=0,
                                                           sticky="w")
        axes_row = ttk.Frame(side)
        axes_row.grid(row=9, column=0, sticky="ew", pady=4)
        ttk.Label(axes_row, text="across").grid(row=0, column=0)
        self.x_choice = ttk.Combobox(axes_row, state="readonly", width=7)
        self.x_choice.grid(row=0, column=1, padx=(4, 10))
        ttk.Label(axes_row, text="up").grid(row=0, column=2)
        self.y_choice = ttk.Combobox(axes_row, state="readonly", width=7)
        self.y_choice.grid(row=0, column=3, padx=4)
        self.x_choice.bind("<<ComboboxSelected>>", lambda _e: self.redraw())
        self.y_choice.bind("<<ComboboxSelected>>", lambda _e: self.redraw())

        self.status = ttk.Label(side, text="", wraplength=230,
                                foreground="#1864ab")
        self.status.grid(row=10, column=0, sticky="w", pady=(12, 0))

    def _build_canvas(self):
        holder = ttk.Frame(self)
        holder.grid(row=1, column=1, sticky="nsew")
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(11.2, 8.0), dpi=100)
        grid = self.figure.add_gridspec(
            2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1],
            left=0.07, right=0.97, top=0.89, bottom=0.08,
            wspace=0.30, hspace=0.42)
        inner = grid[0, :].subgridspec(1, 2, width_ratios=[2.4, 1], wspace=0.02)
        self.ax_bars = self.figure.add_subplot(inner[0, 0])
        self.ax_table = self.figure.add_subplot(inner[0, 1])
        self.ax_scores = self.figure.add_subplot(grid[1, 0])
        self.ax_loadings = self.figure.add_subplot(grid[1, 1])
        for ax in (self.ax_bars, self.ax_table, self.ax_scores,
                   self.ax_loadings):
            ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.figure, master=holder)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_holder = ttk.Frame(holder)
        toolbar_holder.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_holder).update()

    # ---- behavior

    def choose(self):
        chosen = filedialog.askopenfilename(
            parent=self, title="Choose a data file",
            filetypes=[("Data files", "*.csv *.xlsx *.xlsm *.xls"),
                       ("CSV", "*.csv"), ("Excel", "*.xlsx *.xlsm *.xls"),
                       ("All files", "*.*")])
        if chosen:
            self.load(chosen)

    def load(self, path):
        try:
            frame = read_table(path)
        except Exception as problem:
            messagebox.showerror("Could not read that file", str(problem))
            return
        numeric = numeric_columns(frame)
        if len(numeric) < 2:
            messagebox.showerror(
                "Not enough to work with",
                "This file has fewer than two numeric columns.")
            return

        self.frame = frame
        self.path = path
        self.result = None
        self.path_label.configure(text=f"{path}    "
                                       f"{len(frame)} rows, "
                                       f"{len(frame.columns)} columns")
        self.var_list.delete(0, "end")
        for name in numeric:
            self.var_list.insert("end", name)
        self.var_list.selection_set(0, "end")
        self.label_choice.configure(values=["(none)"] + list(frame.columns))
        self.label_choice.set("(none)")
        self.x_choice.configure(values=[])
        self.y_choice.configure(values=[])
        self.status.configure(text="Choose the variables, then run the "
                                   "analysis.")
        for ax in (self.ax_bars, self.ax_table, self.ax_scores,
                   self.ax_loadings):
            ax.clear()
            ax.set_axis_off()
        self.canvas.draw_idle()

    def selected_variables(self):
        chosen = [self.var_list.get(i) for i in self.var_list.curselection()]
        label = self.label_choice.get()
        dropped = None
        if label != "(none)" and label in chosen:
            chosen = [c for c in chosen if c != label]
            dropped = label
        return chosen, dropped

    def run(self):
        if self.frame is None:
            messagebox.showinfo("No file yet", "Choose a data file first.")
            return
        chosen, dropped = self.selected_variables()
        if len(chosen) < 2:
            messagebox.showinfo(
                "Pick at least two",
                "Principal component analysis needs two or more variables.")
            return
        try:
            values = principal_components(self.frame[chosen])
        except Exception as problem:
            messagebox.showerror("The analysis stopped", str(problem))
            return

        eigenvalues, _vectors, scores, loadings = values
        self.result = dict(eigenvalues=eigenvalues, scores=scores,
                           loadings=loadings, names=chosen)

        count = len(eigenvalues)
        options = [f"PC{i}" for i in range(1, count + 1)]
        self.x_choice.configure(values=options)
        self.y_choice.configure(values=options)
        self.x_choice.set("PC1")
        self.y_choice.set("PC2" if count > 1 else "PC1")

        note = (f"{len(self.frame)} rows and {count} variables. "
                f"The eigenvalues sum to {eigenvalues.sum():.4f}.")
        if dropped:
            note += f" {dropped} was left out because it is the label."
        self.status.configure(text=note)
        self.redraw()

    def redraw(self):
        if not self.result:
            return
        eigenvalues = self.result["eigenvalues"]
        scores = self.result["scores"]
        loadings = self.result["loadings"]
        names = self.result["names"]
        count = len(eigenvalues)

        first = self._axis_index(self.x_choice.get(), 0, count)
        second = self._axis_index(self.y_choice.get(), min(1, count - 1), count)
        if first == second and count > 1:
            self.status.configure(
                text="Both axes name the same component, so the score plot "
                     "would be a straight line. Pick a different one.")

        for ax in (self.ax_bars, self.ax_table, self.ax_scores,
                   self.ax_loadings):
            ax.clear()
            ax.set_axis_on()
        for stale in list(self.figure.axes):
            if stale not in (self.ax_bars, self.ax_table, self.ax_scores,
                             self.ax_loadings):
                stale.remove()

        percent, _cumulative = draw_eigenvalues(self.ax_bars, self.ax_table,
                                                eigenvalues)
        labels = None
        label_name = self.label_choice.get()
        if label_name != "(none)":
            labels = self.frame[label_name].to_numpy()
        draw_scores(self.ax_scores, scores, percent, first, second,
                    labels, label_name)
        draw_loadings(self.ax_loadings, loadings, names, percent,
                      first, second)
        self.canvas.draw_idle()

    @staticmethod
    def _axis_index(text, fallback, count):
        try:
            index = int(str(text).replace("PC", "")) - 1
        except ValueError:
            return fallback
        return index if 0 <= index < count else fallback


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else None
    Explorer(start).mainloop()


if __name__ == "__main__":
    main()
