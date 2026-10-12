#!/usr/bin/env python3
"""
Single-window support vector machine explorer.

BIT 5534 lec8_1 demonstration and backup tool, written 2026-10-12 against the
same specification the students send, so that what the room sees on the
projector is what their own tools should show.

A category target is fitted by scikit-learn's SVC, a number target by its SVR.
The kernel is linear or radial basis function (RBF), with the cost C and, for
the RBF kernel, gamma, in the form K(x, x') = exp(-gamma * ||x - x'||^2). A
number target is fitted on the target standardized with the train rows, so
epsilon is in standard deviations of the target, and every figure is reported
in the target's own units.

Every row carries a role: train, validation or test, or for k folds
validation_1 to validation_k plus an optional test. A column the file already
has may use Training and Validation as its labels. The model is fitted on the
train rows only. Test rows play no part in any choice and are only reported.

    python svm_explorer.py                    opens a file dialog
    python svm_explorer.py chess.csv          opens that file
    python svm_explorer.py --selftest FILE    prints reference numbers, opens no window

Required packages: pandas, numpy, matplotlib, scikit-learn (and openpyxl for .xlsx).
"""

import os
import secrets
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR

SELFTEST = "--selftest" in sys.argv
if not SELFTEST:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

ROLE_COLORS = {"train": "#2c6fbb", "validation": "#d68910", "test": "#8e44ad"}
CLASS_COLORS = ["#c0392b", "#2c6fbb", "#27ae60", "#d68910", "#8e44ad", "#16a085"]
DEFAULT_C_GRID = "0.1 1 10 100 1000"
DEFAULT_G_GRID = "0.01 0.1 1 10 100"


# ----------------------------------------------------------------- data
def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def default_kind(series):
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        return "number"
    return "category"


def encode_inputs(df, inputs):
    """Numeric inputs stay as they are. A non-numeric input becomes one 0 or 1 column
    per level after the first, so a column with L levels adds L minus 1 inputs."""
    parts, names = [], []
    for c in inputs:
        if pd.api.types.is_numeric_dtype(df[c]):
            parts.append(df[c].astype(float).to_numpy()[:, None])
            names.append(c)
        else:
            levels = sorted(df[c].astype(str).unique())
            for lv in levels[1:]:
                parts.append((df[c].astype(str) == lv).astype(float).to_numpy()[:, None])
                names.append(f"{c}={lv}")
    return np.hstack(parts), names


def draw_seed(text):
    """An entered seed is used as it is. An empty box draws one, and the tool reports it."""
    text = (text or "").strip()
    if text:
        return int(text), False
    return secrets.randbelow(100000), True


ROLE_ALIASES = {"train": "train", "training": "train", "validation": "validation",
                "valid": "validation", "test": "test", "testing": "test"}


def normal_role(v):
    s = str(v).strip().lower()
    if s.startswith("validation_"):
        return s
    return ROLE_ALIASES.get(s)


def role_like(series):
    vals = set(series.dropna().astype(str).unique())
    return bool(vals) and all(normal_role(v) is not None for v in vals)


def make_roles(n, method, val_share, test_share, k, seed):
    """Assign every row a role. Shares are of all rows. Counts are rounded, and the
    rows are shuffled by the seed, so a seed reproduces the same column."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    roles = np.empty(n, dtype=object)
    n_test = int(round(test_share * n))
    roles[order[:n_test]] = "test"
    rest = order[n_test:]
    if method == "holdout":
        n_val = int(round(val_share * n))
        roles[rest[:n_val]] = "validation"
        roles[rest[n_val:]] = "train"
    else:
        for i, idx in enumerate(rest):
            roles[idx] = f"validation_{i % k + 1}"
    return roles


def folds_of(roles):
    """The validation sets this role column defines, as (label, validation mask, train mask)."""
    roles = np.array([normal_role(r) for r in roles], dtype=object).astype(str)
    test = roles == "test"
    fold_labels = sorted({r for r in roles if r.startswith("validation_")},
                         key=lambda s: int(s.split("_")[1]))
    if fold_labels:
        return [(lab, roles == lab, ~test & (roles != lab)) for lab in fold_labels], test
    val = roles == "validation"
    return [("validation", val, roles == "train")], test


class Prepared:
    pass


def prepare(df, target, inputs, kind, role_col):
    used = [target] + inputs + ([role_col] if role_col else [])
    before = len(df)
    clean = df.dropna(subset=used).reset_index(drop=True)
    p = Prepared()
    p.data, p.dropped = clean, before - len(clean)
    p.X, p.names = encode_inputs(clean, inputs)
    p.kind = kind
    p.target_name = target
    if kind == "number":
        p.y = clean[target].astype(float).to_numpy()
        p.classes = None
    else:
        lab = clean[target].astype(str)
        p.classes = sorted(lab.unique())
        p.y = lab.to_numpy()
    return p


def scale_from(X, rows, on):
    if not on:
        return np.zeros(X.shape[1]), np.ones(X.shape[1])
    mu = X[rows].mean(axis=0)
    sd = X[rows].std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


# ----------------------------------------------------------------- the model
class Settings:
    def __init__(self, kernel="rbf", C=1.0, gamma=None, epsilon=0.1, standardize=True):
        self.kernel, self.C, self.gamma, self.epsilon, self.standardize = kernel, C, gamma, epsilon, standardize


def gamma_used(s, k):
    """An empty gamma box means 1 divided by the number of inputs, and the tool reports it."""
    return s.gamma if s.gamma is not None else 1.0 / k


class FoldFit:
    pass


def fit_fold(p, train, val, s):
    mu, sd = scale_from(p.X, train, s.standardize)
    Xs = (p.X - mu) / sd
    g = gamma_used(s, p.X.shape[1])
    f = FoldFit()
    f.mu, f.sd = mu, sd
    if p.kind == "number":
        ym, ys = p.y[train].mean(), p.y[train].std() or 1.0
        m = SVR(kernel=s.kernel, C=s.C, gamma=g, epsilon=s.epsilon)
        m.fit(Xs[train], (p.y[train] - ym) / ys)
        f.pred = m.predict(Xs) * ys + ym
        f.ym, f.ys = ym, ys
    else:
        m = SVC(kernel=s.kernel, C=s.C, gamma=g)
        m.fit(Xs[train], p.y[train])
        f.pred = m.predict(Xs)
    f.model = m
    sv = np.zeros(len(p.y), dtype=bool)
    sv[np.flatnonzero(train)[m.support_]] = True
    f.sv = sv
    f.n_sv = int(sv.sum())
    f.train_m = metrics(p, f.pred, train)
    f.val_m = metrics(p, f.pred, val) if val.any() else None
    return f


def metrics(p, pred, rows):
    if not rows.any():
        return None
    if p.kind == "number":
        y, yh = p.y[rows], pred[rows]
        sse = float(((y - yh) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        mse = sse / len(y)
        return {"n": int(rows.sum()), "r2": 1 - sse / sst if sst > 0 else float("nan"),
                "rmse": mse ** 0.5, "mse": mse, "choose": mse}
    y = p.y[rows]
    miss = float((pred[rows] != y).mean())
    return {"n": int(rows.sum()), "miss": miss, "acc": 1 - miss, "choose": miss}


class Result:
    pass


def run_model(p, roles, s):
    folds, test = folds_of(roles)
    r = Result()
    r.folds, r.fits, r.settings = folds, [], s
    for lab, val, train in folds:
        r.fits.append(fit_fold(p, train, val, s))
    r.test = test
    r.kfold = folds[0][0] != "validation"
    if not r.kfold:
        r.pred = r.fits[0].pred
    else:
        # Each validation row is predicted by the fold model that did not see it.
        # A test row is predicted by the fold models together: the average for a
        # number target, the majority vote for a category target.
        r.pred = r.fits[0].pred.copy()
        for (lab, val, train), f in zip(folds, r.fits):
            r.pred[val] = f.pred[val]
        if test.any():
            if p.kind == "number":
                r.pred[test] = (sum(f.pred for f in r.fits) / len(r.fits))[test]
            else:
                votes = np.array([f.pred for f in r.fits])
                for i in np.flatnonzero(test):
                    lv, ct = np.unique(votes[:, i], return_counts=True)
                    r.pred[i] = lv[ct.argmax()]
    r.gamma = gamma_used(s, p.X.shape[1])
    return r


def summary(p, r):
    """The validation figure a grid compares: the holdout figure, or the fold average."""
    key = "r2" if p.kind == "number" else "miss"
    vals = [f.val_m[key] for f in r.fits if f.val_m]
    trs = [f.train_m[key] for f in r.fits]
    return (float(np.mean(trs)), float(np.mean(vals)) if vals else float("nan"),
            float(np.mean([f.n_sv for f in r.fits])))


def run_grid(p, roles, base, Cs, gammas):
    """Fit every pair of C and gamma and return train figure, validation figure and
    support vector count for each. Test rows play no part."""
    out = []
    gl = gammas if base.kernel == "rbf" else [None]
    for g in gl:
        row = []
        for C in Cs:
            s = Settings(base.kernel, C, g, base.epsilon, base.standardize)
            row.append(summary(p, run_model(p, roles, s)))
        out.append(row)
    return gl, out


# ----------------------------------------------------------------- self test
def selftest(path):
    df = read_table(path)
    cols = list(df.columns)
    target = cols[0]
    inputs = [c for c in cols[1:] if not role_like(df[c])]
    kind = default_kind(df[target])
    p = prepare(df, target, inputs, kind, None)
    roles = make_roles(len(p.data), "holdout", 1 / 3, 0.0, 5, 123)
    print(f"file {path}: {len(p.data)} rows, target {target} ({kind}), inputs {p.names}")
    for kern, C, g in (("linear", 1, None), ("rbf", 1, None), ("rbf", 10, 1), ("rbf", 1000, 100)):
        r = run_model(p, roles, Settings(kern, C, g))
        f = r.fits[0]
        key = "r2" if kind == "number" else "miss"
        print(f"{kern:6} C {C:<6} gamma {r.gamma:<6.3g} train {f.train_m[key]:.3f}  "
              f"validation {f.val_m[key]:.3f}  support vectors {f.n_sv}")


# ----------------------------------------------------------------- the window
if not SELFTEST:
    class App(tk.Tk):
        def __init__(self, path=None):
            super().__init__()
            self.title("Support vector machine explorer")
            self.geometry("1440x900")
            self.raw = None
            self.result = None
            self.p = None
            self.grid_out = None
            self._build()
            if path:
                self.load(path)
            else:
                self.status.config(text="Open a .csv or .xlsx file to begin.")
                self.after(200, self.open_file)

        # ------------------------------------------------------------- layout
        def _build(self):
            # a side panel that scrolls when it is taller than the window
            holder = ttk.Frame(self)
            holder.pack(side="left", fill="y")
            cv = tk.Canvas(holder, width=300, highlightthickness=0)
            sb = ttk.Scrollbar(holder, orient="vertical", command=cv.yview)
            side = ttk.Frame(cv, padding=8)
            side.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
            cv.create_window((0, 0), window=side, anchor="nw")
            cv.configure(yscrollcommand=sb.set)
            cv.pack(side="left", fill="y", expand=False)
            sb.pack(side="left", fill="y")
            cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta / 120), "units"))

            ttk.Button(side, text="Open file...", command=self.open_file).pack(fill="x")
            self.file_label = ttk.Label(side, text="no file", wraplength=270)
            self.file_label.pack(fill="x", pady=(4, 8))

            ttk.Label(side, text="Target column").pack(anchor="w")
            self.target_var = tk.StringVar()
            self.target_box = ttk.Combobox(side, textvariable=self.target_var, state="readonly")
            self.target_box.pack(fill="x")
            self.target_box.bind("<<ComboboxSelected>>", lambda e: self.target_changed())
            kf = ttk.Frame(side)
            kf.pack(fill="x")
            self.kind_var = tk.StringVar(value="category")
            ttk.Radiobutton(kf, text="number", value="number", variable=self.kind_var).pack(side="left")
            ttk.Radiobutton(kf, text="category", value="category", variable=self.kind_var).pack(side="left")

            ttk.Label(side, text="Input columns").pack(anchor="w", pady=(8, 0))
            self.input_list = tk.Listbox(side, selectmode="multiple", height=8, exportselection=False)
            self.input_list.pack(fill="x")

            rf = ttk.LabelFrame(side, text="Validation column", padding=6)
            rf.pack(fill="x", pady=(8, 0))
            ttk.Label(rf, text="Use the column").grid(row=0, column=0, sticky="w")
            self.role_var = tk.StringVar(value="(make one)")
            self.role_box = ttk.Combobox(rf, textvariable=self.role_var, state="readonly", width=16)
            self.role_box.grid(row=0, column=1, sticky="ew")
            self.role_box.bind("<<ComboboxSelected>>", lambda e: self.show_roles())
            self.method_var = tk.StringVar(value="holdout")
            ttk.Radiobutton(rf, text="holdout", value="holdout", variable=self.method_var).grid(row=1, column=0, sticky="w")
            ttk.Radiobutton(rf, text="k folds", value="kfold", variable=self.method_var).grid(row=1, column=1, sticky="w")
            self.val_var = tk.StringVar(value="0.3333")
            self.test_var = tk.StringVar(value="0")
            self.k_var = tk.StringVar(value="5")
            self.role_seed_var = tk.StringVar(value="123")
            for i, (lab, var) in enumerate([("Validation share", self.val_var), ("Test share", self.test_var),
                                            ("k", self.k_var), ("Seed (empty = random)", self.role_seed_var)]):
                ttk.Label(rf, text=lab).grid(row=2 + i, column=0, sticky="w")
                ttk.Entry(rf, textvariable=var, width=10).grid(row=2 + i, column=1, sticky="w")
            ttk.Button(rf, text="Make the column", command=self.make_column).grid(row=6, column=0, sticky="ew", pady=(4, 0))
            ttk.Button(rf, text="Save file with it...", command=self.save_file).grid(row=6, column=1, sticky="ew", pady=(4, 0))

            mf = ttk.LabelFrame(side, text="Support vector machine", padding=6)
            mf.pack(fill="x", pady=(8, 0))
            ttk.Label(mf, text="Kernel").grid(row=0, column=0, sticky="w")
            self.kernel_var = tk.StringVar(value="rbf")
            kk = ttk.Frame(mf)
            kk.grid(row=0, column=1, sticky="w")
            ttk.Radiobutton(kk, text="RBF", value="rbf", variable=self.kernel_var).pack(side="left")
            ttk.Radiobutton(kk, text="linear", value="linear", variable=self.kernel_var).pack(side="left")
            self.C_var = tk.StringVar(value="1")
            self.gamma_var = tk.StringVar(value="")
            self.eps_var = tk.StringVar(value="0.1")
            for i, (lab, var) in enumerate([("Cost C", self.C_var), ("Gamma (empty = 1/inputs)", self.gamma_var),
                                            ("Epsilon (number target)", self.eps_var)]):
                ttk.Label(mf, text=lab).grid(row=1 + i, column=0, sticky="w")
                ttk.Entry(mf, textvariable=var, width=10).grid(row=1 + i, column=1, sticky="w")
            self.std_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(mf, text="Standardize inputs", variable=self.std_var).grid(row=4, column=0, columnspan=2, sticky="w")
            ttk.Label(mf, text="Epsilon is in standard deviations of the target.", foreground="#555",
                      wraplength=260).grid(row=5, column=0, columnspan=2, sticky="w")

            ttk.Button(side, text="Fit the model", command=self.fit).pack(fill="x", pady=(10, 0))

            gf = ttk.LabelFrame(side, text="Grid of C and gamma", padding=6)
            gf.pack(fill="x", pady=(8, 0))
            self.Cgrid_var = tk.StringVar(value=DEFAULT_C_GRID)
            self.Ggrid_var = tk.StringVar(value=DEFAULT_G_GRID)
            ttk.Label(gf, text="C values").grid(row=0, column=0, sticky="w")
            ttk.Entry(gf, textvariable=self.Cgrid_var, width=22).grid(row=0, column=1, sticky="w")
            ttk.Label(gf, text="gamma values").grid(row=1, column=0, sticky="w")
            ttk.Entry(gf, textvariable=self.Ggrid_var, width=22).grid(row=1, column=1, sticky="w")
            ttk.Button(gf, text="Search the grid", command=self.grid).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

            ttk.Button(side, text="Save predictions...", command=self.save_predictions).pack(fill="x", pady=(8, 0))
            self.status = ttk.Label(side, text="", wraplength=270, foreground="#444")
            self.status.pack(fill="x", pady=(8, 0))

            self.tabs = ttk.Notebook(self)
            self.tabs.pack(side="left", fill="both", expand=True)
            self.fit_text = self._text_tab("Fit")
            self.fig_plot, self.canvas_plot = self._figure_tab("Two input plot")
            self.fig_pred, self.canvas_pred = self._figure_tab("Actual vs predicted")
            self.fig_grid, self.canvas_grid = self._figure_tab("Grid")
            self.roles_text = self._text_tab("Validation column")

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
            return text

        def _message(self, fig, canvas, msg):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, wrap=True)
            canvas.draw()

        # ------------------------------------------------------------- data
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
            self.path = path
            self.file_label.config(text=os.path.basename(path))
            cols = list(self.raw.columns)
            self.target_box["values"] = cols
            self.target_var.set(cols[0])
            self.refresh_roles()
            self.target_changed()
            self.show_roles()
            self.status.config(text=f"{len(self.raw)} rows read. Choose the target and inputs, "
                                    f"make or choose a validation column, then fit.")

        def role_like(self, c):
            return role_like(self.raw[c])

        def refresh_roles(self):
            existing = [c for c in self.raw.columns if self.role_like(c)]
            self.role_box["values"] = ["(make one)"] + existing
            self.role_var.set(existing[0] if existing else "(make one)")

        def target_changed(self):
            t = self.target_var.get()
            self.kind_var.set(default_kind(self.raw[t]))
            self.input_list.delete(0, "end")
            for c in self.raw.columns:
                if c != t and not self.role_like(c):
                    self.input_list.insert("end", c)
            names = list(self.input_list.get(0, "end"))
            for i, c in enumerate(names):
                if pd.api.types.is_numeric_dtype(self.raw[c]):
                    self.input_list.select_set(i)

        def keep_inputs(self):
            chosen = {self.input_list.get(i) for i in self.input_list.curselection()}
            t = self.target_var.get()
            self.input_list.delete(0, "end")
            for c in self.raw.columns:
                if c != t and not self.role_like(c):
                    self.input_list.insert("end", c)
                    if c in chosen:
                        self.input_list.select_set("end")

        def make_column(self):
            if self.raw is None:
                return
            try:
                val, test, k = float(self.val_var.get()), float(self.test_var.get()), int(self.k_var.get())
                seed, drawn = draw_seed(self.role_seed_var.get())
            except ValueError:
                messagebox.showerror("Validation column", "Shares must be numbers, k and the seed whole numbers.")
                return
            method = self.method_var.get()
            if method == "holdout" and val + test >= 1:
                messagebox.showerror("Validation column", "Validation and test shares together must leave rows to train on.")
                return
            if method == "kfold" and k < 2:
                messagebox.showerror("Validation column", "k folds needs k of at least 2.")
                return
            name = "validation_holdout" if method == "holdout" else f"fold{k}"
            self.raw[name] = make_roles(len(self.raw), method, val, test, k, seed)
            self.refresh_roles()
            self.role_var.set(name)
            self.keep_inputs()
            counts = self.raw[name].value_counts().sort_index()
            how = f"seed {seed} drawn at random" if drawn else f"seed {seed}"
            self.status.config(text=f"Made the column {name} with {how}. "
                                    + ", ".join(f"{r} {n}" for r, n in counts.items()) + ".")
            self.show_roles()

        def save_file(self):
            if self.raw is None:
                return
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                self.raw.to_csv(path, index=False)
                self.status.config(text=f"Saved {len(self.raw)} rows with every column to {os.path.basename(path)}.")

        def show_roles(self):
            t = self.roles_text
            t.delete("1.0", "end")
            if self.raw is None:
                return
            role = self.role_var.get()
            if role == "(make one)" or role not in self.raw.columns:
                t.insert("end", "No validation column chosen yet.\n")
                return
            t.insert("end", f"Column {role}\n\n", "head")
            counts = self.raw[role].value_counts()
            for r in sorted(counts.index, key=lambda r: (normal_role(r) == "test", str(r))):
                t.insert("end", f"{str(r):16}{counts[r]:6} rows   {100 * counts[r] / len(self.raw):5.1f}%\n")
            folds, test = folds_of(self.raw[role].astype(str).to_numpy())
            t.insert("end", "\nWhat each fit uses\n", "head")
            for lab, val, train in folds:
                t.insert("end", f"fit on {int(train.sum())} rows, judged on the {int(val.sum())} rows marked {lab}\n")
            if test.any():
                t.insert("end", f"\n{int(test.sum())} test rows are reported only, and no choice uses them.\n")

        # ------------------------------------------------------------- fit
        def read_settings(self):
            g = self.gamma_var.get().strip()
            return Settings(self.kernel_var.get(), float(self.C_var.get()),
                            float(g) if g else None, float(self.eps_var.get()), self.std_var.get())

        def prepared(self):
            role = self.role_var.get()
            if role == "(make one)" or role not in self.raw.columns:
                self.status.config(text="Make a validation column first, or choose one the file already has.")
                return None
            inputs = [self.input_list.get(i) for i in self.input_list.curselection()]
            if not inputs:
                self.status.config(text="Choose at least one input column.")
                return None
            target = self.target_var.get()
            self.p = prepare(self.raw, target, inputs, self.kind_var.get(), role)
            if self.p.kind == "category" and len(self.p.classes) < 2:
                self.status.config(text="The target has one class in these rows. Choose another target.")
                return None
            self.inputs = inputs
            return self.p.data[role].astype(str).to_numpy()

        def fit(self):
            if self.raw is None:
                self.open_file()
                return
            try:
                roles = self.prepared()
                if roles is None:
                    return
                s = self.read_settings()
                self.status.config(text="Fitting...")
                self.update_idletasks()
                self.result = run_model(self.p, roles, s)
                self.result.roles = roles
            except Exception as exc:
                self.status.config(text=f"Could not fit: {type(exc).__name__}: {exc}")
                messagebox.showerror("Could not fit", f"{type(exc).__name__}: {exc}")
                return
            r = self.result
            gtxt = f", gamma {r.gamma:.4g}" + (" (1/inputs)" if s.gamma is None else "") if s.kernel == "rbf" else ""
            self.status.config(text=f"{len(self.p.data)} rows used, {self.p.dropped} dropped for missing values. "
                                    f"{s.kernel} kernel, C {s.C:g}{gtxt}.")
            self.show_fit()
            self.draw_plot()
            self.draw_predictions()
            self.show_roles()

        def show_fit(self):
            r, p, s = self.result, self.p, self.result.settings
            t = self.fit_text
            t.delete("1.0", "end")
            what = "classification (SVC)" if p.kind == "category" else "regression (SVR)"
            t.insert("end", f"Support vector {what}, {s.kernel} kernel\n", "head")
            line = f"C {s.C:g}"
            if s.kernel == "rbf":
                line += f", gamma {r.gamma:.4g}" + (" (1 divided by the number of inputs)" if s.gamma is None else "")
            if p.kind == "number":
                line += f", epsilon {s.epsilon:g} target standard deviations"
            t.insert("end", line + "\n")
            t.insert("end", f"Inputs: {', '.join(p.names)}\n")
            t.insert("end", f"Standardized inputs: {'yes, from the train rows' if s.standardize else 'no'}\n\n")
            if p.kind == "number":
                head = f"{'rows':26}{'n':>6}{'R squared':>12}{'RMSE':>12}{'MSE':>14}{'support vectors':>18}\n"
            else:
                head = f"{'rows':26}{'n':>6}{'misclassification':>20}{'accuracy':>11}{'support vectors':>18}\n"
            t.insert("end", head, "head")

            def row(label, m, nsv=None):
                if m is None:
                    return
                tail = f"{nsv:18d}" if nsv is not None else ""
                if p.kind == "number":
                    t.insert("end", f"{label:26}{m['n']:6}{m['r2']:12.3f}{m['rmse']:12.3f}{m['mse']:14.3f}{tail}\n")
                else:
                    t.insert("end", f"{label:26}{m['n']:6}{m['miss']:20.3f}{m['acc']:11.3f}{tail}\n")

            for (lab, val, train), f in zip(r.folds, r.fits):
                pre = f"{lab}: " if r.kfold else ""
                row(f"{pre}train", f.train_m, f.n_sv)
                row(f"{pre}validation", f.val_m)
            tr, va, nsv = summary(p, r)
            if r.kfold:
                name = "R squared" if p.kind == "number" else "misclassification"
                t.insert("end", f"\nAverage over {len(r.fits)} folds: train {name} {tr:.3f}, validation {name} "
                                f"{va:.3f}, {nsv:.0f} support vectors\n", "head")
            if r.test.any():
                t.insert("end", "\n")
                row("test (reported only)", metrics(p, r.pred, r.test))
            n_train = int(r.folds[0][2].sum())
            t.insert("end", f"\n{r.fits[0].n_sv} of the {n_train} train rows are support vectors"
                            + (" in the first fold." if r.kfold else ".") + "\n")
            if p.kind == "category":
                self.confusion(t)

        def confusion(self, t):
            p, r = self.p, self.result
            groups = [("train", r.folds[0][2])] if not r.kfold else []
            vrows = np.zeros(len(p.y), dtype=bool)
            for lab, val, train in r.folds:
                vrows |= val
            groups.append(("validation", vrows))
            if r.test.any():
                groups.append(("test", r.test))
            for name, rows in groups:
                if not rows.any():
                    continue
                t.insert("end", f"\nConfusion table, {name} rows (actual down, predicted across)\n", "head")
                t.insert("end", f"{'':12}" + "".join(f"{c:>18}" for c in p.classes) + "\n")
                for a in p.classes:
                    rr = rows & (p.y == a)
                    tot = int(rr.sum())
                    t.insert("end", f"{a:12}")
                    for c in p.classes:
                        n = int((rr & (r.pred == c)).sum())
                        t.insert("end", f"{n:>9} ({100 * n / tot if tot else 0:5.1f}%) ")
                    t.insert("end", "\n")

        # ------------------------------------------------------------- plots
        def draw_plot(self):
            r, p = self.result, self.p
            fig = self.fig_plot
            if len(p.names) != 2:
                self._message(fig, self.canvas_plot, "This tab draws when the model has exactly two inputs.\n"
                                                     f"The model has {len(p.names)}.")
                return
            f = r.fits[0]
            lab, val, train = r.folds[0]
            Xs = (p.X - f.mu) / f.sd
            fig.clear()
            ax = fig.add_subplot(111)
            pad = 0.3
            x0, x1 = Xs[:, 0].min() - pad, Xs[:, 0].max() + pad
            y0, y1 = Xs[:, 1].min() - pad, Xs[:, 1].max() + pad
            gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
            G = np.c_[gx.ravel(), gy.ravel()]
            unit = " (standardized)" if r.settings.standardize else ""
            rows = train | val
            if p.kind == "category":
                if len(p.classes) == 2:
                    D = f.model.decision_function(G).reshape(gx.shape)
                    pos = f.model.classes_[1]
                    ax.contourf(gx, gy, D, levels=[-1e9, 0, 1e9],
                                colors=[CLASS_COLORS[0], CLASS_COLORS[1]], alpha=0.12)
                    ax.contour(gx, gy, D, levels=[0], colors="black", linewidths=1.6)
                    ax.contour(gx, gy, D, levels=[-1, 1], colors="black", linewidths=1, linestyles="--")
                else:
                    Z = f.model.predict(G)
                    idx = np.searchsorted(np.array(p.classes), Z).reshape(gx.shape)
                    ax.contourf(gx, gy, idx, levels=np.arange(len(p.classes) + 1) - 0.5,
                                colors=CLASS_COLORS[:len(p.classes)], alpha=0.12)
                    ax.contour(gx, gy, idx, levels=np.arange(len(p.classes)) + 0.5, colors="black", linewidths=1.4)
                for j, c in enumerate(p.classes):
                    col = CLASS_COLORS[j % 6]
                    m = train & (p.y == c)
                    ax.scatter(Xs[m, 0], Xs[m, 1], s=16, color=col, label=f"{p.target_name} = {c}",
                               edgecolor="none", alpha=0.85)
                    m = val & (p.y == c)
                    ax.scatter(Xs[m, 0], Xs[m, 1], s=16, facecolor="none", edgecolor=col, lw=0.9)
                handles, labels = ax.get_legend_handles_labels()
                handles.append(Line2D([], [], color="black", lw=1.6))
                labels.append("decision boundary, f(x) = 0")
                if len(p.classes) == 2:
                    handles.append(Line2D([], [], color="black", lw=1, ls="--"))
                    labels.append("margin, f(x) = -1 and +1")
            else:
                Z = f.model.predict(G).reshape(gx.shape) * f.ys + f.ym
                cf = ax.contourf(gx, gy, Z, levels=20, cmap="coolwarm", alpha=0.55)
                fig.colorbar(cf, ax=ax, label=f"predicted {p.target_name}")
                ax.scatter(Xs[train, 0], Xs[train, 1], c=p.y[train], cmap="coolwarm", s=14, edgecolor="#333",
                           linewidths=0.3, vmin=Z.min(), vmax=Z.max())
                ax.scatter(Xs[val, 0], Xs[val, 1], s=14, facecolor="none", edgecolor="#333", linewidths=0.6)
                handles, labels = [], []
            ax.scatter(Xs[f.sv, 0], Xs[f.sv, 1], s=60, facecolor="none", edgecolor="black", lw=1.1)
            handles.append(Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                                  markeredgecolor="black", markersize=8))
            labels.append(f"support vector ({f.n_sv})")
            if val.any():
                handles.append(Line2D([], [], marker="o", ls="none", markerfacecolor="#888",
                                      markeredgecolor="#888", markersize=5))
                labels.append("train row (filled)")
                handles.append(Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                                      markeredgecolor="#888", markersize=5))
                labels.append(f"{lab} row (hollow)")
            if p.kind == "category":
                ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
            else:
                ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
                          fontsize=9, frameon=False)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_xlabel(p.names[0] + unit)
            ax.set_ylabel(p.names[1] + unit)
            note = ", fold model 1" if r.kfold else ""
            ax.set_title(f"{lab} rows and train rows{note}", fontsize=10)
            fig.tight_layout()
            self.canvas_plot.draw()

        def draw_predictions(self):
            r, p = self.result, self.p
            fig = self.fig_pred
            if p.kind == "category":
                self._message(fig, self.canvas_pred, "For a category target the confusion tables are on the Fit tab.")
                return
            fig.clear()
            groups = [("train", r.folds[0][2]), ("validation", r.folds[0][1])] if not r.kfold \
                else [("validation", ~r.test)]
            if r.test.any():
                groups.append(("test", r.test))
            lo = min(p.y.min(), r.pred.min())
            hi = max(p.y.max(), r.pred.max())
            for i, (name, rows) in enumerate(groups):
                ax = fig.add_subplot(1, len(groups), i + 1)
                ax.scatter(r.pred[rows], p.y[rows], s=12, alpha=0.6, color=ROLE_COLORS[name])
                ax.plot([lo, hi], [lo, hi], color="black", lw=1)
                m = metrics(p, r.pred, rows)
                ax.set_title(f"{name}, n={m['n']}, R squared {m['r2']:.3f}", fontsize=10)
                ax.set_xlabel("predicted")
                if i == 0:
                    ax.set_ylabel("actual")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.grid(alpha=0.3)
            fig.tight_layout()
            self.canvas_pred.draw()

        # ------------------------------------------------------------- grid
        def grid(self):
            if self.raw is None:
                return
            try:
                roles = self.prepared()
                if roles is None:
                    return
                base = self.read_settings()
                Cs = [float(v) for v in self.Cgrid_var.get().replace(",", " ").split()]
                gs = [float(v) for v in self.Ggrid_var.get().replace(",", " ").split()]
                if not Cs or (base.kernel == "rbf" and not gs):
                    self.status.config(text="Type at least one C value and one gamma value.")
                    return
                self.status.config(text=f"Fitting {len(Cs) * (len(gs) if base.kernel == 'rbf' else 1)} settings...")
                self.update_idletasks()
                gl, out = run_grid(self.p, roles, base, Cs, gs)
            except Exception as exc:
                self.status.config(text=f"Could not search: {type(exc).__name__}: {exc}")
                messagebox.showerror("Could not search", f"{type(exc).__name__}: {exc}")
                return
            self.draw_grid(base, Cs, gl, out)

        def draw_grid(self, base, Cs, gl, out):
            p = self.p
            num = p.kind == "number"
            name = "R squared" if num else "misclassification"
            V = np.array([[c[1] for c in row] for row in out])
            T = np.array([[c[0] for c in row] for row in out])
            S = np.array([[c[2] for c in row] for row in out])
            best = np.unravel_index(np.nanargmax(V) if num else np.nanargmin(V), V.shape)
            fig = self.fig_grid
            fig.clear()
            cmap = "viridis" if num else "viridis_r"
            for k, (M, title) in enumerate([(T, f"train {name}"), (V, f"validation {name}")]):
                ax = fig.add_subplot(1, 2, k + 1)
                im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower",
                               vmin=np.nanmin(np.r_[T.ravel(), V.ravel()]), vmax=np.nanmax(np.r_[T.ravel(), V.ravel()]))
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        txt = f"{M[i, j]:.3f}"
                        if k == 1:
                            txt += f"\n{S[i, j]:.0f} sv"
                        mid = 0.5 * (np.nanmin(M) + np.nanmax(M))
                        dark = M[i, j] < mid if num else M[i, j] > mid
                        ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                                color="white" if dark else "black",
                                fontweight="bold" if (i, j) == best else "normal")
                ax.set_xticks(range(len(Cs)))
                ax.set_xticklabels([f"{c:g}" for c in Cs])
                ax.set_xlabel("cost C")
                ax.set_yticks(range(len(gl)))
                ax.set_yticklabels([f"{g:g}" if g is not None else "linear" for g in gl])
                ax.set_ylabel("gamma" if base.kernel == "rbf" else "")
                ax.set_title(title, fontsize=10)
                if k == 1:
                    ax.add_patch(Rectangle(
                        (best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor="red", lw=2))
            fig.tight_layout()
            self.canvas_grid.draw()
            self.tabs.select(3)
            g = gl[best[0]]
            gtxt = f", gamma {g:g}" if g is not None else ""
            self.status.config(text=f"Best validation {name} {V[best]:.3f} at C {Cs[best[1]]:g}{gtxt}, outlined in red. "
                                    f"The validation rows chose it, so only test rows can still judge it.")

        def save_predictions(self):
            if self.result is None:
                return
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            out = self.p.data.copy()
            out["predicted"] = self.result.pred
            out["support_vector"] = np.where(self.result.fits[0].sv, 1, 0)
            out.to_csv(path, index=False)
            self.status.config(text=f"Saved {len(out)} rows to {os.path.basename(path)}.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    if SELFTEST:
        selftest(args[0] if args else "chess.csv")
    else:
        App(args[0] if args else None).mainloop()
