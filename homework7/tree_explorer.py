#!/usr/bin/env python3
"""
Single-window tree explorer: a regression tree for a number target, a decision
tree for a category target.

BIT 5534 lec8_2 tool, written 2026-10-15. It replaces the lec6_1 and lec6_2 tree
explorers with one window that carries the validation column of the neural
network (lec7_1) and support vector machine (lec8_1) explorers, so the four
model families can be compared row for row and stacked.

The tree is grown best first by scikit-learn: each new split is the one, in any
leaf, that most reduces the error of the train rows, measured by entropy for a
category target, which ranks splits the way G squared does, and by the sum of
squares for a number target. The number of splits is the setting. A non-numeric
input becomes one 0 or 1 column per level after the first, so a split on it
sends one level one way and the rest the other, where lec6_1 grouped levels.
The tree is grown on the train rows only.

Every row carries a role: train, validation or test, or for k folds
validation_1 to validation_k plus an optional test. A column the file already
has may use Training and Validation as its labels.

Save predictions and Save probabilities add a column to the data and save the
whole file, so the next explorer can open it and use the column as an input.
With k folds, each row's value comes from the fold model that did not see
that row, and a test row's value is the average of the fold models, so a model
fitted on the saved columns (a stacking meta-model) learns from values no base
model fitted to. With a holdout column the values come from the fit on the
train rows, which saw the train rows themselves. A probability is the share of the positive class among the train rows
in the row's leaf.

    python tree_explorer.py                    opens a file dialog
    python tree_explorer.py diabetes.csv       opens that file
    python tree_explorer.py --selftest FILE    prints reference numbers, opens no window

Required packages: pandas, numpy, matplotlib, scikit-learn (and openpyxl for .xlsx).
"""

import os
import secrets
import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SELFTEST = "--selftest" in sys.argv
if not SELFTEST:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from sklearn.tree import plot_tree

TAG = "TREE"
PRED_TAGS = ("REG ", "TREE ", "NN ", "SVM ")
ROLE_COLORS = {"train": "#2c6fbb", "validation": "#d68910", "test": "#8e44ad"}


# ----------------------------------------------------------------- data
def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def default_kind(series):
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        return "number"
    return "category"


def default_positive(series):
    """The less frequent level, which is usually the one a business wants to find."""
    counts = series.dropna().astype(str).value_counts()
    return str(counts.index[-1]) if len(counts) else ""


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


def prepare(df, target, inputs, kind, role_col, positive=None):
    used = [target] + inputs + ([role_col] if role_col else [])
    before = len(df)
    clean = df.dropna(subset=used)
    p = Prepared()
    p.rows = clean.index.to_numpy()          # where each used row sits in the file
    clean = clean.reset_index(drop=True)
    p.data, p.dropped = clean, before - len(clean)
    p.X, p.names = encode_inputs(clean, inputs)
    p.kind = kind
    p.target_name = target
    if kind == "number":
        p.y = clean[target].astype(float).to_numpy()
        p.classes = None
        p.positive = None
    else:
        lab = clean[target].astype(str)
        p.classes = sorted(lab.unique())
        p.y = lab.to_numpy()
        p.positive = positive if positive in p.classes else default_positive(lab)
    return p


# ----------------------------------------------------------------- the model
class Settings:
    def __init__(self, splits=4, min_leaf=5, history=20):
        self.splits, self.min_leaf, self.history = splits, min_leaf, history


class FoldFit:
    pass


def grow(p, train, splits, min_leaf):
    if p.kind == "number":
        m = DecisionTreeRegressor(max_leaf_nodes=splits + 1, min_samples_leaf=min_leaf, random_state=0)
    else:
        m = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=splits + 1,
                                   min_samples_leaf=min_leaf, random_state=0)
    m.fit(p.X[train], p.y[train])
    return m


def fit_fold(p, train, val, s):
    f = FoldFit()
    m = grow(p, train, s.splits, s.min_leaf)
    f.model = m
    f.pred = m.predict(p.X)
    if p.kind == "category":
        f.prob = m.predict_proba(p.X)[:, list(m.classes_).index(p.positive)] \
            if p.positive in m.classes_ else np.zeros(len(p.y))
    else:
        f.prob = None
    f.splits = int(m.tree_.node_count // 2)
    f.train_m = metrics(p, f, train)
    f.val_m = metrics(p, f, val) if val.any() else None
    return f


def metrics(p, f, rows, pred=None, prob=None):
    pred = f.pred if pred is None else pred
    if not rows.any():
        return None
    if p.kind == "number":
        y, yh = p.y[rows], pred[rows].astype(float)
        sse = float(((y - yh) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        mse = sse / len(y)
        return {"n": int(rows.sum()), "r2": 1 - sse / sst if sst > 0 else float("nan"),
                "rmse": mse ** 0.5, "mse": mse, "choose": mse}
    y = p.y[rows]
    miss = float((pred[rows] != y).mean())
    return {"n": int(rows.sum()), "miss": miss, "acc": 1 - miss, "choose": miss}


def contributions(p, m, n_train):
    """What each input column's splits removed: G squared for a category target
    (entropy is in bits, so 2 ln 2 times the bits removed), the sum of squares for a
    number target."""
    raw = m.tree_.compute_feature_importances(normalize=False) * n_train
    if p.kind == "category":
        raw = raw * 2 * np.log(2)
    return raw


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
        r.pred, r.prob = r.fits[0].pred, r.fits[0].prob
    else:
        # Each validation row is predicted by the fold tree that did not see it.
        # A test row is predicted by the fold trees together: the average for a
        # number target, the average probability for a category target.
        r.pred = r.fits[0].pred.copy()
        r.prob = None if r.fits[0].prob is None else r.fits[0].prob.copy()
        for (lab, val, train), f in zip(folds, r.fits):
            r.pred[val] = f.pred[val]
            if r.prob is not None:
                r.prob[val] = f.prob[val]
        if test.any():
            if p.kind == "number":
                r.pred[test] = (sum(f.pred for f in r.fits) / len(r.fits))[test]
            else:
                votes = np.array([f.pred for f in r.fits])
                for i in np.flatnonzero(test):
                    lv, ct = np.unique(votes[:, i], return_counts=True)
                    r.pred[i] = lv[ct.argmax()]
                r.prob[test] = (sum(f.prob for f in r.fits) / len(r.fits))[test]
    if r.kfold:
        r.final_rows = ~test
        r.final = fit_fold(p, r.final_rows, np.zeros(len(p.y), dtype=bool), s)
    else:
        r.final_rows = folds[0][2]
        r.final = r.fits[0]
    return r


def summary(p, r):
    key = "r2" if p.kind == "number" else "miss"
    vals = [f.val_m[key] for f in r.fits if f.val_m]
    trs = [f.train_m[key] for f in r.fits]
    return float(np.mean(trs)), float(np.mean(vals)) if vals else float("nan")


def split_history(p, roles, s):
    """Train and validation figures for every number of splits from 1 to s.history,
    averaged over the folds for k folds. Test rows play no part."""
    out = []
    for n in range(1, s.history + 1):
        r = run_model_light(p, roles, Settings(n, s.min_leaf, s.history))
        out.append((n,) + summary(p, r))
    return out


def run_model_light(p, roles, s):
    folds, test = folds_of(roles)
    r = Result()
    r.folds, r.fits = folds, [fit_fold(p, train, val, s) for lab, val, train in folds]
    return r


# ----------------------------------------------------------------- self test
def selftest(path, target=None, inputs=None, role=None, splits=4):
    df = read_table(path)
    cols = list(df.columns)
    target = target or cols[0]
    if inputs is None:
        inputs = [c for c in cols if c != target and not role_like(df[c])
                  and pd.api.types.is_numeric_dtype(df[c])]
    if role is None:
        role = next((c for c in cols if role_like(df[c])), None)
    kind = default_kind(df[target])
    p = prepare(df, target, inputs, kind, role)
    roles = p.data[role].astype(str).to_numpy() if role else make_roles(len(p.data), "holdout", 1 / 3, 0, 5, 123)
    s = Settings(splits)
    r = run_model(p, roles, s)
    key = "r2" if kind == "number" else "miss"
    print(f"file {path}: {len(p.data)} rows, target {target} ({kind}), inputs {p.names}")
    for n, tr, va in split_history(p, roles, Settings(splits, 5, 12)):
        print(f"  splits {n:2}  train {key} {tr:.3f}  validation {key} {va:.3f}")
    return p, r


# ----------------------------------------------------------------- the window
if not SELFTEST:
    class App(tk.Tk):
        def __init__(self, path=None):
            super().__init__()
            self.title("Tree explorer: regression and decision trees")
            self.geometry("1440x900")
            self.raw = None
            self.result = None
            self.p = None
            self._build()
            if path:
                self.load(path)
            else:
                self.status.config(text="Open a .csv or .xlsx file to begin.")
                self.after(200, self.open_file)

        # ------------------------------------------------------------- layout
        def _build(self):
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
            ttk.Radiobutton(kf, text="number", value="number", variable=self.kind_var,
                            command=self.kind_changed).pack(side="left")
            ttk.Radiobutton(kf, text="category", value="category", variable=self.kind_var,
                            command=self.kind_changed).pack(side="left")
            pf = ttk.Frame(side)
            pf.pack(fill="x")
            ttk.Label(pf, text="Positive class").pack(side="left")
            self.pos_var = tk.StringVar()
            self.pos_box = ttk.Combobox(pf, textvariable=self.pos_var, state="readonly", width=14)
            self.pos_box.pack(side="left", padx=(6, 0))

            ttk.Label(side, text="Input columns").pack(anchor="w", pady=(8, 0))
            self.input_list = tk.Listbox(side, selectmode="multiple", height=9, exportselection=False)
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

            mf = ttk.LabelFrame(side, text="Tree", padding=6)
            mf.pack(fill="x", pady=(8, 0))
            self.splits_var = tk.StringVar(value="4")
            self.leaf_var = tk.StringVar(value="5")
            self.hist_var = tk.StringVar(value="20")
            for i, (lab, var) in enumerate([("Number of splits", self.splits_var),
                                            ("Smallest leaf (rows)", self.leaf_var),
                                            ("Split history up to", self.hist_var)]):
                ttk.Label(mf, text=lab).grid(row=i, column=0, sticky="w")
                ttk.Entry(mf, textvariable=var, width=8).grid(row=i, column=1, sticky="w")
            bf = ttk.Frame(mf)
            bf.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
            ttk.Button(bf, text="Grow one split", command=lambda: self.step(+1)).pack(side="left", fill="x", expand=True)
            ttk.Button(bf, text="Prune one", command=lambda: self.step(-1)).pack(side="left", fill="x", expand=True, padx=(4, 0))
            ttk.Label(mf, text="A number target grows a regression tree, a category target a decision tree.",
                      foreground="#555", wraplength=260).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Button(side, text="Fit the tree", command=self.fit).pack(fill="x", pady=(8, 0))

            sf = ttk.LabelFrame(side, text="Save for stacking", padding=6)
            sf.pack(fill="x", pady=(8, 0))
            ttk.Button(sf, text="Save predictions...", command=self.save_predictions).pack(fill="x")
            self.prob_button = ttk.Button(sf, text="Save probabilities...", command=self.save_probabilities)
            self.prob_button.pack(fill="x", pady=(4, 0))
            self.status = ttk.Label(side, text="", wraplength=270, foreground="#444")
            self.status.pack(fill="x", pady=(8, 0))

            self.tabs = ttk.Notebook(self)
            self.tabs.pack(side="left", fill="both", expand=True)
            self.fit_text = self._text_tab("Fit")
            self.fig_tree, self.canvas_tree = self._figure_tab("Tree")
            self.fig_hist, self.canvas_hist = self._figure_tab("Split history")
            self.fig_contrib, self.canvas_contrib = self._figure_tab("Column contributions")
            self.fig_pred, self.canvas_pred = self._figure_tab("Actual vs predicted")
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
            self.result = None
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
                # saved predictions start unselected, so a base model never uses another one's
                if pd.api.types.is_numeric_dtype(self.raw[c]) and not c.startswith(PRED_TAGS):
                    self.input_list.select_set(i)
            self.kind_changed()

        def kind_changed(self):
            t = self.target_var.get()
            if self.raw is None or not t:
                return
            if self.kind_var.get() == "category":
                levels = sorted(self.raw[t].dropna().astype(str).unique())
                self.pos_box["values"] = levels
                if self.pos_var.get() not in levels:
                    self.pos_var.set(default_positive(self.raw[t]))
                self.pos_box.config(state="readonly")
                self.prob_button.config(state="normal")
            else:
                self.pos_box["values"] = []
                self.pos_var.set("")
                self.pos_box.config(state="disabled")
                self.prob_button.config(state="disabled")

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
            self.p = prepare(self.raw, target, inputs, self.kind_var.get(), role, self.pos_var.get())
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
                self.status.config(text="Growing...")
                self.update_idletasks()
                self.result = run_model(self.p, roles, s)
                self.history = split_history(self.p, roles, s)
            except Exception as exc:
                self.result = None
                self.status.config(text=f"Could not fit: {type(exc).__name__}: {exc}")
                messagebox.showerror("Could not fit", f"{type(exc).__name__}: {exc}")
                return
            pos = f" Positive class {self.p.positive}." if self.p.kind == "category" else ""
            self.status.config(text=f"{len(self.p.data)} rows used, {self.p.dropped} dropped for missing values.{pos}")
            self.show_fit()
            self.draw_tree()
            self.draw_history()
            self.draw_contributions()
            self.draw_predictions()
            self.show_roles()

        def read_settings(self):
            n = max(1, int(self.splits_var.get()))
            return Settings(n, max(1, int(self.leaf_var.get())), max(n, int(self.hist_var.get())))

        def step(self, delta):
            try:
                n = int(self.splits_var.get())
            except ValueError:
                n = 4
            self.splits_var.set(str(max(1, n + delta)))
            if self.raw is not None:
                self.fit()

        def show_fit(self):
            r, p = self.result, self.p
            t = self.fit_text
            t.delete("1.0", "end")
            s = r.settings
            what = "Regression tree" if p.kind == "number" else "Decision tree"
            t.insert("end", f"{what} for {p.target_name}, {s.splits} splits asked, "
                            f"smallest leaf {s.min_leaf} rows\n", "head")
            if p.kind == "category":
                t.insert("end", f"Positive class for probabilities: {p.positive}\n")
            t.insert("end", f"Inputs: {', '.join(p.names)}\n\n")
            if p.kind == "number":
                head = f"{'rows':26}{'n':>6}{'R squared':>12}{'RMSE':>12}{'MSE':>14}\n"
            else:
                head = f"{'rows':26}{'n':>6}{'misclassification':>20}{'accuracy':>11}\n"
            t.insert("end", head, "head")

            def row(label, m):
                if m is None:
                    return
                if p.kind == "number":
                    t.insert("end", f"{label:26}{m['n']:6}{m['r2']:12.3f}{m['rmse']:12.3f}{m['mse']:14.3f}\n")
                else:
                    t.insert("end", f"{label:26}{m['n']:6}{m['miss']:20.3f}{m['acc']:11.3f}\n")

            for (lab, val, train), f in zip(r.folds, r.fits):
                pre = f"{lab}: " if r.kfold else ""
                row(f"{pre}train", f.train_m)
                row(f"{pre}validation", f.val_m)
            tr, va = summary(p, r)
            if r.kfold:
                name = "R squared" if p.kind == "number" else "misclassification"
                t.insert("end", f"\nAverage over {len(r.fits)} folds: train {name} {tr:.3f}, "
                                f"validation {name} {va:.3f}\n", "head")
            if r.test.any():
                t.insert("end", "\n")
                row("test (reported only)", metrics(p, None, r.test, pred=r.pred, prob=r.prob))
            where = (f"every row that is not a test row ({int(r.final_rows.sum())} rows)" if r.kfold
                     else f"the {int(r.final_rows.sum())} train rows")
            fsp = r.final.splits
            t.insert("end", f"\nThe tree on the Tree tab is grown on {where}: {fsp} splits, "
                            f"{fsp + 1} leaves.\n", "head")
            if fsp < s.splits:
                t.insert("end", f"It stopped at {fsp} splits because no leaf could split again "
                                f"with at least {s.min_leaf} rows on each side.\n")
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
        def draw_tree(self):
            r, p = self.result, self.p
            fig = self.fig_tree
            fig.clear()
            ax = fig.add_subplot(111)
            m = r.final.model
            names = [n if len(n) <= 18 else n[:17] + "." for n in p.names]
            fs = 9 if r.final.splits <= 8 else (7 if r.final.splits <= 16 else 5)
            plot_tree(m, ax=ax, feature_names=names,
                      class_names=[str(c) for c in m.classes_] if p.kind == "category" else None,
                      filled=True, impurity=False, rounded=True, fontsize=fs, precision=2)
            where = "every row that is not a test row" if r.kfold else "the train rows"
            ax.set_title(f"{r.final.splits} splits, grown on {where}. Each box shows its rule, its train rows "
                         f"and {'the class counts' if p.kind == 'category' else 'the mean'}.", fontsize=10)
            fig.tight_layout()
            self.canvas_tree.draw()

        def draw_history(self):
            p, r = self.p, self.result
            h = np.array(self.history, dtype=float)
            fig = self.fig_hist
            fig.clear()
            ax = fig.add_subplot(111)
            name = "R squared" if p.kind == "number" else "misclassification"
            ax.plot(h[:, 0], h[:, 1], "-o", color=ROLE_COLORS["train"], ms=4, label=f"train {name}")
            ax.plot(h[:, 0], h[:, 2], "-o", color=ROLE_COLORS["validation"], ms=4,
                    label=f"validation {name}" + (" (fold average)" if r.kfold else ""))
            best = int(np.nanargmax(h[:, 2]) if p.kind == "number" else np.nanargmin(h[:, 2]))
            ax.plot(h[best, 0], h[best, 2], "s", ms=11, mfc="none", mec="#c0392b", mew=2,
                    label=f"best validation, {int(h[best, 0])} splits")
            ax.axvline(r.settings.splits, color="#555", lw=1, ls="--", label=f"this tree, {r.settings.splits} splits")
            ax.set_xlabel("number of splits")
            ax.set_ylabel(name)
            ax.set_xticks(h[:, 0])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, frameon=False)
            ax.set_title("Split history: every tree from 1 split up, grown on the train rows", fontsize=10)
            fig.tight_layout()
            self.canvas_hist.draw()

        def draw_contributions(self):
            p, r = self.p, self.result
            c = contributions(p, r.final.model, int(r.final_rows.sum()))
            order = np.argsort(c)
            fig = self.fig_contrib
            fig.clear()
            ax = fig.add_subplot(111)
            ax.barh([p.names[i] for i in order], c[order], color="#2c6fbb")
            for k, i in enumerate(order):
                if c[i] > 0:
                    ax.text(c[i], k, f" {c[i]:,.1f}", va="center", fontsize=9)
            what = "G squared" if p.kind == "category" else "sum of squares"
            ax.set_xlabel(f"{what} removed by the column's splits")
            ax.set_title(f"Column contributions: the {what} of every split, summed by the column it used",
                         fontsize=10)
            fig.tight_layout()
            self.canvas_contrib.draw()

        def draw_predictions(self):
            r, p = self.result, self.p
            fig = self.fig_pred
            fig.clear()
            groups = [("train", r.folds[0][2]), ("validation", r.folds[0][1])] if not r.kfold \
                else [("validation", ~r.test)]
            if r.test.any():
                groups.append(("test", r.test))
            if p.kind == "number":
                lo = min(p.y.min(), r.pred.min())
                hi = max(p.y.max(), r.pred.max())
                for i, (name, rows) in enumerate(groups):
                    ax = fig.add_subplot(1, len(groups), i + 1)
                    ax.scatter(r.pred[rows], p.y[rows], s=12, alpha=0.6, color=ROLE_COLORS[name])
                    ax.plot([lo, hi], [lo, hi], color="black", lw=1)
                    m = metrics(p, None, rows, pred=r.pred)
                    ax.set_title(f"{name}, n={m['n']}, R squared {m['r2']:.3f}", fontsize=10)
                    ax.set_xlabel("predicted")
                    if i == 0:
                        ax.set_ylabel("actual")
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)
                    ax.grid(alpha=0.3)
            else:
                # the probability of the positive class, one histogram per actual class
                bins = np.linspace(0, 1, 21)
                for i, (name, rows) in enumerate(groups):
                    ax = fig.add_subplot(1, len(groups), i + 1)
                    for c, col in zip(p.classes, ["#9bb7d8", "#c0392b"] if p.classes[1] == p.positive
                                      else ["#c0392b", "#9bb7d8"]):
                        ax.hist(r.prob[rows & (p.y == c)], bins=bins, alpha=0.6, color=col, label=f"actual {c}")
                    ax.axvline(0.5, color="black", lw=1, ls="--")
                    ax.set_title(f"{name} rows", fontsize=10)
                    ax.set_xlabel(f"predicted probability of {p.positive}")
                    if i == 0:
                        ax.set_ylabel("rows")
                        ax.legend(fontsize=9, frameon=False)
            fig.tight_layout()
            self.canvas_pred.draw()

        # ------------------------------------------------------------- save
        def add_and_save(self, name, values, what):
            """Put the column into the data, in the rows the model used, and save the whole file."""
            if what == "numbers":
                col = np.full(len(self.raw), np.nan)
            else:
                col = np.full(len(self.raw), None, dtype=object)
            col[self.p.rows] = values
            self.raw[name] = col
            self.keep_inputs()
            base = os.path.splitext(os.path.basename(self.path))[0]
            base = base if base.endswith("_stack") else base + "_stack"
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                                initialfile=base + ".csv")
            src = ("the fold model that did not see each row, with test rows from the fold models together"
                   if self.result.kfold else
                   "the fit on the train rows, which saw the train rows themselves; for stacking, use k folds")
            if not path:
                self.status.config(text=f"Added the column {name} from {src}. The file is not saved yet.")
                return
            self.raw.to_csv(path, index=False)
            self.status.config(text=f"Added the column {name}, from {src}, and saved {len(self.raw)} rows "
                                    f"with every column to {os.path.basename(path)}.")

        def save_predictions(self):
            if self.result is None:
                self.status.config(text="Fit the model first.")
                return
            name = f"{TAG} pred {self.p.target_name}"
            vals = self.result.pred
            self.add_and_save(name, vals.astype(float) if self.p.kind == "number" else vals,
                              "numbers" if self.p.kind == "number" else "labels")

        def save_probabilities(self):
            if self.result is None:
                self.status.config(text="Fit the model first.")
                return
            if self.p.kind != "category":
                self.status.config(text="Probabilities are for a category target.")
                return
            name = f"{TAG} prob {self.p.positive}"
            self.add_and_save(name, np.round(self.result.prob.astype(float), 6), "numbers")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    if SELFTEST:
        selftest(args[0] if args else "diabetes.csv")
    else:
        App(args[0] if args else None).mainloop()
