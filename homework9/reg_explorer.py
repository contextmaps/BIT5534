#!/usr/bin/env python3
"""
Single-window regression explorer: multiple regression for a number target,
logistic regression for a category target.

BIT 5534 lec8_2 tool, written 2026-10-15. It joins the multiple regression of
lec2_1 and the logistic regression of lec3_2 in the window the neural network
(lec7_1) and support vector machine (lec8_1) explorers use, with the same
validation column, so the four model families can be compared row for row and
stacked.

A number target is fitted by ordinary least squares, a category target with
two levels by logistic regression for the positive class. Both are fitted on
the train rows only, with statsmodels, and every term gets a logworth,
-log10(p value), drawn as a bar chart.

Every row carries a role: train, validation or test, or for k folds
validation_1 to validation_k plus an optional test. A column the file already
has may use Training and Validation as its labels.

Save predictions and Save probabilities add a column to the data and save the
whole file, so the next explorer can open it and use the column as an input.
With k folds, each row's value comes from the fold model that did not see
that row, and a test row's value is the average of the fold models, so a model
fitted on the saved columns (a stacking meta-model) learns from values no base
model fitted to. With a holdout column the values come from the fit on the
train rows, which saw the train rows themselves.

    python reg_explorer.py                    opens a file dialog
    python reg_explorer.py diabetes.csv       opens that file
    python reg_explorer.py --selftest FILE    prints reference numbers, opens no window

Required packages: pandas, numpy, matplotlib, statsmodels, scipy (and openpyxl for .xlsx).
"""

import os
import secrets
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

SELFTEST = "--selftest" in sys.argv
if not SELFTEST:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

TAG = "REG"
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
class FoldFit:
    pass


def design(X):
    return sm.add_constant(X, has_constant="add")


def fit_fold(p, train, val):
    f = FoldFit()
    Xd = design(p.X)
    if p.kind == "number":
        m = sm.OLS(p.y[train], Xd[train]).fit()
        f.pred = m.predict(Xd)
        f.prob = None
    else:
        if len(p.classes) != 2:
            raise ValueError(f"Logistic regression here takes a target with two levels, "
                             f"and {p.target_name} has {len(p.classes)}.")
        yb = (p.y == p.positive).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sm.Logit(yb[train], Xd[train]).fit(disp=0, maxiter=200)
        prob = m.predict(Xd)
        other = [c for c in p.classes if c != p.positive][0]
        f.prob = prob
        f.pred = np.where(prob >= 0.5, p.positive, other).astype(object)
    f.model = m
    f.train_m = metrics(p, f, train)
    f.val_m = metrics(p, f, val) if val.any() else None
    return f


def metrics(p, f, rows, pred=None, prob=None):
    pred = f.pred if pred is None else pred
    prob = (f.prob if f is not None else None) if prob is None else prob
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
    out = {"n": int(rows.sum()), "miss": miss, "acc": 1 - miss, "choose": miss}
    if prob is not None:
        # RSquare (U): one minus the log likelihood of the model over the log likelihood
        # of predicting every row with the share of positives in these rows.
        yb = (y == p.positive).astype(float)
        pr = np.clip(prob[rows], 1e-12, 1 - 1e-12)
        ll = float((yb * np.log(pr) + (1 - yb) * np.log(1 - pr)).sum())
        base = np.clip(yb.mean(), 1e-12, 1 - 1e-12)
        ll0 = float((yb * np.log(base) + (1 - yb) * np.log(1 - base)).sum())
        out["r2u"] = 1 - ll / ll0 if ll0 < 0 else float("nan")
    return out


def term_table(p, m):
    """Estimate, standard error, statistic, p value and logworth for every term."""
    names = ["Intercept"] + list(p.names)
    est, se = np.asarray(m.params), np.asarray(m.bse)
    stat = np.asarray(m.tvalues)
    pv = np.asarray(m.pvalues)
    lw = -np.log10(np.clip(pv, 1e-300, 1))
    return [(n, est[i], se[i], stat[i], pv[i], lw[i]) for i, n in enumerate(names)]


class Result:
    pass


def run_model(p, roles):
    folds, test = folds_of(roles)
    r = Result()
    r.folds, r.fits = folds, []
    for lab, val, train in folds:
        r.fits.append(fit_fold(p, train, val))
    r.test = test
    r.kfold = folds[0][0] != "validation"
    if not r.kfold:
        r.pred, r.prob = r.fits[0].pred, r.fits[0].prob
    else:
        # Each validation row is predicted by the fold model that did not see it.
        # A test row is predicted by the average of the fold models.
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
                avg = sum(f.prob for f in r.fits) / len(r.fits)
                r.prob[test] = avg[test]
                other = [c for c in p.classes if c != p.positive][0]
                r.pred[test] = np.where(avg[test] >= 0.5, p.positive, other)
    # The fit whose terms are reported: the train rows for a holdout, every row
    # that is not a test row for k folds.
    if r.kfold:
        everything = ~test
        r.final = fit_fold(p, everything, np.zeros(len(p.y), dtype=bool))
        r.final_rows = everything
    else:
        r.final = r.fits[0]
        r.final_rows = folds[0][2]
    return r


def summary(p, r):
    key = "r2" if p.kind == "number" else "miss"
    vals = [f.val_m[key] for f in r.fits if f.val_m]
    trs = [f.train_m[key] for f in r.fits]
    return float(np.mean(trs)), float(np.mean(vals)) if vals else float("nan")


# ----------------------------------------------------------------- self test
def selftest(path, target=None, inputs=None, role=None):
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
    r = run_model(p, roles)
    print(f"file {path}: {len(p.data)} rows, target {target} ({kind}), inputs {p.names}")
    for (lab, val, train), f in zip(r.folds, r.fits):
        key = "r2" if kind == "number" else "miss"
        print(f"{lab}: train {key} {f.train_m[key]:.3f}  validation {key} {f.val_m[key]:.3f}")
    for row in term_table(p, r.final.model):
        print(f"  {row[0]:22} estimate {row[1]:10.4f}  p {row[4]:.3g}  logworth {row[5]:.2f}")
    return p, r


# ----------------------------------------------------------------- the window
if not SELFTEST:
    class App(tk.Tk):
        def __init__(self, path=None):
            super().__init__()
            self.title("Regression explorer: multiple and logistic regression")
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

            ttk.Label(side, text="A number target is fitted by multiple regression, a category target "
                                 "with two levels by logistic regression for the positive class.",
                      foreground="#555", wraplength=270).pack(fill="x", pady=(8, 0))
            ttk.Button(side, text="Fit the model", command=self.fit).pack(fill="x", pady=(8, 0))

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
            self.fig_lw, self.canvas_lw = self._figure_tab("Logworth")
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
                self.status.config(text="Fitting...")
                self.update_idletasks()
                self.result = run_model(self.p, roles)
            except Exception as exc:
                self.result = None
                self.status.config(text=f"Could not fit: {type(exc).__name__}: {exc}")
                messagebox.showerror("Could not fit", f"{type(exc).__name__}: {exc}")
                return
            pos = f" Positive class {self.p.positive}." if self.p.kind == "category" else ""
            self.status.config(text=f"{len(self.p.data)} rows used, {self.p.dropped} dropped for missing values.{pos}")
            self.show_fit()
            self.draw_logworth()
            self.draw_predictions()
            self.show_roles()

        def show_fit(self):
            r, p = self.result, self.p
            t = self.fit_text
            t.delete("1.0", "end")
            if p.kind == "number":
                t.insert("end", f"Multiple regression of {p.target_name}\n", "head")
            else:
                t.insert("end", f"Logistic regression of {p.target_name}, probability of {p.positive}\n", "head")
            t.insert("end", f"Inputs: {', '.join(p.names)}\n\n")
            if p.kind == "number":
                head = f"{'rows':26}{'n':>6}{'R squared':>12}{'RMSE':>12}{'MSE':>14}\n"
            else:
                head = f"{'rows':26}{'n':>6}{'misclassification':>20}{'accuracy':>11}{'RSquare (U)':>14}\n"
            t.insert("end", head, "head")

            def row(label, m):
                if m is None:
                    return
                if p.kind == "number":
                    t.insert("end", f"{label:26}{m['n']:6}{m['r2']:12.3f}{m['rmse']:12.3f}{m['mse']:14.3f}\n")
                else:
                    t.insert("end", f"{label:26}{m['n']:6}{m['miss']:20.3f}{m['acc']:11.3f}{m['r2u']:14.3f}\n")

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
            t.insert("end", f"\nTerms, from the fit on {where}\n", "head")
            stat = "t ratio" if p.kind == "number" else "z"
            t.insert("end", f"{'term':26}{'estimate':>12}{'std error':>12}{stat:>10}{'p value':>12}{'logworth':>10}\n", "head")
            for n, est, se, st, pv, lw in term_table(p, r.final.model):
                t.insert("end", f"{n[:25]:26}{est:12.4f}{se:12.4f}{st:10.2f}{pv:12.3g}{lw:10.2f}\n")
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
        def draw_logworth(self):
            r, p = self.result, self.p
            rows = [x for x in term_table(p, r.final.model) if x[0] != "Intercept"]
            rows.sort(key=lambda x: x[5])
            fig = self.fig_lw
            fig.clear()
            ax = fig.add_subplot(111)
            lw = [min(x[5], 16) for x in rows]
            bars = ax.barh([x[0] for x in rows], lw, color=["#2c6fbb" if x[4] < 0.05 else "#9bb7d8" for x in rows])
            ax.axvline(-np.log10(0.05), color="#c0392b", lw=1, ls="--")
            ax.text(-np.log10(0.05), len(rows) - 0.4, " p = 0.05", color="#c0392b", fontsize=9, va="bottom")
            for b, v, x in zip(bars, lw, rows):
                ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{x[5]:.2f}", va="center", fontsize=9)
            ax.set_xlabel("logworth, -log10(p value)")
            ax.set_xlim(0, max(1.6, max(lw + [0]) * 1.18 + 0.2))
            what = "multiple regression" if p.kind == "number" else f"logistic regression, P({p.positive})"
            ax.set_title(f"{what}: logworth of every term, darker bars below p = 0.05", fontsize=10)
            fig.tight_layout()
            self.canvas_lw.draw()

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
