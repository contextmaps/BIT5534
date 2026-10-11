#!/usr/bin/env python3
"""
Single-window neural network explorer.

BIT 5534 lec7_1 demonstration and backup tool, written 2026-10-05 against the
same specification the students send, so that what the room sees on the
projector is what their own tools should show.

The network is written out in numpy and fitted with scipy's L-BFGS-B, so every
layer can mix activation types, the way JMP's hidden layer table does. One or
two hidden layers. Each hidden neuron computes a bias plus a weighted sum of the
layer before it and passes it through its own activation. A number target gets
one linear output neuron and is fitted by mean squared error. A category target
gets one output neuron per class through a softmax and is fitted by cross
entropy.

Every row carries a role: train, validation or test, or for k folds
validation_1 to validation_k plus an optional test. The model is fitted on the
train rows only. Each start begins from its own random weights, and the start
with the smallest validation error is kept. Test rows play no part in any
choice and are only reported.

    python nn_explorer.py                    opens a file dialog
    python nn_explorer.py car_prices.csv     opens that file
    python nn_explorer.py --selftest FILE    prints reference numbers, opens no window

Required packages: pandas, numpy, matplotlib, scipy (and openpyxl for .xlsx).
"""

import os
import secrets
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SELFTEST = "--selftest" in sys.argv
if not SELFTEST:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

ACTIVATIONS = ["tanh", "relu", "logistic", "identity"]
ACT_COLORS = {"tanh": "#2c6fbb", "relu": "#d68910", "logistic": "#27ae60", "identity": "#7f8c8d"}
ROLE_COLORS = {"train": "#2c6fbb", "validation": "#d68910", "test": "#8e44ad"}
CLASS_COLORS = ["#c0392b", "#2c6fbb", "#27ae60", "#d68910", "#8e44ad", "#16a085"]
MAX_ITER = 1000


# ----------------------------------------------------------------- activations
def act(name, z):
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return z


def act_slope(name, a, z):
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return (z > 0).astype(float)
    if name == "logistic":
        return a * (1.0 - a)
    return np.ones_like(z)


# ----------------------------------------------------------------- the network
def layer_list(counts):
    """counts is [{act: n}, {act: n}]. Returns one list of activation names per hidden layer."""
    layers = []
    for c in counts:
        names = [a for a in ACTIVATIONS for _ in range(int(c.get(a, 0)))]
        if names:
            layers.append(names)
    return layers


def shapes_for(k, layers, s):
    """Weight matrix shapes, each with a bias row on top."""
    shapes, prev = [], k
    for names in layers:
        shapes.append((prev + 1, len(names)))
        prev = len(names)
    shapes.append((prev + 1, s))
    return shapes


def n_params(k, layers, s):
    return sum(r * c for r, c in shapes_for(k, layers, s))


def unpack(w, shapes):
    out, i = [], 0
    for r, c in shapes:
        out.append(w[i:i + r * c].reshape(r, c))
        i += r * c
    return out


def forward(W, X, layers):
    A, Z = [X], []
    for Wl, names in zip(W[:-1], layers):
        z = np.c_[np.ones(len(A[-1])), A[-1]] @ Wl
        a = np.empty_like(z)
        for j, nm in enumerate(names):
            a[:, j] = act(nm, z[:, j])
        Z.append(z)
        A.append(a)
    out = np.c_[np.ones(len(A[-1])), A[-1]] @ W[-1]
    return out, A, Z


def softmax(o):
    o = o - o.max(axis=1, keepdims=True)
    e = np.exp(o)
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grad(w, shapes, X, Y, layers, kind):
    W = unpack(w, shapes)
    out, A, Z = forward(W, X, layers)
    n = len(X)
    if kind == "number":
        r = out - Y
        loss = float((r * r).sum() / n)
        d = 2.0 * r / n
    else:
        P = softmax(out)
        loss = float(-(Y * np.log(np.clip(P, 1e-300, None))).sum() / n)
        d = (P - Y) / n
    g = [None] * len(W)
    g[-1] = np.c_[np.ones(n), A[-1]].T @ d
    d = d @ W[-1][1:].T
    for l in range(len(layers) - 1, -1, -1):
        slope = np.empty_like(Z[l])
        for j, nm in enumerate(layers[l]):
            slope[:, j] = act_slope(nm, A[l + 1][:, j], Z[l][:, j])
        d = d * slope
        g[l] = np.c_[np.ones(n), A[l]].T @ d
        d = d @ W[l][1:].T
    return loss, np.concatenate([x.ravel() for x in g])


def fit_once(X, Y, layers, kind, rng):
    shapes = shapes_for(X.shape[1], layers, Y.shape[1])
    w0 = rng.uniform(-0.5, 0.5, sum(r * c for r, c in shapes))
    res = minimize(loss_and_grad, w0, args=(shapes, X, Y, layers, kind), jac=True,
                   method="L-BFGS-B", options={"maxiter": MAX_ITER})
    return unpack(res.x, shapes)


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
    roles = np.asarray(roles).astype(str)
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


def target_matrix(p, rows):
    if p.kind == "number":
        return p.y[rows][:, None]
    return (p.y[rows][:, None] == np.array(p.classes)[None, :]).astype(float)


class FoldFit:
    pass


def fit_fold(p, train, val, layers, starts, rng, standardize):
    """Fit the network from `starts` random starting points on the train rows and keep the
    start with the smallest validation error. Returns everything a report needs."""
    mu, sd = scale_from(p.X, train, standardize)
    Xs = (p.X - mu) / sd
    Yt = target_matrix(p, train)
    if p.kind == "number":
        ym, ys = Yt.mean(), Yt.std() or 1.0
        Yt = (Yt - ym) / ys
    runs = []
    for s in range(starts):
        W = fit_once(Xs[train], Yt, layers, p.kind, rng)
        pred = predict_raw(W, Xs, layers, p, ym if p.kind == "number" else None,
                           ys if p.kind == "number" else None)
        tr = metrics(p, pred, train)
        va = metrics(p, pred, val) if val.any() else None
        runs.append((W, pred, tr, va))
    key = (lambda r: r[3]["choose"]) if val.any() else (lambda r: r[2]["choose"])
    best = min(range(len(runs)), key=lambda i: key(runs[i]))
    f = FoldFit()
    f.runs, f.best = runs, best
    f.W, f.pred = runs[best][0], runs[best][1]
    f.mu, f.sd = mu, sd
    return f


def predict_raw(W, Xs, layers, p, ym, ys):
    out, _, _ = forward(W, Xs, layers)
    if p.kind == "number":
        return out[:, 0] * ys + ym
    return softmax(out)


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
    P = pred[rows]
    yhat = np.array(p.classes)[P.argmax(axis=1)]
    y = p.y[rows]
    miss = float((yhat != y).mean())
    Y = (y[:, None] == np.array(p.classes)[None, :]).astype(float)
    ll = float(-(Y * np.log(np.clip(P, 1e-300, None))).sum() / len(y))
    return {"n": int(rows.sum()), "miss": miss, "acc": 1 - miss, "logloss": ll,
            "choose": miss + 1e-9 * ll}


class Result:
    pass


def run_model(p, roles, layers, starts, weight_seed, standardize):
    folds, test = folds_of(roles)
    rng = np.random.default_rng(weight_seed)
    r = Result()
    r.folds, r.fits = folds, []
    for lab, val, train in folds:
        r.fits.append(fit_fold(p, train, val, layers, starts, rng, standardize))
    r.test = test
    r.kfold = folds[0][0] != "validation"
    if not r.kfold:
        r.pred = r.fits[0].pred
    else:
        # Each validation row is predicted by the fold model that did not see it.
        # A test row is predicted by the average of the k fold models.
        r.pred = np.zeros_like(r.fits[0].pred)
        for (lab, val, train), f in zip(folds, r.fits):
            r.pred[val] = f.pred[val]
        avg = sum(f.pred for f in r.fits) / len(r.fits)
        r.pred[test] = avg[test]
    r.layers = layers
    r.k = p.X.shape[1]
    r.s = 1 if p.kind == "number" else len(p.classes)
    r.params = n_params(r.k, layers, r.s)
    return r


def predicted_column(p, pred):
    if p.kind == "number":
        return pred
    return np.array(p.classes)[pred.argmax(axis=1)]


# ----------------------------------------------------------------- self test
def selftest(path):
    df = read_table(path)
    target = "Price"
    inputs = ["Mileage", "Liter", "Doors", "Cruise", "Sound", "Leather"]
    p = prepare(df, target, inputs, "number", None)
    print(f"file {path}: {len(p.data)} rows, {p.dropped} dropped, {len(p.names)} inputs")
    roles = make_roles(len(p.data), "holdout", 1 / 3, 0.0, 5, 123)
    print("roles", pd.Series(roles).value_counts().to_dict())
    for counts in ([{"identity": 1}], [{"tanh": 3}], [{"tanh": 10}], [{"tanh": 40}]):
        layers = layer_list(counts)
        r = run_model(p, roles, layers, 1, 123, True)
        f = r.fits[0]
        tr, va = f.runs[f.best][2], f.runs[f.best][3]
        print(f"{[len(l) for l in layers]} params {r.params}  train R2 {tr['r2']:.3f}  "
              f"validation R2 {va['r2']:.3f}  validation RMSE {va['rmse']:.0f}")


# ----------------------------------------------------------------- the window
if not SELFTEST:
    class App(tk.Tk):
        def __init__(self, path=None):
            super().__init__()
            self.title("Neural network explorer")
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
            outer = ttk.Frame(self)
            outer.pack(side="left", fill="y")
            side = ttk.Frame(outer, padding=8)
            side.pack(fill="y")

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
            self.kind_var = tk.StringVar(value="number")
            ttk.Radiobutton(kf, text="number", value="number", variable=self.kind_var).pack(side="left")
            ttk.Radiobutton(kf, text="category", value="category", variable=self.kind_var).pack(side="left")

            ttk.Label(side, text="Input columns").pack(anchor="w", pady=(8, 0))
            self.input_list = tk.Listbox(side, selectmode="multiple", height=8, exportselection=False)
            self.input_list.pack(fill="x")

            # role column
            rf = ttk.LabelFrame(side, text="Validation column", padding=6)
            rf.pack(fill="x", pady=(8, 0))
            ttk.Label(rf, text="Use the column").grid(row=0, column=0, sticky="w")
            self.role_var = tk.StringVar(value="(make one)")
            self.role_box = ttk.Combobox(rf, textvariable=self.role_var, state="readonly", width=16)
            self.role_box.grid(row=0, column=1, sticky="ew")
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

            # network
            nf = ttk.LabelFrame(side, text="Hidden layers, neurons of each activation", padding=6)
            nf.pack(fill="x", pady=(8, 0))
            ttk.Label(nf, text="").grid(row=0, column=0)
            for j, a in enumerate(ACTIVATIONS):
                ttk.Label(nf, text=a).grid(row=0, column=j + 1)
            self.count_vars = []
            for i, lab in enumerate(["First", "Second"]):
                ttk.Label(nf, text=lab).grid(row=i + 1, column=0, sticky="w")
                row = {}
                for j, a in enumerate(ACTIVATIONS):
                    v = tk.StringVar(value="3" if (i == 0 and a == "tanh") else "0")
                    ttk.Spinbox(nf, from_=0, to=50, width=4, textvariable=v).grid(row=i + 1, column=j + 1)
                    row[a] = v
                self.count_vars.append(row)
            ttk.Label(nf, text="First is the layer next to the inputs.", foreground="#555").grid(
                row=3, column=0, columnspan=5, sticky="w")

            of = ttk.Frame(side)
            of.pack(fill="x", pady=(8, 0))
            self.std_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(of, text="Standardize inputs", variable=self.std_var).grid(row=0, column=0, columnspan=2, sticky="w")
            ttk.Label(of, text="Number of starts").grid(row=1, column=0, sticky="w")
            self.starts_var = tk.StringVar(value="1")
            ttk.Spinbox(of, from_=1, to=100, width=6, textvariable=self.starts_var).grid(row=1, column=1, sticky="w")
            ttk.Label(of, text="Seed (empty = random)").grid(row=2, column=0, sticky="w")
            self.weight_seed_var = tk.StringVar(value="123")
            ttk.Entry(of, textvariable=self.weight_seed_var, width=10).grid(row=2, column=1, sticky="w")

            ttk.Button(side, text="Fit the network", command=self.fit).pack(fill="x", pady=(10, 0))
            ttk.Button(side, text="Save predictions...", command=self.save_predictions).pack(fill="x", pady=(4, 0))
            self.status = ttk.Label(side, text="", wraplength=270, foreground="#444")
            self.status.pack(fill="x", pady=(8, 0))

            self.tabs = ttk.Notebook(self)
            self.tabs.pack(side="left", fill="both", expand=True)
            self.fit_text = self._text_tab("Fit")
            self.fig_net, self.canvas_net = self._figure_tab("Network")
            self.fig_pred, self.canvas_pred = self._figure_tab("Actual vs predicted")
            self.starts_text = self._text_tab("Starts")
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
            text.tag_configure("best", foreground="#c0392b", font=("Consolas", 11, "bold"))
            return text

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
            self.target_var.set("Price" if "Price" in cols else cols[0])
            self.refresh_roles()
            self.target_changed()
            self.status.config(text=f"{len(self.raw)} rows read. Make a validation column, "
                                    f"then fit the network.")

        def role_like(self, c):
            vals = set(self.raw[c].dropna().astype(str).unique())
            return bool(vals) and all(v in ("train", "validation", "test") or v.startswith("validation_")
                                      for v in vals)

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
            """Rebuild the input list after a new column, keeping the choices already made."""
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
            name = "validation" if method == "holdout" else f"fold{k}"
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
            role = self.role_var.get()
            if role == "(make one)" or role not in self.raw.columns:
                t.insert("end", "No validation column chosen yet.\n")
                return
            t.insert("end", f"Column {role}\n\n", "head")
            counts = self.raw[role].value_counts()
            order = sorted(counts.index, key=lambda r: (r == "test", r))
            for r in order:
                t.insert("end", f"{r:16}{counts[r]:6} rows   {100 * counts[r] / len(self.raw):5.1f}%\n")
            folds, test = folds_of(self.raw[role].astype(str).to_numpy())
            t.insert("end", "\nWhat each fit uses\n", "head")
            for lab, val, train in folds:
                t.insert("end", f"fit on {int(train.sum())} rows, judged on the {int(val.sum())} rows marked {lab}\n")
            if test.any():
                t.insert("end", f"\n{int(test.sum())} test rows are reported only, and no choice uses them.\n")

        # ------------------------------------------------------------- fit
        def counts(self):
            return [{a: int(v.get() or 0) for a, v in row.items()} for row in self.count_vars]

        def fit(self):
            if self.raw is None:
                self.open_file()
                return
            try:
                self._fit()
            except Exception as exc:
                self.status.config(text=f"Could not fit: {type(exc).__name__}: {exc}")
                messagebox.showerror("Could not fit", f"{type(exc).__name__}: {exc}")

        def _fit(self):
            role = self.role_var.get()
            if role == "(make one)" or role not in self.raw.columns:
                self.status.config(text="Make a validation column first, or choose one the file already has.")
                return
            inputs = [self.input_list.get(i) for i in self.input_list.curselection()]
            if not inputs:
                self.status.config(text="Choose at least one input column.")
                return
            counts = self.counts()
            if sum(counts[0].values()) == 0 and sum(counts[1].values()) > 0:
                self.status.config(text="The second layer has neurons and the first has none. "
                                        "Put neurons in the first layer.")
                return
            layers = layer_list(counts)
            if not layers:
                self.status.config(text="Put at least one neuron in the first hidden layer.")
                return
            starts = max(1, int(self.starts_var.get()))
            seed, drawn = draw_seed(self.weight_seed_var.get())
            target = self.target_var.get()
            self.p = prepare(self.raw, target, inputs, self.kind_var.get(), role)
            roles = self.p.data[role].astype(str).to_numpy()
            self.status.config(text="Fitting...")
            self.update_idletasks()
            self.result = run_model(self.p, roles, layers, starts, seed, self.std_var.get())
            self.result.seed, self.result.drawn, self.result.starts = seed, drawn, starts
            self.result.roles = roles
            how = f"seed {seed} drawn at random" if drawn else f"seed {seed}"
            self.status.config(text=f"{len(self.p.data)} rows used, {self.p.dropped} dropped for missing "
                                    f"values. {starts} start(s) from {how}.")
            self.show_fit()
            self.draw_network()
            self.draw_predictions()
            self.show_starts()
            self.show_roles()

        # ------------------------------------------------------------- outputs
        def show_fit(self):
            r, p = self.result, self.p
            t = self.fit_text
            t.delete("1.0", "end")
            shape = " then ".join(", ".join(f"{n} {a}" for a, n in
                                            pd.Series(names).value_counts().reindex(ACTIVATIONS).dropna().astype(int).items())
                                  for names in r.layers)
            t.insert("end", f"Network: {r.k} inputs, hidden {shape}, {r.s} output(s)\n", "head")
            sizes = [r.k] + [len(l) for l in r.layers]
            terms = [f"{sizes[i + 1]} x ({sizes[i]} + 1)" for i in range(len(r.layers))]
            terms.append(f"{r.s} x ({sizes[-1]} + 1)")
            t.insert("end", f"Parameters: {' + '.join(terms)} = {r.params}\n")
            t.insert("end", f"Inputs: {', '.join(p.names)}\n")
            t.insert("end", f"Standardized inputs: {'yes, from the train rows' if self.std_var.get() else 'no'}\n\n")
            if p.kind == "number":
                head = f"{'rows':26}{'n':>6}{'R squared':>12}{'RMSE':>12}{'MSE':>16}\n"
            else:
                head = f"{'rows':26}{'n':>6}{'misclassification':>20}{'accuracy':>11}{'log loss':>10}\n"
            t.insert("end", head, "head")

            def line(label, m):
                if m is None:
                    return
                if p.kind == "number":
                    t.insert("end", f"{label:26}{m['n']:6}{m['r2']:12.3f}{m['rmse']:12.1f}{m['mse']:16.1f}\n")
                else:
                    t.insert("end", f"{label:26}{m['n']:6}{m['miss']:20.3f}{m['acc']:11.3f}{m['logloss']:10.3f}\n")

            vals = []
            for (lab, val, train), f in zip(r.folds, r.fits):
                tr = f.runs[f.best][2]
                va = f.runs[f.best][3]
                pre = f"{lab}: " if r.kfold else ""
                line(f"{pre}train", tr)
                line(f"{pre}validation", va)
                if va:
                    vals.append(va)
            if r.kfold and vals:
                key = "r2" if p.kind == "number" else "miss"
                avg = np.mean([v[key] for v in vals])
                name = "R squared" if p.kind == "number" else "misclassification"
                t.insert("end", f"\nAverage validation {name} over {len(vals)} folds: {avg:.3f}\n", "head")
            if r.test.any():
                t.insert("end", "\n")
                line("test (reported only)", metrics(p, r.pred, r.test))
            if p.kind == "category":
                self.confusion(t, r.pred, ~r.test if not r.kfold else None)

        def confusion(self, t, pred, _):
            p, r = self.p, self.result
            groups = [("train", r.folds[0][2])] if not r.kfold else []
            groups += [("validation", np.isin(r.roles, [lab for lab, _, _ in r.folds]))]
            if r.test.any():
                groups.append(("test", r.test))
            yhat = predicted_column(p, pred)
            for name, rows in groups:
                if not rows.any():
                    continue
                t.insert("end", f"\nConfusion table, {name} rows (actual down, predicted across)\n", "head")
                t.insert("end", f"{'':12}" + "".join(f"{c:>18}" for c in p.classes) + "\n")
                for a in p.classes:
                    row = rows & (p.y == a)
                    tot = int(row.sum())
                    t.insert("end", f"{a:12}")
                    for c in p.classes:
                        n = int((row & (yhat == c)).sum())
                        t.insert("end", f"{n:>9} ({100 * n / tot if tot else 0:5.1f}%) ")
                    t.insert("end", "\n")

        def draw_network(self):
            r = self.result
            fig = self.fig_net
            fig.clear()
            ax = fig.add_subplot(111)
            ax.axis("off")
            cols = [[("input", n) for n in self.p.names]]
            for names in r.layers:
                cols.append([(a, a) for a in names])
            outs = [self.target_var.get()] if self.p.kind == "number" else list(self.p.classes)
            cols.append([("output", o) for o in outs])
            pos = []
            for i, col in enumerate(cols):
                n = len(col)
                pos.append([(i, (j + 1) / (n + 1)) for j in range(n)])
            for i in range(len(cols) - 1):
                for x0, y0 in pos[i]:
                    for x1, y1 in pos[i + 1]:
                        ax.plot([x0, x1], [y0, y1], color="#bbb", lw=0.6, zorder=1)
            for i, col in enumerate(cols):
                for (kind, label), (x, y) in zip(col, pos[i]):
                    colour = ACT_COLORS.get(kind, "#333" if kind == "output" else "#444")
                    ax.scatter([x], [y], s=380, color="white", edgecolor=colour, lw=2.2, zorder=2)
                    if kind == "input":
                        ax.text(x - 0.08, y, label, ha="right", va="center", fontsize=9)
                    elif kind == "output":
                        ax.text(x + 0.08, y, label, ha="left", va="center", fontsize=9)
            heads = ["inputs"] + [f"hidden layer {i + 1}" for i in range(len(r.layers))] + ["output"]
            for i, h in enumerate(heads):
                ax.text(i, 1.02, h, ha="center", fontsize=10, fontweight="bold")
            for j, a in enumerate(ACTIVATIONS):
                ax.scatter([], [], s=80, color="white", edgecolor=ACT_COLORS[a], lw=2, label=a)
            ax.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
            ax.set_xlim(-0.9, len(cols) - 0.2)
            ax.set_ylim(-0.05, 1.07)
            ax.set_title(f"{r.params} parameters", fontsize=11)
            self.canvas_net.draw()

        def draw_predictions(self):
            r, p = self.result, self.p
            fig = self.fig_pred
            fig.clear()
            if p.kind == "category":
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.5, 0.5, "For a category target the confusion tables are on the Fit tab.",
                        ha="center", va="center", fontsize=12)
                self.canvas_pred.draw()
                return
            groups = []
            if not r.kfold:
                groups = [("train", r.folds[0][2]), ("validation", r.folds[0][1])]
            else:
                groups = [("validation", ~r.test)]
            if r.test.any():
                groups.append(("test", r.test))
            n = len(groups)
            lo = min(p.y.min(), r.pred.min())
            hi = max(p.y.max(), r.pred.max())
            for i, (name, rows) in enumerate(groups):
                ax = fig.add_subplot(1, n, i + 1)
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

        def show_starts(self):
            r, p = self.result, self.p
            t = self.starts_text
            t.delete("1.0", "end")
            name = "MSE" if p.kind == "number" else "misclassification"
            t.insert("end", f"Each start begins from its own random weights. The kept start has the "
                            f"smallest validation {name}.\n\n", "head")
            for (lab, val, train), f in zip(r.folds, r.fits):
                if r.kfold:
                    t.insert("end", f"{lab}\n", "head")
                t.insert("end", f"{'start':>7}{'train ' + name:>24}{'validation ' + name:>26}\n", "head")
                for i, (_, _, tr, va) in enumerate(f.runs):
                    v = va["choose"] if va else float("nan")
                    tag = "best" if i == f.best else None
                    fmt = "{:24.1f}{:26.1f}" if p.kind == "number" else "{:24.3f}{:26.3f}"
                    t.insert("end", f"{i + 1:>7}" + fmt.format(tr["choose"], v)
                             + ("   kept" if i == f.best else "") + "\n", tag)
                t.insert("end", "\n")

        def save_predictions(self):
            if self.result is None:
                return
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not path:
                return
            out = self.p.data.copy()
            out["predicted"] = predicted_column(self.p, self.result.pred)
            if self.p.kind == "category":
                for j, c in enumerate(self.p.classes):
                    out[f"prob_{c}"] = self.result.pred[:, j].round(4)
            out.to_csv(path, index=False)
            self.status.config(text=f"Saved {len(out)} rows to {os.path.basename(path)}.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--selftest"]
    if SELFTEST:
        selftest(args[0] if args else "car_prices.csv")
    else:
        App(args[0] if args else None).mainloop()
