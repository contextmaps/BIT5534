#!/usr/bin/env python3
"""Single-predictor logistic regression GUI.

Dependencies: pandas, numpy, statsmodels, matplotlib, openpyxl (for .xlsx files).
"""

import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import PerfectSeparationError

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class LogisticApp:
    def __init__(self, root, path=None):
        self.root = root
        self.root.title("Single-Predictor Logistic Regression")
        self.root.geometry("650x470")
        self.df = None
        self.event_values = []

        self.response_var = tk.StringVar()
        self.event_var = tk.StringVar()
        self.input_var = tk.StringVar()
        self.step_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Choose a data file to begin.")

        self._build_ui()
        if path:
            self.load_file(path)
        else:
            self.choose_file()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        ttk.Label(outer, text="Data file:").grid(row=0, column=0, sticky="w", pady=5)
        self.file_label = ttk.Label(outer, text="No file loaded", foreground="navy")
        self.file_label.grid(row=0, column=1, sticky="w", pady=5)
        ttk.Button(outer, text="Choose file", command=self.choose_file).grid(row=0, column=2, padx=6)

        ttk.Label(outer, text="Response column:").grid(row=1, column=0, sticky="w", pady=5)
        self.response_box = ttk.Combobox(outer, textvariable=self.response_var, state="readonly")
        self.response_box.grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)
        self.response_box.bind("<<ComboboxSelected>>", self.response_changed)

        ttk.Label(outer, text="Event value:").grid(row=2, column=0, sticky="w", pady=5)
        self.event_box = ttk.Combobox(outer, textvariable=self.event_var, state="readonly")
        self.event_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=5)

        ttk.Label(outer, text="Numeric input column:").grid(row=3, column=0, sticky="w", pady=5)
        self.input_box = ttk.Combobox(outer, textvariable=self.input_var, state="readonly")
        self.input_box.grid(row=3, column=1, columnspan=2, sticky="ew", pady=5)

        note = "Text columns cannot be used as the model input."
        ttk.Label(outer, text=note, foreground="gray").grid(row=4, column=1, columnspan=2, sticky="w")

        ttk.Label(outer, text="Step size:").grid(row=5, column=0, sticky="w", pady=(18, 5))
        ttk.Entry(outer, textvariable=self.step_var, width=15).grid(row=5, column=1, sticky="w", pady=(18, 5))
        ttk.Label(outer, text="(used for the input odds multiplier)", foreground="gray").grid(row=5, column=2, sticky="w", pady=(18, 5))

        ttk.Button(outer, text="Run", command=self.run_model).grid(row=6, column=1, sticky="w", pady=20)
        ttk.Label(outer, textvariable=self.status_var, foreground="darkred", wraplength=590).grid(
            row=7, column=0, columnspan=3, sticky="w"
        )

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("CSV or Excel", "*.csv *.xlsx"), ("CSV", "*.csv"), ("Excel", "*.xlsx")],
        )
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            suffix = path.lower().rsplit(".", 1)[-1]
            if suffix == "csv":
                df = pd.read_csv(path)
            elif suffix == "xlsx":
                df = pd.read_excel(path)
            else:
                raise ValueError("The file must have a .csv or .xlsx extension.")
            if df.shape[1] < 2:
                raise ValueError("The file must contain at least two columns.")
            self.df = df
            self.file_label.config(text=path)
            columns = [str(c) for c in df.columns]
            self.response_box["values"] = columns
            self.response_var.set(columns[0])
            self.response_changed()
            self.status_var.set(f"Loaded {len(df):,} rows and {len(columns):,} columns.")
        except Exception as exc:
            self.df = None
            self.status_var.set(f"File error: {exc}")
            messagebox.showerror("Could not load file", str(exc), parent=self.root)

    def response_changed(self, _event=None):
        if self.df is None or not self.response_var.get():
            return
        col = self.response_var.get()
        vals = list(pd.unique(self.df[col].dropna()))
        try:
            vals = sorted(vals)
        except TypeError:
            vals = sorted(vals, key=lambda v: str(v))
        self.event_values = vals if len(vals) == 2 else vals[:2]
        labels = [self.display_value(v) for v in self.event_values]
        self.event_box["values"] = labels
        self.event_var.set(labels[-1] if labels else "")

        numeric = [str(c) for c in self.df.select_dtypes(include=[np.number]).columns if str(c) != col]
        self.input_box["values"] = numeric
        self.input_var.set(numeric[0] if numeric else "")

    @staticmethod
    def display_value(value):
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    @staticmethod
    def logworth(p):
        if not np.isfinite(p) or p <= 0:
            return "infinite (p numerically 0)"
        return f"{-np.log10(p):.6g}"

    @staticmethod
    def fmt(value):
        return f"{value:.6g}" if np.isfinite(value) else "not finite"

    def run_model(self):
        if self.df is None:
            messagebox.showerror("Missing data", "Load a .csv or .xlsx file first.", parent=self.root)
            return
        response = self.response_var.get()
        predictor = self.input_var.get()
        if not response or not predictor:
            messagebox.showerror("Incomplete choices", "Select a response and a numeric input column.", parent=self.root)
            return
        vals = list(pd.unique(self.df[response].dropna()))
        if len(vals) != 2:
            messagebox.showerror("Response is not binary", "The selected response must contain exactly two distinct nonmissing values.", parent=self.root)
            return
        try:
            step = float(self.step_var.get().strip())
            if not np.isfinite(step):
                raise ValueError
        except (ValueError, AttributeError):
            messagebox.showerror("Invalid step size", "Enter a finite numeric step size.", parent=self.root)
            return

        event_label = self.event_var.get()
        try:
            event = next(v for v in vals if self.display_value(v) == event_label)
        except StopIteration:
            event = vals[0]
        non_event = next(v for v in vals if not self.same_value(v, event))

        data = self.df[[response, predictor]].dropna().copy()
        if data.empty:
            messagebox.showerror("No usable rows", "No rows remain after excluding missing response or input values.", parent=self.root)
            return
        x = pd.to_numeric(data[predictor], errors="coerce")
        if x.isna().any() or not np.isfinite(x).all():
            messagebox.showerror("Invalid input", "The selected input column must contain only finite numeric values.", parent=self.root)
            return
        if x.nunique() < 2:
            messagebox.showerror("Constant predictor", "The selected input has fewer than two distinct values.", parent=self.root)
            return
        y = data[response].map(lambda v: int(self.same_value(v, event))).astype(int)
        if y.nunique() < 2:
            messagebox.showerror("Response is not binary in usable rows", "Both response classes must remain after missing rows are excluded.", parent=self.root)
            return

        X = sm.add_constant(pd.DataFrame({predictor: x.to_numpy()}), has_constant="add")
        try:
            model = sm.Logit(y.to_numpy(), X).fit(disp=False, maxiter=200)
            if not np.isfinite(model.params).all() or not np.isfinite(model.bse).all():
                raise ValueError("The fitted coefficients are not finite.")
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Model fitting failed", f"The logistic model could not be fit.\n\n{exc}", parent=self.root)
            return
        except Exception as exc:
            messagebox.showerror("Model fitting failed", str(exc), parent=self.root)
            return

        null_X = np.ones((len(y), 1))
        try:
            null_model = sm.Logit(y.to_numpy(), null_X).fit(disp=False, maxiter=200)
            lr = 2 * (model.llf - null_model.llf)
            lr_p = chi2.sf(lr, 1)
        except Exception:
            lr, lr_p = np.nan, np.nan

        beta = float(model.params[predictor])
        one_unit_or = np.exp(beta) if beta < 709 else np.inf
        step_or = np.exp(beta * step) if beta * step < 709 else np.inf
        pred_prob = model.predict(X)
        pred = (pred_prob >= 0.5).astype(int)
        table = self.classification_table(y.to_numpy(), pred, event, non_event)
        self.show_results(response, event, non_event, len(y), predictor, step, model, lr, lr_p, one_unit_or, step_or, x, y, table)

    @staticmethod
    def same_value(a, b):
        try:
            return bool(a == b)
        except Exception:
            return repr(a) == repr(b)

    def classification_table(self, actual, predicted, event, non_event):
        labels = [(1, event), (0, non_event)]
        rows = []
        for actual_code, actual_value in labels:
            mask = actual == actual_code
            n = int(mask.sum())
            cells = []
            for pred_code, _ in labels:
                count = int(((predicted == pred_code) & mask).sum())
                pct = 100 * count / n if n else 0.0
                cells.append((count, pct))
            rows.append((actual_value, n, cells))
        return rows

    def show_results(self, response, event, non_event, n, predictor, step, model, lr, lr_p, one_or, step_or, x, y, table):
        win = tk.Toplevel(self.root)
        win.title("Logistic Regression Results")
        win.geometry("980x850")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        frame = ttk.Frame(win)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        text = tk.Text(frame, wrap="none", height=25, font=("Courier New", 10))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        lines = [
            "LOGISTIC REGRESSION RESULTS", "=" * 72,
            f"Response column: {response}",
            f"Event value: {self.display_value(event)} (treated as the event)",
            f"Non-event value: {self.display_value(non_event)} (treated as the non-event)",
            f"Rows used in model: {n:,}", "",
            "Whole-model likelihood-ratio test", "-" * 35,
            f"Likelihood-ratio chi-square: {self.fmt(lr)}",
            "Degrees of freedom: 1",
            f"p-value: {self.fmt(lr_p)}", "",
            f"Coefficient results (input: {predictor})", "-" * 72,
            f"{'Term':<20}{'Estimate':>13}{'Std. Error':>13}{'Odds multiplier':>18}{'p-value':>14}{'Logworth':>18}",
        ]
        for name in ["Intercept", predictor]:
            i = 0 if name == "Intercept" else 1
            p = float(model.pvalues.iloc[i] if hasattr(model.pvalues, "iloc") else model.pvalues[i])
            est = float(model.params.iloc[i] if hasattr(model.params, "iloc") else model.params[i])
            se = float(model.bse.iloc[i] if hasattr(model.bse, "iloc") else model.bse[i])
            odds = np.exp(est) if est < 709 else np.inf
            lines.append(f"{name:<20}{self.fmt(est):>13}{self.fmt(se):>13}{self.fmt(odds):>18}{self.fmt(p):>14}{self.logworth(p):>18}")
        lines += ["", f"Odds multiplier for {predictor} change of {self.fmt(step)}: {self.fmt(step_or)}", "", "Classification table (cutoff = 0.5)", "-" * 44]
        lines.append("Rows are actual. Percentages are of the actual row.")
        lines.append(f"Actual event ({self.display_value(event)}):")
        r = table[0][2]
        lines.append(f"    Predicted event:     {r[0][0]:>8} ({r[0][1]:6.2f}% of actual events)")
        lines.append(f"    Predicted non-event: {r[1][0]:>8} ({r[1][1]:6.2f}% of actual events)")
        lines.append(f"Actual non-event ({self.display_value(non_event)}):")
        r = table[1][2]
        lines.append(f"    Predicted event:     {r[0][0]:>8} ({r[0][1]:6.2f}% of actual non-events)")
        lines.append(f"    Predicted non-event: {r[1][0]:>8} ({r[1][1]:6.2f}% of actual non-events)")
        text.insert("1.0", "\n".join(lines) + "\n")
        text.configure(state="disabled")

        fig = Figure(figsize=(8.5, 5.2), dpi=100)
        ax = fig.add_subplot(111)
        order = np.argsort(x.to_numpy())
        xs = x.to_numpy()[order]
        curve_X = sm.add_constant(pd.DataFrame({predictor: xs}), has_constant="add")
        curve = model.predict(curve_X)
        ax.scatter(x, y, color="black", alpha=0.65, label="Observed response", zorder=2)
        ax.plot(xs, curve, color="tab:blue", linewidth=2, label="Fitted probability")
        ax.set_xlabel(predictor)
        ax.set_ylabel("Probability of event")
        ax.set_ylim(0, 1)
        ax.set_title(f"{response} versus {predictor}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky="nsew", pady=8)
        frame.rowconfigure(1, weight=1)


def main():
    root = tk.Tk()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    LogisticApp(root, path)
    root.mainloop()


if __name__ == "__main__":
    main()
