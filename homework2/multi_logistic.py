"""Desktop logistic regression tool.

Install dependencies once:
    python -m pip install pandas openpyxl statsmodels scipy matplotlib

Double-click this file to launch it. The initial file chooser is followed by
one main setup window containing all model choices.
"""

import math
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import PerfectSeparationError


P_MIN = 1e-300


def display_value(value):
    if pd.isna(value):
        return "<missing>"
    return str(value)


def format_number(value):
    if not np.isfinite(value):
        return "not finite"
    return f"{value:.6g}"


def p_text(p):
    if not np.isfinite(p) or p <= P_MIN:
        return "< 1e-300"
    return f"{p:.6g}"


def logworth(p):
    if not np.isfinite(p) or p <= P_MIN:
        return -math.log10(P_MIN)
    return -math.log10(max(p, P_MIN))


def make_effect_column(values, level_order, level):
    omitted = level_order[0]
    return np.array([
        1.0 if value == level else (-1.0 if value == omitted else 0.0)
        for value in values
    ])


def build_design(data, response, inputs, categorical):
    X = pd.DataFrame(index=data.index)
    X["Intercept"] = 1.0
    omitted = {}

    for name in inputs:
        values = data[name]
        if name in categorical:
            levels = list(pd.unique(values))
            if len(levels) < 2:
                raise ValueError(f"Categorical input '{name}' has fewer than two usable levels.")
            omitted[name] = levels[0]
            for level in levels[1:]:
                term = f"{name}[{display_value(level)}]"
                if term in X.columns:
                    term = f"{name}[{display_value(level)}]#{len(X.columns)}"
                X[term] = make_effect_column(values, levels, level)
        else:
            converted = pd.to_numeric(values, errors="coerce")
            if converted.isna().any():
                bad = int(converted.isna().sum())
                raise ValueError(
                    f"Numeric input '{name}' contains {bad} value(s) that cannot be converted to numeric."
                )
            if converted.nunique() < 2:
                raise ValueError(f"Numeric input '{name}' has no usable variation.")
            X[name] = converted.astype(float)

    if X.shape[1] < 2:
        raise ValueError("The model has no fitted predictor terms.")
    if np.linalg.matrix_rank(X.to_numpy(dtype=float)) < X.shape[1]:
        raise ValueError("The design matrix is singular or contains redundant terms.")
    return X, omitted


def classification_text(actual, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    lines = [
        "Classification table (cutoff = 0.5)",
        "Rows = actual; columns = predicted. Event/positive appears first.",
        "Each cell shows count (row percentage).",
        "",
        "                 Predicted Event    Predicted Non-event",
    ]
    for actual_value, label in [(1, "Actual Event"), (0, "Actual Non-event")]:
        mask = actual == actual_value
        total = int(mask.sum())
        cells = []
        for pred_value in (1, 0):
            count = int(((actual == actual_value) & (predicted == pred_value)).sum())
            pct = 100.0 * count / total if total else 0.0
            cells.append(f"{count} ({pct:.1f}%)")
        lines.append(f"{label:18s} {cells[0]:>18s} {cells[1]:>21s}")
    return "\n".join(lines)


class LogisticApp:
    def __init__(self, root, path, data):
        self.root = root
        self.path = path
        self.data = data
        self.response_var = tk.StringVar()
        self.event_var = tk.StringVar()
        self.status_var = tk.StringVar(value=f"Loaded {len(data):,} rows from {path}")
        self.input_list = None
        self.cat_list = None
        self.event_combo = None
        self.build_ui()

    def build_ui(self):
        self.root.title("Logistic Regression Setup")
        self.root.geometry("900x650")
        self.root.minsize(760, 520)
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Data file:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=self.path).grid(row=0, column=1, columnspan=3, sticky="w", padx=6)
        ttk.Label(top, textvariable=self.status_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 12))

        ttk.Label(top, text="Response column").grid(row=2, column=0, sticky="w")
        response_names = [str(c) for c in self.data.columns]
        response = ttk.Combobox(top, textvariable=self.response_var, values=response_names, state="readonly", width=34)
        response.grid(row=3, column=0, sticky="ew", padx=(0, 12))
        response.bind("<<ComboboxSelected>>", self.response_changed)

        ttk.Label(top, text="Event value treated as 1").grid(row=2, column=1, sticky="w")
        self.event_combo = ttk.Combobox(top, textvariable=self.event_var, state="readonly", width=28)
        self.event_combo.grid(row=3, column=1, sticky="ew", padx=(0, 12))

        ttk.Label(top, text="Input columns (Ctrl/Shift-click for multiple)").grid(row=4, column=0, sticky="w", pady=(15, 4))
        ttk.Label(top, text="Categorical inputs (must also be selected)").grid(row=4, column=1, sticky="w", pady=(15, 4))

        self.input_list = tk.Listbox(top, selectmode=tk.EXTENDED, exportselection=False, height=19)
        self.cat_list = tk.Listbox(top, selectmode=tk.EXTENDED, exportselection=False, height=19)
        self.input_list.grid(row=5, column=0, sticky="nsew", padx=(0, 12))
        self.cat_list.grid(row=5, column=1, sticky="nsew", padx=(0, 12))
        for widget in (self.input_list, self.cat_list):
            scroll = ttk.Scrollbar(top, orient="vertical", command=widget.yview)
            widget.configure(yscrollcommand=scroll.set)
            scroll.grid(row=5, column=2 if widget is self.input_list else 3, sticky="ns")

        help_text = ("Categorical effect coding: the first observed level is omitted. "
                     "Each remaining level gets a column: +1 for that level, -1 for the omitted level, 0 otherwise.")
        ttk.Label(top, text=help_text, wraplength=820).grid(row=6, column=0, columnspan=4, sticky="w", pady=12)
        ttk.Button(top, text="Run", command=self.run_model).grid(row=7, column=0, sticky="w")
        ttk.Button(top, text="Choose different file", command=self.choose_file).grid(row=7, column=1, sticky="w")
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(5, weight=1)

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv *.xlsx"), ("CSV", "*.csv"), ("Excel", "*.xlsx")])
        if not path:
            return
        try:
            data = read_file(path)
        except Exception as exc:
            messagebox.showerror("File error", str(exc))
            return
        self.path, self.data = path, data
        self.status_var.set(f"Loaded {len(data):,} rows from {path}")
        self.response_var.set("")
        self.event_var.set("")
        self.response_changed()

    def response_changed(self, _event=None):
        response = self.response_var.get()
        self.input_list.delete(0, tk.END)
        self.cat_list.delete(0, tk.END)
        self.event_combo["values"] = []
        self.event_var.set("")
        if not response or response not in self.data.columns:
            return
        levels = list(pd.unique(self.data[response].dropna()))
        if len(levels) == 2:
            self.event_combo["values"] = [display_value(x) for x in levels]
            self.event_var.set(display_value(levels[0]))
        for col in self.data.columns:
            if col != response:
                self.input_list.insert(tk.END, str(col))
                self.cat_list.insert(tk.END, str(col))

    def run_model(self):
        try:
            report, chart_data = fit_model(self.data, self.response_var.get(), self.event_var.get(),
                                           list(self.input_list.get(i) for i in self.input_list.curselection()),
                                           list(self.cat_list.get(i) for i in self.cat_list.curselection()))
        except Exception as exc:
            messagebox.showerror("Model error", str(exc))
            return
        show_results(self.root, report, chart_data)


def read_file(path):
    if not path:
        raise ValueError("No file selected.")
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Select a .csv or .xlsx file.")


def fit_model(data, response, event_display, inputs, categorical):
    if not response:
        raise ValueError("Select a response column.")
    if response not in data.columns:
        raise ValueError("The selected response column is not available.")
    levels = list(pd.unique(data[response].dropna()))
    if len(levels) != 2:
        raise ValueError(f"The response must have exactly two nonmissing levels; found {len(levels)}.")
    if not inputs:
        raise ValueError("Select at least one input column.")
    if not set(categorical).issubset(inputs):
        raise ValueError("Every categorical selection must also be selected as an input.")
    event_matches = [x for x in levels if display_value(x) == event_display]
    if len(event_matches) != 1:
        raise ValueError("Select one of the two observed response levels as the event value.")
    event_value = event_matches[0]
    non_event = next(x for x in levels if x != event_value)

    complete = data[[response] + inputs].notna().all(axis=1)
    used = data.loc[complete, [response] + inputs].copy()
    if used.empty:
        raise ValueError("No complete rows remain after excluding missing response/input values.")
    y = (used[response] == event_value).astype(int).to_numpy()
    X, omitted = build_design(used, response, inputs, set(categorical))
    if len(np.unique(y)) < 2:
        raise ValueError("The complete-case response has only one observed class.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = sm.Logit(y, X).fit(disp=False, maxiter=200)
            intercept_model = sm.Logit(y, X[["Intercept"]]).fit(disp=False, maxiter=200)
        except PerfectSeparationError:
            raise ValueError("Perfect separation was detected; maximum-likelihood estimates do not exist.")
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"The model could not be fit because the design matrix is singular: {exc}")
        except Exception as exc:
            raise ValueError(f"Model fitting failed: {exc}")
    if not getattr(model, "mle_retvals", {}).get("converged", True):
        raise ValueError("The logistic regression did not converge.")

    llr = 2 * (model.llf - intercept_model.llf)
    df = X.shape[1] - 1
    lr_p = chi2.sf(llr, df)
    report = []
    report += ["LOGISTIC REGRESSION REPORT", "=" * 80, "", "Response coding", "----------------"]
    report += [f"Response column: {response}", f"Event value treated as 1: {display_value(event_value)}",
               f"Non-event value treated as 0: {display_value(non_event)}", "", "Omitted categorical levels", "--------------------------"]
    if omitted:
        report += [f"{name}: {display_value(level)}" for name, level in omitted.items()]
    else:
        report.append("None")
    report += ["", "Row counts", "----------", f"Original rows: {len(data):,}",
               f"Excluded for missingness: {len(data) - len(used):,}", f"Rows used in model: {len(used):,}", "",
               "Whole-model likelihood-ratio test", "--------------------------------", 
               f"Likelihood-ratio chi-square: {format_number(llr)}; df = {df}; p = {p_text(lr_p)}", "",
               "Term-level results", "------------------", 
               f"{'Term':40s} {'Estimate':>12s} {'Std. Error':>12s} {'Exp(Estimate)':>15s} {'Wald p':>14s} {'Logworth':>12s}"]
    pvals = model.pvalues.to_numpy()
    chart = []
    for i, term in enumerate(X.columns):
        estimate, se, p = model.params.iloc[i], model.bse.iloc[i], pvals[i]
        odds = math.exp(estimate) if estimate < 709 else float("inf")
        lw = logworth(p)
        report.append(f"{term[:40]:40s} {format_number(estimate):>12s} {format_number(se):>12s} {format_number(odds):>15s} {p_text(p):>14s} {lw:12.4f}")
        if p < 0.05:
            chart.append((term, lw))
    chart.sort(key=lambda x: x[1], reverse=True)
    report += ["", classification_text(y, model.predict(X))]
    return "\n".join(report), chart


def show_results(parent, report, chart_data):
    win = tk.Toplevel(parent)
    win.title("Logistic Regression Results")
    win.geometry("1100x700")
    text = tk.Text(win, wrap="none", font=("Courier New", 10))
    yscroll = ttk.Scrollbar(win, orient="vertical", command=text.yview)
    xscroll = ttk.Scrollbar(win, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll.grid(row=1, column=0, sticky="ew")
    buttons = ttk.Frame(win, padding=6)
    buttons.grid(row=2, column=0, columnspan=2, sticky="w")
    text.insert("1.0", report)
    text.configure(state="disabled")

    def save_report():
        path = filedialog.asksaveasfilename(parent=win, defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            with open(path, "w", encoding="utf-8") as file:
                file.write(report)

    def chart():
        if not chart_data:
            messagebox.showinfo("Logworth chart", "No term has a p-value below 0.05.", parent=win)
            return
        import matplotlib.pyplot as plt
        labels = [x[0] for x in chart_data][::-1]
        values = [x[1] for x in chart_data][::-1]
        height = max(4, 0.38 * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(9, height))
        ax.barh(labels, values, color="#4472C4")
        ax.set_xlabel("Logworth = -log10(p-value)")
        ax.set_title("Significant-term logworths (p < 0.05)")
        fig.tight_layout()
        plt.show()

    ttk.Button(buttons, text="Save report", command=save_report).pack(side="left", padx=(0, 6))
    ttk.Button(buttons, text="Show significant-term chart", command=chart).pack(side="left")
    win.columnconfigure(0, weight=1)
    win.rowconfigure(0, weight=1)


def main():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(parent=root, title="Select data file",
                                      filetypes=[("Data files", "*.csv *.xlsx"), ("CSV", "*.csv"), ("Excel", "*.xlsx")])
    if not path:
        messagebox.showinfo("No file selected", "No file was selected. The program will close.")
        root.destroy()
        return
    try:
        data = read_file(path)
    except Exception as exc:
        messagebox.showerror("File error", str(exc))
        root.destroy()
        return
    if data.empty:
        messagebox.showerror("File error", "The selected file contains no rows.")
        root.destroy()
        return
    root.deiconify()
    LogisticApp(root, path, data)
    root.mainloop()


if __name__ == "__main__":
    main()
