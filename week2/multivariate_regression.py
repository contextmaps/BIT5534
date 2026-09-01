import os
import sys
import math
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox

root = None


def show_error(message, tb=None):
    text = message if tb is None else f"{message}\n\nFull traceback:\n{tb}"
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "error_log.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass
    try:
        if root is not None and root.winfo_exists():
            win = tk.Toplevel(root)
            standalone = False
        else:
            win = tk.Tk()
            standalone = True
        win.title("Error")
        win.geometry("850x550")
        win.minsize(450, 250)
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        frame = tk.Frame(win)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        scroll = tk.Scrollbar(frame)
        scroll.grid(row=0, column=1, sticky="ns")
        box = tk.Text(frame, wrap="word", yscrollcommand=scroll.set, cursor="xterm")
        box.insert("1.0", text)
        box.configure(state="disabled")
        box.grid(row=0, column=0, sticky="nsew")
        scroll.configure(command=box.yview)

        def copy_text():
            win.clipboard_clear()
            win.clipboard_append(text)
            win.update()

        buttons = tk.Frame(win)
        buttons.grid(row=1, column=0, pady=(0, 10))
        tk.Button(buttons, text="Copy to clipboard", command=copy_text).pack(side="left", padx=6)
        tk.Button(buttons, text="Close", command=win.destroy).pack(side="left", padx=6)
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.lift()
        win.focus_force()
        if standalone:
            win.mainloop()
        else:
            win.wait_window()
    except Exception:
        try:
            messagebox.showerror("Error", text)
        except Exception:
            pass


def fail(message):
    raise RuntimeError(message)


def main():
    global root
    root = tk.Tk()
    root.withdraw()
    try:
        try:
            import pandas as pd
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "A required library could not be imported. Install them with:\n\n"
                f'"{sys.executable}" -m pip install pandas statsmodels matplotlib openpyxl xlrd'
            ) from exc

        input_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("Excel or CSV files", ("*.xlsx", "*.xls", "*.csv")),
                       ("All files", "*.*")])
        if not input_path:
            return

        extension = os.path.splitext(input_path)[1].lower()
        if extension == ".csv":
            data = pd.read_csv(input_path)
        elif extension in (".xlsx", ".xls"):
            data = pd.read_excel(input_path, sheet_name=0)
        else:
            fail("Please select an .xlsx, .xls, or .csv file.")

        if data.shape[1] < 2:
            fail("The data must contain at least two columns.")
        bad = [str(c) for c, t in data.dtypes.items() if not pd.api.types.is_numeric_dtype(t)]
        if bad:
            fail("Every column must be numeric. Non-numeric columns: " + ", ".join(bad))

        original_rows = len(data)
        data = data.dropna(axis=0, how="any").copy()
        dropped = original_rows - len(data)

        predictor_names = [str(c) for c in data.columns[1:]]
        predictors = data.iloc[:, 1:]
        p = predictors.shape[1]
        if len(data) <= p + 1:
            fail(f"After dropping missing rows there must be more rows than predictors "
                 f"plus one. Rows available: {len(data)}; predictors plus one: {p + 1}.")
        constant = [n for n, u in predictors.nunique(dropna=False).items() if u <= 1]
        if constant:
            fail("A predictor never changes: " + ", ".join(map(str, constant)))

        y = data.iloc[:, 0].astype(float)
        x = predictors.astype(float)
        x_const = sm.add_constant(x, has_constant="add")
        model = sm.OLS(y, x_const).fit()
        vifs = [variance_inflation_factor(x_const.to_numpy(), i)
                for i in range(1, x_const.shape[1])]
        pvalues = model.pvalues.iloc[1:]
        logworth = [min(16.0, -math.log10(max(float(v), sys.float_info.min))) for v in pvalues]

        output_dir = os.path.splitext(input_path)[0] + "_outputs"
        os.makedirs(output_dir, exist_ok=True)

        plot_data = pd.DataFrame({"predictor": predictor_names, "logworth": logworth})
        plot_data = plot_data.sort_values("logworth", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(plot_data) + 1.5)))
        bars = ax.barh(plot_data["predictor"], plot_data["logworth"], color="#4472C4")
        ax.axvline(1.30103, color="#C00000", linestyle="--", linewidth=1.2, label="p = 0.05")
        ax.set_xlabel("Logworth, -log10(p)")
        ax.set_title("Predictor significance")
        ax.legend(loc="lower right")
        xmax = max(1.6, float(plot_data["logworth"].max()) * 1.18 + 0.15)
        ax.set_xlim(0, xmax)
        for bar, value in zip(bars, plot_data["logworth"]):
            ax.text(bar.get_width() + xmax * 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}", va="center", ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "logworth.png"), dpi=160)
        plt.close(fig)

        residuals = model.resid
        fitted = model.fittedvalues
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        axes[0].scatter(fitted, residuals, color="#4472C4", alpha=0.85)
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_xlabel("Fitted value")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Residual vs fitted")
        axes[1].scatter(range(1, len(residuals) + 1), residuals, color="#70AD47", alpha=0.85)
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_xlabel("Row number in file")
        axes[1].set_title("Residual vs row number")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "residuals.png"), dpi=160)
        plt.close(fig)

        model_table = pd.DataFrame({
            "metric": ["R-squared", "Adjusted R-squared", "F statistic", "F-test p-value"],
            "value": [model.rsquared, model.rsquared_adj, model.fvalue, model.f_pvalue]})
        term_table = pd.DataFrame({
            "term": ["intercept"] + predictor_names,
            "estimate": model.params.to_numpy(),
            "standard_error": model.bse.to_numpy(),
            "p_value": model.pvalues.to_numpy(),
            "variance_inflation_factor": pd.Series([""] + [round(v, 2) for v in vifs],
                                                   dtype="object")})
        report = ("MODEL\n" + model_table.round(4).to_string(index=False)
                  + "\n\nTERMS\n" + term_table.round(4).to_string(index=False) + "\n")
        with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        messagebox.showinfo(
            "Regression complete",
            f"Folder written to:\n{output_dir}\n\n"
            f"Rows used: {len(data)}\nRows dropped: {dropped}\n\n"
            f"R-squared: {model.rsquared:.4f}\n"
            f"Adjusted R-squared: {model.rsquared_adj:.4f}\n"
            f"F statistic: {model.fvalue:.4f}\n"
            f"F-test p-value: {model.f_pvalue:.4f}")
    except Exception as exc:
        show_error(str(exc), traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            show_error("Unexpected error", traceback.format_exc())
        except Exception:
            pass
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass
