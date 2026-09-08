import math
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


class UserError(Exception):
    """An expected, user-facing error."""


def display_value(value):
    """Return a readable representation without exposing pandas NA values."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def value_key(value):
    """Normalize ordinary numeric representations and trimmed text consistently."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number):
            return f"NUMBER:{number:.17g}"
    except (TypeError, ValueError):
        pass
    return f"TEXT:{text}"


def ordered_levels(series):
    """Return [(normalized key, first-seen display value)] for nonmissing values."""
    levels = []
    seen = set()
    for value in series:
        key = value_key(value)
        if key is not None and key not in seen:
            seen.add(key)
            levels.append((key, display_value(value)))
    return levels


def choose_from_list(root, title, prompt, items):
    """Show a modal list-selection dialog and return the selected item index."""
    result = {"index": None}
    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(dialog, text=prompt, padx=12, pady=10, justify="left").pack()
    listbox = tk.Listbox(dialog, height=min(max(len(items), 2), 12), width=70,
                         exportselection=False)
    for item in items:
        listbox.insert(tk.END, item)
    listbox.pack(padx=12, pady=(0, 10))
    listbox.selection_set(0)
    listbox.focus_set()

    def accept(event=None):
        selection = listbox.curselection()
        if selection:
            result["index"] = selection[0]
            dialog.destroy()

    def cancel(event=None):
        dialog.destroy()

    buttons = tk.Frame(dialog)
    buttons.pack(pady=(0, 12))
    tk.Button(buttons, text="OK", width=10, command=accept).pack(side="left", padx=5)
    tk.Button(buttons, text="Cancel", width=10, command=cancel).pack(side="left", padx=5)
    dialog.bind("<Return>", accept)
    dialog.bind("<Escape>", cancel)
    dialog.protocol("WM_DELETE_WINDOW", cancel)
    dialog.wait_window()

    if result["index"] is None:
        raise UserError("Selection cancelled.")
    return result["index"]


def choose_csv(root):
    path = filedialog.askopenfilename(
        parent=root,
        title="Choose a CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        raise UserError("No CSV file was selected.")
    return path


def clean_response(df, response_col):
    response = df[response_col]
    nonmissing = response.notna()
    levels = ordered_levels(response[nonmissing])
    if len(levels) != 2:
        raise UserError(
            f"The response column must have exactly two observed values after missing "
            f"values are removed; found {len(levels)}: "
            + ", ".join(label for _, label in levels)
        )
    return response, nonmissing, levels


def prepare_input(series):
    """Return a numeric predictor and metadata describing its interpretation."""
    nonmissing = series.notna()
    raw_levels = ordered_levels(series[nonmissing])
    numeric = pd.to_numeric(series, errors="coerce")
    all_numeric = numeric[nonmissing].notna().all()

    if len(raw_levels) == 2:
        labels = [label for _, label in raw_levels]
        keys = [key for key, _ in raw_levels]
        return {
            "kind": "binary",
            "numeric": numeric,
            "valid": nonmissing,
            "keys": keys,
            "labels": labels,
        }

    if len(raw_levels) > 2 and all_numeric:
        return {
            "kind": "numeric",
            "numeric": numeric,
            "valid": nonmissing & numeric.notna(),
            "keys": None,
            "labels": None,
        }

    if len(raw_levels) > 2:
        raise UserError(
            f"The selected input has {len(raw_levels)} observed levels and is categorical. "
            "Only numeric or binary inputs are supported."
        )

    if len(raw_levels) == 1:
        raise UserError("The selected input has only one observed level and cannot be fitted.")

    raise UserError("The selected input has no usable observed values.")


def fmt_pvalue(p):
    if not np.isfinite(p):
        return "not available"
    if p == 0:
        return "0 (underflow)"
    return f"{p:.6g}"


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        path = choose_csv(root)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise UserError(f"Could not read the CSV file: {exc}") from exc

        if df.shape[1] < 2:
            raise UserError("The CSV must contain a response column and at least one candidate input column.")
        if df.columns.duplicated().any():
            raise UserError("The CSV contains duplicate column names; rename them and try again.")

        response_col = str(df.columns[0])
        candidates = [str(c) for c in df.columns[1:]]
        input_col = candidates[choose_from_list(
            root, "Choose input column", "Select one candidate input column:", candidates
        )]

        response, response_nonmissing, response_levels = clean_response(df, response_col)
        response_labels = [label for _, label in response_levels]
        event_index = choose_from_list(
            root,
            "Choose response event",
            f"Response column: {response_col}\nSelect the value to treat as the event (coded 1):",
            response_labels,
        )
        event_key = response_levels[event_index][0]

        input_info = prepare_input(df[input_col])
        if input_info["kind"] == "binary":
            ref_index = choose_from_list(
                root,
                "Choose binary reference",
                f"Input column: {input_col}\nSelect the level to treat as the reference (coded 0):",
                input_info["labels"],
            )
            ref_key = input_info["keys"][ref_index]
            one_key = input_info["keys"][1 - ref_index]
            input_values = df[input_col].map(
                lambda value: np.nan if value_key(value) is None
                else (0.0 if value_key(value) == ref_key else
                      1.0 if value_key(value) == one_key else np.nan)
            )
        else:
            ref_key = one_key = None
            input_values = input_info["numeric"]

        valid = response_nonmissing & input_info["valid"] & input_values.notna()
        excluded_total = int((~valid).sum())
        excluded_response = int((~response_nonmissing).sum())
        excluded_input = int((response_nonmissing & (~input_info["valid"] | input_values.isna())).sum())

        if valid.sum() < 3:
            raise UserError("Fewer than three usable rows remain after exclusions; the model cannot be fitted.")

        y = response.loc[valid].map(lambda value: 1.0 if value_key(value) == event_key else 0.0).astype(float)
        x = pd.to_numeric(input_values.loc[valid], errors="coerce").astype(float)
        if y.nunique() != 2:
            raise UserError("After exclusions, the response contains only one event status; the model cannot be fitted.")
        if x.nunique() < 2:
            raise UserError("After exclusions, the selected input has no variation; the model cannot be fitted.")

        X = sm.add_constant(pd.DataFrame({input_col: x}), has_constant="add")
        X_null = pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
        try:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                fitted = sm.Logit(y, X).fit(disp=False, maxiter=200)
                null_model = sm.Logit(y, X_null).fit(disp=False, maxiter=200)
        except Exception as exc:
            raise UserError(
                "The logistic regression could not converge. Check for perfect separation, "
                "a nearly constant input, or too few observations in one response group."
            ) from exc

        lr_stat = 2.0 * (fitted.llf - null_model.llf)
        df_lr = int(len(fitted.params) - len(null_model.params))
        lr_p = float(stats.chi2.sf(lr_stat, df_lr))
        if lr_p == 0 or (0 < lr_p < np.finfo(float).tiny):
            logworth = -math.log10(np.finfo(float).tiny)
            logworth_text = f"> {logworth:.6g} (p underflowed below floating-point range)"
        else:
            logworth = -math.log10(lr_p)
            logworth_text = f"{logworth:.6g}"

        predictions = np.asarray(fitted.predict(X), dtype=float)
        tolerance = 1e-12
        in_range = bool(np.all((predictions >= -tolerance) & (predictions <= 1 + tolerance)))

        print("\n" + "=" * 72)
        print("SINGLE-PREDICTOR LOGISTIC REGRESSION")
        print("=" * 72)
        print(f"Response column: {response_col}")
        print(f"Event value (coded 1): {response_labels[event_index]}")
        print(f"Selected input column: {input_col}")
        if input_info["kind"] == "binary":
            print(f"Binary reference level (coded 0): {input_info['labels'][ref_index]}")
            print(f"Binary level coded 1: {input_info['labels'][1 - ref_index]}")
        else:
            print("Input type: numeric; no binary reference level applies.")

        print(f"\nUsable observations: {len(y)}")
        print(f"Rows excluded: {excluded_total} (response missing: {excluded_response}; input missing/nonconvertible: {excluded_input})")
        print("\nModel results")
        print(f"Intercept: estimate={fitted.params['const']:.6g}, SE={fitted.bse['const']:.6g}, p-value={fmt_pvalue(fitted.pvalues['const'])}, odds multiplier={math.exp(fitted.params['const']):.6g}")
        print(f"Slope ({input_col}): estimate={fitted.params[input_col]:.6g}, SE={fitted.bse[input_col]:.6g}, p-value={fmt_pvalue(fitted.pvalues[input_col])}, odds multiplier={math.exp(fitted.params[input_col]):.6g}")
        print(f"Whole-model likelihood-ratio chi-square: {lr_stat:.6g}, df={df_lr}, p-value={fmt_pvalue(lr_p)}")
        print(f"Logworth (-log10(p)): {logworth_text}")

        output = pd.DataFrame({
            "original_row": df.index[valid] + 2,
            "response_value": response.loc[valid].map(display_value).to_numpy(),
            "input_value_or_coded": x.to_numpy(),
            "predicted_probability": predictions,
        })
        print("\nPredicted probabilities")
        print(output.to_string(index=False))
        print("\nProbability check")
        print(f"All predicted probabilities in [0, 1] within tolerance {tolerance:g}: {in_range}")
        print(f"Minimum predicted probability: {predictions.min():.6g}")
        print(f"Maximum predicted probability: {predictions.max():.6g}")
        print("=" * 72)

    except UserError as exc:
        messagebox.showerror("Logistic regression", str(exc), parent=root)
        print(f"Error: {exc}")
    except Exception as exc:
        messagebox.showerror("Unexpected error", str(exc), parent=root)
        print(f"Unexpected error: {exc}")
    finally:
        root.destroy()
        if sys.stdin.isatty():
            input("\nPress Enter to close...")


if __name__ == "__main__":
    main()


# Installation and usage:
# 1. Install Python 3.10+ and run: python -m pip install pandas numpy scipy statsmodels
# 2. Save this file as single_predictor_logistic_regression.py.
# 3. Double-click it, choose a CSV, choose an input, and answer the dialogs.
# 4. To keep a Windows console open after double-clicking, run it from Command Prompt
#    or create a shortcut whose target begins with: cmd /k python "...path..."
