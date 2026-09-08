# single_predictor_logistic.py
#
# Installation instructions (run once in Command Prompt or PowerShell):
# pip install pandas statsmodels scipy
#
# This program fits a binary logistic regression with one selected numeric
# predictor from a CSV file. The first CSV column is the binary response;
# remaining columns are candidate predictors.

import math
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationError,
)


def show_error(root, title, message):
    """Display an error dialog."""
    messagebox.showerror(title, message, parent=root)


def choose_predictor(root, predictor_names):
    """
    Show a dialog containing candidate predictor columns.
    Return the selected column name, or None if the user cancels.
    """
    selection_window = tk.Toplevel(root)
    selection_window.title("Choose Logistic Regression Input")
    selection_window.geometry("500x350")
    selection_window.resizable(True, True)
    selection_window.transient(root)
    selection_window.grab_set()

    selected_name = {"value": None}

    instructions = tk.Label(
        selection_window,
        text="Select exactly one numeric input variable:",
        padx=10,
        pady=10,
    )
    instructions.pack(anchor="w")

    frame = tk.Frame(selection_window)
    frame.pack(fill="both", expand=True, padx=10, pady=5)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        frame,
        selectmode=tk.SINGLE,
        yscrollcommand=scrollbar.set,
        font=("Courier New", 10),
    )
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    for name in predictor_names:
        listbox.insert(tk.END, name)

    def confirm_selection():
        selection = listbox.curselection()

        if not selection:
            messagebox.showwarning(
                "No Input Selected",
                "Please select one input variable.",
                parent=selection_window,
            )
            return

        selected_name["value"] = predictor_names[selection[0]]
        selection_window.destroy()

    def cancel_selection():
        selection_window.destroy()

    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=10)

    tk.Button(
        button_frame,
        text="Use Selected Input",
        command=confirm_selection,
        width=20,
    ).pack(side="left", padx=5)

    tk.Button(
        button_frame,
        text="Cancel",
        command=cancel_selection,
        width=12,
    ).pack(side="left", padx=5)

    selection_window.protocol("WM_DELETE_WINDOW", cancel_selection)

    # Allow pressing Enter after selecting a variable.
    listbox.bind("<Double-Button-1>", lambda event: confirm_selection())

    root.wait_window(selection_window)
    return selected_name["value"]


def format_number(value, decimals=6):
    """Format ordinary numeric output consistently."""
    if pd.isna(value):
        return "NA"

    if math.isinf(value):
        return "infinity"

    return f"{value:.{decimals}f}"


def display_results(root, results_text):
    """Show model output in a scrollable Tkinter text window."""
    results_window = tk.Toplevel(root)
    results_window.title("Single-Predictor Logistic Regression Results")
    results_window.geometry("1050x700")
    results_window.minsize(750, 450)

    main_frame = tk.Frame(results_window)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    vertical_scrollbar = tk.Scrollbar(main_frame)
    vertical_scrollbar.pack(side="right", fill="y")

    horizontal_scrollbar = tk.Scrollbar(main_frame, orient="horizontal")
    horizontal_scrollbar.pack(side="bottom", fill="x")

    text_box = tk.Text(
        main_frame,
        wrap="none",
        font=("Courier New", 10),
        yscrollcommand=vertical_scrollbar.set,
        xscrollcommand=horizontal_scrollbar.set,
    )
    text_box.pack(side="left", fill="both", expand=True)

    vertical_scrollbar.config(command=text_box.yview)
    horizontal_scrollbar.config(command=text_box.xview)

    text_box.insert("1.0", results_text)
    text_box.config(state="disabled")

    close_button = tk.Button(
        results_window,
        text="Close",
        command=root.destroy,
        width=12,
    )
    close_button.pack(pady=(0, 10))

    results_window.protocol("WM_DELETE_WINDOW", root.destroy)


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the otherwise empty main Tkinter window.

    csv_path = filedialog.askopenfilename(
        parent=root,
        title="Choose a CSV File",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )

    if not csv_path:
        root.destroy()
        return

    # Read the selected CSV file.
    try:
        data = pd.read_csv(csv_path)
    except Exception as error:
        show_error(
            root,
            "CSV Read Error",
            f"Could not read the selected CSV file.\n\nDetails:\n{error}",
        )
        root.destroy()
        return

    # A valid file needs one response column and at least one predictor column.
    if data.shape[1] < 2:
        show_error(
            root,
            "Insufficient Columns",
            "The CSV file must contain at least two columns:\n"
            "the first column as the binary response and at least one input column.",
        )
        root.destroy()
        return

    if data.shape[0] == 0:
        show_error(
            root,
            "No Data Rows",
            "The CSV file contains column names but no data rows.",
        )
        root.destroy()
        return

    response_name = data.columns[0]
    predictor_names = list(data.columns[1:])

    # Convert response to numeric. Values that are nonmissing but cannot be
    # converted to numbers are invalid.
    response_original = data[response_name]
    response_numeric = pd.to_numeric(response_original, errors="coerce")

    invalid_response = response_original.notna() & response_numeric.isna()

    if invalid_response.any():
        show_error(
            root,
            "Invalid Response Variable",
            f"The response column '{response_name}' must be numeric.\n\n"
            "At least one nonmissing response value could not be converted "
            "to a number.",
        )
        root.destroy()
        return

    # Check the two observed response levels before choosing the predictor.
    response_levels_before_missing_removal = sorted(
        response_numeric.dropna().unique()
    )

    if len(response_levels_before_missing_removal) != 2:
        show_error(
            root,
            "Response Is Not Binary",
            f"The response column '{response_name}' must contain exactly "
            "two distinct nonmissing numeric values.\n\n"
            f"Number of distinct nonmissing values found: "
            f"{len(response_levels_before_missing_removal)}",
        )
        root.destroy()
        return

    selected_predictor = choose_predictor(root, predictor_names)

    if selected_predictor is None:
        root.destroy()
        return

    predictor_original = data[selected_predictor]
    predictor_numeric = pd.to_numeric(predictor_original, errors="coerce")

    # A nonmissing original value that becomes missing after conversion is text
    # or otherwise invalid for this numeric-only version of the program.
    invalid_predictor = predictor_original.notna() & predictor_numeric.isna()

    if invalid_predictor.any():
        show_error(
            root,
            "Invalid Input Variable",
            f"The selected input column '{selected_predictor}' must be numeric.\n\n"
            "At least one nonmissing value could not be converted to a number. "
            "Text/categorical inputs are not supported by this version.",
        )
        root.destroy()
        return

    # Build a working data set and preserve the original pandas row index.
    working_data = pd.DataFrame(
        {
            "original_index": data.index,
            "response_original": response_original,
            "response_numeric": response_numeric,
            "predictor_original": predictor_original,
            "predictor_numeric": predictor_numeric,
        }
    )

    original_row_count = len(working_data)

    # Remove rows missing either the response or the chosen predictor.
    retained_data = working_data.dropna(
        subset=["response_numeric", "predictor_numeric"]
    ).copy()

    used_row_count = len(retained_data)
    excluded_row_count = original_row_count - used_row_count

    if used_row_count == 0:
        show_error(
            root,
            "No Usable Rows",
            "No rows remain after removing rows with missing response or input values.",
        )
        root.destroy()
        return

    # Re-check response levels after missing-value removal.
    response_levels = sorted(retained_data["response_numeric"].unique())

    if len(response_levels) != 2:
        show_error(
            root,
            "Response Has Only One Level After Row Removal",
            "After removing rows with missing values, the response no longer "
            "contains both binary outcome levels.",
        )
        root.destroy()
        return

    response_lower = response_levels[0]
    response_event = response_levels[1]  # Higher response value is the event.

    # Internal response coding: lower = 0, higher/event = 1.
    retained_data["response_model"] = (
        retained_data["response_numeric"] == response_event
    ).astype(int)

    predictor_levels = sorted(retained_data["predictor_numeric"].unique())

    if len(predictor_levels) < 2:
        show_error(
            root,
            "Input Has No Variation",
            f"The selected input column '{selected_predictor}' has fewer than "
            "two distinct values after missing rows were removed.",
        )
        root.destroy()
        return

    predictor_is_binary = len(predictor_levels) == 2

    if predictor_is_binary:
        predictor_reference = predictor_levels[0]
        predictor_higher = predictor_levels[1]

        # Internal binary predictor coding: lower/reference = 0, higher = 1.
        retained_data["predictor_model"] = (
            retained_data["predictor_numeric"] == predictor_higher
        ).astype(int)

    else:
        # Continuous numeric predictor: retain original numeric values.
        retained_data["predictor_model"] = retained_data["predictor_numeric"]

    # Use a named DataFrame so statsmodels gives the slope a useful name.
    x_matrix = pd.DataFrame(
        {selected_predictor: retained_data["predictor_model"]}
    )
    x_matrix = sm.add_constant(x_matrix, has_constant="add")
    y_vector = retained_data["response_model"]

    # Fit the logistic regression and catch common fitting problems.
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")

            fitted_model = sm.Logit(y_vector, x_matrix).fit(disp=False)
            null_matrix = pd.DataFrame({"const": [1.0] * len(y_vector)})
            null_model = sm.Logit(y_vector, null_matrix).fit(disp=False)

        warning_categories = [warning.category for warning in captured_warnings]

        if any(
            issubclass(category, ConvergenceWarning)
            or issubclass(category, HessianInversionWarning)
            for category in warning_categories
        ):
            show_error(
                root,
                "Model Fitting Problem",
                "The logistic regression did not converge reliably or could not "
                "estimate its uncertainty measures.\n\n"
                "This can occur with perfect or quasi-complete separation, "
                "very small samples, or extreme predictor values.",
            )
            root.destroy()
            return

        if not fitted_model.mle_retvals.get("converged", False):
            show_error(
                root,
                "Model Did Not Converge",
                "The logistic regression did not converge.\n\n"
                "This may indicate perfect or quasi-complete separation, "
                "too little data, or an unstable model.",
            )
            root.destroy()
            return

        if not pd.Series(fitted_model.bse).notna().all():
            show_error(
                root,
                "Standard Error Estimation Failed",
                "The model was fitted, but valid coefficient standard errors "
                "could not be estimated.\n\n"
                "This often occurs because of separation or insufficient data.",
            )
            root.destroy()
            return

    except PerfectSeparationError:
        show_error(
            root,
            "Perfect Separation Detected",
            "The selected input perfectly predicts the response in this data.\n\n"
            "Ordinary logistic-regression coefficient estimates are not finite "
            "under perfect separation.",
        )
        root.destroy()
        return

    except Exception as error:
        show_error(
            root,
            "Logistic Regression Fitting Error",
            "The model could not be fitted.\n\n"
            "Possible causes include perfect or quasi-complete separation, "
            "too few observations, or numerical instability.\n\n"
            f"Details:\n{error}",
        )
        root.destroy()
        return

    # Extract coefficient information.
    intercept_estimate = fitted_model.params["const"]
    intercept_se = fitted_model.bse["const"]
    intercept_pvalue = fitted_model.pvalues["const"]

    slope_estimate = fitted_model.params[selected_predictor]
    slope_se = fitted_model.bse[selected_predictor]
    slope_pvalue = fitted_model.pvalues[selected_predictor]

    try:
        odds_multiplier = math.exp(slope_estimate)
    except OverflowError:
        odds_multiplier = math.inf

    # Calculate the whole-model likelihood ratio chi-square test.
    log_likelihood_fitted = fitted_model.llf
    log_likelihood_null = null_model.llf

    lr_chi_square = 2 * (log_likelihood_fitted - log_likelihood_null)
    lr_degrees_freedom = 1
    lr_pvalue = chi2.sf(lr_chi_square, lr_degrees_freedom)

    if lr_pvalue == 0:
        logworth = math.inf
    else:
        logworth = -math.log10(lr_pvalue)

    # Predicted probabilities are probabilities of the event (higher response).
    retained_data["predicted_probability"] = fitted_model.predict(x_matrix)

    probability_minimum = retained_data["predicted_probability"].min()
    probability_maximum = retained_data["predicted_probability"].max()

    # Build readable output for the scrollable results window.
    output = []
    output.append("SINGLE-PREDICTOR BINARY LOGISTIC REGRESSION")
    output.append("=" * 78)
    output.append("")

    # This response/event statement appears before other reported results.
    output.append("RESPONSE EVENT CODING")
    output.append("-" * 78)
    output.append(f"Response column:                 {response_name}")
    output.append(
        f"Observed response values:          "
        f"{format_number(response_lower)} and {format_number(response_event)}"
    )
    output.append(
        f"Event definition:                  "
        f"{format_number(response_event)} (the higher response value)"
    )
    output.append(
        f"Internal model coding:             "
        f"{format_number(response_lower)} = 0, "
        f"{format_number(response_event)} = 1"
    )
    output.append("")

    output.append("ROWS USED")
    output.append("-" * 78)
    output.append(f"Rows in original CSV:            {original_row_count}")
    output.append(f"Rows excluded for missing data:  {excluded_row_count}")
    output.append(f"Rows used in model:              {used_row_count}")
    output.append("")

    output.append("INPUT VARIABLE CODING")
    output.append("-" * 78)
    output.append(f"Selected input column:           {selected_predictor}")

    if predictor_is_binary:
        output.append("Input type:                      Binary numeric")
        output.append(
            f"Observed input values:            "
            f"{format_number(predictor_reference)} and "
            f"{format_number(predictor_higher)}"
        )
        output.append(
            f"Reference/baseline level:         "
            f"{format_number(predictor_reference)} (the lower value)"
        )
        output.append(
            f"Comparison represented by slope:  "
            f"{format_number(predictor_higher)} versus "
            f"{format_number(predictor_reference)}"
        )
        output.append(
            f"Internal model coding:            "
            f"{format_number(predictor_reference)} = 0, "
            f"{format_number(predictor_higher)} = 1"
        )
    else:
        output.append("Input type:                      Continuous numeric")
        output.append(
            "Interpretation of slope:          Change in event log-odds for "
            "a one-unit increase in the input"
        )
        output.append(
            "Interpretation of odds multiplier: Multiplication of event odds "
            "for a one-unit increase"
        )

    output.append("")
    output.append("COEFFICIENTS")
    output.append("-" * 78)
    output.append(
        f"{'Term':<25}{'Estimate':>15}{'Std. Error':>15}{'Wald p-value':>18}"
    )
    output.append("-" * 78)
    output.append(
        f"{'Intercept':<25}"
        f"{format_number(intercept_estimate):>15}"
        f"{format_number(intercept_se):>15}"
        f"{format_number(intercept_pvalue):>18}"
    )
    output.append(
        f"{selected_predictor:<25}"
        f"{format_number(slope_estimate):>15}"
        f"{format_number(slope_se):>15}"
        f"{format_number(slope_pvalue):>18}"
    )
    output.append("")
    output.append(
        f"Odds multiplier, exp(slope):       {format_number(odds_multiplier)}"
    )

    if predictor_is_binary:
        output.append(
            "Odds multiplier interpretation:   Event odds for the higher input "
            "level divided by event odds for the lower reference level"
        )
    else:
        output.append(
            "Odds multiplier interpretation:   Multiplication of event odds "
            "for a one-unit input increase"
        )

    output.append("")
    output.append("WHOLE-MODEL LIKELIHOOD RATIO TEST")
    output.append("-" * 78)
    output.append(
        "Comparison: fitted intercept + slope model versus intercept-only model"
    )
    output.append(
        f"Log-likelihood, null model:        "
        f"{format_number(log_likelihood_null)}"
    )
    output.append(
        f"Log-likelihood, fitted model:      "
        f"{format_number(log_likelihood_fitted)}"
    )
    output.append(
        f"LR chi-square:                     {format_number(lr_chi_square)}"
    )
    output.append(f"Degrees of freedom:               {lr_degrees_freedom}")
    output.append(f"LR-test p-value:                  {format_number(lr_pvalue)}")

    if math.isinf(logworth):
        output.append("LogWorth = -log10(LR p-value):   infinity (p-value underflow)")
    else:
        output.append(
            f"LogWorth = -log10(LR p-value):   {format_number(logworth)}"
        )

    output.append("")
    output.append("PREDICTED PROBABILITIES")
    output.append("-" * 78)
    output.append(
        "Each probability below is the predicted probability of the event: "
        f"response = {format_number(response_event)}"
    )
    output.append(
        f"Verification -- minimum probability: {format_number(probability_minimum)}, "
        f"maximum probability: {format_number(probability_maximum)}"
    )
    output.append("All predicted probabilities should be between 0 and 1.")
    output.append("")
    output.append(
        f"{'Original Index':>14}  "
        f"{'Observed Response':>20}  "
        f"{'Input Value':>18}  "
        f"{'Predicted P(Event)':>20}"
    )
    output.append("-" * 78)

    for _, row in retained_data.iterrows():
        output.append(
            f"{int(row['original_index']):>14}  "
            f"{format_number(row['response_numeric']):>20}  "
            f"{format_number(row['predictor_numeric']):>18}  "
            f"{format_number(row['predicted_probability']):>20}"
        )

    display_results(root, "\n".join(output))
    root.mainloop()


if __name__ == "__main__":
    main()