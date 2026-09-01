import sys
import tkinter as tk
from tkinter import ttk

# Constants
BETA0 = 5.0
BETA1 = 3.0
SIGMA = 5.0
X_LOW, X_HIGH = 0.0, 10.0
N_POP = 10000
N_REPS = 1000


def show_missing_module(module_name, error_text):
    root = tk.Tk()
    root.title("Missing Python module")
    root.resizable(False, False)

    interpreter = sys.executable
    package = module_name.split(".")[0] if module_name else "required-package"
    if package in {"matplotlib", "numpy"}:
        install_package = package
    else:
        install_package = package
    pip_command = f'"{interpreter}" -m pip install {install_package}'

    frame = ttk.Frame(root, padding=14)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=f"Missing module: {module_name or error_text}",
              font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 8))
    ttk.Label(frame, text="Python interpreter:").pack(anchor="w")
    ttk.Label(frame, text=interpreter, wraplength=560).pack(anchor="w", pady=(0, 8))
    ttk.Label(frame, text="Install command:").pack(anchor="w")
    ttk.Label(frame, text=pip_command, wraplength=560).pack(anchor="w", pady=(0, 8))
    ttk.Label(frame, text="Close this window after installation, then run the script again.").pack(anchor="w")
    root.mainloop()


try:
    import numpy as np
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception as exc:
    show_missing_module(getattr(exc, "name", None), str(exc))
    raise SystemExit


class RegressionSamplingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression sampling")
        self.root.geometry("1120x820")
        self.root.minsize(900, 680)
        self.rng = np.random.default_rng()

        controls = ttk.Frame(root, padding=(10, 8, 10, 2))
        controls.pack(fill="x")
        ttk.Label(controls, text="Sample size (n)").pack(side="left")
        self.n_var = tk.StringVar(value="25")
        self.n_entry = ttk.Entry(controls, textvariable=self.n_var, width=12)
        self.n_entry.pack(side="left", padx=(7, 7))
        self.resample_button = ttk.Button(controls, text="Resample", command=self.run_from_input)
        self.resample_button.pack(side="left")
        self.status_var = tk.StringVar(value="")
        ttk.Label(controls, textvariable=self.status_var, foreground="#9b1c1c").pack(side="left", padx=(14, 0))

        self.figure, self.axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
        self.figure.subplots_adjust(top=0.88, bottom=0.14, left=0.07, right=0.98,
                                    hspace=0.72, wspace=0.24)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.n_entry.bind("<Return>", lambda event: self.run_from_input())

        self.draw_run(25)
        self.n_entry.focus_set()

    def run_from_input(self):
        raw = self.n_var.get().strip()
        try:
            n = int(float(raw))
            if not np.isfinite(float(raw)):
                raise ValueError
        except (ValueError, OverflowError):
            self.status_var.set("Enter a real number from 1 to 10000.")
            return
        if not 1 <= n <= N_POP:
            self.status_var.set("Sample size must be from 1 to 10000.")
            return
        if n == 1:
            self.status_var.set("n = 1 cannot produce a nonzero slope denominator; choose n ≥ 2.")
            return
        self.draw_run(n)

    def make_population(self):
        x = self.rng.uniform(X_LOW, X_HIGH, N_POP)
        y = BETA0 + BETA1 * x + self.rng.normal(0.0, SIGMA, N_POP)
        return x, y

    def draw_run(self, n):
        x, y = self.make_population()
        slopes = np.empty(N_REPS)
        intercepts = np.empty(N_REPS)
        first_b1 = first_b0 = None
        completed = 0

        while completed < N_REPS:
            indices = self.rng.integers(0, N_POP, size=n)
            xs = x[indices]
            ys = y[indices]
            x_mean = xs.mean()
            y_mean = ys.mean()
            centered_x = xs - x_mean
            denominator = np.sum(centered_x ** 2)
            if denominator == 0.0:
                continue
            b1 = np.sum(centered_x * (ys - y_mean)) / denominator
            b0 = y_mean - b1 * x_mean
            slopes[completed] = b1
            intercepts[completed] = b0
            if completed == 0:
                first_b1, first_b0 = b1, b0
            completed += 1

        self.render(n, x, y, slopes, intercepts, first_b1, first_b0)
        self.status_var.set("")

    @staticmethod
    def stats(values):
        return float(np.mean(values)), float(np.var(values))

    def add_stats(self, ax, values, se_text=None):
        mean, variance = self.stats(values)
        text = f"mean = {mean:.4f}    variance = {variance:.4f}"
        ax.text(0.5, -0.22 if se_text else -0.14, text,
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
        if se_text:
            ax.text(0.5, -0.35, se_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=9)

    def render(self, n, x, y, slopes, intercepts, first_b1, first_b0):
        for ax in self.axes.flat:
            ax.clear()
            ax.tick_params(axis="y", labelleft=False)
            ax.axhline(0, color="#555555", linewidth=0.8, zorder=1)

        self.axes[0, 0].hist(x, bins=45, color="#78a9d1", edgecolor="white")
        self.axes[0, 0].set_title("Population of X, wait at the stop, minutes")
        self.add_stats(self.axes[0, 0], x)

        self.axes[0, 1].hist(y, bins=45, color="#78a9d1", edgecolor="white")
        self.axes[0, 1].set_title("Population of Y, ad revenue, cents")
        self.add_stats(self.axes[0, 1], y)

        slope_ax = self.axes[1, 0]
        slope_ax.hist(slopes, bins=45, color="#78a9d1", edgecolor="white")
        slope_ax.set_xlim(0.5, 6.0)
        slope_ax.set_title("1000 estimates of the slope b1")
        slope_ax.axvline(BETA1, color="#184e77", linewidth=2.2, zorder=4)
        slope_ax.plot(first_b1, 0, "o", color="red", markersize=11, zorder=10, clip_on=False)
        slope_ax.annotate(f"{first_b1:.4f}", (first_b1, 0), xytext=(0, 12),
                          textcoords="offset points", ha="center", color="red",
                          fontsize=9, fontweight="bold", clip_on=False)
        var_x = (X_HIGH - X_LOW) ** 2 / 12.0
        mean_x = (X_LOW + X_HIGH) / 2.0
        empirical_b1 = float(np.std(slopes))
        theoretical_b1 = SIGMA / np.sqrt(n * var_x)
        self.add_stats(slope_ax, slopes,
                       f"empirical SE = {empirical_b1:.4f}    theoretical SE = {theoretical_b1:.4f}")

        intercept_ax = self.axes[1, 1]
        intercept_ax.hist(intercepts, bins=45, color="#78a9d1", edgecolor="white")
        intercept_ax.set_xlim(-15, 25)
        intercept_ax.set_title("1000 estimates of the intercept b0")
        intercept_ax.axvline(BETA0, color="#184e77", linewidth=2.2, zorder=4)
        intercept_ax.plot(first_b0, 0, "o", color="red", markersize=11, zorder=10, clip_on=False)
        intercept_ax.annotate(f"{first_b0:.4f}", (first_b0, 0), xytext=(0, 12),
                              textcoords="offset points", ha="center", color="red",
                              fontsize=9, fontweight="bold", clip_on=False)
        empirical_b0 = float(np.std(intercepts))
        theoretical_b0 = SIGMA * np.sqrt(1 / n + mean_x ** 2 / (n * var_x))
        self.add_stats(intercept_ax, intercepts,
                       f"empirical SE = {empirical_b0:.4f}    theoretical SE = {theoretical_b0:.4f}")

        self.figure.suptitle(f"Y = 5 + 3X + noise, {N_REPS} samples of n = {n}", fontsize=15)
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    RegressionSamplingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
