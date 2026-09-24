#!/usr/bin/env python3
"""
Single-window k means cluster analysis application.

BIT 5534 lec5_2 backup tool. Built by Hokie AI from the lec5_2 specification
during authoring on 2026-09-23, run 1 of five sampled runs, and kept because its
silhouette agrees with scikit-learn to five decimals on every k from 2 to 10.
Two changes were made to the generated script: the default restarts per k went
from 10 to 50, because ten cannot resolve the four thousandth margin between
k=5 and k=6 on distance_test, and a --selftest path was added so the numbers can
be checked without opening a window.

Required packages:
    pandas, numpy, matplotlib, scipy

Note: pandas.read_excel() requires an Excel reader engine available in the
Python environment (commonly openpyxl) to read .xlsx files.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform

# The window is imported only when a window is wanted. Running with --selftest
# has to work on a machine with no tkinter and no display, since that is how the
# numbers get checked before the tool is handed to anybody.
if "--selftest" in sys.argv:
    class _Absent:
        """Stands in for a GUI module so the class statements below still execute."""
        Tk = object

        def __getattr__(self, name):
            return object

    tk = ttk = filedialog = messagebox = _Absent()
    FigureCanvasTkAgg = Figure = Line2D = object
else:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


MAX_ITERATIONS = 300
CENTROID_TOLERANCE = 1e-6


def kmeans_one_run(data, k, rng):
    """
    Run ordinary Lloyd k-means once.

    Empty clusters are reassigned to observations farthest from their currently
    assigned centroid. This avoids invalid centroids while preserving k clusters.
    """
    n_rows = data.shape[0]
    initial_indices = rng.choice(n_rows, size=k, replace=False)
    centers = data[initial_indices].copy()
    empty_reinitializations = 0

    for _ in range(MAX_ITERATIONS):
        squared_distances = cdist(data, centers, metric="sqeuclidean")
        assignments = np.argmin(squared_distances, axis=1)

        new_centers = np.empty_like(centers)
        nearest_distances = squared_distances[np.arange(n_rows), assignments]

        for cluster_id in range(k):
            members = data[assignments == cluster_id]

            if len(members) > 0:
                new_centers[cluster_id] = members.mean(axis=0)
            else:
                # Reinitialize empty cluster at currently least-well-represented row.
                farthest_index = int(np.argmax(nearest_distances))
                new_centers[cluster_id] = data[farthest_index]
                nearest_distances[farthest_index] = -np.inf
                empty_reinitializations += 1

        centroid_shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
        centers = new_centers

        if centroid_shift <= CENTROID_TOLERANCE:
            break

    squared_distances = cdist(data, centers, metric="sqeuclidean")
    assignments = np.argmin(squared_distances, axis=1)
    inertia = float(squared_distances[np.arange(n_rows), assignments].sum())

    return assignments, centers, inertia, empty_reinitializations


def best_kmeans(data, k, restarts, seed):
    """Run reproducible random restarts and retain the lowest-SSE solution."""
    # Different k values receive deterministic but distinct random streams.
    rng = np.random.default_rng(seed + 100003 * k)

    best = None
    total_empty_reinitializations = 0

    for _ in range(restarts):
        assignments, centers, inertia, empty_count = kmeans_one_run(data, k, rng)
        total_empty_reinitializations += empty_count

        if best is None or inertia < best["inertia"]:
            best = {
                "assignments": assignments,
                "centers": centers,
                "inertia": inertia,
            }

    best["empty_reinitializations"] = total_empty_reinitializations
    return best


def average_silhouette(data, assignments, k):
    """
    Compute the ordinary mean silhouette score from pairwise Euclidean distances.

    Singleton-cluster observations receive silhouette value zero.
    """
    n_rows = len(assignments)
    distances = squareform(pdist(data, metric="euclidean"))
    silhouettes = np.zeros(n_rows, dtype=float)

    for i in range(n_rows):
        own_cluster = assignments[i]
        own_members = np.where(assignments == own_cluster)[0]

        if len(own_members) <= 1:
            silhouettes[i] = 0.0
            continue

        own_others = own_members[own_members != i]
        a_value = distances[i, own_others].mean()

        b_value = np.inf
        for other_cluster in range(k):
            if other_cluster == own_cluster:
                continue

            other_members = np.where(assignments == other_cluster)[0]
            if len(other_members) > 0:
                b_value = min(b_value, distances[i, other_members].mean())

        denominator = max(a_value, b_value)
        silhouettes[i] = 0.0 if denominator == 0 else (b_value - a_value) / denominator

    return float(silhouettes.mean())


class ClusterApp(tk.Tk):
    def __init__(self, startup_path=None):
        super().__init__()
        self.title("Cluster Analysis")
        self.geometry("1450x850")
        self.minsize(1100, 650)

        self.dataframe = None
        self.source_path = None
        self.numeric_columns = []
        self.results = {}
        self.cleaned_data = None
        self.cleaned_original_rows = None
        self.cleaned_labels = None
        self.selected_columns = []
        self.z_data = None
        self.dropped_rows = 0
        self.colors = []
        self.current_display_k = None

        self.file_var = tk.StringVar(value="No file loaded")
        self.label_var = tk.StringVar(value="(None: use original row number)")
        self.standardize_var = tk.BooleanVar(value=True)
        self.min_k_var = tk.StringVar(value="2")
        self.max_k_var = tk.StringVar(value="6")
        self.restarts_var = tk.StringVar(value="50")
        self.seed_var = tk.StringVar(value="5534")
        self.display_k_var = tk.StringVar(value="2")
        self.status_var = tk.StringVar(value="Load a CSV or XLSX file to begin.")

        self._build_interface()

        if startup_path and Path(startup_path).is_file():
            self.after(100, lambda: self.load_file(startup_path))
        else:
            self.after(150, self.choose_file)

    def _build_interface(self):
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        side = ttk.Frame(container, padding=10, width=320)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        main = ttk.Frame(container, padding=(0, 10, 10, 10))
        main.pack(side="left", fill="both", expand=True)

        ttk.Label(side, text="Cluster Analysis", font=("TkDefaultFont", 14, "bold")).pack(
            anchor="w", pady=(0, 8)
        )

        ttk.Button(side, text="Load CSV or XLSX...", command=self.choose_file).pack(
            fill="x", pady=(0, 4)
        )
        ttk.Label(
            side, textvariable=self.file_var, wraplength=295, justify="left"
        ).pack(anchor="w", pady=(0, 10))

        ttk.Separator(side).pack(fill="x", pady=4)
        ttk.Label(side, text="Analysis columns (numeric only):").pack(anchor="w")

        columns_frame = ttk.Frame(side)
        columns_frame.pack(fill="both", expand=False, pady=(2, 6))

        self.columns_listbox = tk.Listbox(
            columns_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=9,
        )
        columns_scroll = ttk.Scrollbar(
            columns_frame, orient="vertical", command=self.columns_listbox.yview
        )
        self.columns_listbox.configure(yscrollcommand=columns_scroll.set)
        self.columns_listbox.pack(side="left", fill="both", expand=True)
        columns_scroll.pack(side="right", fill="y")

        ttk.Label(side, text="Optional label column:").pack(anchor="w")
        self.label_combo = ttk.Combobox(
            side, textvariable=self.label_var, state="readonly"
        )
        self.label_combo.pack(fill="x", pady=(2, 8))

        ttk.Checkbutton(
            side,
            text="Standardize for clustering",
            variable=self.standardize_var,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Separator(side).pack(fill="x", pady=4)
        params = ttk.Frame(side)
        params.pack(fill="x", pady=4)

        self._parameter_row(params, "Minimum k:", self.min_k_var, 0)
        self._parameter_row(params, "Maximum k:", self.max_k_var, 1)
        self._parameter_row(params, "Restarts per k:", self.restarts_var, 2)
        self._parameter_row(params, "Random seed:", self.seed_var, 3)
        self._parameter_row(params, "Displayed k:", self.display_k_var, 4)

        ttk.Button(side, text="Run / Re-run Analysis", command=self.run_analysis).pack(
            fill="x", pady=(8, 3)
        )
        ttk.Button(side, text="Update Displayed k", command=self.update_display).pack(
            fill="x", pady=3
        )
        ttk.Button(side, text="Save Assignments CSV...", command=self.save_assignments).pack(
            fill="x", pady=(3, 8)
        )

        ttk.Separator(side).pack(fill="x", pady=4)
        ttk.Label(side, text="Status:", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        status_label = ttk.Label(
            side,
            textvariable=self.status_var,
            wraplength=295,
            justify="left",
            foreground="#333333",
        )
        status_label.pack(anchor="w", fill="x", pady=(3, 0))

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)

        self.silhouette_tab = ttk.Frame(self.notebook)
        self.parallel_tab = ttk.Frame(self.notebook)
        self.summary_tab = ttk.Frame(self.notebook)
        self.assignments_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.silhouette_tab, text="Silhouette Scores")
        self.notebook.add(self.parallel_tab, text="Parallel Coordinates")
        self.notebook.add(self.summary_tab, text="Cluster Summary")
        self.notebook.add(self.assignments_tab, text="Assignments")

        self.silhouette_figure = Figure(figsize=(8, 6), dpi=100)
        self.silhouette_canvas = FigureCanvasTkAgg(self.silhouette_figure, self.silhouette_tab)
        self.silhouette_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.parallel_figure = Figure(figsize=(8, 6), dpi=100)
        self.parallel_canvas = FigureCanvasTkAgg(self.parallel_figure, self.parallel_tab)
        self.parallel_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.summary_tree = self._make_tree(self.summary_tab)
        self.assignments_tree = self._make_tree(self.assignments_tab)

    @staticmethod
    def _parameter_row(parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable, width=14).grid(
            row=row, column=1, sticky="ew", padx=(8, 0), pady=2
        )
        parent.columnconfigure(1, weight=1)

    @staticmethod
    def _make_tree(parent):
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True)

        tree = ttk.Treeview(outer, show="headings")
        y_scroll = ttk.Scrollbar(outer, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(outer, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        return tree

    def choose_file(self):
        path = filedialog.askopenfilename(
            title="Choose data file",
            filetypes=[
                ("Data files", "*.csv *.xlsx"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_file(path)

    def load_file(self, path):
        suffix = Path(path).suffix.lower()

        if suffix not in (".csv", ".xlsx"):
            self._error("Unsupported file type. Select a .csv or .xlsx file.")
            return

        try:
            if suffix == ".csv":
                dataframe = pd.read_csv(path)
            else:
                dataframe = pd.read_excel(path)
        except Exception as exc:
            self._error(f"Could not read file:\n{exc}")
            return

        if dataframe.empty:
            self._error("The selected file contains no data rows.")
            return

        self.dataframe = dataframe.copy()
        self.source_path = path
        self.numeric_columns = list(
            dataframe.select_dtypes(include=[np.number]).columns
        )
        self.results = {}
        self.cleaned_data = None
        self.current_display_k = None

        self.file_var.set(path)
        self.columns_listbox.delete(0, tk.END)
        for column in self.numeric_columns:
            self.columns_listbox.insert(tk.END, str(column))

        label_options = ["(None: use original row number)"] + [
            str(column) for column in dataframe.columns
        ]
        self.label_combo["values"] = label_options
        self.label_var.set("(None: use original row number)")

        if not self.numeric_columns:
            self.status_var.set(
                f"Loaded {Path(path).name}, but it contains no numeric columns."
            )
            self._clear_outputs()
            return

        self.status_var.set(
            f"Loaded {Path(path).name}: {len(dataframe)} rows, "
            f"{len(self.numeric_columns)} numeric column(s)."
        )
        self._clear_outputs()

    def _parse_integer(self, variable, name, minimum=None):
        try:
            value = int(variable.get().strip())
        except ValueError:
            raise ValueError(f"{name} must be an integer.")

        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")

        return value

    def _selected_analysis_columns(self):
        selected_indices = self.columns_listbox.curselection()
        return [self.numeric_columns[i] for i in selected_indices]

    def run_analysis(self):
        if self.dataframe is None:
            self._error("Load a data file before running analysis.")
            return

        selected_columns = self._selected_analysis_columns()
        if not selected_columns:
            self._error("Select at least one numeric analysis column.")
            return

        label_name = self.label_var.get()
        if label_name != "(None: use original row number)" and label_name in selected_columns:
            self._error(
                "The selected label column is also selected for analysis. "
                "Remove it from the analysis-column selection."
            )
            return

        try:
            requested_min = self._parse_integer(self.min_k_var, "Minimum k")
            requested_max = self._parse_integer(self.max_k_var, "Maximum k")
            restarts = self._parse_integer(self.restarts_var, "Restarts per k", 1)
            seed = self._parse_integer(self.seed_var, "Random seed")
        except ValueError as exc:
            self._error(str(exc))
            return

        if requested_min > requested_max:
            self._error("Minimum k cannot be greater than maximum k.")
            return

        source = self.dataframe
        complete_mask = source[selected_columns].notna().all(axis=1)
        cleaned = source.loc[complete_mask, selected_columns].copy()
        original_rows = source.index.to_numpy() + 1
        retained_original_rows = original_rows[complete_mask.to_numpy()]
        dropped_rows = int((~complete_mask).sum())

        if len(cleaned) < 2:
            self._error(
                "Not enough data for clustering: fewer than two rows remain "
                "after dropping rows with missing analysis values."
            )
            return

        feasible_low = 2
        feasible_high = min(10, len(cleaned) - 1)

        if feasible_high < feasible_low:
            self._error(
                "Not enough data for clustering: at least three usable rows "
                "are required because k must be at least 2 and less than n."
            )
            return

        effective_min = int(np.clip(requested_min, feasible_low, feasible_high))
        effective_max = int(np.clip(requested_max, feasible_low, feasible_high))
        effective_ks = sorted(set(range(effective_min, effective_max + 1)))

        raw_data = cleaned.to_numpy(dtype=float)
        means = raw_data.mean(axis=0)
        standard_deviations = raw_data.std(axis=0, ddof=0)
        zero_variance = standard_deviations == 0

        safe_standard_deviations = standard_deviations.copy()
        safe_standard_deviations[zero_variance] = 1.0
        z_data = (raw_data - means) / safe_standard_deviations
        z_data[:, zero_variance] = 0.0

        clustering_data = z_data if self.standardize_var.get() else raw_data

        if label_name == "(None: use original row number)":
            labels = retained_original_rows.astype(str)
        else:
            raw_labels = source.loc[complete_mask, label_name]
            labels = np.array(
                [
                    str(original_row) if pd.isna(value) else str(value)
                    for value, original_row in zip(raw_labels, retained_original_rows)
                ],
                dtype=object,
            )

        self.status_var.set("Running k-means and silhouette calculations...")
        self.update_idletasks()

        try:
            results = {}
            empty_cluster_events = 0

            for k in effective_ks:
                result = best_kmeans(clustering_data, k, restarts, seed)
                result["silhouette"] = average_silhouette(
                    clustering_data, result["assignments"], k
                )
                results[k] = result
                empty_cluster_events += result["empty_reinitializations"]

        except Exception as exc:
            self._error(f"Analysis failed due to a numerical error:\n{exc}")
            return

        self.selected_columns = selected_columns
        self.cleaned_data = cleaned.reset_index(drop=True)
        self.cleaned_original_rows = retained_original_rows
        self.cleaned_labels = labels
        self.z_data = z_data
        self.results = results
        self.dropped_rows = dropped_rows

        requested_display = self._try_parse_display_k()
        if requested_display is None:
            requested_display = effective_ks[0]

        effective_display = int(np.clip(requested_display, feasible_low, feasible_high))

        messages = [
            f"Analyzed k={effective_ks}.",
            f"Dropped {dropped_rows} row(s) with missing analysis values.",
        ]

        if requested_min != effective_min or requested_max != effective_max:
            messages.append(
                f"Requested k range {requested_min}–{requested_max}; "
                f"effective range {effective_min}–{effective_max} "
                f"(feasible range {feasible_low}–{feasible_high})."
            )

        if zero_variance.any():
            zero_names = ", ".join(str(selected_columns[i]) for i in np.where(zero_variance)[0])
            messages.append(f"Zero-variance columns assigned z-score 0: {zero_names}.")

        if empty_cluster_events:
            messages.append(
                f"Empty clusters were reinitialized {empty_cluster_events} time(s) "
                "during candidate runs."
            )

        best_silhouette_k = min(
            results,
            key=lambda k: (-results[k]["silhouette"], k),
        )
        messages.append(
            f"Best silhouette k={best_silhouette_k} "
            f"({results[best_silhouette_k]['silhouette']:.4f}); ties use smallest k."
        )

        self.status_var.set(" ".join(messages))
        self._draw_silhouette()

        if effective_display not in results:
            self.current_display_k = None
            self._clear_display_outputs()
            self.status_var.set(
                self.status_var.get()
                + f" Displayed k request {requested_display} clamps to "
                f"{effective_display}, which was not analyzed. Re-run with a range "
                "containing that k."
            )
            return

        if requested_display != effective_display:
            self.status_var.set(
                self.status_var.get()
                + f" Displayed k requested {requested_display}; using effective "
                f"k={effective_display}."
            )

        self.display_solution(effective_display)

    def _try_parse_display_k(self):
        try:
            return int(self.display_k_var.get().strip())
        except ValueError:
            return None

    def update_display(self):
        if not self.results:
            self._error("Run analysis before choosing a displayed k.")
            return

        try:
            requested_k = self._parse_integer(self.display_k_var, "Displayed k")
        except ValueError as exc:
            self._error(str(exc))
            return

        n_rows = len(self.cleaned_data)
        feasible_low, feasible_high = 2, min(10, n_rows - 1)
        effective_k = int(np.clip(requested_k, feasible_low, feasible_high))

        if effective_k not in self.results:
            self._error(
                f"Displayed k request {requested_k} has effective value {effective_k}, "
                "but that k was not analyzed. Re-run analysis with a range containing it."
            )
            return

        if requested_k != effective_k:
            self.status_var.set(
                f"Displayed k requested {requested_k}; clamped to feasible k={effective_k}. "
                f"{self.dropped_rows} row(s) were dropped for missing analysis values."
            )

        self.display_solution(effective_k)

    def display_solution(self, k):
        self.current_display_k = k
        self.colors = self._cluster_colors(k)
        result = self.results[k]

        self._draw_parallel(k)
        self._populate_summary(k)
        self._populate_assignments(k)

        self.status_var.set(
            self.status_var.get()
            + f" Displaying k={k}: inertia={result['inertia']:.4f}, "
            f"average silhouette={result['silhouette']:.4f}."
        )

    @staticmethod
    def _cluster_colors(k):
        cmap = matplotlib.colormaps["tab10"]
        return [matplotlib.colors.to_hex(cmap(i % 10)) for i in range(k)]

    def _draw_silhouette(self):
        self.silhouette_figure.clear()
        axis = self.silhouette_figure.add_subplot(111)

        ks = sorted(self.results)
        scores = [self.results[k]["silhouette"] for k in ks]
        best_k = min(ks, key=lambda k: (-self.results[k]["silhouette"], k))
        best_score = self.results[best_k]["silhouette"]

        axis.plot(ks, scores, marker="o", linewidth=1.8, color="#1f77b4")
        axis.scatter(
            [best_k], [best_score], s=110, color="#d62728",
            edgecolor="black", zorder=3, label=f"Best: k={best_k}"
        )
        axis.axvline(best_k, color="#d62728", linestyle="--", alpha=0.55)
        axis.annotate(
            f"k={best_k}\n{best_score:.4f}",
            (best_k, best_score),
            xytext=(8, 8),
            textcoords="offset points",
        )

        for k, score in zip(ks, scores):
            axis.annotate(
                f"{score:.3f}", (k, score),
                xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=8,
            )

        axis.set_title("Average Silhouette Score by Number of Clusters")
        axis.set_xlabel("Number of clusters (k)")
        axis.set_ylabel("Average silhouette score")
        axis.set_xticks(ks)
        axis.grid(True, alpha=0.3)
        axis.legend()
        self.silhouette_figure.tight_layout()
        self.silhouette_canvas.draw()

    def _draw_parallel(self, k):
        self.parallel_figure.clear()
        axis = self.parallel_figure.add_subplot(111)

        assignments = self.results[k]["assignments"]
        n_columns = len(self.selected_columns)
        x_values = np.arange(n_columns)

        for row_index, profile in enumerate(self.z_data):
            cluster_id = assignments[row_index]
            axis.plot(
                x_values,
                profile,
                color=self.colors[cluster_id],
                linewidth=0.8,
                alpha=0.30,
            )

        legend_handles = []
        for cluster_id in range(k):
            members = self.z_data[assignments == cluster_id]
            mean_profile = members.mean(axis=0)
            axis.plot(
                x_values,
                mean_profile,
                color=self.colors[cluster_id],
                linewidth=3.2,
                alpha=1.0,
            )
            legend_handles.append(
                Line2D(
                    [0], [0],
                    color=self.colors[cluster_id],
                    linewidth=3.2,
                    label=f"Cluster {cluster_id + 1} mean",
                )
            )

        axis.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
        axis.set_xticks(x_values)
        axis.set_xticklabels([str(c) for c in self.selected_columns], rotation=25, ha="right")
        axis.set_ylabel("Standardized value (z-score)")
        axis.set_title(
            f"Parallel Coordinates: Displayed k={k}\n"
            "Thin lines = retained rows; thick lines = cluster means"
        )
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(handles=legend_handles, loc="best", fontsize=8)
        self.parallel_figure.tight_layout()
        self.parallel_canvas.draw()

    def _configure_tree(self, tree, headings, widths):
        for item in tree.get_children():
            tree.delete(item)

        tree["columns"] = headings
        for heading, width in zip(headings, widths):
            tree.heading(heading, text=heading)
            tree.column(heading, width=width, minwidth=80, anchor="center")

    def _populate_summary(self, k):
        headings = ["Color", "Cluster", "Size"] + [
            f"Mean: {column} (z)" for column in self.selected_columns
        ]
        widths = [70, 85, 80] + [145] * len(self.selected_columns)
        self._configure_tree(self.summary_tree, headings, widths)

        assignments = self.results[k]["assignments"]

        for cluster_id in range(k):
            member_mask = assignments == cluster_id
            cluster_size = int(member_mask.sum())
            mean_z = self.z_data[member_mask].mean(axis=0)

            values = [
                "■",
                str(cluster_id + 1),
                str(cluster_size),
            ] + [f"{value:.3f}" for value in mean_z]

            tag = f"cluster_{cluster_id}"
            self.summary_tree.insert("", "end", values=values, tags=(tag,))
            self.summary_tree.tag_configure(tag, foreground=self.colors[cluster_id])

    def _populate_assignments(self, k):
        headings = ["Color", "Original Row", "Row Label", "Cluster"] + [
            str(column) for column in self.selected_columns
        ]
        widths = [60, 105, 170, 85] + [130] * len(self.selected_columns)
        self._configure_tree(self.assignments_tree, headings, widths)

        assignments = self.results[k]["assignments"]

        for row_position, (_, row) in enumerate(self.cleaned_data.iterrows()):
            cluster_id = int(assignments[row_position])
            values = [
                "■",
                str(self.cleaned_original_rows[row_position]),
                str(self.cleaned_labels[row_position]),
                str(cluster_id + 1),
            ] + [self._format_value(row[column]) for column in self.selected_columns]

            tag = f"cluster_{cluster_id}"
            self.assignments_tree.insert("", "end", values=values, tags=(tag,))
            self.assignments_tree.tag_configure(tag, foreground=self.colors[cluster_id])

    @staticmethod
    def _format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)

    def save_assignments(self):
        if self.current_display_k is None or self.current_display_k not in self.results:
            self._error("There is no displayed clustering result to save.")
            return

        path = filedialog.asksaveasfilename(
            title="Save Cluster Assignments",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        try:
            k = self.current_display_k
            assignments = self.results[k]["assignments"] + 1

            output = pd.DataFrame({
                "original_input_row_number": self.cleaned_original_rows,
                "row_label": self.cleaned_labels,
                f"cluster_k_{k}": assignments,
            })

            for column in self.selected_columns:
                output[str(column)] = self.cleaned_data[column].to_numpy()

            output.to_csv(path, index=False)

            self.status_var.set(
                f"Saved {len(output)} clustered row(s) for displayed k={k} to "
                f"{path}. Excluded {self.dropped_rows} row(s) with missing "
                "analysis values."
            )
        except Exception as exc:
            self._error(f"Could not save CSV:\n{exc}")

    def _clear_outputs(self):
        self.silhouette_figure.clear()
        self.silhouette_canvas.draw()
        self.parallel_figure.clear()
        self.parallel_canvas.draw()
        self._clear_tree(self.summary_tree)
        self._clear_tree(self.assignments_tree)

    def _clear_display_outputs(self):
        self.parallel_figure.clear()
        self.parallel_canvas.draw()
        self._clear_tree(self.summary_tree)
        self._clear_tree(self.assignments_tree)

    @staticmethod
    def _clear_tree(tree):
        for item in tree.get_children():
            tree.delete(item)
        tree["columns"] = ()

    def _error(self, message):
        self.status_var.set(message.replace("\n", " "))
        messagebox.showerror("Cluster Analysis", message)


def selftest(path, cols, kmin=2, kmax=10, restarts=50, seed=5534):
    """Run the analysis path with no window and print the silhouette curve.

    Exists so the shipped tool can be checked against a known answer on a machine
    with no display. On distance_test.csv with x and y the curve should read
    0.403 0.500 0.489 0.532 0.536 0.524 0.507 0.504 0.426 and peak at k=6, four
    thousandths ahead of k=5.
    """
    frame = pd.read_csv(path) if str(path).lower().endswith(".csv") else pd.read_excel(path)
    data = frame[list(cols)].apply(pd.to_numeric, errors="coerce").dropna()
    dropped = len(frame) - len(data)
    values = data.to_numpy(dtype=float)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    values = (values - values.mean(axis=0)) / std
    print(f"{len(values)} rows, {values.shape[1]} columns, {dropped} dropped, "
          f"{restarts} restarts, seed {seed}")
    best_k, best_s = None, -2.0
    for k in range(kmin, kmax + 1):
        result = best_kmeans(values, k, restarts, seed)
        labels = np.asarray(result["assignments"]).ravel()
        s = average_silhouette(values, labels, k)
        sizes = np.bincount(labels.astype(int), minlength=k)
        flag = " <-- largest" if (s is not None and s > best_s) else ""
        if s is not None and s > best_s:
            best_k, best_s = k, s
        print(f"  k={k:2d}  silhouette={'n/a' if s is None else format(s, '.4f')}  "
              f"sizes={sorted(sizes.tolist(), reverse=True)}{flag}")
    print(f"largest silhouette at k={best_k}, {best_s:.4f}")
    return best_k


def main():
    startup_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = ClusterApp(startup_path)
    app.mainloop()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _a = [x for x in sys.argv[1:] if x != "--selftest"]
        _p = _a[0] if _a else "distance_test.csv"
        _c = _a[1].split(",") if len(_a) > 1 else ["x", "y"]
        selftest(_p, _c)
        sys.exit(0)
    main()