#!/usr/bin/env python3
"""
reg_server.py, the regression explorer as a browser page.

BIT 5534 lec12_1 reference. The model code is reg_explorer.py's own: this file
imports prepare, run_model, metrics, summary and term_table from it and changes
none of them, so the numbers match the desktop tool. Only the window is new. A
small server built into Python (http.server) runs the model, and the page
reg_page.html, in the same folder, shows the settings, the results and the
team's agent.

    python reg_server.py            then open http://localhost:8000 in a browser

Keep reg_explorer.py and reg_page.html in the same folder as this file.
Stop the server with Ctrl+C in the terminal.
"""
import base64
import io
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.argv.append("--selftest")          # import reg_explorer without opening its window
import reg_explorer as rx
sys.argv.remove("--selftest")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8000
STATE = {"df": None, "name": None, "report": ""}


# ----------------------------------------------------------------- the file
def load_csv(name, text):
    df = pd.read_csv(io.StringIO(text))
    STATE["df"], STATE["name"] = df, name
    cols = []
    for c in df.columns:
        cols.append({"name": c,
                     "kind": rx.default_kind(df[c]),
                     "numeric": bool(pd.api.types.is_numeric_dtype(df[c])),
                     "levels": int(df[c].nunique()),
                     "role_like": rx.role_like(df[c]),
                     "positive": rx.default_positive(df[c])})
    return {"file": name, "rows": len(df), "columns": cols}


# ----------------------------------------------------------------- the fit
def fit(req):
    df = STATE["df"]
    if df is None:
        raise ValueError("Open a .csv file first.")
    target, inputs, role = req["target"], req["inputs"], req.get("role")
    kind = req.get("kind") or rx.default_kind(df[target])
    if not inputs:
        raise ValueError("Choose at least one input column.")
    if not role:
        raise ValueError("Choose the validation column.")
    p = rx.prepare(df, target, inputs, kind, role, req.get("positive"))
    if kind == "category" and len(p.classes) != 2:
        raise ValueError(f"Logistic regression here takes a target with two levels, "
                         f"and {target} has {len(p.classes)}.")
    roles = p.data[role].astype(str).to_numpy()
    r = rx.run_model(p, roles)
    text = fit_text(p, r)
    STATE["report"] = report_text(p, r, text, role)
    return {"fit": text,
            "logworth": png(draw_logworth(p, r)),
            "predictions": png(draw_predictions(p, r)),
            "dropped": int(p.dropped)}


def fit_text(p, r):
    """The Fit tab of reg_explorer.py, line for line."""
    out = []
    if p.kind == "number":
        out.append(f"Multiple regression of {p.target_name}")
    else:
        out.append(f"Logistic regression of {p.target_name}, probability of {p.positive}")
    out.append(f"Inputs: {', '.join(p.names)}")
    out.append("")
    if p.kind == "number":
        out.append(f"{'rows':26}{'n':>6}{'R squared':>12}{'RMSE':>12}{'MSE':>14}")
    else:
        out.append(f"{'rows':26}{'n':>6}{'misclassification':>20}{'accuracy':>11}{'RSquare (U)':>14}")

    def row(label, m):
        if m is None:
            return
        if p.kind == "number":
            out.append(f"{label:26}{m['n']:6}{m['r2']:12.3f}{m['rmse']:12.3f}{m['mse']:14.3f}")
        else:
            out.append(f"{label:26}{m['n']:6}{m['miss']:20.3f}{m['acc']:11.3f}{m['r2u']:14.3f}")

    for (lab, val, train), f in zip(r.folds, r.fits):
        pre = f"{lab}: " if r.kfold else ""
        row(f"{pre}train", f.train_m)
        row(f"{pre}validation", f.val_m)
    if r.kfold:
        tr, va = rx.summary(p, r)
        name = "R squared" if p.kind == "number" else "misclassification"
        out.append(f"\nAverage over {len(r.fits)} folds: train {name} {tr:.3f}, validation {name} {va:.3f}")
    if r.test.any():
        out.append("")
        row("test (reported only)", rx.metrics(p, None, r.test, pred=r.pred, prob=r.prob))
    where = (f"every row that is not a test row ({int(r.final_rows.sum())} rows)" if r.kfold
             else f"the {int(r.final_rows.sum())} train rows")
    out.append(f"\nTerms, from the fit on {where}")
    stat = "t ratio" if p.kind == "number" else "z"
    out.append(f"{'term':26}{'estimate':>12}{'std error':>12}{stat:>10}{'p value':>12}{'logworth':>10}")
    for n, est, se, st, pv, lw in rx.term_table(p, r.final.model):
        out.append(f"{n[:25]:26}{est:12.4f}{se:12.4f}{st:10.2f}{pv:12.3g}{lw:10.2f}")
    if p.kind == "category":
        out.extend(confusion(p, r))
    return "\n".join(out)


def confusion(p, r):
    out = []
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
        out.append(f"\nConfusion table, {name} rows (actual down, predicted across)")
        out.append(f"{'':12}" + "".join(f"{c:>18}" for c in p.classes))
        for a in p.classes:
            rr = rows & (p.y == a)
            tot = int(rr.sum())
            line = f"{a:12}"
            for c in p.classes:
                n = int((rr & (r.pred == c)).sum())
                line += f"{n:>9} ({100 * n / tot if tot else 0:5.1f}%) "
            out.append(line)
    return out


def report_text(p, r, text, role):
    head = [f"Regression explorer report, browser version",
            f"File: {STATE['name']}",
            f"Target: {p.target_name} ({p.kind})" + (f", positive class {p.positive}" if p.kind == "category" else ""),
            f"Validation column: {role}",
            f"Rows used: {len(p.y)}, rows dropped for missing values: {p.dropped}",
            ""]
    return "\n".join(head) + text + "\n"


# ----------------------------------------------------------------- the figures
def png(fig):
    buf = io.BytesIO()
    FigureCanvasAgg(fig).print_png(buf)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_logworth(p, r):
    rows = [x for x in rx.term_table(p, r.final.model) if x[0] != "Intercept"]
    rows.sort(key=lambda x: x[5])
    fig = Figure(figsize=(8, max(3, 0.32 * len(rows) + 1.2)), dpi=100)
    ax = fig.add_subplot(111)
    lw = [min(x[5], 16) for x in rows]
    bars = ax.barh([x[0] for x in rows], lw, color=["#2c6fbb" if x[4] < 0.05 else "#9bb7d8" for x in rows])
    ax.axvline(-np.log10(0.05), color="#c0392b", lw=1, ls="--")
    for b, v, x in zip(bars, lw, rows):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{x[5]:.2f}", va="center", fontsize=9)
    ax.set_xlabel("logworth, -log10(p value)")
    ax.set_xlim(0, max(1.6, max(lw + [0]) * 1.18 + 0.2))
    what = "multiple regression" if p.kind == "number" else f"logistic regression, P({p.positive})"
    ax.set_title(f"{what}: logworth of every term, darker bars below p = 0.05", fontsize=10)
    fig.tight_layout()
    return fig


def draw_predictions(p, r):
    groups = [("train", r.folds[0][2]), ("validation", r.folds[0][1])] if not r.kfold \
        else [("validation", ~r.test)]
    if r.test.any():
        groups.append(("test", r.test))
    fig = Figure(figsize=(4 * len(groups), 4), dpi=100)
    if p.kind == "number":
        lo, hi = min(p.y.min(), r.pred.min()), max(p.y.max(), r.pred.max())
        for i, (name, rows) in enumerate(groups):
            ax = fig.add_subplot(1, len(groups), i + 1)
            ax.scatter(r.pred[rows], p.y[rows], s=12, alpha=0.6, color=rx.ROLE_COLORS[name])
            ax.plot([lo, hi], [lo, hi], color="black", lw=1)
            m = rx.metrics(p, None, rows, pred=r.pred)
            ax.set_title(f"{name}, n={m['n']}, R squared {m['r2']:.3f}", fontsize=10)
            ax.set_xlabel("predicted")
            if i == 0:
                ax.set_ylabel("actual")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(alpha=0.3)
    else:
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
    return fig


# ----------------------------------------------------------------- the server
class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "reg_page.html"), encoding="utf-8") as f:
                self.send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/report.txt":
            self.send(200, STATE["report"] or "Fit a model first.\n", "text/plain; charset=utf-8")
        else:
            self.send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/open":
                out = load_csv(req["name"], req["text"])
            elif self.path == "/fit":
                out = fit(req)
            else:
                return self.send(404, "not found", "text/plain")
            self.send(200, json.dumps(out), "application/json")
        except Exception as e:
            self.send(400, json.dumps({"error": str(e)}), "application/json")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    try:
        server = HTTPServer(("localhost", PORT), Handler)
        print(f"Regression explorer running at http://localhost:{PORT}  (Ctrl+C stops it)")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception:
        import traceback
        traceback.print_exc()
        input("The server stopped with the error above. Press Enter to close this window.")
