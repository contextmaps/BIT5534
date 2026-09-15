"""
Plant floor simulator for BIT 5534, week 4 lecture 1.

Draws the thing the lecture is about, in one window:

  left        the floor, a 100 meter square bay. Every recorded event is a dot,
              red if the response crew can reach it inside the service target
              and blue if they cannot. The twenty acoustic monitors are the
              green diamonds. Reaching an event means walking the aisles, so
              the reachable set is every point within 50 meters of the center
              measured along the grid, which draws a diamond.

  top right   the two monitors whose columns agree most, plotted against each
              other. Two monitors standing near each other repeat one another.

  bottom mid  the two monitors whose columns agree least, plotted against each
              other. No better. Neither pair recovers the floor.

  right       the twenty by twenty correlation matrix of the monitor columns.

Two sliders change the simulation while the room watches. Noise is the error in
each reading. Spread pushes the monitors away from the floor, and it is the
knob that matters most, though the lecture does not say why.

Run it by double clicking, or from a terminal. Without a display, or with
--save, it writes the four panels to plant_floor.png instead and exits.

    python plant_floor.py
    python plant_floor.py --save
    python plant_floor.py --write-csv my_floor.csv
"""

import argparse
import sys

import numpy as np

HALF = 50.0          # the bay runs from -50 to +50 meters on each side
REACH = 50.0         # the service target, in meters walked along the aisles
N_EVENTS = 1000
N_MONITORS = 20
DEFAULT_NOISE = 0.5  # a reading is the true distance times 1 plus or minus a quarter
DEFAULT_SPREAD = 1.0
DEFAULT_SEED = 4


def simulate(seed=DEFAULT_SEED, n_events=N_EVENTS, n_monitors=N_MONITORS,
             noise=DEFAULT_NOISE, spread=DEFAULT_SPREAD):
    """One draw of the floor. Returns the events, the monitors and the readings.

    A reading is the straight line distance from the event to the monitor,
    multiplied by one plus a uniform error of half the noise setting either
    way. At the default noise of 0.5 that is plus or minus 25 percent.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-HALF, HALF, n_events)
    y = rng.uniform(-HALF, HALF, n_events)
    reachable = (np.abs(x) + np.abs(y) < REACH).astype(int)

    mx = rng.uniform(-HALF, HALF, n_monitors) * spread
    my = rng.uniform(-HALF, HALF, n_monitors) * spread

    true_distance = np.sqrt((x[:, None] - mx) ** 2 + (y[:, None] - my) ** 2)
    error = (rng.random((n_events, n_monitors)) - 0.5) * noise
    readings = true_distance * (1.0 + error)

    return x, y, reachable, mx, my, readings, true_distance


def extreme_pairs(readings):
    """The most agreeing pair of monitors and the least agreeing pair."""
    corr = np.corrcoef(readings, rowvar=False)
    off = corr.copy()
    np.fill_diagonal(off, -np.inf)
    hi = np.unravel_index(np.argmax(off), off.shape)
    np.fill_diagonal(off, np.inf)
    lo = np.unravel_index(np.argmin(off), off.shape)
    return corr, tuple(sorted(hi)), tuple(sorted(lo))


def draw(fig, axes, state):
    x, y, reachable, mx, my, readings, true_distance = state
    ax_floor, ax_hi, ax_lo, ax_corr, ax_noise = axes
    corr, hi, lo = extreme_pairs(readings)

    for ax in axes:
        ax.clear()

    red = reachable == 1
    ax_floor.scatter(x[~red], y[~red], s=7, c="#2c6fbb", alpha=0.75, linewidths=0)
    ax_floor.scatter(x[red], y[red], s=7, c="#d23b2e", alpha=0.75, linewidths=0)
    ax_floor.plot([REACH, 0, -REACH, 0, REACH], [0, REACH, 0, -REACH, 0],
                  color="#333333", linewidth=1.2, linestyle="--")
    ax_floor.scatter(mx, my, s=150, marker="D", c="#2f9e44",
                     edgecolors="black", linewidths=0.8, zorder=5)
    for i, (a, b) in enumerate(zip(mx, my), start=1):
        ax_floor.annotate(f"m{i}", (a, b), xytext=(7, 5),
                          textcoords="offset points", fontsize=9, zorder=6)
    lim = max(HALF, np.abs(np.r_[mx, my]).max()) * 1.08
    ax_floor.set_xlim(-lim, lim)
    ax_floor.set_ylim(-lim, lim)
    ax_floor.set_aspect("equal")
    ax_floor.set_title(f"The floor. {red.sum()} of {len(x)} events reachable",
                       fontsize=12)
    ax_floor.set_xlabel("meters east")
    ax_floor.set_ylabel("meters north")

    for ax, (i, j), label in ((ax_hi, hi, "closest agreement"),
                              (ax_lo, lo, "loosest agreement")):
        ax.scatter(readings[~red, i], readings[~red, j], s=7,
                   c="#2c6fbb", alpha=0.7, linewidths=0)
        ax.scatter(readings[red, i], readings[red, j], s=7,
                   c="#d23b2e", alpha=0.7, linewidths=0)
        ax.set_xlabel(f"m{i + 1}")
        ax.set_ylabel(f"m{j + 1}")
        ax.set_title(f"{label}, r = {corr[i, j]:.2f}", fontsize=11)

    im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_corr.set_title("correlation of the twenty columns", fontsize=11)
    if not hasattr(fig, "_corr_bar"):
        fig._corr_bar = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.03)
    ax_corr.set_xticks(range(0, N_MONITORS, 4))
    ax_corr.set_xticklabels([f"m{k + 1}" for k in range(0, N_MONITORS, 4)], fontsize=8)
    ax_corr.set_yticks(range(0, N_MONITORS, 4))
    ax_corr.set_yticklabels([f"m{k + 1}" for k in range(0, N_MONITORS, 4)], fontsize=8)

    center = int(np.argmin(mx ** 2 + my ** 2))
    ax_noise.scatter(true_distance[:, center], readings[:, center], s=6,
                     c="#6c757d", alpha=0.55, linewidths=0)
    top = float(true_distance[:, center].max())
    ax_noise.plot([0, top], [0, top], color="black", linewidth=1.3)
    ax_noise.set_xlabel(f"true distance to m{center + 1}, meters")
    ax_noise.set_ylabel("what m%d reports" % (center + 1))
    ax_noise.set_title("one monitor, reading against truth", fontsize=11)
    return im


def write_csv(path, reachable, readings):
    import csv
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"m{i}" for i in range(1, readings.shape[1] + 1)] + ["reachable"])
        for row, flag in zip(readings, reachable):
            writer.writerow([f"{v:.9g}" for v in row] + [int(flag)])
    print(f"wrote {path}")


def build_figure():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig = plt.figure(figsize=(16, 8))
    try:
        fig.canvas.manager.set_window_title("BIT 5534 week 4, the plant floor")
    except Exception:
        pass
    grid = fig.add_gridspec(2, 3, width_ratios=[2.0, 1, 1],
                            left=0.04, right=0.97, top=0.93, bottom=0.16,
                            wspace=0.32, hspace=0.38)
    ax_floor = fig.add_subplot(grid[:, 0])
    ax_hi = fig.add_subplot(grid[0, 1])
    ax_lo = fig.add_subplot(grid[0, 2])
    ax_corr = fig.add_subplot(grid[1, 1])
    ax_noise_panel = fig.add_subplot(grid[1, 2])
    return fig, (ax_floor, ax_hi, ax_lo, ax_corr, ax_noise_panel)


def main():
    parser = argparse.ArgumentParser(description="Plant floor simulator, BIT 5534 week 4.")
    parser.add_argument("--save", action="store_true",
                        help="write plant_floor.png and exit, no window")
    parser.add_argument("--write-csv", metavar="PATH",
                        help="write the monitor readings to PATH and exit")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--spread", type=float, default=DEFAULT_SPREAD)
    args = parser.parse_args()

    import matplotlib
    headless = args.save or args.write_csv
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state = simulate(seed=args.seed, noise=args.noise, spread=args.spread)

    if args.write_csv:
        write_csv(args.write_csv, state[2], state[5])
        return

    fig, axes = build_figure()
    draw(fig, axes, state)

    if args.save:
        fig.savefig("plant_floor.png", dpi=140)
        print("wrote plant_floor.png")
        return

    from matplotlib.widgets import Button, Slider

    seed_box = {"value": args.seed}
    ax_noise = fig.add_axes([0.08, 0.07, 0.34, 0.03])
    ax_spread = fig.add_axes([0.08, 0.025, 0.34, 0.03])
    ax_again = fig.add_axes([0.56, 0.035, 0.12, 0.05])
    ax_save = fig.add_axes([0.71, 0.035, 0.14, 0.05])

    s_noise = Slider(ax_noise, "noise", 0.0, 2.0, valinit=args.noise, valstep=0.05)
    s_spread = Slider(ax_spread, "monitor spread", 0.5, 10.0,
                      valinit=args.spread, valstep=0.5)
    b_again = Button(ax_again, "New draw")
    b_save = Button(ax_save, "Write CSV")

    def refresh(_=None):
        new_state = simulate(seed=seed_box["value"], noise=s_noise.val,
                             spread=s_spread.val)
        state_box["value"] = new_state
        draw(fig, axes, new_state)
        fig.canvas.draw_idle()

    state_box = {"value": state}

    def new_draw(_):
        seed_box["value"] += 1
        refresh()

    def save_csv(_):
        current = state_box["value"]
        write_csv("plant_floor.csv", current[2], current[5])

    s_noise.on_changed(refresh)
    s_spread.on_changed(refresh)
    b_again.on_clicked(new_draw)
    b_save.on_clicked(save_csv)

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:                      # a demo that dies silently is worse
        print(f"plant_floor.py stopped: {exc}", file=sys.stderr)
        raise
