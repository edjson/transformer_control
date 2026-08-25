"""
plot_cartpole.py — visualize the cartpole eval CSVs (grid_results.csv / eval_results.csv).

Produces five figures and prints the numbers that go in the memo:
  1. success_vs_track.png      success rate reconstructed as a function of track limit
                               (the curve that shows the 95.6% collapsing toward the spec)
  2. peak_x_hist.png           transient cart excursion over SUCCESSFUL runs, vs the
                               +-0.3 m objective and a ~+-0.4 m rail
  3. success_heatmap.png       success rate over cart mass x pole length (mean over pole mass)
  4. outcome_by_cart.png       outcome breakdown (success / out_of_bounds / timeout) by cart mass
  5. actuation_hist.png        |u_scaled| distribution, showing the ceiling at the control scale

USAGE
    python plot_cartpole.py                       # defaults to grid_results.csv
    python plot_cartpole.py eval_results.csv
    python plot_cartpole.py grid_results.csv --outdir plots --objective 0.3 --rail 0.4
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # file output, no display needed
import matplotlib.pyplot as plt

# the only columns we rely on; script degrades gracefully if optional ones are missing
REQUIRED = ["cartmass", "polemass", "polelength", "outcome", "success", "peak_abs_x_m"]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return df



def plot_success_vs_track(df, track_max, objective, rail, outpath):
    """
    Reconstruct success rate vs track limit from the recorded transient peak excursion.
    A run counts as a success at candidate limit L iff it already succeeded AND its
    peak |x| stayed under L. This turns the single 20 m run into a full curve and shows
    where the spec-compliant number actually lands.
    """
    limits = np.linspace(0.1, track_max, 300)
    n = len(df)
    succ_x = df.loc[df["success"] == 1, "peak_abs_x_m"].to_numpy()
    rates = np.array([(succ_x <= L).sum() / n for L in limits])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(limits, 100 * rates, lw=2, color="#1f4e8c",
            label="success rate vs track limit")                    # <- label added
    for x, c, lbl, ytxt in [(objective, "#c0392b", f"objective {objective:g} m", 30),
                            (rail, "#e67e22", f"rail {rail:g} m", 50)]:
        r = (succ_x <= x).sum() / n
        ax.axvline(x, ls="--", lw=1.3, color=c,
                   label=f"{lbl}: {100*r:.1f}%")                     # <- label added
        ax.annotate(f"{lbl}: {100*r:.1f}%", xy=(x, 100 * r),
                    xytext=(x + 0.06 * track_max, ytxt),
                    color=c, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1))
    full = df["success"].mean()
    ax.axhline(100 * full, ls=":", lw=1, color="gray",
               label=f"as-run @ {track_max:g} m: {100*full:.1f}%")   # <- label added
    ax.text(track_max * 0.5, 100 * full + 1.5,
            f"as-run @ {track_max:g} m track: {100*full:.1f}%", color="gray", fontsize=9)
    ax.set_xlabel("track limit  |x|  (m)")
    ax.set_ylabel("stabilization success rate (%)")
    ax.set_title("Success rate vs track limit (reconstructed from transient peak |x|)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")                         # <- legend added
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_peak_x_hist(df, objective, rail, outpath):
    succ_x = df.loc[df["success"] == 1, "peak_abs_x_m"].to_numpy()
    if succ_x.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(succ_x, bins=40, color="#1f4e8c", alpha=0.85, edgecolor="white", linewidth=0.4)
    for x, c, lbl in [(objective, "#c0392b", f"objective {objective:g} m"),
                      (rail, "#e67e22", f"rail {rail:g} m")]:
        ax.axvline(x, ls="--", lw=1.3, color=c, label=lbl)
    med, p95 = np.median(succ_x), np.percentile(succ_x, 95)
    ax.axvline(med, ls="-", lw=1.2, color="black", label=f"median {med:.2f} m")
    ax.axvline(p95, ls="-.", lw=1.2, color="dimgray", label=f"95th pct {p95:.2f} m")
    ax.set_xlabel("peak transient cart excursion  |x|  (m), successful runs")
    ax.set_ylabel("count")
    ax.set_title("Swing-up cart excursion vs displacement spec")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return med, p95, succ_x.min(), succ_x.max()


def plot_success_heatmap(df, outpath):
    piv = df.pivot_table(values="success", index="cartmass",
                         columns="polelength", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in piv.index])
    ax.set_xlabel("pole length (m)")
    ax.set_ylabel("cart mass (kg)")
    ax.set_title("Success rate (mean over pole mass)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if 0.2 < v < 0.85 else "white")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_outcome_by_cart(df, outpath):
    outcomes = ["success", "out_of_bounds", "timeout", "quit"]
    present = [o for o in outcomes if (df["outcome"] == o).any()]
    ct = df.pivot_table(index="cartmass", columns="outcome",
                       values="idx" if "idx" in df.columns else "success",
                       aggfunc="count", fill_value=0)
    for o in present:
        if o not in ct.columns:
            ct[o] = 0
    ct = ct[present]
    colors = {"success": "#2e7d32", "out_of_bounds": "#c0392b",
              "timeout": "#e67e22", "quit": "gray"}
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(ct))
    for o in present:
        ax.bar([f"{c:.2f}" for c in ct.index], ct[o].values, bottom=bottom,
               label=o, color=colors.get(o, None))
        bottom += ct[o].values
    ax.set_xlabel("cart mass (kg)")
    ax.set_ylabel("count")
    ax.set_title("Outcome breakdown by cart mass")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_actuation_hist(df, outpath):
    if "peak_abs_uscaled" not in df.columns:
        return
    u = df["peak_abs_uscaled"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(u, bins=40, color="#6a3d9a", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, ls="--", lw=1.4, color="#c0392b",
               label="control-scale ceiling (|u_scaled| = 1)")
    ax.set_xlabel("peak |u_scaled|  (1.0 = CONTROL_SCALE, ~15 N)")
    ax.set_ylabel("count")
    ax.set_title("Actuation: peak normalized output pins at the control scale")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="grid_results.csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--objective", type=float, default=0.3, help="displacement objective (m)")
    ap.add_argument("--rail", type=float, default=0.4, help="physical rail half-length (m)")
    ap.add_argument("--track-max", type=float, default=None,
                    help="max track for the sweep curve (defaults to data max peak |x|)")
    args = ap.parse_args()

    df = load(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    track_max = args.track_max or float(np.ceil(df["peak_abs_x_m"].max()))

    # ---- printed summary (these are the memo numbers) ----
    n = len(df)
    k = int(df["success"].sum())
    lo, hi = wilson_ci(k, n)
    print(f"\nfile        : {args.csv}")
    print(f"systems     : {n}")
    print(f"success     : {k}/{n} = {100*k/n:.1f}%  (95% Wilson {100*lo:.1f}-{100*hi:.1f}%)")
    print("outcomes    : " + ", ".join(f"{o}={c}" for o, c in
                                        df["outcome"].value_counts().items()))

    succ_x = df.loc[df["success"] == 1, "peak_abs_x_m"]
    if len(succ_x):
        print("\ntransient peak |x| over successful runs:")
        print(f"  min {succ_x.min():.2f} | median {succ_x.median():.2f} | "
              f"95th {np.percentile(succ_x, 95):.2f} | max {succ_x.max():.2f} m")
        for L in (args.objective, args.rail, 1.0, 5.0):
            r = (succ_x <= L).sum() / n
            print(f"  success if track = {L:>4.1f} m : {100*r:5.1f}%")

    # ---- figures ----
    plot_success_vs_track(df, track_max, args.objective, args.rail,
                          os.path.join(args.outdir, "success_vs_track.png"))
    plot_peak_x_hist(df, args.objective, args.rail,
                     os.path.join(args.outdir, "peak_x_hist.png"))
    plot_success_heatmap(df, os.path.join(args.outdir, "success_heatmap.png"))
    plot_outcome_by_cart(df, os.path.join(args.outdir, "outcome_by_cart.png"))
    plot_actuation_hist(df, os.path.join(args.outdir, "actuation_hist.png"))
    print(f"\nfigures written to {os.path.abspath(args.outdir)}/\n")


if __name__ == "__main__":
    main()
