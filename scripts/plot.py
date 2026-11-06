#!/usr/bin/env python3
"""
plot.py -- make plots/roofline.png and plots/time_N<N>_d<d>.png from results.csv.

Roofline parameters are the Q1 numbers (machine_spec.txt): theoretical peak FP32
for one P-core and for all cores, and theoretical DRAM bandwidth.

Usage (from the repo root):  python3 scripts/plot.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- Q1 numbers (machine_spec.txt) ----
BW = 68.3        # GB/s: 128-bit LPDDR4X-4266 = 16 B x 4.266 GT/s
PEAK_1 = 102.5   # GFLOP/s: 1 P-core = 4 lanes x 2 FLOP x 4 FMA x 3.204 GHz
PEAK_ALL = 542.2 # GFLOP/s: 4 x 102.5 + 4 E-cores x (4 x 2 x 2 x 2.064)
OMP_THREADS = 8

# ---- style (reference palette, light mode) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2 = "#2a78d6", "#eb6834"  # categorical slots 1 (single core) and 2 (all cores)
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

ORDER = ["naive", "loops", "tiled", "fused", "best", "omp"]


def load():
    rows = list(csv.DictReader(open("results.csv")))
    res = {}
    for r in rows:
        t = int(r["threads"])
        if r["variant"] == "omp" and t != OMP_THREADS:
            continue
        res[(r["variant"], int(r["N"]), int(r["d"]))] = (
            float(r["time_ms"]), float(r["gflops"]), float(r["arithmetic_intensity_flop_per_byte"]))
    return res


def roofline(res, n=4096, d=64):
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ai = [2 ** (e / 4) for e in range(-12, 4 * 12 + 1)]  # 0.125 .. 4096
    for peak, color, style, label in [
        (PEAK_1, C1, "-", "1 P-core peak: %.1f GFLOP/s" % PEAK_1),
        (PEAK_ALL, C2, "-", "8 cores peak: %.1f GFLOP/s" % PEAK_ALL),
    ]:
        ax.plot(ai, [min(peak, BW * x) for x in ai], style, color=color, lw=2, label=label)
    for peak, color in [(PEAK_1, C1), (PEAK_ALL, C2)]:
        ridge = peak / BW
        ax.plot([ridge], [peak], "o", ms=5, color=color)
        ax.annotate("ridge %.2f FLOP/B" % ridge, (ridge, peak), textcoords="offset points",
                    xytext=(6, -14), color=INK2, fontsize=8.5)
    ax.text(0.14, BW * 0.14 * 1.35, "DRAM bandwidth %.1f GB/s" % BW, rotation=33, color=INK2,
            fontsize=8.5)

    offsets = {"naive": (-8, -4), "loops": (8, -4), "tiled": (8, -4), "fused": (8, -4),
               "best": (8, -4), "omp": (8, -4)}
    for v in ORDER:
        t, g, x = res[(v, n, d)]
        color = C2 if v == "omp" else C1
        ax.plot([x], [g], "o", ms=9, color=color, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=5)
        name = "omp, 8 threads" if v == "omp" else v
        dx, dy = offsets[v]
        ax.annotate("%s  %.1f" % (name, g), (x, g), textcoords="offset points", xytext=(dx, dy),
                    ha="left" if dx > 0 else "right", fontsize=9, color=INK, zorder=6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlim(0.125, 4096)
    ax.set_ylim(1, 1000)
    ax.set_xlabel("Arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Roofline, Apple M1, attention variants at N=%d, d=%d" % (n, d), loc="left",
                 fontsize=11)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig("plots/roofline.png", dpi=200)
    plt.close(fig)


def time_plot(res, n, d):
    variants = [v for v in ORDER if (v, n, d) in res]
    times = [res[(v, n, d)][0] for v in variants]
    base = res[("naive", n, d)][0]
    names = ["omp, %d threads" % OMP_THREADS if v == "omp" else v for v in variants]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = list(range(len(variants)))[::-1]  # naive on top
    ax.barh(y, times, height=0.62, color=C1, edgecolor=SURFACE, linewidth=2)
    for yi, t in zip(y, times):
        ax.text(t * 1.08, yi, "%.1f ms,  %.1f×" % (t, base / t), va="center", fontsize=9,
                color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlim(min(times) / 2, max(times) * 8)
    ax.set_xlabel("Time in ms, log scale. Label: time and speedup over naive")
    ax.set_title("Time per variant, N=%d, d=%d" % (n, d), loc="left", fontsize=11)
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig("plots/time_N%d_d%d.png" % (n, d), dpi=200)
    plt.close(fig)


def main():
    os.makedirs("plots", exist_ok=True)
    res = load()
    roofline(res)
    shapes = sorted({(n, d) for (_, n, d) in res})
    for n, d in shapes:
        time_plot(res, n, d)
    print("wrote plots/roofline.png and", ", ".join("plots/time_N%d_d%d.png" % s for s in shapes))


if __name__ == "__main__":
    main()
