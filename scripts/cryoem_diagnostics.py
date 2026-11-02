"""Diagnostic figures summarizing the dim=10 investigation recorded in
todo.txt: why the old rank=4/degree=13 default failed the along-path metric
(under-ranking, not overfitting) and why the frequency-aware Tikhonov
regularizer, once its per-axis calibration bug was fixed, still doesn't
close the gap to naive on this dataset. These plot HARDCODED numbers from
that investigation's sweeps (not re-run here -- each sweep took 10s of
minutes on a GPU node; see todo.txt for the exact commands/scripts used),
so this script only needs matplotlib, no jax/GPU.

Usage: python scripts/cryoem_diagnostics.py
"""

import matplotlib.pyplot as plt
import numpy as np
from cryoem_common import FIGS_DIR

# Fixed categorical palette (see the dataviz skill's references/palette.md):
# slot order held constant across every figure in this project so the same
# method always has the same color.
COLOR_KDE = "#2a78d6"      # slot 1 blue
COLOR_NAIVE = "#eb6834"    # slot 2 orange
COLOR_DECONV = "#1baf7a"   # slot 3 aqua
COLOR_BAD = "#e34948"      # slot 8 red (used only to flag NaN-unstable runs)
GRID_KW = {"color": "#c8c8c3", "linewidth": 0.6, "alpha": 0.6}

L2_KDE = 0.0757


def plot_rank_degree_sweep():
    # rank/degree sweep, naive TT only, dim=10, noise1 (see todo.txt STATUS
    # "dim=10's REAL problem: found -- it was under-ranking, not overfitting").
    data = {
        2: {5: 0.5263, 9: 1.1266, 13: 0.7254},
        3: {5: 0.8168, 9: 1.0410, 13: 1.2019},
        4: {5: 0.6319, 7: 0.7986, 9: 0.7973, 13: 1.0102},
        6: {5: 0.3164, 9: 0.3666},
        8: {3: 0.1445, 5: 0.3015, 7: 0.3356, 9: None},
        10: {3: 0.1315, 5: 0.2694, 7: 0.2332},
        12: {3: 0.1391, 5: 0.2682, 7: 0.1839},
        16: {2: 1.0309, 3: 0.1109, 5: 0.2342},
        20: {2: 1.0247, 3: 0.1293, 4: 0.1611},
        24: {2: 1.0217, 3: 0.2089},
    }
    degrees = sorted({d for per_rank in data.values() for d in per_rank})
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for i, degree in enumerate(degrees):
        ranks, l2s = [], []
        for rank, per_degree in sorted(data.items()):
            if degree in per_degree and per_degree[degree] is not None:
                ranks.append(rank)
                l2s.append(per_degree[degree])
        if len(ranks) < 2:
            continue
        color = cmap(0.35 + 0.55 * i / max(len(degrees) - 1, 1))
        ax.plot(ranks, l2s, marker="o", markersize=5, linewidth=2, color=color, label=f"degree={degree}")
    ax.axhline(L2_KDE, color=COLOR_KDE, linewidth=2, linestyle="--", label=f"KDE baseline ({L2_KDE:.3f})")
    ax.axhline(1.0102, color="#9a9a94", linewidth=1.5, linestyle=":", label="old default: rank=4, degree=13")
    ax.scatter([16], [0.1109], s=110, facecolors="none", edgecolors=COLOR_NAIVE, linewidths=2, zorder=5)
    ax.annotate(
        "rank=16, degree=3\n(best found)", xy=(16, 0.1109), xytext=(17.5, 0.32),
        fontsize=9, color=COLOR_NAIVE, arrowprops={"arrowstyle": "->", "color": COLOR_NAIVE, "lw": 1.2},
    )
    ax.set(
        xlabel="TT rank", ylabel="relative L2 error (along-path metric)",
        title="dim=10 naive squared-TT: rank vs. degree\n(low degree + high rank is the fix, not overfitting)",
        yscale="log",
    )
    ax.grid(True, which="both", **GRID_KW)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    out = FIGS_DIR / "diagnostic_rank_degree_sweep_dim10.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_reg_lambda_sweep():
    # Tikhonov reg_lambda sweep AFTER fixing the per-axis calibration bug
    # (see todo.txt). Two panels: the old, badly under-ranked rank=4/
    # degree=13 operating point, and the corrected rank=8/degree=5 point.
    old_point = {  # rank=4, degree=13
        0.0: 1.0102, 0.03: 1.0091, 0.1: 1.0073, 0.3: 1.0044, 1.0: 1.0010, 3.0: 1.0005, 10.0: 1.0071,
    }
    good_point = {  # rank=8, degree=5
        0.0: 0.3377, 0.3: 0.3624, 1.0: 0.3552,
    }
    naive_old, naive_good = 0.8077, 0.3015

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True, sharey=False)

    ax = axes[0]
    xs = sorted(old_point)
    ax.plot(range(len(xs)), [old_point[x] for x in xs], marker="o", color=COLOR_DECONV, linewidth=2, label="deconv")
    ax.axhline(naive_old, color=COLOR_NAIVE, linewidth=2, linestyle="--", label=f"naive ({naive_old:.3f})")
    ax.axhline(L2_KDE, color=COLOR_KDE, linewidth=2, linestyle=":", label=f"KDE ({L2_KDE:.3f})")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set(xlabel="reg_lambda", ylabel="relative L2 error", title="old default (rank=4, degree=13)\nregularization barely moves the needle")
    ax.grid(True, axis="y", **GRID_KW)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    xs = sorted(good_point)
    ax.plot(range(len(xs)), [good_point[x] for x in xs], marker="o", color=COLOR_DECONV, linewidth=2, label="deconv")
    ax.axhline(naive_good, color=COLOR_NAIVE, linewidth=2, linestyle="--", label=f"naive ({naive_good:.3f})")
    ax.axhline(L2_KDE, color=COLOR_KDE, linewidth=2, linestyle=":", label=f"KDE ({L2_KDE:.3f})")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs])
    ax.set(xlabel="reg_lambda", ylabel="relative L2 error", title="corrected point (rank=8, degree=5)\ndeconv still doesn't beat naive")
    ax.grid(True, axis="y", **GRID_KW)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("dim=10: Tikhonov reg_lambda sweep, before vs. after fixing the per-axis calibration bug", fontsize=11)
    out = FIGS_DIR / "diagnostic_reg_lambda_sweep_dim10.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_method_summary():
    # Headline comparison across the whole investigation: every operating
    # point tried at dim=10, KDE vs naive vs best deconv found there.
    rows = [
        ("old default\n(rank=4, deg=13)", 0.8077, 1.0005),
        ("rank=8, deg=5\n(stable, exact conv)", 0.3015, 0.3377),
        ("rank=16, deg=3\n(best naive; deconv unstable)", 0.1109, 0.5177),
    ]
    labels = [r[0] for r in rows]
    naive_vals = [r[1] for r in rows]
    deconv_vals = [r[2] for r in rows]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.bar(x - width / 2, naive_vals, width, color=COLOR_NAIVE, label="naive (no deconv)")
    ax.bar(x + width / 2, deconv_vals, width, color=COLOR_DECONV, label="deconv (best reg_lambda found)")
    ax.axhline(L2_KDE, color=COLOR_KDE, linewidth=2, linestyle="--", label=f"KDE baseline ({L2_KDE:.3f})")
    ax.annotate(
        "NaN-unstable during training\n(pre_conv_max_rank rounding)", xy=(2 + width / 2, deconv_vals[2]),
        xytext=(1.15, 0.75), fontsize=8, color=COLOR_BAD,
        arrowprops={"arrowstyle": "->", "color": COLOR_BAD, "lw": 1.1},
    )
    ax.set(
        xlabel=None, ylabel="relative L2 error (along-path metric, lower is better)",
        title="dim=10 cryo-EM along-path accuracy: every operating point tried\n(naive consistently beats deconvolution on this dataset/metric)",
    )
    ax.set_xticks(x, labels, fontsize=9)
    ax.grid(True, axis="y", **GRID_KW)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    out = FIGS_DIR / "diagnostic_method_summary_dim10.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    plot_rank_degree_sweep()
    plot_reg_lambda_sweep()
    plot_method_summary()
