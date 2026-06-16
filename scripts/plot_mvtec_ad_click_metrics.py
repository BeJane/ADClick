import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLICKS = ["2-click", "3-click", "5-click"]

METRICS = {
    "AP": {
        "FocSAM (CVPR'24)": [43.1, 61.5, 77.6],
        "GPCIS (CVPR'23)": [58.8, 71.1, 84.6],
        "SimpleClick (ICCV'23)": [69.4, 78.6, 87.0],
        "Ours (full-shot)": [90.1, 93.3, 96.1],
        "Ours (10-shot)": [84.9, 88.8, 94.1],
        "Ours (5-shot)": [83.3, 87.8, 92.9],
        "Ours (1-shot)": [76.7, 80.8, 84.7],
    },
    "PRO": {
        "FocSAM (CVPR'24)": [65.6, 72.3, 81.0],
        "GPCIS (CVPR'23)": [79.9, 85.2, 90.9],
        "SimpleClick (ICCV'23)": [90.1, 92.7, 95.5],
        "Ours (full-shot)": [95.9, 97.2, 98.3],
        "Ours (10-shot)": [93.6, 94.9, 97.2],
        "Ours (5-shot)": [93.8, 94.8, 96.6],
        "Ours (1-shot)": [91.7, 93.9, 95.3],
    },
    "Pixel AUROC": {
        "FocSAM (CVPR'24)": [85.7, 88.2, 92.2],
        "GPCIS (CVPR'23)": [87.4, 93.0, 97.6],
        "SimpleClick (ICCV'23)": [95.4, 97.2, 98.3],
        "Ours (full-shot)": [99.0, 99.4, 99.7],
        "Ours (10-shot)": [98.0, 98.9, 99.5],
        "Ours (5-shot)": [98.1, 98.9, 99.5],
        "Ours (1-shot)": [97.8, 98.6, 99.1],
    },
    "mIoU": {
        "FocSAM (CVPR'24)": [58.6, 68.3, 78.9],
        "GPCIS (CVPR'23)": [49.0, 57.5, 68.7],
        "SimpleClick (ICCV'23)": [54.5, 61.5, 72.7],
        "Ours (full-shot)": [69.2, 75.0, 81.1],
        "Ours (10-shot)": [65.4, 71.4, 78.4],
        "Ours (5-shot)": [63.9, 70.0, 77.1],
        "Ours (1-shot)": [59.0, 64.6, 70.0],
    },
}

STYLE = {
    "FocSAM (CVPR'24)": {"color": "#0072B2", "marker": "o"},
    "GPCIS (CVPR'23)": {"color": "#E69F00", "marker": "s"},
    "SimpleClick (ICCV'23)": {"color": "#009E73", "marker": "^"},
    "Ours (full-shot)": {"color": "#D55E00", "marker": "D", "linestyle": "-"},
    "Ours (10-shot)": {"color": "#CC79A7", "marker": "D", "linestyle": "--"},
    "Ours (5-shot)": {"color": "#56B4E9", "marker": "D", "linestyle": "--"},
    "Ours (1-shot)": {"color": "#F0E442", "marker": "D", "linestyle": "--"},
}

YLIMS = {
    "AP": (40, 99),
    "PRO": (64, 100),
    "Pixel AUROC": (85, 100.4),
    "mIoU": (47, 83),
}

YTICKS = {
    "AP": [60, 80],
    "PRO": [70, 80, 90],
    "Pixel AUROC": [90, 100],
    "mIoU": [60, 80],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the MVTec AD click-metric comparison figure."
    )
    parser.add_argument(
        "--output",
        default="mvtec_ad_click_metrics",
        help="Output path without suffix, or with .png/.pdf/.svg suffix.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output DPI.",
    )
    return parser.parse_args()


def save_figure(fig, output, dpi):
    output_path = Path(output)
    if output_path.suffix:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        return

    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def main():
    args = parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    metric_order = ["AP", "PRO", "Pixel AUROC", "mIoU"]

    for ax, metric in zip(axes.flat, metric_order):
        for method, values in METRICS[metric].items():
            ax.plot(
                CLICKS,
                values,
                label=method,
                linewidth=4.0,
                markersize=7.0,
                markeredgewidth=1.0,
                **STYLE[method],
            )

        ax.set_title(metric, pad=8)
        ax.set_ylim(*YLIMS[metric])
        ax.set_yticks(YTICKS[metric])
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        ax.tick_params(axis="both", direction="out", length=3.5, width=0.8, pad=2)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#333333")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.045),
        handlelength=1.5,
        columnspacing=1.2,
        handletextpad=0.45,
        labelspacing=0.35,
    )

    fig.subplots_adjust(left=0.075, right=0.98, top=0.91, bottom=0.15, wspace=0.18, hspace=0.43)
    save_figure(fig, args.output, args.dpi)


if __name__ == "__main__":
    main()
