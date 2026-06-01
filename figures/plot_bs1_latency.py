"""Plot latency vs k for several batch sizes from latency_1m.csv."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "latency_1m.csv"
OUT_PDF = ROOT / "figures" / "latency_by_bs.pdf"
OUT_SVG = ROOT / "figures" / "latency_by_bs.svg"


BS_VALUES = [1, 8, 64, 128]
K_MIN = 100
K_MAX = 20000
SERIES = [
    ("torch_ms", "Torch", "#D55E00", "o", "-"),
    ("raft_ms", "RAFT", "#0072B2", "^", "-."),
    ("faiss_ms", "FAISS", "#009E73", "D", ":"),
    ("flashlib_ms", "FlashLib", "#CC79A7", "*", (0, (3, 5, 1, 5))),
    ("flat_ms", "efficienttopk", "#000000", "s", "--"),
]


def _read_rows() -> dict[int, list[dict[str, str]]]:
    with CSV_PATH.open(newline="") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if int(row["bs"]) in BS_VALUES and K_MIN <= int(row["k"]) <= K_MAX
        ]
    rows_by_bs = {bs: [] for bs in BS_VALUES}
    for row in rows:
        rows_by_bs[int(row["bs"])].append(row)
    for bs in BS_VALUES:
        rows_by_bs[bs].sort(key=lambda row: int(row["k"]))
    return rows_by_bs


def _parse(value: str) -> float | None:
    return None if value in ("", "NA") else float(value)


def main() -> None:
    rows_by_bs = _read_rows()

    plt.rcParams.update(
        {
            "axes.linewidth": 1.1,
            "font.size": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axs = plt.subplots(
        1,
        len(BS_VALUES),
        figsize=(22, 5.6),
        sharey=True,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.16, top=0.78, wspace=0.06)
    for ax, bs in zip(axs, BS_VALUES):
        rows = rows_by_bs[bs]
        for key, label, color, marker, linestyle in SERIES:
            xs: list[int] = []
            ys: list[float] = []
            for row in rows:
                value = _parse(row[key])
                if value is None:
                    continue
                xs.append(int(row["k"]))
                ys.append(value)
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                label=label,
                linewidth=2.4 if key == "flat_ms" else 2,
                markersize=8,
                color=color,
                markerfacecolor="none",
                markeredgewidth=1.6,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(90, 23000)
        ax.set_ylim(1, 200)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks([1, 10, 100])
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"bs = {bs}", fontsize=22, pad=10)
        ax.tick_params(axis="both", which="major", labelsize=18, length=6, width=1.0)
        ax.tick_params(axis="both", which="minor", length=3, width=0.8)
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, color="#bdbdbd", alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="#d9d9d9", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axs[0].set_ylabel("Latency (ms)", fontsize=24)
    fig.supxlabel("k", fontsize=24, y=0.035)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            lw=2.4 if key == "flat_ms" else 2,
            marker=marker,
            linestyle=linestyle,
            markersize=8,
            markerfacecolor="none",
            markeredgewidth=1.6,
        )
        for key, _, color, marker, linestyle in SERIES
    ]
    fig.legend(
        handles,
        [label for _, label, _, _, _ in SERIES],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=5,
        fontsize=19,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.2,
        handletextpad=0.5,
    )

    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.04)
    print(OUT_PDF)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
