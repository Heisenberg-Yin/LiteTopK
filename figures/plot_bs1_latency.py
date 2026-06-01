"""Plot bs=1 latency vs k from latency_1m.csv."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "latency_1m.csv"
OUT_PDF = ROOT / "figures" / "bs1_latency.pdf"
OUT_SVG = ROOT / "figures" / "bs1_latency.svg"


LINESTYLES = ["-", "-.", ":", (0, (3, 5, 1, 5)), "--"]
POINTSTYLES = ["o", "^", "D", "*", "s"]
COLORS = ["r", "b", "g", "m", "#17becf"]
SERIES = [
    ("torch_ms", "Torch"),
    ("raft_ms", "RAFT"),
    ("faiss_ms", "FAISS"),
    ("flashlib_ms", "FlashLib"),
    ("flat_ms", "efficienttopk"),
]


def _read_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(row["bs"]) == 1]
    return sorted(rows, key=lambda row: int(row["k"]))


def _parse(value: str) -> float | None:
    return None if value in ("", "NA") else float(value)


def main() -> None:
    rows = _read_rows()

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for i, (key, label) in enumerate(SERIES):
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
            marker=POINTSTYLES[i],
            linestyle=LINESTYLES[i],
            label=label,
            linewidth=2,
            markersize=10,
            color=COLORS[i],
            markerfacecolor="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(1, 100)
    ax.set_yticks([1, 10, 100])
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("k", fontsize=40)
    ax.set_ylabel("Latency (ms)", fontsize=40)
    ax.tick_params(axis="both", which="major", labelsize=30)
    ax.tick_params(axis="both", which="minor", length=4, width=0.8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=COLORS[i],
            lw=2,
            marker=POINTSTYLES[i],
            linestyle=LINESTYLES[i],
            markersize=10,
            markerfacecolor="none",
        )
        for i in range(len(SERIES))
    ]
    ax.legend(
        handles,
        [label for _, label in SERIES],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        fontsize=18,
        frameon=False,
        columnspacing=1,
        handletextpad=0.5,
    )

    plt.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    print(OUT_PDF)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
