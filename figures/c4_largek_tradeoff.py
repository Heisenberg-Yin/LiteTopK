import matplotlib.pyplot as plt
import numpy as np
import math

baseline1_time = [
    [
        10159.5,
        13457.9,
        15709.5,
        17812.5,
        19580.9,
        21201.5,
        22459.5,
        24550.7,
        25539.1,
        26978.1,
        28450.6,
        29899.6,
    ],
    [
        19231.5,
        22190.3,
        24855.7,
        27143.5,
        29212.8,
        31104.9,
        32910.1,
        34626.5,
        36276.1,
        37876,
        39433.9,
        40956.6,
        42455.7,
    ],
    [
        29892.8,
        38044.5,
        43906.6,
        48622.7,
        52706.1,
        56411.3,
        59841.1,
        63117.2,
        66187.0,
    ],
    [
        58095.9,
        67167.5,
        74018.1,
        79655.4,
        84554.7,
        88909.3,
        92873.1,
        96570.1,
        100053.0,
        103373.0,
    ],
    [
        97124.4,
        115760,
        128308,
        138600,
        147194,
        154445,
        160852,
        166699,
        171761,
        176229,
        181882,
        188998,
    ],
]


baseline1_recall = [
    [
        71.5341,
        82.9385,
        88.2498,
        91.3233,
        93.2998,
        94.6933,
        95.699,
        96.458,
        97.034,
        97.4913,
        97.8546,
        98.1576,
    ],
    [
        78.4298,
        84.7511,
        88.5364,
        91.041,
        92.8247,
        94.1333,
        95.1345,
        95.9051,
        96.5275,
        97.0217,
        97.4362,
        97.77,
        98.0515,
    ],
    [
        72.7579,
        84.8096,
        90.2415,
        93.2721,
        95.1494,
        96.3976,
        97.2574,
        97.8704,
        98.3271,
    ],
    [
        79.8257,
        86.6294,
        90.5836,
        93.1064,
        94.8275,
        96.0328,
        96.908,
        97.5667,
        98.063,
        98.4404,
    ],
    [
        70.8456,
        79.6895,
        85.1685,
        88.8489,
        91.4568,
        93.3430,
        94.7484,
        95.8246,
        96.6534,
        97.2966,
        97.8171,
        98.5423,
    ],
]


baseline2_time = [
    [
        32351.1,
        35697.4,
        37870.4,
        39727.8,
        41367.9,
        42928.9,
        44489.1,
        46006.6,
        47458.5,
        48932.0,
        50365.6,
        51806.9,
        53236.9,
        54666.9,
        56082.0,
        57506.6,
        58854.3,
        60283.4,
        61717.8,
    ],
    [57219, 63500, 67778, 71585.5, 75627.4, 78771, 82290, 85767],
    [
        93998.6,
        105505.0,
        111333.0,
        116052.0,
        120065.0,
        122728.0,
        126370.0,
        129624.0,
        133787.0,
        136902.0,
    ],
    [
        169999.0,
        178507.0,
        184828.0,
        190004.0,
        194532.0,
        198773.0,
        202883.0,
        206502.0,
        210206.0,
        213753.0,
        217363.0,
    ],
    [328474, 364207, 381057, 393990, 404390, 410850, 418809, 425855],
]

baseline2_recall = [
    [
        71.5614,
        81.3817,
        86.4591,
        89.5679,
        91.6517,
        93.1611,
        94.2364,
        95.0591,
        95.7129,
        96.2267,
        96.6392,
        96.988,
        97.269,
        97.5033,
        97.7002,
        97.8633,
        98.0083,
        98.1278,
        98.2318,
    ],
    [78.4248, 88.4198, 92.5735, 94.7456, 96.0271, 96.8342, 97.3705, 97.734],
    [
        72.7482,
        84.7461,
        90.0951,
        93.014,
        94.7855,
        95.9276,
        96.7006,
        97.2355,
        97.6159,
        97.9018,
    ],
    [
        79.7829,
        86.5554,
        90.4316,
        92.8542,
        94.4599,
        95.5696,
        96.3567,
        96.9291,
        97.3577,
        97.6821,
        97.9268,
    ],
    [70.7801, 85.1375, 91.2913, 94.4244, 96.1679, 97.2, 97.8237, 98.2107],
]


improved1_time = [
    [6484.82, 8807.34, 15264.0, 21090.9, 26796.0],
    [12565.9, 20562.4, 26810.1, 33167.7, 38231.1],
    [
        18367.9,
        28659.2,
        35961.8,
        42491.1,
        48764.8,
        51426.8,
    ],
    [36242.8, 47509.6, 55837.5, 62825.8, 69394.1, 76318.4],
    [61374.3, 89186.1, 114649.0, 122364.0, 132591],
]


improved1_recall = [
    [71.4472, 82.7533, 94.325, 97.0553, 98.117],
    [78.291, 92.5012, 96.1264, 97.6097, 98.3339],
    [
        72.6588,
        89.9733,
        94.7909,
        96.8527,
        97.893,
        98.242,
    ],
    [79.6613, 90.3096, 94.4694, 96.5024, 97.6293, 98.3031],
    [70.7692, 88.6316, 96.2807, 97.4056, 98.3698],
]

improved2_time = [
    [19180.4, 22003.6, 25740.9, 35146.6, 44369.8],
    [31622.2, 36461.4, 46125.6, 61815.1],
    [57562.1, 68597.5, 83619.4, 89800.9],
    [90865.8, 106595.0, 120638.0, 135601.0],
    [148677.0, 203009.0, 221695.0, 234056.0],
]

improved2_recall = [
    [71.48, 82.7988, 91.1906, 97.0635, 98.3023],
    [78.101, 88.3574, 96.0294, 98.1801],
    [84.6115, 94.7739, 97.9039, 98.2736],
    [79.5106, 92.8008, 97.3633, 98.375],
    [70.4848, 88.627, 96.1947, 98.0519],
]

opt_time = [
    [
        11818.1,
        14912.3,
        17225.9,
        19210.3,
        20978.8,
        22744.9,
        24147.4,
        26005.2,
        27322.0,
        28654.2,
        30846.2,
        31648.6,
        33139.9,
    ],
    [23300.9, 29092.8, 33347.7, 37165.7, 40525.5, 43869.2, 46923.5, 49928.7],
    [
        37507.5,
        47069.2,
        53230.2,
        59070.5,
        63184.2,
        66765.9,
        70308.5,
        73742.3,
        75381.7,
        78559.1,
    ],
    [
        78228.8,
        88451.6,
        95400.8,
        101300.0,
        106345.0,
        110967.0,
        115064.0,
        120368.0,
        122905.0,
        126363.0,
    ],
    [154676, 192023, 216573, 231490, 244485, 254299, 263163],
]

opt_recall = [
    [
        71.4755,
        82.8042,
        88.0606,
        91.0939,
        93.042,
        94.4119,
        95.3988,
        96.1393,
        96.6984,
        97.1393,
        97.4859,
        97.7716,
        97.9951,
    ],
    [78.3487, 88.3682, 92.6062, 94.8829, 96.2491, 97.1334, 97.7213, 98.1223],
    [
        72.7138,
        84.6927,
        90.0731,
        93.0673,
        94.9184,
        96.1447,
        96.9826,
        97.5698,
        97.9761,
        98.2929,
    ],
    [
        79.7548,
        86.5053,
        90.4173,
        92.9084,
        94.6058,
        95.7928,
        96.6517,
        97.2929,
        97.7678,
        98.1203,
    ],
    [70.8255, 85.0752, 91.3001, 94.5489, 96.4254, 97.5548, 98.2511],
]

# BFC baseline: same units as other *_time lists; recall in %.
bfc_time = [[2.33838e+06], [2.33838e+06], [2.33838e+06], [2.33838e+06], [2.33838e+06]]
bfc_recall = [[100], [100], [100], [100], [100]]


fig, axs = plt.subplots(1, 5, figsize=(5 * 7, 7))
linestyles = ["-", "-.", ":", (0, (3, 5, 1, 5)), "--", (0, (5, 2))]
pointstyles = ["o", "^", "D", "*", "p", "s"]
colors = ["r", "b", "g", "m", "orange", "#17becf"]  # last = BFC
methods = ["IVF+RaBitQ", "IVF+PQ", "IVF+RaBitQ+BBC", "IVF+PQ+BBC", "IVF+RaBitQ+Min"]

xlabels = [
    "Recall@5000",
    "Recall@10000",
    "Recall@20000",
    "Recall@40000",
    "Recall@100000",
]

for i, xlabel in enumerate(xlabels):
    qps1 = [(1e6 / value) for value in baseline1_time[i]]
    qps2 = [(1e6 / value) for value in baseline2_time[i]]
    qps3 = [(1e6 / value) for value in improved1_time[i]]
    qps4 = [(1e6 / value) for value in improved2_time[i]]
    qps5 = [(1e6 / value) for value in opt_time[i]]

    axs[i].plot(
        [value / 100 for value in baseline1_recall[i]],
        qps1,
        marker=pointstyles[0],
        linestyle=linestyles[0],
        label="RabitQ",
        linewidth=2,
        markersize=10,
        color=colors[0],
        markerfacecolor="none",
    )

    axs[i].plot(
        [value / 100 for value in baseline2_recall[i]],
        qps2,
        marker=pointstyles[1],
        linestyle=linestyles[1],
        label="IVF+OPQ",
        linewidth=2,
        markersize=10,
        color=colors[1],
        markerfacecolor="none",
    )
    axs[i].plot(
        [value / 100 for value in improved1_recall[i]],
        qps3,
        marker=pointstyles[2],
        linestyle=linestyles[2],
        label="RabitQ-improve",
        linewidth=2,
        markersize=10,
        color=colors[2],
        markerfacecolor="none",
    )  # 新增：绘制 baseline 3 的曲线
    axs[i].plot(
        [value / 100 for value in improved2_recall[i]],
        qps4,
        marker=pointstyles[3],
        linestyle=linestyles[3],
        label="IVF+OPQ-improve",
        linewidth=2,
        markersize=10,
        color=colors[3],
        markerfacecolor="none",
    )
    axs[i].plot(
        [value / 100 for value in opt_recall[i]],
        qps5,
        marker=pointstyles[4],
        linestyle=linestyles[4],
        label="RaBitQ-OP",
        linewidth=2,
        markersize=10,
        color=colors[4],
        markerfacecolor="none",
    )

    qps_bfc = [(1e6 / value) for value in bfc_time[i]]
    axs[i].plot(
        [value / 100 for value in bfc_recall[i]],
        qps_bfc,
        marker=pointstyles[5],
        linestyle=linestyles[5],
        label="BFC",
        linewidth=2,
        markersize=10,
        color=colors[5],
        markerfacecolor="none",
    )
    axs[i].set_xlabel(f"{xlabel}", fontsize=40)
    if i == 0:
        axs[i].set_yticks([0, 30, 60, 90, 120])
        axs[i].set_ylim([0 - 1, 120 + 1])
        axs[i].set_ylabel("QPS", fontsize=40)
    elif i == 1:
        axs[i].set_yticks([0, 20, 40, 60, 80])
        axs[i].set_ylim([0 - 2, 80 + 2])
    elif i == 2:
        axs[i].set_yticks([0, 10, 20, 30, 40])
        axs[i].set_ylim([0 - 1, 40 + 1])
    elif i == 3:
        axs[i].set_yticks([0, 8, 16, 24, 32])
        axs[i].set_ylim([0 - 0.8, 32 + 0.8])
    elif i == 4:
        axs[i].set_yticks([0, 4, 8, 12, 16])
        axs[i].set_ylim([0 - 0.4, 16 + 0.4])

    axs[i].set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    axs[i].set_xlim([0.8 - 0.005, 1 + 0.005])

    axs[i].tick_params(axis="both", which="major", labelsize=30)
    axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)
    # axs[i].set_title(f'{xlabel}', fontsize=50, pad=20)


# 创建图例并右移
# handles = [
#     plt.Line2D(
#         [0],
#         [0],
#         color=colors[i],
#         lw=2,
#         marker=pointstyles[i],
#         linestyle=linestyles[i],
#         markersize=10,
#         markerfacecolor="none",
#     )
#     for i in range(len(methods))
# ]
# fig.legend(
#     handles,
#     methods,
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.2),
#     ncol=5,
#     fontsize=45,
#     frameon=False,
#     columnspacing=1,
#     handletextpad=1,
# )

plt.tight_layout()
plt.savefig("exp_qps_recall_c4_large.pdf", bbox_inches="tight", format="pdf")
exit(1)
