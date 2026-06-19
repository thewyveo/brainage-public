import matplotlib.pyplot as plt

# -----------------------------
# Global style
# -----------------------------
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.edgecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "font.size": 10,
})

# Colors
GREEN = "#22C55E"
BLUE = "#3B82F6"
RED = "#EF4444"
WHITE = "white"
GRID = "#444444"


def horizontal_bar_plot(
    labels,
    values,
    title,
    xlabel,
    output_name,
    positive_color,
    negative_color,
):
    colors = [
        positive_color if v > 0 else negative_color
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.axvline(0, color=WHITE, linewidth=1)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID)

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)

    # Value labels
    x_min, x_max = ax.get_xlim()
    span = x_max - x_min
    offset = span * 0.015

    for bar, value in zip(bars, values):
        y = bar.get_y() + bar.get_height() / 2

        if value >= 0:
            ax.text(
                value + offset,
                y,
                f"{value:+.2f}",
                va="center",
                ha="left",
                color=WHITE,
                fontsize=9,
            )
        else:
            ax.text(
                value - offset,
                y,
                f"{value:+.2f}",
                va="center",
                ha="right",
                color=WHITE,
                fontsize=9,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_name}.pdf", bbox_inches="tight")
    plt.show()


# =========================================================
# Experiment 0
# Positive = GREEN
# Negative = RED
# =========================================================

exp0_labels = [
    "BNX+CM",
    "BNX+GLI",
    "BNX+USB",
    "JOOS+CM",
    "JOOS+GLI",
    "JOOS+USB",
]

exp0_values = [
    0.18,
    1.67,
    -4.90,
    2.21,
    0.77,
    21.98,
]

horizontal_bar_plot(
    labels=exp0_labels,
    values=exp0_values,
    title="Experiment 0 signed prediction shift",
    xlabel=r"$\Delta$BAG relative to healthy baseline (years)",
    output_name="exp0_signed_delta_bag",
    positive_color=RED,
    negative_color=BLUE,
)


# =========================================================
# Experiment 1
# Positive = GREEN
# Negative = BLUE
# =========================================================

exp1_labels = [
    "BNX+CM+BID",
    "BNX+CM+LIT",
    "BNX+CM+USB",
    "BNX+GLI+BID",
    "BNX+GLI+LIT",
    "BNX+GLI+USB",
    "JOOS+CM+BID",
    "JOOS+CM+LIT",
    "JOOS+CM+USB",
    "JOOS+GLI+BID",
    "JOOS+GLI+LIT",
    "JOOS+GLI+USB",
]

exp1_values = [
    -4.17,
    -0.98,
    -10.63,
    -2.66,
    0.31,
    -10.36,
    -4.45,
    -1.36,
    -19.62,
    -4.09,
    -0.41,
    -21.07,
]

horizontal_bar_plot(
    labels=exp1_labels,
    values=exp1_values,
    title="Experiment 1 inpainting recovery",
    xlabel="Recovery toward healthy baseline (years)",
    output_name="exp1_inpainting_recovery",
    positive_color=GREEN,
    negative_color=BLUE,
)