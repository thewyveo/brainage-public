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

GREEN = "#22C55E"
BLUE = "#3B82F6"
RED = "#EF4444"
WHITE = "white"
GRID = "#444444"


# =========================================================
# 1) Tumor -> inpaint trajectory plot
# =========================================================

conditions = [
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

tumor_abs = [
    1.53, 1.53, 1.53,
    2.26, 2.26, 2.26,
    2.94, 2.94, 2.94,
    1.46, 1.46, 1.46,
]

inpaint_abs = [
    5.69, 2.50, 12.16,
    4.92, 1.95, 12.62,
    7.39, 4.31, 22.56,
    5.55, 1.87, 22.52,
]

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = range(len(conditions))

for y, t, i in zip(y_pos, tumor_abs, inpaint_abs):
    color = GREEN if i < t else BLUE
    ax.plot([t, i], [y, y], color=color, linewidth=2, alpha=0.85)
    ax.scatter(t, y, color=WHITE, s=35, zorder=3, label="Tumor" if y == 0 else "")
    ax.scatter(i, y, color=color, s=45, zorder=3, label="Inpaint" if y == 0 else "")

    ax.text(t, y - 0.18, f"{t:.2f}", color=WHITE, fontsize=8, ha="center")
    ax.text(i, y + 0.18, f"{i:.2f}", color=WHITE, fontsize=8, ha="center")

ax.axvline(0, color=WHITE, linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(conditions)
ax.invert_yaxis()
ax.set_xlabel(r"$|\Delta$BAG$|$ relative to healthy baseline (years)")
ax.set_title("Tumor-to-inpaint perturbation trajectory", fontweight="bold", pad=12)
ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("trajectory_tumor_to_inpaint.png", dpi=300, bbox_inches="tight")
plt.savefig("trajectory_tumor_to_inpaint.pdf", bbox_inches="tight")
plt.show()


# =========================================================
# 2) Proportion closer-to-healthy plot
# =========================================================

prop_closer = [
    0.150,
    0.415,
    0.052,
    0.299,
    0.608,
    0.092,
    0.150,
    0.447,
    0.048,
    0.134,
    0.445,
    0.035,
]

percent_closer = [p * 100 for p in prop_closer]
colors = [GREEN if p > 50 else BLUE for p in percent_closer]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(conditions, percent_closer, color=colors)

ax.axvline(50, color=WHITE, linewidth=1, linestyle="--", alpha=0.8)
ax.set_xlabel("Subjects closer to healthy after inpainting (%)")
ax.set_title("Proportion of subjects improved after inpainting", fontweight="bold", pad=12)
ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID)
ax.invert_yaxis()

for bar, value in zip(bars, percent_closer):
    ax.text(
        value + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}%",
        va="center",
        ha="left",
        color=WHITE,
        fontsize=9,
    )

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("proportion_closer_to_healthy.png", dpi=300, bbox_inches="tight")
plt.savefig("proportion_closer_to_healthy.pdf", bbox_inches="tight")
plt.show()