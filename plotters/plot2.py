import json
import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# plain BPE with different data mixes (from mix.py)
fractions = [0, 5, 10, 25, 50, 100]
english = [67221, 69966, 72884, 78878, 84608, 91342]
kannada = [5.58, 2.29, 1.88, 1.47, 1.27, 1.11]

with open(os.path.join(ROOT, "results", "fair_bpe_results.json")) as f:
    runs = json.load(f)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(english, kannada, "o-", color="gray", label="plain BPE, varying data mix")
for x, y, frac in zip(english, kannada, fractions):
    ax.annotate(f"{frac}%", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

style = {"current": ("*", "tab:red", "Case 2"), "start": ("D", "tab:blue", "Case 3")}
for r in runs:
    if r["mode"] in style:
        marker, color, name = style[r["mode"]]
        ax.plot(r["tokens"]["English"], r["parity_kn"], marker, color=color, markersize=10,
                label=f"{name}, {int(r['frac'] * 100)}% data")

ax.axhline(1, color="black", linestyle="--", linewidth=0.8, label="perfect parity")
ax.set_ylim(0.9, 2.5)
ax.set_xlabel("English tokens on devtest")
ax.set_ylabel("Kannada parity (worst language)")
ax.set_title("Weighted merge rules on imbalanced data, 200 merges")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "results", "plot2_rules.png"), dpi=200)
plt.show()