import json
import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")

with open(os.path.join(OUT, "sweep_results.json")) as f:
    runs = json.load(f)

def unfairness(tokens):
    return max(tokens.values()) / min(tokens.values())

# plain BPE with more and more Hindi/Kannada data (from mix.py and measure.py)
mix_frac = [0, 5, 10, 25, 50, 100]
mix_en = [67221, 69966, 72884, 78878, 84608, 91342]
mix_unfair = [5.58, 2.29, 1.88, 1.47, 1.27, 1.11]

# ---------- Plot A: the trade-off ----------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(mix_en, mix_unfair, "o-", color="gray", label="plain BPE, more data")
for x, y, fr in zip(mix_en, mix_unfair, mix_frac):
    ax.annotate(f"{fr}%", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, color="gray")

colors = {0.05: "tab:green", 0.1: "tab:purple"}
for frac, color in colors.items():
    sym = sorted([r for r in runs if r["mode"] == "sym" and r["frac"] == frac], key=lambda r: r["c"])
    if sym:
        ax.plot([r["tokens"]["English"] for r in sym], [unfairness(r["tokens"]) for r in sym],
                "^-", color=color, label=f"symmetric min-max, {frac:.0%} data")
    for mode, marker in [("current", "*"), ("start", "D")]:
        for r in runs:
            if r["mode"] == mode and r["frac"] == frac:
                ax.plot(r["tokens"]["English"], unfairness(r["tokens"]), marker, color=color,
                        markersize=10, label=f"Case {2 if mode == 'current' else 3}, {frac:.0%} data")

ax.axhline(1, color="black", linestyle="--", linewidth=0.8, label="perfectly fair")
ax.set_ylim(0.95, 2.4)
ax.set_xlabel("English tokens on devtest (lower = better English compression)")
ax.set_ylabel("unfairness: most tokens / fewest tokens")
ax.set_title("Compression vs fairness, 200 merges")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plotA_tradeoff.png"), dpi=200)

# ---------- Plot B: what the knob c does ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for frac, color in colors.items():
    sym = sorted([r for r in runs if r["mode"] == "sym" and r["frac"] == frac], key=lambda r: r["c"])
    if sym:
        cs = [r["c"] for r in sym]
        ax1.plot(cs, [unfairness(r["tokens"]) for r in sym], "o-", color=color, label=f"{frac:.0%} data")
        ax2.plot(cs, [r["tokens"]["English"] for r in sym], "o-", color=color, label=f"{frac:.0%} data")
ax1.axhline(1, color="black", linestyle="--", linewidth=0.8)
ax1.set_xlabel("c"); ax1.set_ylabel("unfairness (most / fewest tokens)"); ax1.set_title("Fairness vs c")
ax2.set_xlabel("c"); ax2.set_ylabel("English tokens on devtest"); ax2.set_title("English cost vs c")
for a in (ax1, ax2):
    a.grid(alpha=0.3); a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "plotB_knob.png"), dpi=200)

# ---------- Plot C: who gets the merges ----------
sym = sorted([r for r in runs if r["mode"] == "sym" and r["frac"] == 0.05], key=lambda r: r["c"])
if sym:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [str(r["c"]) for r in sym]
    bottom = [0] * len(sym)
    for lang, color in [("English", "tab:gray"), ("Hindi", "tab:orange"), ("Kannada", "tab:blue")]:
        vals = [r["merges"].get(lang, 0) for r in sym]
        ax.bar(labels, vals, bottom=bottom, color=color, label=lang)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xlabel("c"); ax.set_ylabel("number of merges (out of 200)")
    ax.set_title("Who gets the merges, symmetric min-max, 5% data")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plotC_merges.png"), dpi=200)

print("saved plots in", OUT)
plt.show()