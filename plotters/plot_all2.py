import json
import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LANGS = ["English", "Hindi", "Kannada"]
CODES = {"English": "eng_Latn", "Hindi": "hin_Deva", "Kannada": "kan_Knda"}
COLOR = {"English": "tab:gray", "Hindi": "tab:orange", "Kannada": "tab:blue"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------- inputs ----------
with open(os.path.join(RES, "sweep_results.json")) as f:
    runs = json.load(f)

# plain BPE with all English data + a share of Hindi/Kannada data (from measure.py and mix.py)
mix = {0: {"English": 67221, "Hindi": 336766, "Kannada": 375069},
       5: {"English": 69966, "Hindi": 148233, "Kannada": 159988},
       10: {"English": 72884, "Hindi": 127870, "Kannada": 137000},
       25: {"English": 78878, "Hindi": 111213, "Kannada": 116292},
       50: {"English": 84608, "Hindi": 104404, "Kannada": 107734},
       100: {"English": 91342, "Hindi": 100063, "Kannada": 101094}}
BEST_ENGLISH = mix[0]["English"]      # English tokens when all 200 merges go to English

# number of words in each language's test set, for fertility
words = {}
for lang, code in CODES.items():
    with open(os.path.join(ROOT, "data", f"{code}_devtest.txt"), encoding="utf-8") as f:
        words[lang] = sum(len(line.split()) for line in f)
print("words in devtest:", words)

def unfairness(t):        # F = most tokens / fewest tokens (1 = perfectly fair)
    return max(t.values()) / min(t.values())

def fairness(t):          # 1/F = fewest tokens / most tokens (1 = perfectly fair, higher = fairer)
    return min(t.values()) / max(t.values())

def english_cost(t):      # % more English tokens than with English-only merges
    return 100 * (t["English"] - BEST_ENGLISH) / BEST_ENGLISH

def fertility(t, lang):   # tokens per word
    return t[lang] / words[lang]

def pick(mode, frac, c=None):
    for r in runs:
        if r["mode"] == mode and abs(r["frac"] - frac) < 1e-9 and (c is None or r["c"] == c):
            return r
    return None

def sym_runs(frac):
    return sorted([r for r in runs if r["mode"] == "sym" and abs(r["frac"] - frac) < 1e-9], key=lambda r: r["c"])

# ---------- Figure 1: plain BPE, more data ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
shares = list(mix.keys())
for lang in ["Hindi", "Kannada"]:
    ax.plot(shares, [mix[s][lang] / mix[s]["English"] for s in shares], "o-", color=COLOR[lang], label=lang)
ax.axhline(1, color="black", linestyle="--", linewidth=0.8, label="fair (parity = 1)")
ax.set_xscale("symlog", linthresh=5)
ax.set_xticks(shares); ax.set_xticklabels([f"{s}%" for s in shares])
ax.set_xlabel("share of Hindi/Kannada training data (English data always 100%)")
ax.set_ylabel("parity vs English (tokens / English tokens)")
ax.set_title("Plain BPE: more Hindi/Kannada data makes it fairer")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_more_data.png"), dpi=200)

# ---------- Figure 2: the price of fairness ----------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
xs = [fairness(mix[s]) for s in shares if s > 0]
ys = [english_cost(mix[s]) for s in shares if s > 0]
ax.plot(xs, ys, "o-", color="gray", label="plain BPE, more data")
for x, y, s in zip(xs, ys, [s for s in shares if s > 0]):
    ax.annotate(f"{s}% data", (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8, color="gray")
for frac, color in [(0.05, "tab:green"), (0.1, "tab:purple")]:
    sr = sym_runs(frac)
    if sr:
        ax.plot([fairness(r["tokens"]) for r in sr], [english_cost(r["tokens"]) for r in sr],
                "^-", color=color, label=f"symmetric min-max, {frac:.0%} data")
        for r in sr:
            if r["c"] in (0, 2, 5, 10, 30):
                ax.annotate(f"c={r['c']}", (fairness(r["tokens"]), english_cost(r["tokens"])),
                            textcoords="offset points", xytext=(-8, 8), fontsize=8, color=color)
    for mode, marker, name in [("current", "*", "Case 2"), ("start", "D", "Case 3")]:
        r = pick(mode, frac)
        if r:
            ax.plot(fairness(r["tokens"]), english_cost(r["tokens"]), marker, color=color,
                    markersize=11, label=f"{name}, {frac:.0%} data")
ax.plot(1, 0, "X", color="black", markersize=10, label="ideal (fair, no extra cost)")
ax.set_xlabel("fairness 1/F = fewest tokens / most tokens   (1 = perfectly fair)")
ax.set_ylabel("extra English tokens vs English-only BPE (%)")
ax.set_title("The price of fairness, 200 merges")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_price_of_fairness.png"), dpi=200)

# ---------- Figure 3: the knob c ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
for frac, color in [(0.05, "tab:green"), (0.1, "tab:purple")]:
    sr = sym_runs(frac)
    if sr:
        cs = [r["c"] for r in sr]
        a1.plot(cs, [fairness(r["tokens"]) for r in sr], "o-", color=color, label=f"{frac:.0%} data")
        a2.plot(cs, [english_cost(r["tokens"]) for r in sr], "o-", color=color, label=f"{frac:.0%} data")
a1.axhline(1, color="black", linestyle="--", linewidth=0.8)
a1.set_xlabel("fairness weight c"); a1.set_ylabel("fairness 1/F (1 = fair)"); a1.set_title("Larger c: fairer")
a2.set_xlabel("fairness weight c"); a2.set_ylabel("extra English tokens (%)"); a2.set_title("Larger c: English pays more")
for a in (a1, a2):
    a.legend(); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_knob.png"), dpi=200)

# ---------- Figure 4: who gets the merges ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, frac in zip(axes, [0.05, 0.1]):
    sr = sym_runs(frac)
    labels = [str(r["c"]) for r in sr]
    bottom = [0] * len(sr)
    for lang in LANGS:
        vals = [r["merges"].get(lang, 0) for r in sr]
        a.bar(labels, vals, bottom=bottom, color=COLOR[lang], label=lang)
        bottom = [b + v for b, v in zip(bottom, vals)]
    a.set_xlabel("fairness weight c"); a.set_title(f"{frac:.0%} Hindi/Kannada data")
axes[0].set_ylabel("merges (out of 200)")
axes[1].legend(loc="upper right", fontsize=9)
fig.suptitle("Who gets the merges as c grows")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_merges.png"), dpi=200)

# ---------- Figure 5: fertility ----------
methods = [("plain BPE\n5% data", pick("sym", 0.05, 0)),
           ("Case 2\n5% data", pick("current", 0.05)),
           ("Case 3\n5% data", pick("start", 0.05)),
           ("symmetric c=10\n5% data", pick("sym", 0.05, 10)),
           ("symmetric c=30\n5% data", pick("sym", 0.05, 30)),
           ("plain BPE\n100% data", {"tokens": mix[100]})]
methods = [(n, r) for n, r in methods if r]
fig, ax = plt.subplots(figsize=(10, 4.2))
width = 0.27
for i, lang in enumerate(LANGS):
    vals = [fertility(r["tokens"], lang) for _, r in methods]
    pos = [j + (i - 1) * width for j in range(len(methods))]
    ax.bar(pos, vals, width, color=COLOR[lang], label=lang)
ax.set_xticks(range(len(methods))); ax.set_xticklabels([n for n, _ in methods], fontsize=9)
ax.set_ylabel("fertility (tokens per word)")
ax.set_title("Fertility on devtest: lower is better, equal bars are fair")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig5_fertility.png"), dpi=200)

print("\nfertility (tokens per word):")
for name, r in methods:
    print(f"  {name.replace(chr(10), ', '):28s}", {l: round(fertility(r['tokens'], l), 2) for l in LANGS})

# ---------- Figure 6: total fertility (all three languages together) ----------
def total_fertility(t):   # all tokens / all words, over the three languages
    return sum(t[l] for l in LANGS) / sum(words[l] for l in LANGS)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})

# left: one bar per method
names = [n for n, _ in methods]
vals = [total_fertility(r["tokens"]) for _, r in methods]
a1.bar(range(len(vals)), vals, color="tab:green")
for i, v in enumerate(vals):
    a1.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
a1.set_xticks(range(len(names))); a1.set_xticklabels(names, fontsize=8)
a1.set_ylabel("total fertility (all tokens / all words)")
a1.set_title("Total fertility by method")
a1.grid(axis="y", alpha=0.3)

# right: as the knob c grows, compared with plain BPE
plotted = []
for frac, color in [(0.05, "tab:green"), (0.1, "tab:purple")]:
    sr = sym_runs(frac)
    if sr:
        ys = [total_fertility(r["tokens"]) for r in sr]
        plotted += ys
        a2.plot([r["c"] for r in sr], ys, "o-", color=color, label=f"symmetric, {frac:.0%} data")

balanced = total_fertility(mix[100])
plotted.append(balanced)
a2.axhline(balanced, color="black", linestyle="--", label="plain BPE, 100% data")

# English-only is far off the scale, so report it in the legend instead of drawing it
a2.plot([], [], " ", label=f"plain BPE, English-only: {total_fertility(mix[0]):.2f} (off scale)")

lo, hi = min(plotted), max(plotted)
pad = 0.1 * (hi - lo)
a2.set_ylim(lo - pad, hi + pad)

a2.set_xlabel("fairness weight c"); a2.set_ylabel("total fertility")
a2.set_title("Total fertility as c grows")
a2.legend(fontsize=8); a2.grid(alpha=0.3)
print("saved figures in", RES)


plt.show()


