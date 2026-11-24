import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# results from experiments/measure.py (0% and 100%) and experiments/mix.py (the rest)
fractions = [0, 5, 10, 25, 50, 100]          # % of Hindi/Kannada sentences in training
english = [67221, 69966, 72884, 78878, 84608, 91342]
hindi_parity = [5.01, 2.12, 1.75, 1.41, 1.23, 1.10]
kannada_parity = [5.58, 2.29, 1.88, 1.47, 1.27, 1.11]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(english, hindi_parity, "o-", label="Hindi")
ax.plot(english, kannada_parity, "s-", label="Kannada")

for x, y, f in zip(english, kannada_parity, fractions):
    ax.annotate(f"{f}%", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

ax.axhline(1, color="gray", linestyle="--", linewidth=1, label="perfect parity")
ax.set_xlabel("English tokens on devtest (lower = better English compression)")
ax.set_ylabel("Parity vs English (lower = fairer)")
ax.set_title("English compression vs parity, 200 merges")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

out = os.path.join(ROOT, "results", "plot1_tradeoff.png")
fig.savefig(out, dpi=200)
print("saved", out)
plt.show()