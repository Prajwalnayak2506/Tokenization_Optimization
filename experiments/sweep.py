import json
import os
import time
from collections import Counter

from measure import ROOT, load, total_tokens, script_of    # first: adds bpe/ to the path
from bpe_text import corpus_from_texts
from fair_bpe import train_weighted
from sym_bpe import train_symmetric

languages = {"English": "eng_Latn", "Hindi": "hin_Deva", "Kannada": "kan_Knda"}
K = 200
english, hindi, kannada = (load(code, "dev") for code in languages.values())
test = {name: load(code, "devtest") for name, code in languages.items()}
out_path = os.path.join(ROOT, "results", "sweep_results.json")
runs = []

def record(mode, frac, c, merges, seconds):
    tokens = {name: total_tokens(test[name], merges) for name in languages}
    row = {"mode": mode, "frac": frac, "c": c, "tokens": tokens,
           "F_sym": round(max(tokens.values()) / min(tokens.values()), 3),
           "merges": dict(Counter(script_of(a + b) for a, b in merges)),
           "seconds": round(seconds)}
    runs.append(row)
    with open(out_path, "w") as f:          # save after every run
        json.dump(runs, f, indent=2)
    print(f"{mode:8s} frac={frac} c={c} | English {tokens['English']} | "
          f"F_sym {row['F_sym']} | {row['seconds']} s")

start_all = time.time()
for frac in [0.05, 0.1]:
    n = int(frac * len(hindi))
    corpora = {"English": corpus_from_texts(english),
               "Hindi": corpus_from_texts(hindi[:n]),
               "Kannada": corpus_from_texts(kannada[:n])}
    parallel = {"English": corpus_from_texts(english[:n]),
                "Hindi": corpus_from_texts(hindi[:n]),
                "Kannada": corpus_from_texts(kannada[:n])}

    for mode in ["current", "start"]:                       # Cases 2 and 3
        t = time.time()
        record(mode, frac, None, train_weighted(corpora, K, mode), time.time() - t)

    for c in [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:        # symmetric min-max
        t = time.time()
        record("sym", frac, c, train_symmetric(corpora, parallel, K, c), time.time() - t)

print(f"all done in {(time.time() - start_all) / 60:.1f} minutes")