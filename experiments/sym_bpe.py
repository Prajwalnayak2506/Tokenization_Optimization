import json
import os
from collections import Counter

from measure import ROOT, load, total_tokens, script_of    # first: adds bpe/ to the path
from bpe_text import corpus_from_texts
from bpe_bytes import get_vocab, get_pair_counts, merge_pair, total_symbols

def train_symmetric(corpora, parallel, num_merges, c):
    vocabs = {lang: get_vocab(cor) for lang, cor in corpora.items()}   # all training data
    par = {lang: get_vocab(cor) for lang, cor in parallel.items()}     # parallel sentences only
    merges = []
    for step in range(num_merges):
        size = {lang: total_symbols(v) for lang, v in par.items()}
        worst = max(size, key=size.get)        # W: needs the most tokens
        best_off = min(size, key=size.get)     # B: needs the fewest tokens
        F = size[worst] / size[best_off]

        score = Counter()
        for lang, v in vocabs.items():                     # compression
            for pair, n in get_pair_counts(v).items():
                score[pair] += n
        for pair, n in get_pair_counts(par[worst]).items():     # help W
            score[pair] += c * n
        for pair, n in get_pair_counts(par[best_off]).items():  # compressing B hurts fairness
            score[pair] -= c * F * n

        if not score:
            break
        best = max(score, key=score.get)
        vocabs = {lang: merge_pair(v, best) for lang, v in vocabs.items()}
        par = {lang: merge_pair(v, best) for lang, v in par.items()}
        merges.append(best)
    return merges

if __name__ == "__main__":
    languages = {"English": "eng_Latn", "Hindi": "hin_Deva", "Kannada": "kan_Knda"}
    K, frac = 200, 0.05
    english, hindi, kannada = (load(code, "dev") for code in languages.values())
    n = int(frac * len(hindi))
    corpora = {"English": corpus_from_texts(english),
               "Hindi": corpus_from_texts(hindi[:n]),
               "Kannada": corpus_from_texts(kannada[:n])}
    parallel = {"English": corpus_from_texts(english[:n]),
                "Hindi": corpus_from_texts(hindi[:n]),
                "Kannada": corpus_from_texts(kannada[:n])}

    results = []
    for c in [0, 2, 5, 10, 30, 100]:
        merges = train_symmetric(corpora, parallel, K, c)
        tokens = {name: total_tokens(load(code, "devtest"), merges)
                  for name, code in languages.items()}
        row = {"c": c, "frac": frac, "tokens": tokens,
               "parity_hi": round(tokens["Hindi"] / tokens["English"], 2),
               "parity_kn": round(tokens["Kannada"] / tokens["English"], 2),
               "F_sym": round(max(tokens.values()) / min(tokens.values()), 2),
               "merges": dict(Counter(script_of(a + b) for a, b in merges))}
        results.append(row)
        print(row)

    with open(os.path.join(ROOT, "results", "sym_results.json"), "w") as f:
        json.dump(results, f, indent=2)