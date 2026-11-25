import json
import os
from collections import Counter

from measure import ROOT, load, total_tokens, script_of    # first: adds bpe/ to the path
from bpe_text import corpus_from_texts
from bpe_bytes import get_vocab, get_pair_counts, merge_pair, total_symbols

def train_weighted(corpora, num_merges, mode):
    # corpora: {language: corpus}. mode: "plain" (Case 1), "current" (Case 2), "start" (Case 3)
    vocabs = {lang: get_vocab(c) for lang, c in corpora.items()}
    start = {lang: total_symbols(v) for lang, v in vocabs.items()}       # T^0, fixed
    merges = []
    for step in range(num_merges):
        current = {lang: total_symbols(v) for lang, v in vocabs.items()}  # T, changes every step
        if mode == "plain":
            weight = {lang: 1.0 for lang in vocabs}
        elif mode == "current":
            weight = {lang: 1.0 / current[lang] for lang in vocabs}
        else:
            weight = {lang: 1.0 / start[lang] for lang in vocabs}

        score = Counter()
        for lang, v in vocabs.items():
            for pair, n in get_pair_counts(v).items():
                score[pair] += weight[lang] * n          # weighted saving, summed over languages
        if not score:
            break
        best = max(score, key=score.get)
        vocabs = {lang: merge_pair(v, best) for lang, v in vocabs.items()}
        merges.append(best)
    return merges

if __name__ == "__main__":
    languages = {"English": "eng_Latn", "Hindi": "hin_Deva", "Kannada": "kan_Knda"}
    K = 200
    english, hindi, kannada = (load(c, "dev") for c in languages.values())
    all_results = []

    for frac in [0.05, 0.1]:
        n = int(frac * len(hindi))
        corpora = {"English": corpus_from_texts(english),
                   "Hindi": corpus_from_texts(hindi[:n]),
                   "Kannada": corpus_from_texts(kannada[:n])}
        for mode in ["plain", "current", "start"]:
            merges = train_weighted(corpora, K, mode)
            tokens = {name: total_tokens(load(code, "devtest"), merges)
                      for name, code in languages.items()}
            allocation = Counter(script_of(a + b) for a, b in merges)
            row = {"frac": frac, "mode": mode, "tokens": tokens,
                   "parity_hi": round(tokens["Hindi"] / tokens["English"], 2),
                   "parity_kn": round(tokens["Kannada"] / tokens["English"], 2),
                   "merges": dict(allocation)}
            all_results.append(row)
            print(row)

    with open(os.path.join(ROOT, "results", "fair_bpe_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)