import json
import os
from collections import Counter

from measure import ROOT, load, total_tokens, script_of    # first: adds bpe/ to the path
from bpe_text import corpus_from_texts
from bpe_bytes import get_vocab, get_pair_counts, merge_pair, total_symbols

def train_minmax(corpora, parallel_english, num_merges, c):
    vocabs = {lang: get_vocab(cor) for lang, cor in corpora.items()}   # all training data
    par_en = get_vocab(parallel_english)     # English sentences that Hindi/Kannada translate
    merges = []
    for step in range(num_merges):
        # training-time parity, on the parallel sentences
        en_par = total_symbols(par_en)
        rho = {lang: total_symbols(vocabs[lang]) / en_par for lang in ["Hindi", "Kannada"]}
        worst = max(rho, key=rho.get)

        counts = {lang: get_pair_counts(v) for lang, v in vocabs.items()}
        score = Counter()
        for lang, cnt in counts.items():                 # compression: raw saving, all languages
            for pair, n in cnt.items():
                score[pair] += n
        for pair, n in counts[worst].items():            # fairness: helping the worst language
            score[pair] += c * n
        for pair, n in get_pair_counts(par_en).items():  # fairness: compressing English hurts parity
            score[pair] -= c * rho[worst] * n

        if not score:
            break
        best = max(score, key=score.get)
        vocabs = {lang: merge_pair(v, best) for lang, v in vocabs.items()}
        par_en = merge_pair(par_en, best)
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
    parallel_english = corpus_from_texts(english[:n])

    results = []
    for c in [0, 2, 5, 10, 30]:
        merges = train_minmax(corpora, parallel_english, K, c)
        tokens = {name: total_tokens(load(code, "devtest"), merges)
                  for name, code in languages.items()}
        row = {"c": c, "frac": frac, "tokens": tokens,
               "parity_hi": round(tokens["Hindi"] / tokens["English"], 2),
               "parity_kn": round(tokens["Kannada"] / tokens["English"], 2),
               "merges": dict(Counter(script_of(a + b) for a, b in merges))}
        results.append(row)
        print(row)

    with open(os.path.join(ROOT, "results", "minmax_results.json"), "w") as f:
        json.dump(results, f, indent=2)