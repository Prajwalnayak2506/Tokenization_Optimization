import os
import pickle
import time
from collections import Counter

from measure import ROOT, load, total_tokens, script_of   # must come first: it adds bpe/ to the path
from bpe_text import corpus_from_texts
from bpe_bytes import train

languages = {"English": "eng_Latn", "Hindi": "hin_Deva", "Kannada": "kan_Knda"}
K = 200

english = load("eng_Latn", "dev")
hindi = load("hin_Deva", "dev")
kannada = load("kan_Knda", "dev")

def nbytes(sents):
    return sum(len(s.encode("utf-8")) for s in sents)

for frac in [0.05, 0.1, 0.25, 0.5]:
    n = int(frac * len(hindi))                    # how many Hindi and Kannada sentences to keep
    texts = english + hindi[:n] + kannada[:n]
    english_share = nbytes(english) / nbytes(texts)

    start = time.time()
    merges, _ = train(corpus_from_texts(texts), K)
    minutes = (time.time() - start) / 60

    with open(os.path.join(ROOT, "results", f"merges_frac{frac}_K{K}.pkl"), "wb") as f:
        pickle.dump(merges, f)

    results = {name: total_tokens(load(code, "devtest"), merges)
               for name, code in languages.items()}
    allocation = Counter(script_of(a + b) for a, b in merges)

    print("=" * 60)
    print(f"Hindi/Kannada sentences: {n} ({frac:.0%}) | English share of training bytes: {english_share:.0%}")
    print(f"Training took {minutes:.1f} minutes")
    for name in languages:
        print(f"  {name}: {results[name]} tokens | parity {results[name] / results['English']:.2f}")
    print("  merges per language:", dict(allocation))