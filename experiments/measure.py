import os
import sys
from collections import Counter
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "bpe"))

def load(code, split):
    path = os.path.join(ROOT, "data", f"{code}_{split}.txt")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

from bpe_text import corpus_from_texts
from bpe_bytes import train, encode_word

def total_tokens(sents, merges):
    corpus = corpus_from_texts(sents)
    return sum(count * len(encode_word(piece, merges)) for piece, count in corpus.items())

def script_of(token):
    '''
    This reads which language a merged token belongs to, using what you proved about the bytes:

All bytes below 128: English (ASCII).
A 224 followed by 164 or 165: a Devanagari letter, so Hindi.
A 224 followed by 178 or 179: a Kannada letter.
Otherwise “unclear”: for example a token made only of continuation bytes, which could belong to either script.
    '''
    if all(b < 128 for b in token):
        return "English"
    for i in range(len(token) - 1):
        if token[i] == 224 and token[i + 1] in (164, 165):
            return "Hindi"
        if token[i] == 224 and token[i + 1] in (178, 179):
            return "Kannada"
    return "unclear"


if __name__ == "__main__" :
    languages = {"English": "eng_Latn", "Hindi": "hin_Deva", "Kannada": "kan_Knda"}
    for name, code in languages.items():
        sents = load(code, "devtest")
        chars = sum(len(s) for s in sents)
        nbytes = sum(len(s.encode("utf-8")) for s in sents)
        print(name, "| sentences:", len(sents), "| characters:", chars, "| bytes:", nbytes)
        print("   first sentence:", sents[0])
    print("-------------------------------------------------------------------------------------------------------------------------------------")
    languages = {"English": "eng_Latn", "Hindi": "hin_Deva", "Kannada": "kan_Knda"}
    K = 200
    train_corpus = corpus_from_texts(load("eng_Latn", "dev"))
    merges, _ = train(train_corpus, K)
    results = {}
    for name, code in languages.items():
        results[name] = total_tokens(load(code, "devtest"), merges)
    for name in languages:
        print(name, "| tokens:", results[name],
            "| parity vs English:", round(results[name] / results["English"], 2))
    print("1. English drops a lot: ✓. It almost halved, a 49% cut, from just 200 merges.")
    print("2. Hindi and Kannada stay exactly the same: almost, but not exactly! Hindi dropped by 307 tokens and Kannada by 311. That’s tiny (0.1%), but not zero. Can you work out why? Remember the exception in this morning’s proof: English merges can fire wherever English bytes sit next to each other inside a Hindi or Kannada piece. Look through your merge list for merges that could appear in Hindi news text. There are a few clear ones: think numbers and brackets.")
    print("3. Parity gets worse: ✓. It roughly doubled, to 5.01 and 5.58. A Hindi reader pays five times as much as an English reader for the same news sentence.")
    print("====================================================================================================================================================================================")
    texts = load("eng_Latn", "dev") + load("hin_Deva", "dev") + load("kan_Knda", "dev")
    merges, _ = train(corpus_from_texts(texts), K)

    with open(os.path.join(ROOT, "results", "merges_mixed_200.pkl"), "wb") as f:
        pickle.dump(merges, f)
    print("=================================================================Saved pkl file==========================================================================")
    with open(os.path.join(ROOT, "results", "merges_mixed_200.pkl"), "rb") as f:
        merges = pickle.load(f)

    results = {}
    for name, code in languages.items():
        results[name] = total_tokens(load(code, "devtest"), merges)
    for name in languages:
        print(name, "| tokens:", results[name],
              "| parity vs English:", round(results[name] / results["English"], 2))

    print(Counter(script_of(a + b) for a, b in merges))#This counts how many of the 200 merges went to each language.
    print("The split is roughly proportional: ✓, more or less. In the training corpus, English is about 16% of the bytes and got 20% of the merges. Hindi and Kannada split their share about 46/54, close to their byte sizes. Greedy never computes anything like “language size”. It only picks the most frequent pair each time. But big languages have more frequent pairs, so the split comes out roughly proportional anyway. Your intuition was good.")
    print("English gets worse: ✓. It went from 67,221 to 91,342 tokens, because it now gets only 40 merges instead of 200.")
    print("Parity improves but stays above 1: ✓. It went from about 5 down to 1.10, which is nearly fair, but not below 1.")
    print("The important insight. FLORES is a parallel corpus: every language has the same sentences, so the training data is perfectly balanced. With balanced data, greedy BPE ends up almost fair by itself.")
    print("Real tokenizers are trained on data that is overwhelmingly English. So the unfairness doesn’t come from BPE alone; it comes from BPE plus imbalanced data. And the amount of each language in the training data is a knob we can turn.")
    print("The next experiment: turn the knob and trace a trade-off curve. Keep all the English training sentences, and include only a fraction of the Hindi and Kannada ones: 10%, 25%, 50%. For each mix, measure English tokens and Hindi/Kannada parity. You already have the two ends: 0% (English only) and 100% (mixed).")
    print("Plot English tokens against parity for these five points, and you get a trade-off curve between compression for English and fairness for Indian languages. That’s the first real picture of the multi-objective problem, and it would be a strong slide for tomorrow.")




