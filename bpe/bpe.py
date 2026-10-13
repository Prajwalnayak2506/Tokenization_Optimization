from collections import Counter

corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

def get_vocab(corpus):
    # each word becomes a tuple of characters: "low" -> ('l', 'o', 'w')
    return {tuple(word): count for word, count in corpus.items()}

def get_pair_counts(vocab):
    pairs = Counter()
    for symbols, count in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += count
    return pairs
def total_symbols(vocab):
    return sum(len(symbols) * count for symbols, count in vocab.items())
vocab = get_vocab(corpus)
print("The total number of single characters with repitition is ",total_symbols(vocab))   # 79
print("The vocabulary is ",vocab)
pairs = get_pair_counts(vocab)
print("The top three most common pairs with their counts are as follows", pairs.most_common(3))
# BPE’s goal is to represent the corpus in as few tokens as possible.
# Right now every character is a token, so the total is:

# low: 3 symbols × 5 = 15
# lower: 5 × 2 = 10
# newest: 6 × 6 = 36
# widest: 6 × 3 = 18
# Total: 79 tokens
# When you merge a pair into one symbol, every place that pair occurs shrinks by one token. So the pair count is exactly how many tokens that merge saves. Merging ('e','s') saves 9, taking the total from 79 to 70.

def merge_pair(vocab, pair):
    a, b = pair
    new_vocab = {}
    for symbols, count in vocab.items():
        new_symbols = []
        i = 0
        while i < len(symbols):
            # if this symbol and the next one form the pair, join them
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(a + b)
                i += 2          # skip both, they're now one symbol
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = count
    return new_vocab

best = pairs.most_common(1)[0][0]    # the winning pair: ('e', 's')
print("Merging:", best)
vocab = merge_pair(vocab, best)
print(vocab)
print(total_symbols(vocab))