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

def merge_symbols(symbols, pair):
    # merge one pair inside a single word
    a, b = pair
    new_symbols = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            new_symbols.append(a + b)
            i += 2
        else:
            new_symbols.append(symbols[i])
            i += 1
    return tuple(new_symbols)

def merge_pair(vocab, pair):
    # merge one pair in every word of the vocabulary
    return {merge_symbols(symbols, pair): count for symbols, count in vocab.items()}

best = pairs.most_common(1)[0][0]    # the winning pair: ('e', 's')
print("Merging:", best)
vocab = merge_pair(vocab, best)
print(vocab)
print(total_symbols(vocab))

def train(corpus, num_merges):
    vocab = get_vocab(corpus)
    merges = []
    for step in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:          # every word is one token, nothing left to merge
            break
        best = pairs.most_common(1)[0][0]
        print("The best is ",best," with number ",pairs[best],". The pairs as for now are : ",pairs)
        vocab = merge_pair(vocab, best)
        merges.append(best)
        print(f"Merge {step + 1}: {best} -> total symbols {total_symbols(vocab)}")
    return merges, vocab
number_of_merges = 4
merges, vocab = train(corpus, number_of_merges)
print("Learned merges:", merges)
print("Final words:", vocab)

def encode_word(word, merges):
    symbols = tuple(word)
    for pair in merges:           # replay merges in learned order
        symbols = merge_symbols(symbols, pair)
    return symbols

def decode(tokens):
    return "".join(tokens)

for w in ["lowest", "slow", "newer", "widest"]:
    tokens = encode_word(w, merges)
    print(w, "->", tokens, "->", decode(tokens))


# version 2: token IDs and the round-trip test
# Why IDs? Your tokenizer outputs strings like ('low', 'est'), but a model only takes numbers. So we give every token in the vocabulary a fixed number, like a dictionary that maps each token to a page number.
def build_vocab(corpus, merges):
    alphabet = sorted(set("".join(corpus)))          # the base characters, Σ
    tokens = alphabet + [a + b for a, b in merges]   # plus one new token per merge
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return token_to_id, id_to_token

def encode(word, merges, token_to_id):
    return [token_to_id[t] for t in encode_word(word, merges)]

def decode_ids(ids, id_to_token):
    return "".join(id_to_token[i] for i in ids)

token_to_id, id_to_token = build_vocab(corpus, merges)
print(token_to_id)

for w in ["lowest", "slow", "newer", "widest","zebra"]:
    ids = encode(w, merges, token_to_id)
    back = decode_ids(ids, id_to_token)
    print(w, "->", ids, "->", back, "| round trip OK:", back == w)