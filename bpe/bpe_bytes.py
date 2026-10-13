
from collections import Counter

def to_bytes(word):
    # "low" -> (b'l', b'o', b'w')
    return tuple(bytes([b]) for b in word.encode("utf-8"))

def get_vocab(corpus):
    return {to_bytes(word): count for word, count in corpus.items()}

def get_pair_counts(vocab):
    pairs = Counter()
    for symbols, count in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += count
    return pairs

def total_symbols(vocab):
    return sum(len(symbols) * count for symbols, count in vocab.items())

def merge_symbols(symbols, pair):
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
    return {merge_symbols(symbols, pair): count for symbols, count in vocab.items()}

def train(corpus, num_merges):
    vocab = get_vocab(corpus)
    merges = []
    print("Start: total symbols", total_symbols(vocab))
    for step in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        vocab = merge_pair(vocab, best)
        merges.append(best)
        print(f"Merge {step + 1}: {best} -> total symbols {total_symbols(vocab)}")
    return merges, vocab


def show(token):
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return "fragment"




def encode_word(word, merges):
    symbols = to_bytes(word)              # was tuple(word)
    for pair in merges:                   # replay merges in learned order
        symbols = merge_symbols(symbols, pair)
    return symbols

def decode(tokens):
    return b"".join(tokens).decode("utf-8")


def build_vocab(merges):
    tokens = [bytes([i]) for i in range(256)] + [a + b for a, b in merges]
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}
    return token_to_id, id_to_token

def encode(word, merges, token_to_id):
    return [token_to_id[t] for t in encode_word(word, merges)]

def decode_ids(ids, id_to_token):
    return b"".join(id_to_token[i] for i in ids).decode("utf-8")




if __name__ == "__main__":
    # all your test code goes here, indented by 4 spaces
    word = "low"
    raw = word.encode("utf-8")
    print(raw) #b'low' is a bytes object. It’s like a string, but made of bytes instead of characters. The b in front is Python saying “this is bytes”. For English letters, Python shows them as letters to be readable.
    print(list(raw))
    symbols = tuple(bytes([b]) for b in raw)
    print(symbols)
    # Why bytes([b]) instead of just the numbers?
    print(224 + 164)
    print(bytes([224]) + bytes([164]))
    # Numbers add up: 224 + 164 = 388. That’s useless; we lose the two bytes.
    # Bytes objects glue together, like strings do. b'\xe0' + b'\xa4' gives b'\xe0\xa4', both bytes side by side.
    print(tuple(bytes([b]) for b in "न".encode("utf-8")))
    print("=========================================================================================================================================================================")
    english = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    merges, vocab = train(english, 4)
    hindi = {"नमस्ते": 5, "नमक": 3, "कमल": 2}
    merges, vocab = train(hindi, 6)
    for a, b in merges:
        print(list(a), "+", list(b), "->", list(a + b), show(a + b))
    
    for w in ["नमस्ते", "नमक", "कमल", "मन", "नमन", "hello"]:
        tokens = encode_word(w, merges)
        print(w, "->", [list(t) for t in tokens])
        print("   readable?", [show(t) for t in tokens])
        print("   decoded:", decode(tokens), "| round trip OK:", decode(tokens) == w)

    token_to_id, id_to_token = build_vocab(merges)
    print("vocabulary size:", len(token_to_id))
    for i in range(256, len(token_to_id)):
        print(i, list(id_to_token[i]), show(id_to_token[i]))

    for w in ["नमस्ते", "नमक", "मन", "zebra", "hello", "🙂"]:
        ids = encode(w, merges, token_to_id)
        print(w, "->", ids, "| round trip OK:", decode_ids(ids, id_to_token) == w)
    
    # But there’s one important catch: “can encode any language” is not the same as “encodes every language equally well.”
    print("========================================================================================")
    english = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    hindi = {"नमस्ते": 5, "नमक": 3, "कमल": 2}
    mixed = {**english, **hindi}        # both corpora in one dictionary

    for name, corpus, k in [("English only", english, 4),
                            ("Hindi only", hindi, 6),
                            ("Mixed", mixed, 6)]:
        m, _ = train(corpus, k)
        print(name, "| hello:", len(encode_word("hello", m)),
            "tokens | नमस्ते:", len(encode_word("नमस्ते", m)), "tokens")