from collections import Counter
from bpe_bytes import train, encode_word, decode

def split_text(text):
    pieces = text.split(" ")
    return [pieces[0]] + [" " + p for p in pieces[1:]]

def corpus_from_texts(texts):
    corpus = Counter()
    for text in texts:
        for chunk in split_text(text):
            corpus[chunk] += 1
    return corpus



print("the cat sat.".split(" "))
print(split_text("the cat sat."))
# Why the space goes in front of the word, rather than after the one before it: it’s GPT-2’s convention, and it makes the model see " cat" (a word in the middle of a sentence) as one unit.
print(split_text("नमस्ते दुनिया"))
print(split_text("two  spaces"))
print("".join(split_text("two  spaces")))



print(corpus_from_texts(["the cat sat on the mat.", "the dog sat on the log."]))