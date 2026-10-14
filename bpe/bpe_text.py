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

def encode_text(text, merges):
    tokens = []
    for chunk in split_text(text):
        tokens.extend(encode_word(chunk, merges))
    return tokens

if __name__ == "__main__":
    print("the cat sat.".split(" "))
    print(split_text("the cat sat."))
    # Why the space goes in front of the word, rather than after the one before it: it’s GPT-2’s convention, and it makes the model see " cat" (a word in the middle of a sentence) as one unit.
    print(split_text("नमस्ते दुनिया"))
    print(split_text("two  spaces"))
    print("".join(split_text("two  spaces")))
    print(corpus_from_texts(["the cat sat on the mat.", "the dog sat on the log."]))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    corpus = corpus_from_texts(["the cat sat on the mat.", "the dog sat on the log."])
    merges, _ = train(corpus, 10)
    sentences = {
        "English": "the cat sat on the mat.",
        "Hindi":   "बिल्ली चटाई पर बैठी।",
        "Kannada": "ಬೆಕ್ಕು ಚಾಪೆಯ ಮೇಲೆ ಕುಳಿತಿತು.",
    }
    for lang, s in sentences.items():
        tokens = encode_text(s, merges)
        print(lang, "| characters:", len(s), "| bytes:", len(s.encode("utf-8")),
            "| tokens:", len(tokens), "| round trip OK:", decode(tokens) == s)

# we trained on english and not on hindi and kannada so nuber of bytes = number of tokens but number of tokens for englihs< number of bytes for englihs, it would have reduced for hindi / kannada too if by coincidece the same tokens come up for these two, but idk if that possibility is there or not.
# It’s impossible, and we can prove why. No coincidence can happen here.
# The argument, in three steps:
# English merges contain only “English bytes”. Every character in the training sentences is a basic English letter, space or full stop, and all of those are bytes below 128. So every merged token is built only from bytes below 128.
# Hindi and Kannada letters contain no such bytes. Every byte of a Hindi or Kannada letter is 128 or more: the lead byte 224, then 164/165 or 178/179, then a continuation byte. The only bytes below 128 in those sentences are the spaces and the . at the end.
# A merge needs both of its halves side by side. The space at the start of " चटाई" is always followed by 224, never by another English byte. So no English merge can ever fire inside a Hindi or Kannada letter.