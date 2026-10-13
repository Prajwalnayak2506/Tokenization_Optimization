# version 3: byte-level BPE
# The idea in three layers

# Characters. Every character in every language has an official number, called its Unicode code point. a is 97, न is 2344.
# Bytes. To store that number in a file, computers use UTF-8, which writes it as 1 to 4 bytes. A byte is just a number from 0 to 255. English letters fit in 1 byte; Devanagari characters need 3.
# What you see. नमस्ते looks like 3 units (न, म, स्ते). Underneath, it’s 6 Unicode characters, because the halant ् and the vowel sign े are separate characters. And those 6 characters are 18 bytes.

# Byte-level BPE works on layer 2, the bytes. Since every text in every language is made of bytes from 0 to 255, nothing can ever be unknown.

for text in ["hello", "नमस्ते"]:
    b = text.encode("utf-8")
    print(text, "| characters:", len(text), "| bytes:", len(b))
    print("   byte values:", list(b))

print()
for ch in "नमस्ते":
    print(ch, "code point:", ord(ch), "| bytes:", list(ch.encode("utf-8")))

print()
b = "नमस्ते".encode("utf-8")
print("all bytes decoded:", b.decode("utf-8"))
print("first 2 bytes decoded:", b[:3].decode("utf-8"))

# How 2344 becomes 224, 164, 168
# A single byte can only hold numbers up to 255, and 2344 is too big. So UTF-8 splits it into pieces, in the same way you’d split 150 seconds into “2 minutes and 30 seconds”. Except UTF-8 uses groups of 64 instead of 60

# Step 1: split 2344 into groups of 64.
# 2344 ÷ 64 = 36, remainder 40. So 2344 = 36 × 64 + 40.
# The pieces are 0, 36, 40 (with a 0 in front, because a 3-byte character always has three pieces).

# Step 2: add a label to each piece, so the computer knows how to read it:

# First byte: add 224. That’s the label for “start of a 3-byte character”. 224 + 0 = 224
# Second byte: add 128. That’s the label for “continuation of a character”. 128 + 36 = 164
# Third byte: add 128 again. 128 + 40 = 168