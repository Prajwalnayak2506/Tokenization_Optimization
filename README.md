# Multi-Objective Tokenization for Indian Languages
 
Course project for **CS 769: Optimization in Machine Learning**

Prajwal Nayak (22B4246)
 
## Overview
 
Language models read text as *tokens*, and the tokenizer decides how many tokens a piece of text costs. The standard method, **Byte Pair Encoding (BPE)**, is a greedy algorithm that optimizes a single objective: compressing the training text into as few tokens as possible.
 
For Indian languages this single objective causes problems. The same sentence can cost several times more tokens in Hindi or Kannada than in English, and BPE happily creates tokens that cut letters and syllables in half. This project:
 
1. implements BPE from scratch, first on characters and then on bytes;
2. studies it as an optimization problem, with proofs of its basic properties;
3. measures three conflicting objectives on English, Hindi and Kannada:
   - **compression**: total number of tokens,
   - **parity**: how many more tokens a language needs than English for the same meaning,
   - **integrity**: how often token boundaries cut through characters;
4. works towards tokenization as a **multi-objective optimization** problem.
## Repository structure
 
```
.
├── README.md
├── bpe/                   tokenizer code
│   ├── bpe.py             character-level BPE: training, encoding, decoding, token IDs
│   ├── bpe_bytes.py       byte-level BPE: works for any language and script
│   ├── bpe_text.py        running text: splitting sentences, building a corpus
│   └── explore_bytes.py   how text becomes UTF-8 bytes
├── data/
│   └── download_flores.py downloads the FLORES+ parallel corpus
├── experiments/           training and measurement scripts (in progress)
└── results/               charts and numbers (in progress)
```
 
## Running the code
 
Requires Python 3.
 
```
cd bpe
py bpe.py          # character-level BPE on a toy corpus
py bpe_bytes.py    # byte-level BPE on English and Hindi examples
py bpe_text.py     # encoding whole sentences
```
 
To download the corpus (requires `pip install datasets` and a Hugging Face account that has accepted the FLORES+ terms):
 
```
cd data
py download_flores.py
```
 
## Data
 
Experiments use **FLORES+** (`openlanguagedata/flores_plus` on Hugging Face): the same sentences professionally translated into 200+ languages. We use English (`eng_Latn`), Hindi (`hin_Deva`) and Kannada (`kan_Knda`), training on the `dev` split and evaluating on `devtest`. As requested by the dataset authors, the downloaded files are not committed to this repository.
 
## Progress
 
- [x] Character-level BPE: training, encoding, decoding
- [x] Token IDs and the round-trip property
- [x] Byte-level BPE, working for any script
- [x] Toy experiments on Hindi: fragment tokens and character-boundary violations
- [x] Parity and the shared merge budget
- [x] Splitting running text and encoding whole sentences
- [ ] Measurements on FLORES+ for English, Hindi and Kannada
- [ ] Multi-objective formulation and experiments
## References
 
- P. Gage. A new algorithm for data compression. *C Users Journal*, 1994.
- R. Sennrich, B. Haddow and A. Birch. Neural machine translation of rare words with subword units. *ACL*, 2016.
- A. Petrov et al. Language model tokenizers introduce unfairness between languages. *NeurIPS*, 2023.
- V. Zouhar et al. A formal perspective on byte-pair encoding. *Findings of ACL*, 2023.
- J. P. Lim, D. Choo and H. W. Lauw. A partition cover approach to tokenization. *NeurIPS*, 2025.
- H. Foroutan et al. Parity-aware byte-pair encoding: improving cross-lingual fairness in tokenization. *ACL*, 2026.
- NLLB Team. No Language Left Behind (FLORES-200), 2022.