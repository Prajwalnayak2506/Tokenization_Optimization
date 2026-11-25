# Multi-Objective Tokenization for Indian Languages

Course project for **CS 769: Optimization in Machine Learning**

Prajwal Nayak (22B4246)

## Overview

Language models read text as *tokens*, and the tokenizer decides how many tokens a piece of text costs. The standard method, **Byte Pair Encoding (BPE)**, is a greedy algorithm that optimizes a single objective: compressing the training text into as few tokens as possible. Because it always merges the most frequent pair, the language with the most training data wins most of the merges.

Real tokenizers are trained mostly on English. With 200 merges learned from English, the same sentences need about **5.0× more tokens in Hindi and 5.6× more in Kannada**. This project asks whether that can be fixed by changing **how BPE chooses merges**, while keeping the imbalanced training data.

It:

1. implements BPE from scratch, first on characters and then on bytes, so that any script can be encoded;
2. measures compression, cross-lingual **parity** and **fertility** on English, Hindi and Kannada with the FLORES+ parallel corpus;
3. derives merge rules from explicit fairness objectives, using a first-order Taylor approximation: each language's saving is weighted by the derivative of the objective;
4. combines compression with a symmetric min-max fairness objective, giving one knob `c` that trades English compression for fairness.

## Main results

All results use 200 merges, train on FLORES+ `dev` with all English sentences plus only 5% or 10% of the Hindi and Kannada sentences, and are measured on the unseen `devtest` split.

- **More data helps, with diminishing returns:** with plain BPE, the first 5% of Hindi/Kannada data lowers Kannada parity from 5.6 to 2.3.
- **Weighted rules reach the same trade-off with far less data:** with 5% data, the current-size rule (Case 2) reaches parity about 1.16, which plain BPE needs roughly 90% of the data for. All rules lie on the same trade-off curve as plain BPE given more data; they match it, not beat it.
- **One knob controls the trade-off:** `c = 0` is plain BPE; increasing `c` moves smoothly towards fairness, and the symmetric objective stops near parity 1 instead of overshooting. Near-perfect fairness costs English about 45–50% more tokens than English-only BPE.
- **Fertility:** total fertility (all tokens / all words) falls from 5.94 for plain BPE on 5% data to about 4.6–4.7 for every weighted rule, close to plain BPE on balanced data (4.60). Per word, Kannada stays higher than English because it is agglutinative and uses fewer, longer words.

## Repository structure

```
.
├── README.md
├── bpe/                       tokenizer code
│   ├── bpe.py                 character-level BPE: training, encoding, decoding, token IDs
│   ├── explore_bytes.py       how text becomes UTF-8 bytes
│   ├── bpe_bytes.py           byte-level BPE: works for any language and script
│   └── bpe_text.py            running text: splitting sentences, building a corpus
├── data/
│   └── download_flores.py     downloads the FLORES+ parallel corpus (files not committed)
├── experiments/
│   ├── measure.py             loading data, counting tokens, labelling merges by script
│   ├── mix.py                 plain BPE with different shares of Hindi/Kannada data
│   ├── analyze_merges.py      which merges each language receives, and how much they save
│   ├── fair_bpe.py            weighted merge rules (Cases 1–3)
│   ├── minmax_bpe.py          min-max fairness plus compression
│   ├── sym_bpe.py             symmetric min-max fairness
│   └── sweep.py               full sweep over the fairness weight c
├── plotters/
│   ├── plot1.py               early trade-off plot
│   ├── plot2.py               weighted rules against plain BPE
│   └── plot_all.py            all final figures, including fertility
└── results/                   saved merges (pickle), results (JSON) and figures
```

## Running the code

Requires Python 3 and `matplotlib`.

**Tokenizer demos:**

```
cd bpe
py bpe.py          # character-level BPE on a toy corpus
py bpe_bytes.py    # byte-level BPE on English and Hindi examples
py bpe_text.py     # encoding whole sentences
```

**Data:** requires `pip install datasets` and a Hugging Face account that has accepted the FLORES+ terms.

```
cd data
py download_flores.py
```

**Experiments and figures**, from the project folder:

```
py experiments\measure.py      # English-only vs all-language training
py experiments\mix.py          # plain BPE with 5%, 10%, 25%, 50% Hindi/Kannada data
py experiments\fair_bpe.py     # weighted rules (Cases 2 and 3)
py experiments\sweep.py        # full sweep over c (about 15 minutes)
py plotters\plot_all.py        # final figures in results/
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
- [x] Measurements on FLORES+ for English, Hindi and Kannada
- [x] Merge rules derived from fairness objectives (Cases 1–3)
- [x] Min-max and symmetric min-max fairness combined with compression
- [x] Full sweep over the fairness weight, with parity and fertility
- [ ] Direct comparison with Parity-aware BPE
- [ ] Larger scale: more merges and more data
- [ ] Optimizing integrity (tokens that split letters or syllables)

## References

- P. Gage. A new algorithm for data compression. *C Users Journal*, 1994.
- R. Sennrich, B. Haddow and A. Birch. Neural machine translation of rare words with subword units. *ACL*, 2016.
- A. Petrov et al. Language model tokenizers introduce unfairness between languages. *NeurIPS*, 2023.
- V. Zouhar et al. A formal perspective on byte-pair encoding. *Findings of ACL*, 2023.
- J. P. Lim, D. Choo and H. W. Lauw. A partition cover approach to tokenization. *NeurIPS*, 2025.
- H. Foroutan et al. Parity-aware byte-pair encoding: improving cross-lingual fairness in tokenization. *ACL*, 2026.
- F. Kelly, A. Maulloo and D. Tan. Rate control for communication networks: shadow prices, proportional fairness and stability. *Journal of the Operational Research Society*, 1998.
- NLLB Team. No Language Left Behind: scaling human-centered machine translation. arXiv:2207.04672, 2022.
- FLORES+ dataset, Open Language Data Initiative: `openlanguagedata/flores_plus` on Hugging Face.
