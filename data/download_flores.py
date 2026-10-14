from datasets import load_dataset

for code in ["eng_Latn", "hin_Deva", "kan_Knda"]:
    for split in ["dev", "devtest"]:
        ds = load_dataset("openlanguagedata/flores_plus", code, split=split)
        rows = sorted(ds, key=lambda r: int(r["id"]))
        with open(f"{code}_{split}.txt", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(r["text"].replace("\n", " ") + "\n")
        print(code, split, len(rows), "sentences")