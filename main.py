# check_dataset.py — RDD2022
from pathlib import Path
from collections import Counter

base = Path(r"C:\kursach\rdd2022")

SPLITS = {"train": "train", "val": "val", "test": "test"}

for label, folder in SPLITS.items():
    imgs = list((base / folder / "images").glob("*.*"))
    lbls = list((base / folder / "labels").glob("*.txt"))
    print(f"{label}: {len(imgs)} изображений, {len(lbls)} файлов разметки")

names = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "other_corruption",
    4: "pothole",
}

counter = Counter()
for txt in (base / "train" / "labels").glob("*.txt"):
    for line in txt.read_text().splitlines():
        if line.strip():
            counter[int(line.split()[0])] += 1

print("\nРаспределение классов в train:")
total = sum(counter.values())
for cls, count in sorted(counter.items()):
    pct = count / total * 100
    print(f"  [{cls}] {names.get(cls, '?'):<22}: {count:>6} bbox  ({pct:.1f}%)")
print(f"  {'ИТОГО':<24}: {total:>6} bbox")