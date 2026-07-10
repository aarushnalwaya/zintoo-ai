"""
Step 1 — Prepare the Kaggle Fashion Product Images dataset.

Real-world quirks in this dataset that this script handles explicitly:
  * `styles.csv` has malformed rows (unescaped commas inside
    `productDisplayName`) -> pandas must skip bad lines, not crash.
  * Some `id`s in the CSV have **no corresponding image file**. Training on
    those produces a silent crash mid-epoch. We validate existence up front.
  * `articleType` has a savage long tail: ~140 classes, many with <10 images.
    Training on a class with 3 examples teaches nothing and wrecks macro-F1.
    We drop classes below a threshold and *report the coverage we kept*.
  * `baseColour` / `season` / `usage` contain NaNs.

Outputs:
  data/manifests/{train,val,test}.csv   -> image_path + integer label per task
  data/manifests/label_maps.json        -> task -> [class names] (index = label)

Run:
    ZINTOO_DATASET_DIR=/path/to/fashion-dataset python -m ml.prepare_data
"""

from __future__ import annotations

import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from ml import config


def load_styles(styles_csv, images_dir, require_images: bool = True) -> pd.DataFrame:
    if not styles_csv.exists():
        sys.exit(
            f"❌ styles.csv not found at {styles_csv}\n"
            f"   Download the dataset and set ZINTOO_DATASET_DIR. See VISION.md."
        )

    # engine='python' + on_bad_lines='skip' survives the malformed rows.
    df = pd.read_csv(styles_csv, on_bad_lines="skip", engine="python")
    print(f"  loaded {len(df):,} rows from styles.csv")

    needed = ["id", *config.TASKS]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"❌ styles.csv missing expected columns: {missing}")

    before = len(df)
    df = df.dropna(subset=needed)
    print(f"  dropped {before - len(df):,} rows with null labels -> {len(df):,}")

    df["id"] = df["id"].astype(int)
    df["image_path"] = df["id"].map(lambda i: str(images_dir / f"{i}.jpg"))

    if require_images:
        before = len(df)
        # Path.exists() per row is slow on 44k files but runs once. Use a set.
        present = {p.stem for p in images_dir.glob("*.jpg")}
        df = df[df["id"].astype(str).isin(present)]
        print(f"  dropped {before - len(df):,} rows with no image file -> {len(df):,}")

    if df.empty:
        sys.exit("❌ No usable rows after cleaning. Check the dataset layout.")
    return df


def filter_rare_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop long-tail classes per task; report retained coverage."""
    report = {}
    for task in config.TASKS:
        min_n = config.MIN_SAMPLES_PER_CLASS[task]
        counts = df[task].value_counts()
        keep = set(counts[counts >= min_n].index)
        dropped_classes = len(counts) - len(keep)
        before = len(df)
        df = df[df[task].isin(keep)]
        report[task] = {
            "classes_before": int(len(counts)),
            "classes_kept": int(len(keep)),
            "classes_dropped": int(dropped_classes),
            "rows_dropped": int(before - len(df)),
            "min_samples": min_n,
        }
        print(
            f"  {task:<16} {len(counts):>3} -> {len(keep):>3} classes "
            f"(dropped {dropped_classes} rare, {before - len(df):,} rows)"
        )
    return df, report


def build_label_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    return {task: sorted(df[task].unique().tolist()) for task in config.TASKS}


def encode(df: pd.DataFrame, label_maps: dict[str, list[str]]) -> pd.DataFrame:
    out = df[["id", "image_path"]].copy()
    for task, classes in label_maps.items():
        lookup = {c: i for i, c in enumerate(classes)}
        out[task] = df[task].map(lookup).astype(int)
    return out


def stratified_split(df: pd.DataFrame):
    """Stratify on the primary task so rare-ish classes appear in every split."""
    strat = df[config.PRIMARY_TASK]
    train, tmp = train_test_split(
        df, test_size=config.VAL_FRAC + config.TEST_FRAC,
        stratify=strat, random_state=config.SEED,
    )
    rel = config.TEST_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val, test = train_test_split(
        tmp, test_size=rel, stratify=tmp[config.PRIMARY_TASK], random_state=config.SEED,
    )
    return train, val, test


def main(require_images: bool = True) -> None:
    print("=" * 62)
    print("STEP 1 — Preparing dataset")
    print("=" * 62)

    df = load_styles(config.STYLES_CSV, config.IMAGES_DIR, require_images)
    print("\nFiltering long-tail classes:")
    df, report = filter_rare_classes(df)

    label_maps = build_label_maps(df)
    encoded = encode(df, label_maps)
    train, val, test = stratified_split(encoded)

    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = config.MANIFEST_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"\n  {name:<5} {len(part):>7,} rows -> {path}")

    (config.MANIFEST_DIR / "label_maps.json").write_text(json.dumps(label_maps, indent=2))
    (config.MANIFEST_DIR / "prep_report.json").write_text(json.dumps(report, indent=2))

    print("\nClasses per task:")
    for task, classes in label_maps.items():
        print(f"  {task:<16} {len(classes):>3}")
    print(f"\n✅ Manifests written to {config.MANIFEST_DIR}")


if __name__ == "__main__":
    main()
