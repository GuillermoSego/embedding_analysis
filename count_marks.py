"""
Contabiliza textos con signos ('!' y '?') por emoción: train, test y hard-100.
Salida: tabla en stdout y CSV (marks_counts.csv).

Eliminados = textos del hard-100 original (con marcas) que contienen '!' y por tanto
desaparecen de hard_{emocion}_without_marks.txt (se reponen con los siguientes más difíciles).
"""

import argparse
import numpy as np
import pandas as pd

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]


def read_texts(path: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"], quoting=3, dtype=str)
    return pd.Series(df["text"]).fillna("")


def n_with(texts: pd.Series, mark: str) -> int:
    return int(texts.str.contains(mark, regex=False).sum())


def count_emotion(sentiment: str, hard_dir: str, test_dir: str, threshold: float) -> dict:
    meta = pd.read_csv(f"sentiments/{sentiment}/metadata.csv")
    scores = np.load(f"eval_src_scores/{sentiment}/eval_src_scores.npy")
    train = meta["text"].astype(str)
    pool = train[(scores > 0.001) & (scores < threshold)]

    hard = read_texts(f"{hard_dir}/hard_{sentiment}.txt")
    hard_nm = read_texts(f"{hard_dir}/hard_{sentiment}_without_marks.txt")
    test = read_texts(f"{test_dir}/eng_test_{sentiment}.txt")
    removed = set(hard) - set(hard_nm)

    return {
        "emotion": sentiment,
        "train_n": len(train),
        "train_bang": n_with(train, "!"),
        "train_q": n_with(train, "?"),
        "pool_hard_n": len(pool),
        "pool_hard_bang": n_with(pool, "!"),
        "pool_hard_q": n_with(pool, "?"),
        "hard100_bang": n_with(hard, "!"),
        "hard100_q": n_with(hard, "?"),
        "hard100_removed": len(removed),
        "hard100_nomarks_bang": n_with(hard_nm, "!"),
        "test_n": len(test),
        "test_bang": n_with(test, "!"),
        "test_q": n_with(test, "?"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-dir", default="hard_texts")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("-o", "--output", default="marks_counts.csv")
    args = parser.parse_args()

    df = pd.DataFrame(
        [count_emotion(s, args.hard_dir, args.test_dir, args.threshold) for s in SENTIMENTS]
    )
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
