"""
Exporta el conjunto "hardness x signos de interrogación" por emoción.

    hardness = 1 - score          (score de eval_src_scores; bajo = difícil)
    combined = hardness * (1 + alpha * n_question_marks)

Los textos con '?' reciben un impulso multiplicativo; los demás se ordenan por hardness
puro, de modo que el top-N se completa aun cuando hay pocos textos con '?'
(en train solo ~61 textos tienen '?', máx. 1 cada uno).

Salida: hard_texts/hard_{emocion}_qmarks[_without_marks].txt  (text<TAB>label)
"""

import argparse
import os
import numpy as np
import pandas as pd

from analyze_hard_texts import load_train_labels

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]


def export(sentiment: str, n: int, alpha: float, exclude_bang: bool, out_dir: str) -> None:
    meta = pd.read_csv(f"sentiments/{sentiment}/metadata.csv")
    scores = np.load(f"eval_src_scores/{sentiment}/eval_src_scores.npy")

    df = meta[scores > 0.001].copy()  # score ~0 = texto no evaluado
    df["score"] = scores[scores > 0.001]
    df["text"] = df["text"].astype(str)
    if exclude_bang:
        df = df[~df["text"].str.contains("!", regex=False)]

    df["n_q"] = df["text"].str.count(r"\?")
    df["hardness"] = 1.0 - df["score"]
    df["combined"] = df["hardness"] * (1.0 + alpha * df["n_q"])
    top = df.sort_values("combined", ascending=False).head(n)

    labels = load_train_labels(sentiment)

    os.makedirs(out_dir, exist_ok=True)
    suffix = "_without_marks" if exclude_bang else ""
    path = os.path.join(out_dir, f"hard_{sentiment}_qmarks{suffix}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for idx, text in zip(top.index, top["text"]):
            f.write(f"{text}\t{labels[idx]}\n")
    print(
        f"{sentiment:<9} -> {path}  n={len(top)}  con '?'={int((top['n_q'] > 0).sum())}  "
        f"con '!'={int(top['text'].str.contains('!', regex=False).sum())}  "
        f"score medio={top['score'].mean():.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0, help="peso del '?' (default 1.0)")
    parser.add_argument(
        "--keep-bang",
        action="store_true",
        help="no excluir textos con '!' (por defecto se excluyen para aislar el efecto de '?')",
    )
    parser.add_argument("--export-dir", default="hard_texts")
    args = parser.parse_args()

    for s in SENTIMENTS:
        export(s, args.top_n, args.alpha, not args.keep_bang, args.export_dir)


if __name__ == "__main__":
    main()
