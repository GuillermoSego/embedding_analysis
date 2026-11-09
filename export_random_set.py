"""
Exporta un conjunto aleatorio FIJO de N textos de train por emoción (control de hard-100).

Los índices se sortean una sola vez con una semilla fija y se aplican a las 5 emociones, de modo
que los textos son los mismos y solo cambian las etiquetas (los train de las 5 emociones son
casi el mismo texto fila a fila; si alguna fila difiere, se avisa).

Salida: random_texts/random_{emocion}.txt  (text<TAB>label)
"""

import argparse
import os
import numpy as np
import pandas as pd

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--export-dir", default="random_texts")
    args = parser.parse_args()

    # Lectura cruda: pd.read_csv con comillas por defecto pierde filas (ver analyze_hard_texts.load_train_labels)
    frames = {}
    for s_ in SENTIMENTS:
        with open(f"data/eng_train_{s_}.txt", encoding="utf-8") as f:
            rows = [line.rsplit("\t", 1) for line in f.read().split("\n") if line]
        frames[s_] = pd.DataFrame(rows, columns=["text", "label"]).astype({"label": int})
    n_rows = min(len(df) for df in frames.values())
    idx = np.sort(np.random.default_rng(args.seed).choice(n_rows, size=args.n, replace=False))

    os.makedirs(args.export_dir, exist_ok=True)
    ref = frames[SENTIMENTS[0]]["text"].iloc[idx].tolist()
    for s in SENTIMENTS:
        sub = frames[s].iloc[idx]
        diff = sum(a != b for a, b in zip(sub["text"], ref))
        path = os.path.join(args.export_dir, f"random_{s}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for text, label in zip(sub["text"], sub["label"]):
                f.write(f"{text}\t{label}\n")
        print(
            f"{s:<9} -> {path}  n={len(sub)}  positivos={int(sub['label'].sum())}  "
            f"con '!'={int(sub['text'].str.contains('!', regex=False).sum())}  "
            f"con '?'={int(sub['text'].str.contains('?', regex=False).sum())}  "
            f"textos distintos al de {SENTIMENTS[0]}={diff}"
        )


if __name__ == "__main__":
    main()
