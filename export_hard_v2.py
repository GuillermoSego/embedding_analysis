"""
Conjuntos de entrenamiento v2 (C1/C2/C4), a partir de los scores v2 (pool de 30 prompts, promedio simple).

  score  = fracción de los 30 prompts que aciertan el texto (eval_src_scores_v2/{e}/eval_src_scores.npy)
  hard   = los n/2 textos de menor score de cada clase (positivos y negativos), con los score 0 INCLUIDOS.
           Balanceado por clase: de otro modo el hard-100 es ~90% positivos (falsos negativos de todos
           los prompts) y el F1 del dev casi no mide nada. Desempates: sorteo con semilla fija.
  hard_without_marks = igual, tras excluir textos con '!'.
  qmarks = igual a without_marks pero ordenando por (1-score)*(1+alpha*n_'?').
  random = n/2 por clase al azar (control con el mismo balance).

Salida: hard_texts_v2/{hardv2_,randomv2_}{e}[_qmarks][_without_marks].txt (text<TAB>label) + MANIFEST.csv
con el sha256 de cada archivo.
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]


def load_train(sentiment):
    with open(f"data/eng_train_{sentiment}.txt", encoding="utf-8") as f:
        rows = [line.rsplit("\t", 1) for line in f.read().split("\n") if line]
    return pd.DataFrame({"text": [r[0] for r in rows], "label": [int(r[1]) for r in rows]})


def pick(df, key, n, rng):
    """n/2 por clase con menor `key` (desempate aleatorio)."""
    out = []
    for lab in (0, 1):
        sub = df[df["label"] == lab]
        order = np.lexsort((rng.random(len(sub)), sub[key].to_numpy()))
        out.append(sub.iloc[order[: n // 2]])
    return pd.concat(out)


def write(sub, path, manifest, name, sentiment):
    with open(path, "w", encoding="utf-8") as f:
        for t, l in zip(sub["text"], sub["label"]):
            f.write(f"{t}\t{l}\n")
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest.append(
        dict(file=os.path.basename(path), emotion=sentiment, set=name, n=len(sub), pos=int(sub["label"].sum()),
             bang=int(sub["text"].str.contains("!", regex=False).sum()),
             qmark=int(sub["text"].str.contains("?", regex=False).sum()),
             mean_score=round(float(sub["score"].mean()), 4), sha256=sha)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="hard_texts_v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = []
    for s in SENTIMENTS:
        df = load_train(s)
        df["score"] = np.load(f"eval_src_scores_v2/{s}/eval_src_scores.npy")
        assert len(df) == len(df["score"])
        df["bang"] = df["text"].str.contains("!", regex=False)
        df["nq"] = df["text"].str.count(r"\?")
        df["hardness_q"] = -(1 - df["score"]) * (1 + args.alpha * df["nq"])  # menor = más difícil
        rng = np.random.default_rng(args.seed)
        nb = df[~df["bang"]]
        sets = {
            f"hardv2_{s}": pick(df, "score", args.n, rng),
            f"hardv2_{s}_without_marks": pick(nb, "score", args.n, rng),
            f"hardv2_{s}_qmarks_without_marks": pick(nb, "hardness_q", args.n, rng),
            f"randomv2_{s}": pick(df.assign(r=rng.random(len(df))), "r", args.n, rng),
        }
        for name, sub in sets.items():
            write(sub, os.path.join(args.out, name + ".txt"), manifest, name, s)
    m = pd.DataFrame(manifest)
    m.to_csv(os.path.join(args.out, "MANIFEST.csv"), index=False)
    print(m.drop(columns="sha256").to_string(index=False))


if __name__ == "__main__":
    main()
