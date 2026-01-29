import os
from typing import List
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError, Timeout
from dotenv import load_dotenv

try:
    from tiktoken import get_encoding

    Has_tiktoken = True
except Exception:
    Has_tiktoken = False
    get_encoding = None


# -------- util: backoff para embeddings --------
def _embed_batch_with_retry(
    client: OpenAI, model: str, batch: List[str], dims: int, max_retries: int = 5
):
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return client.embeddings.create(model=model, input=batch, dimensions=dims)
        except (RateLimitError, APIError, APIConnectionError, Timeout) as e:
            print(f"Working on: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 16.0)


def build_embeddings(texts: List[str], sentiment: str):
    """
    Convierte una lista de textos en embeddings y guarda tanto los vectores
    como la metadata asociada (texto limpio, tokens, índice).
    """
    load_dotenv()
    client = OpenAI()

    MODEL = "text-embedding-3-small"
    DIMENSIONS = 512
    BATCH_SIZE = 100

    os.makedirs(sentiment, exist_ok=True)

    # === Generar embeddings por lotes ===
    embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Embedding {sentiment}"):
        batch = texts[i : i + BATCH_SIZE]
        resp = _embed_batch_with_retry(client, MODEL, batch, DIMENSIONS)
        batch_emb = [d.embedding for d in resp.data]
        embeddings.extend(batch_emb)

    V = np.array(embeddings, dtype=np.float32)
    np.save(f"{sentiment}/embeddings_raw_{sentiment}.npy", V)

    # === Crear metadata ===
    text_clean = [t.strip().replace("\n", " ") for t in texts]

    if Has_tiktoken:
        enc = get_encoding("cl100k_base")
        n_tokens = [len(enc.encode(t)) for t in text_clean]
    else:
        # fallback simple (aprox) si no hay tiktoken
        n_tokens = [len(t.split()) for t in text_clean]

    meta = pd.DataFrame(
        {
            "id": range(len(texts)),
            "text": texts,
            "text_clean": text_clean,
            "n_tokens": n_tokens,
            "embedding_index": range(len(texts)),
        }
    )
    meta.to_csv(f"{sentiment}/metadata.csv", index=False)

    print(
        f"Guardados {V.shape[0]} embeddings de {V.shape[1]} dimensiones en '{sentiment}/'"
    )
    print(f"Archivos generados:\n - embeddings_raw_{sentiment}.npy\n - metadata.csv")


def build_S_matrix(sentiment: str):
    # Cargar embeddings sin normalizar
    v = np.load(f"{sentiment}/embeddings_raw_{sentiment}.npy")

    # Normalización L2 (cada fila tiene longitud 1) con epsilon
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v = v / norms

    # Convertir a float16 para ahorrar espacio
    v = v.astype(np.float16)
    np.save(f"{sentiment}/embeddings_norm_{sentiment}.npy", v)

    # Producto punto: v · v^T  → matriz de similitud coseno
    S = np.matmul(v, v.T).astype(np.float16)
    np.save(f"{sentiment}/matrix_S_{sentiment}.npy", S)

    print(f"[{sentiment}] Matriz S guardada con forma: {S.shape}")


def _read_labelled_txt(path: str) -> List[str]:
    """
    Lee archivos 'texto\\tlabel' tolerando líneas mal formadas.
    Devuelve solo la lista de textos.
    """
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 1:
                # si no hay tab, toma todo como texto
                text = parts[0]
            else:
                text = parts[0]
            if text:
                texts.append(text)
    return texts


def process_embeddings():
    sentiments = ["anger", "fear", "joy", "sadness", "surprise"]
    for sentiment in sentiments:
        # NO crear aquí; se crea dentro de build_embeddings con exist_ok=True
        # os.mkdir(sentiment)

        path = f"eng_train_{sentiment}.txt"
        texts = _read_labelled_txt(path)
        if not texts:
            print(f"[WARN] No se encontraron textos en {path}")
            continue

        # Crear embeddings
        build_embeddings(texts=texts, sentiment=sentiment)
        # Construir matriz S
        build_S_matrix(sentiment=sentiment)


if __name__ == "__main__":
    process_embeddings()
