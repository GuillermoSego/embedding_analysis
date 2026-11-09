"""
K-means cluster analysis of sentiment embeddings.
Clusters in PCA-reduced space for quality, visualizes in PCA-2 space.
Identifies and analyzes clusters of texts that are harder for the model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import Counter
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]

COLORS = {
    "anger": "#e63946",
    "fear": "#6a4c93",
    "joy": "#f4a261",
    "sadness": "#457b9d",
    "surprise": "#2a9d8f",
}

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    "would", "could", "might", "must", "shall", "get", "got", "like", "one",
    "also", "even", "really", "much", "back", "go", "went", "felt", "feel",
    "know", "think", "thought", "time", "day", "said", "told", "made", "make",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(sentiment: str, normalized: bool = True) -> np.ndarray:
    """Load embeddings for a sentiment. Shape: (2768, 512), float32."""
    suffix = "norm" if normalized else "raw"
    path = f"sentiments/{sentiment}/embeddings_{suffix}_{sentiment}.npy"
    return np.load(path).astype(np.float32)


def load_eval_scores(sentiment: str) -> np.ndarray:
    """Load evaluation scores. Shape: (2768,), float32. 0 = not evaluated."""
    path = f"eval_src_scores/{sentiment}/eval_src_scores.npy"
    return np.load(path).astype(np.float32)


def load_metadata(sentiment: str) -> pd.DataFrame:
    """Load metadata CSV. Columns: id, text, text_clean, n_tokens, embedding_index."""
    return pd.read_csv(f"sentiments/{sentiment}/metadata.csv")


# ---------------------------------------------------------------------------
# TF-IDF representation
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    texts: list[str],
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
) -> tuple[np.ndarray, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
    )
    tfidf_sparse = vectorizer.fit_transform(texts)
    return tfidf_sparse.toarray().astype(np.float32), vectorizer


def get_top_tfidf_terms_per_cluster(
    tfidf_matrix: np.ndarray,
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    n_terms: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    feature_names = vectorizer.get_feature_names_out()
    cluster_ids = sorted(np.unique(labels))
    result = {}
    for cid in cluster_ids:
        mask = labels == cid
        mean_tfidf = tfidf_matrix[mask].mean(axis=0)
        top_indices = mean_tfidf.argsort()[::-1][:n_terms]
        result[cid] = [(feature_names[i], float(mean_tfidf[i])) for i in top_indices]
    return result


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_embeddings(
    embeddings: np.ndarray, n_components: int = 50
) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA on embeddings for clustering.

    Returns:
        proj: (N, n_components) float32 array
        pca: fitted PCA object
    """
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    proj = pca.fit_transform(embeddings)
    return proj.astype(np.float32), pca


def project_2d(embeddings: np.ndarray) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA(2) on raw 512-dim embeddings for visualization only.

    Always fitted on the original embedding space (NOT on PCA-reduced data)
    so the 2D projection captures maximum variance from the full space.

    Returns:
        proj2d: (N, 2) float32 array
        pca2d: fitted PCA object
    """
    pca2d = PCA(n_components=2, random_state=42)
    proj2d = pca2d.fit_transform(embeddings)
    return proj2d.astype(np.float32), pca2d


# ---------------------------------------------------------------------------
# Optimal K selection
# ---------------------------------------------------------------------------

def find_optimal_k(
    proj: np.ndarray,
    k_range: range = range(2, 13),
    n_init: int = 10,
    random_state: int = 42,
) -> tuple[list[float], list[float], int]:
    """
    Compute inertia and silhouette scores for a range of k values.

    Returns:
        inertias: list of inertia values (one per k)
        silhouettes: list of silhouette scores (one per k)
        best_k: k with highest silhouette score

    Note:
        Silhouette scores are typically low (~0.03-0.05) for dense
        semantic embedding spaces. This is normal behavior.
    """
    inertias = []
    silhouettes = []
    sample_size = min(500, len(proj))

    print(f"  Searching optimal k in range {list(k_range)[0]}-{list(k_range)[-1]}...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(proj)
        inertias.append(km.inertia_)
        sil = silhouette_score(proj, labels, sample_size=sample_size, random_state=random_state)
        silhouettes.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    best_sil = max(silhouettes)
    print(f"  Best k={best_k} (silhouette={best_sil:.4f})")

    if best_sil < 0.1:
        print(
            f"  Note: max silhouette score is {best_sil:.3f}, suggesting embeddings "
            "are not strongly clustered. Results may vary with different k values."
        )

    return inertias, silhouettes, best_k


# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------

def run_kmeans(
    proj: np.ndarray, k: int, n_init: int = 10, random_state: int = 42
) -> tuple[np.ndarray, KMeans]:
    """
    Run K-means on PCA-reduced embeddings.

    Returns:
        labels: (N,) integer cluster assignment array
        km: fitted KMeans object
    """
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(proj)
    return labels, km


# ---------------------------------------------------------------------------
# Per-cluster analysis
# ---------------------------------------------------------------------------

def _extract_words(texts: list[str]) -> list[str]:
    """Tokenize texts to lowercase alphabetic words."""
    words = []
    for text in texts:
        words.extend(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return words


def analyze_cluster(
    cluster_id: int,
    labels: np.ndarray,
    scores: np.ndarray,
    meta: pd.DataFrame,
    threshold: float = 0.5,
    n_top_words: int = 15,
    n_examples: int = 5,
) -> dict:
    """
    Compute statistics for a single cluster.

    Texts with score == 0 (not evaluated) are excluded from score statistics,
    word analysis, and example selection.

    Returns dict with keys:
        cluster_id, n_total, n_evaluated, score_mean, score_std,
        score_min, score_max, is_hard, hard_count, top_words, hardest_examples
    """
    mask_cluster = labels == cluster_id
    cluster_scores = scores[mask_cluster]
    cluster_meta = meta[mask_cluster].copy().reset_index(drop=True)

    # Exclude unevaluated texts (score == 0)
    evaluated_mask = cluster_scores > 0.001
    eval_scores = cluster_scores[evaluated_mask]
    eval_meta = cluster_meta[evaluated_mask].copy().reset_index(drop=True)

    n_total = int(mask_cluster.sum())
    n_evaluated = int(evaluated_mask.sum())

    if n_evaluated > 0:
        score_mean = float(eval_scores.mean())
        score_std = float(eval_scores.std())
        score_min = float(eval_scores.min())
        score_max = float(eval_scores.max())
        is_hard = score_mean < threshold
        hard_count = int((eval_scores < threshold).sum())
    else:
        score_mean = score_std = score_min = score_max = float("nan")
        is_hard = False
        hard_count = 0

    # Word frequency on evaluated texts only
    eval_meta["_score"] = eval_scores
    texts = eval_meta["text_clean"].tolist()
    words = _extract_words(texts)
    word_freq = Counter(w for w in words if w not in STOPWORDS and len(w) > 2)
    top_words = word_freq.most_common(n_top_words)

    # Hardest examples: lowest scored evaluated texts
    hardest = eval_meta.nsmallest(n_examples, "_score")[["text_clean", "_score"]].values.tolist()

    return {
        "cluster_id": cluster_id,
        "n_total": n_total,
        "n_evaluated": n_evaluated,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_min": score_min,
        "score_max": score_max,
        "is_hard": is_hard,
        "hard_count": hard_count,
        "top_words": top_words,
        "hardest_examples": hardest,
    }


def analyze_all_clusters(
    labels: np.ndarray,
    scores: np.ndarray,
    meta: pd.DataFrame,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Analyze every cluster. Returns list sorted by score_mean ascending (hardest first).
    Clusters with no evaluated texts (NaN mean) are placed last.
    """
    cluster_ids = sorted(np.unique(labels))
    results = [analyze_cluster(cid, labels, scores, meta, threshold) for cid in cluster_ids]
    results.sort(
        key=lambda r: r["score_mean"] if not np.isnan(r["score_mean"]) else float("inf")
    )
    return results


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_cluster_report(
    cluster_stats: list[dict],
    sentiment: str,
    k: int,
    threshold: float,
    pca_dims: int,
    pca_variance: float,
    best_sil: float | None = None,
    representation: str = "embeddings",
    tfidf_terms: dict[int, list[tuple[str, float]]] | None = None,
):
    """Print formatted cluster analysis to stdout."""
    print(f"\n{'#' * 70}")
    print(f"  CLUSTER ANALYSIS — {sentiment.upper()} ({representation.upper()})")
    print(f"  k={k}  |  pca_dims={pca_dims}  |  threshold={threshold}")
    print(f"  PCA variance explained: {pca_variance:.1%}")
    if best_sil is not None:
        print(f"  Silhouette score: {best_sil:.4f}")
        if best_sil < 0.1:
            print(f"  ⚠ Low silhouette — clusters overlap significantly in embedding space")
    print(f"{'#' * 70}")

    hard_clusters = [c for c in cluster_stats if c["is_hard"]]
    n_hard = len(hard_clusters)
    print(f"\n  Hard clusters (mean score < {threshold}): {n_hard} / {k}")

    for c in cluster_stats:
        cid = c["cluster_id"]
        flag = " [HARD]" if c["is_hard"] else ""
        print(f"\n{'=' * 70}")
        print(f"  CLUSTER {cid}{flag}")
        print(f"{'=' * 70}")
        print(f"  Size: {c['n_total']} texts  |  Evaluated: {c['n_evaluated']}")

        if c["n_evaluated"] == 0:
            print("  No evaluated texts in this cluster.")
            continue

        print(f"\n  SCORE DISTRIBUTION:")
        print(f"    Mean: {c['score_mean']:.3f}  |  Std: {c['score_std']:.3f}")
        print(f"    Min:  {c['score_min']:.3f}  |  Max: {c['score_max']:.3f}")
        print(f"    Hard texts (score < {threshold}): {c['hard_count']} / {c['n_evaluated']}")

        if c["is_hard"]:
            print(f"\n  TOP CONTENT WORDS:")
            for word, count in c["top_words"]:
                print(f"    {word:<20} {count:>4}")

            print(f"\n  HARDEST EXAMPLES (lowest scores):")
            for i, (text, score) in enumerate(c["hardest_examples"], 1):
                text_display = text[:80] + "..." if len(text) > 80 else text
                print(f"    {i}. [{score:.3f}] {text_display}")

        if tfidf_terms and cid in tfidf_terms:
            print(f"\n  TOP TF-IDF TERMS:")
            for term, score in tfidf_terms[cid]:
                print(f"    {term:<20} {score:.4f}")

    if n_hard == 0:
        print(f"\n  No hard clusters found at threshold {threshold}.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _get_cluster_colormap(k: int):
    """Return a discrete colormap suitable for k clusters."""
    name = "tab10" if k <= 10 else "tab20"
    return plt.colormaps[name].resampled(k)


def plot_k_selection(
    k_range: range,
    inertias: list[float],
    silhouettes: list[float],
    best_k: int,
    sentiment: str,
    output_path: str | None = None,
):
    """Two-panel figure: inertia (elbow) and silhouette scores."""
    ks = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(ks, inertias, "o-", color=COLORS.get(sentiment, "#333333"), linewidth=2)
    ax1.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"best k={best_k}")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.legend()
    ax1.set_xticks(ks)

    ax2.plot(ks, silhouettes, "o-", color=COLORS.get(sentiment, "#333333"), linewidth=2)
    ax2.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"best k={best_k}")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette Score")
    ax2.legend()
    ax2.set_xticks(ks)

    fig.suptitle(f"K Selection — {sentiment.capitalize()}", fontsize=13)
    plt.tight_layout()

    if output_path:
        stem, ext = os.path.splitext(output_path)
        save_path = f"{stem}_k_selection{ext}"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved k-selection plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_cluster_analysis(
    proj2d: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    km: KMeans,
    pca50: PCA,
    pca2d: PCA,
    cluster_stats: list[dict],
    sentiment: str,
    k: int,
    threshold: float,
    best_sil: float | None = None,
    output_path: str | None = None,
    representation: str = "embeddings",
):
    """
    Three-panel cluster analysis figure:
      Top-left:  2D PCA colored by cluster ID
      Top-right: 2D PCA colored by eval score (RdYlGn)
      Bottom:    Bar chart of mean score per cluster (sorted ascending)
    """
    cmap = _get_cluster_colormap(k)
    cluster_colors = [cmap(i) for i in range(k)]

    # Map each point to its cluster color
    point_colors = np.array([cluster_colors[lbl] for lbl in labels])

    # Project cluster centroids from PCA-50 space → 512-dim → 2D
    centroids_512 = pca50.inverse_transform(km.cluster_centers_)
    centroids_2d = pca2d.transform(centroids_512)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    ax_clusters = fig.add_subplot(gs[0, 0])
    ax_scores = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    var_total = pca2d.explained_variance_ratio_.sum()
    var1 = pca2d.explained_variance_ratio_[0]
    var2 = pca2d.explained_variance_ratio_[1]

    # --- Left panel: cluster colors ---
    ax_clusters.scatter(
        proj2d[:, 0], proj2d[:, 1],
        c=point_colors, alpha=0.5, s=10,
    )
    # Cluster centroids
    for i in range(k):
        ax_clusters.scatter(
            centroids_2d[i, 0], centroids_2d[i, 1],
            marker="X", s=200, c=[cluster_colors[i]],
            edgecolors="black", linewidths=0.8, zorder=5,
        )
        ax_clusters.annotate(
            str(i),
            (centroids_2d[i, 0], centroids_2d[i, 1]),
            fontsize=8, ha="center", va="center", fontweight="bold",
            color="white", zorder=6,
        )

    ax_clusters.set_xlabel(f"PC1 ({var1:.1%})")
    ax_clusters.set_ylabel(f"PC2 ({var2:.1%})")
    ax_clusters.set_title(f"{sentiment.capitalize()} — Clusters (k={k})\n(var: {var_total:.1%})")

    # Simple legend for cluster IDs
    handles = [mpatches.Patch(facecolor=cluster_colors[i], label=f"Cluster {i}") for i in range(k)]
    ax_clusters.legend(handles=handles, loc="best", fontsize=7, ncol=2 if k > 6 else 1)

    # --- Right panel: eval scores ---
    not_evaluated = scores <= 0.001
    evaluated = ~not_evaluated

    if not_evaluated.any():
        ax_scores.scatter(
            proj2d[not_evaluated, 0], proj2d[not_evaluated, 1],
            c="grey", alpha=0.15, s=8, label="Not evaluated",
        )

    sc = ax_scores.scatter(
        proj2d[evaluated, 0], proj2d[evaluated, 1],
        c=scores[evaluated], cmap="RdYlGn",
        alpha=0.7, s=10, vmin=0, vmax=1,
    )
    cbar = fig.colorbar(sc, ax=ax_scores, fraction=0.046, pad=0.04)
    cbar.set_label("Eval score", fontsize=9)

    mean_score = scores[evaluated].mean() if evaluated.any() else 0.0
    ax_scores.set_xlabel(f"PC1 ({var1:.1%})")
    ax_scores.set_ylabel(f"PC2 ({var2:.1%})")
    ax_scores.set_title(
        f"{sentiment.capitalize()} — Eval Scores\n(mean: {mean_score:.3f}  |  Red=Hard, Green=Easy)"
    )

    # --- Bottom panel: bar chart of mean score per cluster ---
    # Sort clusters by mean score ascending (hardest first)
    sorted_stats = sorted(
        cluster_stats,
        key=lambda r: r["score_mean"] if not np.isnan(r["score_mean"]) else float("inf"),
    )
    sorted_cids = [c["cluster_id"] for c in sorted_stats]
    means = [c["score_mean"] for c in sorted_stats]
    stds = [c["score_std"] if not np.isnan(c["score_std"]) else 0 for c in sorted_stats]
    bar_colors = [cluster_colors[cid] for cid in sorted_cids]
    x = np.arange(len(sorted_cids))

    bars = ax_bar.bar(
        x, means, color=bar_colors, edgecolor="black", linewidth=0.6,
        yerr=stds, capsize=4, error_kw={"linewidth": 1},
    )

    # Threshold line
    ax_bar.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
                   alpha=0.8, label=f"Threshold ({threshold})")

    # Annotate hard clusters
    for i, (stat, bar) in enumerate(zip(sorted_stats, bars)):
        if stat["is_hard"]:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                (stat["score_mean"] or 0) + (stat["score_std"] or 0) + 0.02,
                "HARD",
                ha="center", va="bottom", fontsize=8,
                color="red", fontweight="bold",
            )
        if np.isnan(stat["score_mean"]):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, 0.02,
                "No data", ha="center", va="bottom", fontsize=7, color="grey",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"Cluster {cid}" for cid in sorted_cids], rotation=30, ha="right")
    ax_bar.set_ylabel("Mean eval score")
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title("Mean Eval Score per Cluster (sorted ascending — hardest left)")
    ax_bar.legend()

    # Overall title
    sil_info = f"  |  silhouette={best_sil:.4f}" if best_sil is not None else ""
    rep_info = f"  |  repr={representation}" if representation != "embeddings" else ""
    fig.suptitle(
        f"K-means Cluster Analysis — {sentiment.capitalize()}  "
        f"(k={k}, pca_dims={pca50.n_components_}{sil_info}{rep_info})",
        fontsize=13,
    )

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_sentiment(
    sentiment: str,
    k: int | None,
    pca_dims: int,
    threshold: float,
    output_path: str | None,
    k_range: range = range(2, 13),
    n_init: int = 10,
    representation: str = "embeddings",
):
    """Full clustering pipeline for one sentiment."""
    print(f"\nLoading {sentiment} (representation={representation})...")
    scores = load_eval_scores(sentiment)
    meta = load_metadata(sentiment)

    vectorizer = None
    if representation == "tfidf":
        print(f"  Building TF-IDF matrix from text_clean...")
        data_matrix, vectorizer = build_tfidf_matrix(meta["text_clean"].tolist())
        print(f"  TF-IDF matrix: {data_matrix.shape}")
    else:
        data_matrix = load_embeddings(sentiment)
        print(f"  Embeddings: {data_matrix.shape}")

    print(f"  Evaluated: {(scores > 0.001).sum()}/{len(scores)}")

    # PCA for clustering
    print(f"  Reducing to {pca_dims} PCA components...")
    proj50, pca50 = reduce_embeddings(data_matrix, pca_dims)
    pca50_variance = pca50.explained_variance_ratio_.sum()
    print(f"  PCA-{pca_dims} variance explained: {pca50_variance:.1%}")

    # PCA for visualization (2D, on original space)
    proj2d, pca2d = project_2d(data_matrix)

    # Determine k
    best_sil = None
    if k is None:
        inertias, silhouettes, best_k = find_optimal_k(proj50, k_range, n_init)
        best_sil = max(silhouettes)
        plot_k_selection(k_range, inertias, silhouettes, best_k, sentiment, output_path)
        k = best_k
    else:
        print(f"  Using k={k} (specified by user)")

    # K-means
    print(f"  Running K-means (k={k})...")
    labels, km = run_kmeans(proj50, k, n_init)

    # Analyze clusters
    cluster_stats = analyze_all_clusters(labels, scores, meta, threshold)

    # TF-IDF top terms per cluster
    tfidf_terms = None
    if vectorizer is not None:
        tfidf_terms = get_top_tfidf_terms_per_cluster(data_matrix, labels, vectorizer)

    # Console report
    print_cluster_report(
        cluster_stats, sentiment, k, threshold, pca_dims, pca50_variance,
        best_sil, representation, tfidf_terms,
    )

    # Visualization
    plot_cluster_analysis(
        proj2d, labels, scores, km, pca50, pca2d,
        cluster_stats, sentiment, k, threshold, best_sil, output_path,
        representation,
    )


def analyze_all_sentiments(
    k: int | None,
    pca_dims: int,
    threshold: float,
    output_path: str | None,
    k_range: range = range(2, 13),
    n_init: int = 10,
    representation: str = "embeddings",
):
    """Run analyze_sentiment for all 5 sentiments."""
    for sentiment in SENTIMENTS:
        sent_output = None
        if output_path:
            stem, ext = os.path.splitext(output_path)
            sent_output = f"{stem}_{sentiment}{ext}"

        analyze_sentiment(
            sentiment, k, pca_dims, threshold, sent_output, k_range, n_init,
            representation,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="K-means cluster analysis of sentiment embeddings"
    )
    parser.add_argument(
        "--emotion",
        choices=SENTIMENTS + ["all"],
        default="all",
        help="Sentiment to analyze (default: all)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of clusters. If not set, auto-selected via silhouette score.",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=50,
        dest="pca_dims",
        help="PCA dimensions for clustering space (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold to flag a cluster as 'hard' (default: 0.5)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help=(
            "Output file path (PNG/PDF). "
            "For --emotion all, used as stem: 'out.png' → 'out_anger.png'."
        ),
    )
    parser.add_argument(
        "--k-range-max",
        type=int,
        default=12,
        dest="k_range_max",
        help="Maximum k to test during auto K selection (default: 12)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        dest="n_init",
        help="KMeans n_init (default: 10)",
    )
    parser.add_argument(
        "--representation",
        choices=["embeddings", "tfidf"],
        default="embeddings",
        help="Text representation for clustering: OpenAI embeddings (default) or TF-IDF vectors",
    )

    args = parser.parse_args()
    k_range = range(2, args.k_range_max + 1)

    if args.emotion == "all":
        analyze_all_sentiments(
            args.k, args.pca_dims, args.threshold, args.output, k_range,
            args.n_init, args.representation,
        )
    else:
        analyze_sentiment(
            args.emotion, args.k, args.pca_dims, args.threshold,
            args.output, k_range, args.n_init, args.representation,
        )


if __name__ == "__main__":
    main()
