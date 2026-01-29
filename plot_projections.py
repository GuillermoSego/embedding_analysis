"""
Plot 2D (R2) projections of embedding matrices using PCA.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]
COLORS = {
    "anger": "#e63946",
    "fear": "#6a4c93",
    "joy": "#f4a261",
    "sadness": "#457b9d",
    "surprise": "#2a9d8f",
}


def load_eval_scores(sentiment: str) -> np.ndarray:
    """Load evaluation scores for a sentiment."""
    path = f"eval_src_scores/{sentiment}/eval_src_scores.npy"
    return np.load(path).astype(np.float32)


def load_embeddings(sentiment: str, normalized: bool = True) -> np.ndarray:
    """Load embeddings for a sentiment."""
    suffix = "norm" if normalized else "raw"
    path = f"sentiments/{sentiment}/embeddings_{suffix}_{sentiment}.npy"
    return np.load(path).astype(np.float32)


def plot_single_sentiment(sentiment: str, ax: plt.Axes, normalized: bool = True):
    """Plot 2D PCA projection for a single sentiment."""
    embeddings = load_embeddings(sentiment, normalized)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)

    ax.scatter(proj[:, 0], proj[:, 1], c=COLORS[sentiment], alpha=0.6, s=10)
    ax.set_title(
        f"{sentiment.capitalize()}\n(var: {pca.explained_variance_ratio_.sum():.1%})"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_all_sentiments(normalized: bool = True, output_path: str | None = None):
    """Plot 2D projections for all sentiments in a grid."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, sentiment in enumerate(SENTIMENTS):
        plot_single_sentiment(sentiment, axes[i], normalized)

    # Hide unused subplot
    axes[-1].axis("off")

    plt.suptitle("PCA Projections (R²) of Sentiment Embeddings", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_single_sentiment_scores(
    sentiment: str, ax: plt.Axes, normalized: bool = True
) -> plt.cm.ScalarMappable:
    """Plot 2D PCA projection colored by evaluation scores."""
    embeddings = load_embeddings(sentiment, normalized)
    scores = load_eval_scores(sentiment)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)

    scatter = ax.scatter(
        proj[:, 0], proj[:, 1], c=scores, cmap="RdYlGn", alpha=0.7, s=15, vmin=0, vmax=1
    )
    ax.set_title(f"{sentiment.capitalize()}\n(mean score: {scores.mean():.2f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return scatter


def plot_scores_grid(normalized: bool = True, output_path: str | None = None):
    """Plot 2D projections colored by eval scores for all sentiments."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    scatter = None
    for i, sentiment in enumerate(SENTIMENTS):
        scatter = plot_single_sentiment_scores(sentiment, axes[i], normalized)

    # Use last subplot for colorbar
    axes[-1].axis("off")
    cbar = fig.colorbar(scatter, ax=axes[-1], orientation="vertical", fraction=0.8)
    cbar.set_label("Evaluation Score", fontsize=12)

    plt.suptitle(
        "PCA Projections Colored by Algorithm Score\n(Red=Low, Green=High)", fontsize=14
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_scores_combined(normalized: bool = True, output_path: str | None = None):
    """Plot all sentiments in one space, colored by evaluation scores."""
    all_embeddings = []
    all_scores = []
    labels = []

    for sentiment in SENTIMENTS:
        emb = load_embeddings(sentiment, normalized)
        scores = load_eval_scores(sentiment)
        all_embeddings.append(emb)
        all_scores.append(scores)
        labels.extend([sentiment] * len(emb))

    combined = np.vstack(all_embeddings)
    combined_scores = np.concatenate(all_scores)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(12, 9))

    scatter = ax.scatter(
        proj[:, 0],
        proj[:, 1],
        c=combined_scores,
        cmap="RdYlGn",
        alpha=0.6,
        s=10,
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Evaluation Score")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(
        "Combined PCA - Colored by Algorithm Score\n(Red=Difficult, Green=Easy)"
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def plot_combined(normalized: bool = True, output_path: str | None = None):
    """Plot all sentiments in a single 2D space using combined PCA."""
    all_embeddings = []
    labels = []

    for sentiment in SENTIMENTS:
        emb = load_embeddings(sentiment, normalized)
        all_embeddings.append(emb)
        labels.extend([sentiment] * len(emb))

    combined = np.vstack(all_embeddings)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(10, 8))

    for sentiment in SENTIMENTS:
        mask = np.array(labels) == sentiment
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            c=COLORS[sentiment],
            alpha=0.5,
            s=10,
            label=sentiment.capitalize(),
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Combined PCA Projection of All Sentiments")
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def print_stats():
    """Print evaluation statistics for all sentiments."""
    print("=" * 60)
    print("EVALUATION SCORES STATISTICS")
    print("=" * 60)

    # Per-emotion stats
    all_scores = []
    all_evaluated = []

    print(f"\n{'Emotion':<12} {'Total':<8} {'Evaluated':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 60)

    for sentiment in SENTIMENTS:
        scores = load_eval_scores(sentiment)
        evaluated_mask = scores > 0
        n_evaluated = evaluated_mask.sum()
        evaluated_scores = scores[evaluated_mask]

        all_scores.append(scores)
        all_evaluated.append(n_evaluated)

        mean_eval = evaluated_scores.mean() if n_evaluated > 0 else 0
        std_eval = evaluated_scores.std() if n_evaluated > 0 else 0
        min_eval = evaluated_scores.min() if n_evaluated > 0 else 0
        max_eval = evaluated_scores.max() if n_evaluated > 0 else 0

        print(
            f"{sentiment.capitalize():<12} {len(scores):<8} {n_evaluated:<12} "
            f"{mean_eval:<8.3f} {std_eval:<8.3f} {min_eval:<8.3f} {max_eval:<8.3f}"
        )

    # General stats (same texts across emotions)
    print("\n" + "=" * 60)
    print("GENERAL STATISTICS (same texts, different emotion labels)")
    print("=" * 60)

    n_texts = len(all_scores[0])
    stacked = np.stack(all_scores)  # shape: (5, n_texts)

    # Per-text: how many emotions were evaluated
    evaluated_per_text = (stacked > 0).sum(axis=0)

    print(f"\nTotal texts: {n_texts}")
    print(f"Texts evaluated in all 5 emotions: {(evaluated_per_text == 5).sum()}")
    print(f"Texts evaluated in at least 1 emotion: {(evaluated_per_text >= 1).sum()}")
    print(f"Texts never evaluated (all zeros): {(evaluated_per_text == 0).sum()}")

    print(f"\nMean score across all emotions: {np.nanmean(stacked[stacked > 0]):.3f}")
    print(f"Std score across all emotions: {np.nanstd(stacked[stacked > 0]):.3f}")

    # Distribution of evaluations per emotion
    print(f"\nEvaluated texts per emotion:")
    for i, sentiment in enumerate(SENTIMENTS):
        pct = all_evaluated[i] / n_texts * 100
        print(f"  {sentiment.capitalize():<12}: {all_evaluated[i]:>5} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Plot R2 projections of embeddings")
    parser.add_argument(
        "--mode",
        choices=["grid", "combined", "scores", "scores_combined", "stats"],
        default="grid",
        help="grid: separate plots, combined: all in one, scores: colored by eval scores, scores_combined: all in one colored by scores, stats: print evaluation statistics",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw embeddings instead of normalized",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path (PNG/PDF). If not specified, displays plot.",
    )

    args = parser.parse_args()
    normalized = not args.raw

    if args.mode == "stats":
        print_stats()
    elif args.mode == "grid":
        plot_all_sentiments(normalized, args.output)
    elif args.mode == "combined":
        plot_combined(normalized, args.output)
    elif args.mode == "scores":
        plot_scores_grid(normalized, args.output)
    elif args.mode == "scores_combined":
        plot_scores_combined(normalized, args.output)


if __name__ == "__main__":
    main()
