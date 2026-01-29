"""
Semantic analysis of texts that are harder for the algorithm (lower scores).
"""

import argparse
import numpy as np
import pandas as pd
from collections import Counter
import re

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]


def load_data(sentiment: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load metadata and scores for a sentiment."""
    meta = pd.read_csv(f"sentiments/{sentiment}/metadata.csv")
    scores = np.load(f"eval_src_scores/{sentiment}/eval_src_scores.npy")
    return meta, scores


def get_hard_texts(
    meta: pd.DataFrame, scores: np.ndarray, threshold: float = 0.5, min_score: float = 0.001
) -> pd.DataFrame:
    """Get texts with scores below threshold (excluding zeros/unevaluated)."""
    mask = (scores > min_score) & (scores < threshold)
    hard_df = meta[mask].copy()
    hard_df["score"] = scores[mask]
    return hard_df.sort_values("score")


def extract_words(texts: list[str]) -> list[str]:
    """Extract lowercase words from texts."""
    words = []
    for text in texts:
        words.extend(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return words


def analyze_texts(df: pd.DataFrame, label: str) -> dict:
    """Analyze characteristics of a set of texts."""
    if len(df) == 0:
        return {"count": 0}

    texts = df["text_clean"].tolist()
    tokens = df["n_tokens"].values
    scores = df["score"].values
    char_lens = [len(t) for t in texts]

    # Word frequency
    all_words = extract_words(texts)
    word_freq = Counter(all_words)

    # Filter stopwords for top words
    stopwords = {
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
    content_words = {w: c for w, c in word_freq.items() if w not in stopwords and len(w) > 2}

    return {
        "count": len(df),
        "score_mean": scores.mean(),
        "score_std": scores.std(),
        "score_min": scores.min(),
        "score_max": scores.max(),
        "tokens_mean": tokens.mean(),
        "tokens_std": tokens.std(),
        "tokens_min": tokens.min(),
        "tokens_max": tokens.max(),
        "chars_mean": np.mean(char_lens),
        "chars_std": np.std(char_lens),
        "top_words": Counter(content_words).most_common(15),
        "examples": df.nsmallest(5, "score")[["text_clean", "score"]].values.tolist(),
    }


def print_analysis(stats: dict, label: str):
    """Print formatted analysis."""
    print(f"\n{'=' * 70}")
    print(f"  {label.upper()}")
    print(f"{'=' * 70}")

    if stats["count"] == 0:
        print("  No hard texts found with the given threshold.")
        return

    print(f"\n  Count: {stats['count']} hard texts")
    print(f"\n  SCORE DISTRIBUTION:")
    print(f"    Mean: {stats['score_mean']:.3f}  |  Std: {stats['score_std']:.3f}")
    print(f"    Min:  {stats['score_min']:.3f}  |  Max: {stats['score_max']:.3f}")

    print(f"\n  TEXT LENGTH (tokens):")
    print(f"    Mean: {stats['tokens_mean']:.1f}  |  Std: {stats['tokens_std']:.1f}")
    print(f"    Min:  {stats['tokens_min']}  |  Max: {stats['tokens_max']}")

    print(f"\n  TEXT LENGTH (characters):")
    print(f"    Mean: {stats['chars_mean']:.1f}  |  Std: {stats['chars_std']:.1f}")

    print(f"\n  TOP CONTENT WORDS:")
    for word, count in stats["top_words"]:
        print(f"    {word:<20} {count:>4}")

    print(f"\n  HARDEST EXAMPLES (lowest scores):")
    for i, (text, score) in enumerate(stats["examples"], 1):
        text_display = text[:80] + "..." if len(text) > 80 else text
        print(f"    {i}. [{score:.3f}] {text_display}")


def analyze_emotion(sentiment: str, threshold: float):
    """Analyze hard texts for a single emotion."""
    meta, scores = load_data(sentiment)
    hard_df = get_hard_texts(meta, scores, threshold)
    stats = analyze_texts(hard_df, sentiment)
    print_analysis(stats, f"{sentiment} - Hard Texts (score < {threshold})")
    return hard_df


def analyze_general(threshold: float):
    """Analyze texts that are hard across multiple emotions."""
    print(f"\n{'#' * 70}")
    print(f"  GENERAL ANALYSIS - Texts Hard Across Multiple Emotions")
    print(f"{'#' * 70}")

    # Load all data
    all_meta = {}
    all_scores = {}
    for sentiment in SENTIMENTS:
        meta, scores = load_data(sentiment)
        all_meta[sentiment] = meta
        all_scores[sentiment] = scores

    n_texts = len(all_meta["anger"])

    # Stack scores: shape (5, n_texts)
    stacked = np.stack([all_scores[s] for s in SENTIMENTS])

    # Find texts hard in multiple emotions
    hard_mask = (stacked > 0.001) & (stacked < threshold)
    hard_count_per_text = hard_mask.sum(axis=0)

    print(f"\n  Texts hard in multiple emotions (score < {threshold}):")
    for n in range(5, 0, -1):
        count = (hard_count_per_text >= n).sum()
        if count > 0:
            print(f"    Hard in {n}+ emotions: {count} texts")

    # Analyze texts that are hard in at least 3 emotions
    multi_hard_mask = hard_count_per_text >= 3
    if multi_hard_mask.sum() > 0:
        print(f"\n  {'=' * 60}")
        print(f"  TEXTS HARD IN 3+ EMOTIONS")
        print(f"  {'=' * 60}")

        multi_hard_indices = np.where(multi_hard_mask)[0]
        meta = all_meta["anger"]  # Use anger metadata (same texts)

        # Build dataframe with average score across hard emotions
        rows = []
        for idx in multi_hard_indices:
            text = meta.iloc[idx]["text_clean"]
            tokens = meta.iloc[idx]["n_tokens"]
            emotion_scores = {s: all_scores[s][idx] for s in SENTIMENTS}
            hard_emotions = [s for s in SENTIMENTS if hard_mask[SENTIMENTS.index(s), idx]]
            avg_score = np.mean([emotion_scores[s] for s in hard_emotions])
            rows.append({
                "text_clean": text,
                "n_tokens": tokens,
                "score": avg_score,
                "hard_emotions": ", ".join(hard_emotions),
                "n_hard": len(hard_emotions),
            })

        df = pd.DataFrame(rows).sort_values("score")
        stats = analyze_texts(df, "multi-emotion hard")

        print(f"\n  Count: {len(df)} texts hard in 3+ emotions")
        print(f"\n  TOP CONTENT WORDS in multi-hard texts:")
        for word, count in stats["top_words"][:10]:
            print(f"    {word:<20} {count:>4}")

        print(f"\n  EXAMPLES (hardest across multiple emotions):")
        for i, row in df.head(10).iterrows():
            text = row["text_clean"]
            text_display = text[:60] + "..." if len(text) > 60 else text
            print(f"    [{row['score']:.3f}] ({row['hard_emotions']})")
            print(f"           {text_display}")

    # Find consistently hard texts (hard in most evaluated emotions)
    evaluated_mask = stacked > 0.001
    evaluated_count = evaluated_mask.sum(axis=0)
    with np.errstate(invalid="ignore"):
        hard_ratio = np.where(evaluated_count > 0, hard_count_per_text / evaluated_count, 0)

    consistently_hard = (hard_ratio > 0.5) & (evaluated_count >= 3)
    print(f"\n  Consistently hard texts (hard in >50% of evaluated emotions): {consistently_hard.sum()}")


def main():
    parser = argparse.ArgumentParser(description="Analyze hard texts for the algorithm")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for 'hard' texts (default: 0.5)",
    )
    parser.add_argument(
        "--emotion",
        choices=SENTIMENTS + ["all"],
        default="all",
        help="Analyze specific emotion or all",
    )

    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print(f"  SEMANTIC ANALYSIS OF HARD TEXTS")
    print(f"  Threshold: score < {args.threshold}")
    print(f"{'#' * 70}")

    if args.emotion == "all":
        for sentiment in SENTIMENTS:
            analyze_emotion(sentiment, args.threshold)
        analyze_general(args.threshold)
    else:
        analyze_emotion(args.emotion, args.threshold)


if __name__ == "__main__":
    main()
