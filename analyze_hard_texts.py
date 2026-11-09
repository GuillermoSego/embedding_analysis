"""
Semantic analysis of texts that are harder for the algorithm (lower scores).
Computes linguistic features and compares hard vs. all evaluated texts.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

SENTIMENTS = ["anger", "fear", "joy", "sadness", "surprise"]

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

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nor", "nowhere",
    "hardly", "barely", "scarcely",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}


def load_data(sentiment: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(f"sentiments/{sentiment}/metadata.csv")
    scores = np.load(f"eval_src_scores/{sentiment}/eval_src_scores.npy")
    return meta, scores


def load_train_labels(sentiment: str) -> list[str]:
    """Etiquetas del train crudo, una por fila y alineadas con metadata.csv.

    No usar pd.read_csv con comillas por defecto: las comillas sueltas fusionan líneas
    (2129 filas en lugar de 2768) y desalinean/pierden etiquetas.
    """
    with open(f"data/eng_train_{sentiment}.txt", encoding="utf-8") as f:
        return [line.rsplit("\t", 1)[1] for line in f.read().split("\n") if line]


def get_hard_texts(
    meta: pd.DataFrame, scores: np.ndarray, threshold: float = 0.5, min_score: float = 0.001
) -> pd.DataFrame:
    mask = (scores > min_score) & (scores < threshold)
    hard_df = meta[mask].copy()
    hard_df["score"] = scores[mask]
    return hard_df.sort_values("score")


def extract_words(texts: list[str]) -> list[str]:
    words = []
    for text in texts:
        words.extend(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    return words


def _count_syllables(word: str) -> int:
    vowel_groups = re.findall(r"[aeiouy]+", word.lower())
    return max(1, len(vowel_groups))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Linguistic features
# ---------------------------------------------------------------------------

def compute_linguistic_features(texts: list[str]) -> dict:
    """Compute per-text linguistic metrics and return aggregate statistics."""
    sentence_counts = []
    avg_sentence_lens = []
    ttrs = []
    avg_word_lens = []
    question_counts = []
    exclamation_counts = []
    comma_densities = []
    negation_densities = []
    rare_word_ratios = []
    flesch_kincaid_grades = []
    first_person_densities = []

    for text in texts:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        n_words = len(words) if words else 1

        sentences = _split_sentences(text)
        n_sentences = len(sentences) if sentences else 1

        sentence_counts.append(n_sentences)
        avg_sentence_lens.append(n_words / n_sentences)

        unique_words = set(words)
        ttrs.append(len(unique_words) / n_words if n_words > 0 else 0)

        avg_word_lens.append(np.mean([len(w) for w in words]) if words else 0)

        question_counts.append(text.count("?"))
        exclamation_counts.append(text.count("!"))

        n_commas = text.count(",")
        comma_densities.append(n_commas / n_words if n_words > 0 else 0)

        n_negations = sum(1 for w in words if w in NEGATION_WORDS)
        negation_densities.append(n_negations / n_words if n_words > 0 else 0)

        n_rare = sum(1 for w in words if len(w) > 8)
        rare_word_ratios.append(n_rare / n_words if n_words > 0 else 0)

        total_syllables = sum(_count_syllables(w) for w in words)
        fk = 0.39 * (n_words / n_sentences) + 11.8 * (total_syllables / n_words) - 15.59
        flesch_kincaid_grades.append(fk)

        n_fp = sum(1 for w in words if w in FIRST_PERSON)
        first_person_densities.append(n_fp / n_words if n_words > 0 else 0)

    def _stats(values):
        arr = np.array(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    metrics = {}
    for name, values in [
        ("sentence_count", sentence_counts),
        ("avg_sentence_len", avg_sentence_lens),
        ("lexical_diversity", ttrs),
        ("avg_word_len", avg_word_lens),
        ("question_marks", question_counts),
        ("exclamation_marks", exclamation_counts),
        ("comma_density", comma_densities),
        ("negation_density", negation_densities),
        ("rare_word_ratio", rare_word_ratios),
        ("flesch_kincaid", flesch_kincaid_grades),
        ("first_person_density", first_person_densities),
    ]:
        mean, std = _stats(values)
        metrics[f"{name}_mean"] = mean
        metrics[f"{name}_std"] = std

    return metrics


# ---------------------------------------------------------------------------
# Text analysis
# ---------------------------------------------------------------------------

def analyze_texts(df: pd.DataFrame, label: str) -> dict:
    if len(df) == 0:
        return {"count": 0}

    texts = df["text_clean"].tolist()
    tokens = df["n_tokens"].values
    scores = df["score"].values
    char_lens = [len(t) for t in texts]

    all_words = extract_words(texts)
    word_freq = Counter(all_words)
    content_words = {w: c for w, c in word_freq.items() if w not in STOPWORDS and len(w) > 2}

    linguistic = compute_linguistic_features(texts)

    result = {
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
    result.update(linguistic)
    return result


# ---------------------------------------------------------------------------
# Comparison: hard vs all
# ---------------------------------------------------------------------------

COMPARISON_METRICS = [
    ("tokens_mean", "Tokens (mean)"),
    ("chars_mean", "Characters (mean)"),
    ("sentence_count_mean", "Sentence count"),
    ("avg_sentence_len_mean", "Avg sentence length"),
    ("lexical_diversity_mean", "Lexical diversity (TTR)"),
    ("avg_word_len_mean", "Avg word length"),
    ("question_marks_mean", "Question marks"),
    ("exclamation_marks_mean", "Exclamation marks"),
    ("comma_density_mean", "Comma density"),
    ("negation_density_mean", "Negation density"),
    ("rare_word_ratio_mean", "Rare word ratio"),
    ("flesch_kincaid_mean", "Flesch-Kincaid grade"),
    ("first_person_density_mean", "First-person density"),
]


def compute_comparison(hard_stats: dict, all_stats: dict) -> list[dict]:
    rows = []
    for key, label in COMPARISON_METRICS:
        hard_val = hard_stats.get(key, 0.0)
        all_val = all_stats.get(key, 0.0)
        delta = hard_val - all_val
        pct = (delta / all_val * 100) if all_val != 0 else 0.0
        rows.append({
            "metric": label,
            "key": key,
            "hard": hard_val,
            "all": all_val,
            "delta": delta,
            "pct": pct,
        })
    rows.sort(key=lambda r: abs(r["pct"]), reverse=True)
    return rows


def print_comparison(comparison: list[dict], sentiment: str):
    print(f"\n{'=' * 70}")
    print(f"  HARD vs ALL COMPARISON — {sentiment.upper()}")
    print(f"{'=' * 70}")
    header = f"  {'Metric':<28} {'Hard':>8} {'All':>8} {'Delta':>8} {'%':>7}"
    print(header)
    print(f"  {'─' * 63}")
    for row in comparison:
        flag = " ***" if abs(row["pct"]) > 10 else ""
        print(
            f"  {row['metric']:<28} {row['hard']:>8.3f} {row['all']:>8.3f} "
            f"{row['delta']:>+8.3f} {row['pct']:>+6.1f}%{flag}"
        )


def plot_comparison(
    comparison: list[dict], sentiment: str, output_path: str | None = None
):
    top = comparison[:10]
    labels = [r["metric"] for r in top][::-1]
    pcts = [r["pct"] for r in top][::-1]
    colors = ["#2a9d8f" if p > 0 else "#e63946" for p in pcts]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, pcts, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("% difference (hard vs all)")
    ax.set_title(f"Hard vs All — Top Differentiating Features\n{sentiment.capitalize()}")
    plt.tight_layout()

    if output_path:
        stem, ext = os.path.splitext(output_path)
        save_path = f"{stem}_comparison_{sentiment}{ext}"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_analysis(stats: dict, label: str):
    print(f"\n{'=' * 70}")
    print(f"  {label.upper()}")
    print(f"{'=' * 70}")

    if stats["count"] == 0:
        print("  No hard texts found with the given threshold.")
        return

    print(f"\n  Count: {stats['count']} texts")
    print(f"\n  SCORE DISTRIBUTION:")
    print(f"    Mean: {stats['score_mean']:.3f}  |  Std: {stats['score_std']:.3f}")
    print(f"    Min:  {stats['score_min']:.3f}  |  Max: {stats['score_max']:.3f}")

    print(f"\n  TEXT LENGTH (tokens):")
    print(f"    Mean: {stats['tokens_mean']:.1f}  |  Std: {stats['tokens_std']:.1f}")
    print(f"    Min:  {stats['tokens_min']}  |  Max: {stats['tokens_max']}")

    print(f"\n  TEXT LENGTH (characters):")
    print(f"    Mean: {stats['chars_mean']:.1f}  |  Std: {stats['chars_std']:.1f}")

    print(f"\n  LINGUISTIC FEATURES:")
    print(f"    Sentence count:      {stats['sentence_count_mean']:.2f} +/- {stats['sentence_count_std']:.2f}")
    print(f"    Avg sentence length: {stats['avg_sentence_len_mean']:.2f} +/- {stats['avg_sentence_len_std']:.2f}")
    print(f"    Lexical diversity:   {stats['lexical_diversity_mean']:.3f} +/- {stats['lexical_diversity_std']:.3f}")
    print(f"    Avg word length:     {stats['avg_word_len_mean']:.2f} +/- {stats['avg_word_len_std']:.2f}")
    print(f"    Flesch-Kincaid:      {stats['flesch_kincaid_mean']:.2f} +/- {stats['flesch_kincaid_std']:.2f}")

    print(f"\n  STYLISTIC FEATURES:")
    print(f"    Question marks:      {stats['question_marks_mean']:.3f} +/- {stats['question_marks_std']:.3f}")
    print(f"    Exclamation marks:   {stats['exclamation_marks_mean']:.3f} +/- {stats['exclamation_marks_std']:.3f}")
    print(f"    Comma density:       {stats['comma_density_mean']:.4f} +/- {stats['comma_density_std']:.4f}")
    print(f"    Negation density:    {stats['negation_density_mean']:.4f} +/- {stats['negation_density_std']:.4f}")
    print(f"    Rare word ratio:     {stats['rare_word_ratio_mean']:.4f} +/- {stats['rare_word_ratio_std']:.4f}")
    print(f"    First-person density:{stats['first_person_density_mean']:.4f} +/- {stats['first_person_density_std']:.4f}")

    print(f"\n  TOP CONTENT WORDS:")
    for word, count in stats["top_words"]:
        print(f"    {word:<20} {count:>4}")

    print(f"\n  HARDEST EXAMPLES (lowest scores):")
    for i, (text, score) in enumerate(stats["examples"], 1):
        text_display = text[:80] + "..." if len(text) > 80 else text
        print(f"    {i}. [{score:.3f}] {text_display}")


# ---------------------------------------------------------------------------
# Per-emotion analysis
# ---------------------------------------------------------------------------

def analyze_emotion(sentiment: str, threshold: float, output_path: str | None = None):
    meta, scores = load_data(sentiment)
    hard_df = get_hard_texts(meta, scores, threshold)
    hard_stats = analyze_texts(hard_df, sentiment)
    print_analysis(hard_stats, f"{sentiment} - Hard Texts (score < {threshold})")

    if hard_stats["count"] == 0:
        return hard_df

    evaluated_mask = scores > 0.001
    all_eval_df = meta[evaluated_mask].copy()
    all_eval_df["score"] = scores[evaluated_mask]
    all_stats = analyze_texts(all_eval_df, f"{sentiment} - all evaluated")

    comparison = compute_comparison(hard_stats, all_stats)
    print_comparison(comparison, sentiment)
    plot_comparison(comparison, sentiment, output_path)

    return hard_df


# ---------------------------------------------------------------------------
# Cross-emotion analysis
# ---------------------------------------------------------------------------

def analyze_general(threshold: float, output_path: str | None = None):
    print(f"\n{'#' * 70}")
    print(f"  GENERAL ANALYSIS - Texts Hard Across Multiple Emotions")
    print(f"{'#' * 70}")

    all_meta = {}
    all_scores = {}
    for sentiment in SENTIMENTS:
        meta, scores = load_data(sentiment)
        all_meta[sentiment] = meta
        all_scores[sentiment] = scores

    stacked = np.stack([all_scores[s] for s in SENTIMENTS])

    hard_mask = (stacked > 0.001) & (stacked < threshold)
    hard_count_per_text = hard_mask.sum(axis=0)

    print(f"\n  Texts hard in multiple emotions (score < {threshold}):")
    for n in range(5, 0, -1):
        count = (hard_count_per_text >= n).sum()
        if count > 0:
            print(f"    Hard in {n}+ emotions: {count} texts")

    multi_hard_mask = hard_count_per_text >= 3
    if multi_hard_mask.sum() > 0:
        print(f"\n  {'=' * 60}")
        print(f"  TEXTS HARD IN 3+ EMOTIONS")
        print(f"  {'=' * 60}")

        multi_hard_indices = np.where(multi_hard_mask)[0]
        meta = all_meta["anger"]

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
        hard_stats = analyze_texts(df, "multi-emotion hard")

        print(f"\n  Count: {len(df)} texts hard in 3+ emotions")

        print_analysis(hard_stats, "Multi-emotion hard texts")

        # Compare against all evaluated texts (using anger as reference)
        ref_scores = all_scores["anger"]
        eval_mask = ref_scores > 0.001
        all_eval_df = all_meta["anger"][eval_mask].copy()
        all_eval_df["score"] = ref_scores[eval_mask]
        all_stats = analyze_texts(all_eval_df, "all evaluated")

        comparison = compute_comparison(hard_stats, all_stats)
        print_comparison(comparison, "multi-emotion")
        plot_comparison(comparison, "multi-emotion", output_path)

        print(f"\n  EXAMPLES (hardest across multiple emotions):")
        for i, row in df.head(10).iterrows():
            text = row["text_clean"]
            text_display = text[:60] + "..." if len(text) > 60 else text
            print(f"    [{row['score']:.3f}] ({row['hard_emotions']})")
            print(f"           {text_display}")

    evaluated_mask = stacked > 0.001
    evaluated_count = evaluated_mask.sum(axis=0)
    with np.errstate(invalid="ignore"):
        hard_ratio = np.where(evaluated_count > 0, hard_count_per_text / evaluated_count, 0)

    consistently_hard = (hard_ratio > 0.5) & (evaluated_count >= 3)
    print(f"\n  Consistently hard texts (hard in >50% of evaluated emotions): {consistently_hard.sum()}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_hard_texts(
    sentiment: str,
    threshold: float,
    n: int,
    output_dir: str,
    exclude_marks: bool = False,
) -> None:
    meta, scores = load_data(sentiment)
    hard = get_hard_texts(meta, scores, threshold=threshold)
    if exclude_marks:
        # Drop texts containing '!' so the top-N refills with the next-hardest
        # texts that never had an exclamation mark (scores stay valid as-is).
        hard = hard[~hard["text"].astype(str).str.contains("!", regex=False)]
    hard = hard.sort_values("score").head(n)

    labels = load_train_labels(sentiment)

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_without_marks" if exclude_marks else ""
    out_path = os.path.join(output_dir, f"hard_{sentiment}{suffix}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in hard.iterrows():
            f.write(f"{row['text']}\t{labels[idx]}\n")
    print(f"Saved {len(hard)} texts → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path for comparison plots (PNG/PDF)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Save the N hardest texts per emotion to files",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of hardest texts to export (default: 100)",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="hard_texts",
        help="Output directory for exported text files (default: hard_texts/)",
    )
    parser.add_argument(
        "--exclude-marks",
        action="store_true",
        help="Exclude texts containing '!' from the export and add a '_without_marks' suffix",
    )

    args = parser.parse_args()

    print(f"\n{'#' * 70}")
    print(f"  SEMANTIC ANALYSIS OF HARD TEXTS")
    print(f"  Threshold: score < {args.threshold}")
    print(f"{'#' * 70}")

    if args.emotion == "all":
        for sentiment in SENTIMENTS:
            analyze_emotion(sentiment, args.threshold, args.output)
        analyze_general(args.threshold, args.output)
    else:
        analyze_emotion(args.emotion, args.threshold, args.output)

    if args.export:
        print(f"\n{'#' * 70}")
        print(f"  EXPORTING TOP-{args.top_n} HARDEST TEXTS")
        print(f"{'#' * 70}")
        sentiments_to_export = SENTIMENTS if args.emotion == "all" else [args.emotion]
        for sentiment in sentiments_to_export:
            export_hard_texts(
                sentiment,
                args.threshold,
                args.top_n,
                args.export_dir,
                exclude_marks=args.exclude_marks,
            )


if __name__ == "__main__":
    main()
