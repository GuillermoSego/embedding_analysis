# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embedding generation pipeline for sentiment analysis texts using OpenAI's embedding API. Processes 5 emotion categories (anger, fear, joy, sadness, surprise) and generates cosine similarity matrices.

## Commands

```bash
# Install dependencies (UV package manager)
uv sync

# Run the embedding pipeline
python main.py

# Plot 2D projections (grid of all sentiments)
python plot_projections.py

# Plot all sentiments combined in one space
python plot_projections.py --mode combined

# Save plot to file
python plot_projections.py -o output.png

# Analyze hard texts (low scores)
python analyze_hard_texts.py --threshold 0.5

# Analyze specific emotion
python analyze_hard_texts.py --emotion anger --threshold 0.3

# Sync similarity matrices to remote server
./cp.sh
```

## Environment Setup

Requires `.env` file with:
```
OPENAI_API_KEY=<your_key>
```

## Architecture

### Pipeline Stages (main.py)

1. **build_embeddings(texts, sentiment)** - Generates embeddings using OpenAI `text-embedding-3-small` (512 dims), processes in batches of 100 with exponential backoff retry. Outputs raw embeddings and metadata CSV.

2. **build_S_matrix(sentiment)** - L2 normalizes embeddings and computes cosine similarity matrix (S = v·vᵀ). Stores as float16 for efficiency.

3. **process_embeddings()** - Main entry point, iterates through all 5 sentiments.

### Data Flow

```
data/eng_train_[sentiment].txt (tab-separated: text\tlabel)
    ↓
sentiments/[sentiment]/
    ├── embeddings_raw_[sentiment].npy   (float32)
    ├── embeddings_norm_[sentiment].npy  (float16, L2 normalized)
    ├── matrix_S_[sentiment].npy         (float16, cosine similarity)
    └── metadata.csv                     (id, text, text_clean, n_tokens, embedding_index)
```

### Key Technical Details

- Token counting uses tiktoken (cl100k_base) with word count fallback
- API retry logic: max 5 attempts, exponential backoff 1s-16s
- Handles RateLimitError, APIError, APIConnectionError, Timeout

## Module Structure

- `main.py` - Complete embedding pipeline
- `plot_projections.py` - 2D PCA visualization of embeddings (grid, combined, scores modes)
- `analyze_hard_texts.py` - Semantic analysis of texts difficult for the algorithm
- `src/evaluation/` - Evaluation framework (stub, needs implementation)
