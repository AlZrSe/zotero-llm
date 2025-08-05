# Zotero LLM Integration

This project provides a simple yet powerful integration between Zotero and LLMs (Large Language Models) to enhance research workflow.

## Features

- Connect to Zotero Desktop library
- Analyze research papers and their metadata
- Generate insights and summaries using LLM
- Interactive CLI interface for easy interaction

## Setup

1. Run a Zotero Desktop. Go to Edit > Settings > Advanced and check "Allow other application on this computer communicate with Zotero".
2. Copy `.env.example` to `.env` file and edit with your credentials.

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python zotero_llm/main.py
```

## Evaluation

To evaluate the RAG retrieval performance with different embedding models:

1. Configure the embedding models to evaluate in `evaluation/rag_list.json`:
```json
[
    {
        "embedding_model": "jinaai/jina-embeddings-v2-base-en",
        "embedding_model_size": 768
    }
]
```

2. Set up test queries with expected DOIs in `evaluation/query_list.json` (ground truth or golden standard):
```json
[
    {
        "query": "Which LLM's are helpful for developing cloud-native software?",
        "context_dois": [
            "10.1016/j.future.2025.107947"
        ]
    }
]
```

3. Run the evaluation script:
```bash
python evaluation/evaluation-rag.py
```

The script will:
- Test each embedding model with the provided queries
- Calculate metrics:
  - Mean Reciprocal Rank (MRR)
  - Hit Rate@K (for K = 1, 3, 5, 10)
  - Response times
  - Mean upsert times
- Generate confidence intervals for all metrics
- Append results to:
  - `evaluation/rag_evaluation_results.csv`: Detailed results for each query
  - `evaluation/model_stats.csv`: Aggregated statistics per model
  - `evaluation/evaluation_log.json`: Detailed evaluation logs

## License

GPL v3.0
