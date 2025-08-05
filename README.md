# Zotero LLM Integration

This project provides a simple yet powerful integration between Zotero and LLMs (Large Language Models) to enhance research workflow.

## Features

- Connect to Zotero Desktop library
- Analyze research papers and their metadata
- Generate insights and summaries using LLM
- Interactive web interface for easy querying

## Setup

1. Run a Zotero Desktop. Go to Edit > Settings > Advanced and check "Allow other application on this computer communicate with Zotero".
2. Copy `.env.example` to `.env` file and edit with your credentials.

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare LLM config file `llm_config.json` with the following configuration sections:

- `answer_llm` - parameters for the primary LLM that generates answers
- `review_llm` - parameters for the LLM that reviews generated answers
- `judge_llm` - parameters for the LLM that makes final judgments

Each block have parameters:
- `model_name`: The name of the LLM model to be used (e.g., "mistral/mistral-medium-latest", "openrouter/deepseek/deepseek-r1-0528:free"). Model name must have a prefix with corresponding provider. For information look liteLLM documentation.
- `system_prompt`: Initial instructions/context provided to the LLM that defines its behavior and role
- `base_url`: API endpoint URL for the model
- `timeout`: Maximum time in seconds to wait between retries of the LLM response, in seconds (e.g., 5)
- `retries`: Number of retry attempts if the request fails

- `embedding_model` - parameters for the vector embeddings
    - `embedding_model`: The model to use for generating embeddings (e.g., "jinaai/jina-embeddings-v2-base-en")
    - `embedding_model_size`: Dimension size of the embedding vectors (e.g., 768)

## Usage

1. Run Zotero Desktop

2. Run Qdrant Docker image:

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
   -v "./qdrant_storage:/qdrant/storage:z" \
   qdrant/qdrant
```

3. Run the main script:
```bash
python zotero_llm/main.py
```

4. Go to [http://0.0.0.0:7860](http://0.0.0.0:7860) to open web interface and start querying.

## Evaluation of RAG

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

2. For evaluation purposes with provided query list (p. 3) you must add public group library into your account [Review LLM](https://www.zotero.org/groups/6056275/review_llm), which have a large library of review articles about LLM with titles and abstracts. For those click in menu File > New Library > New Group... , authorize if needed, select `Search for Groups` and in search box enter `Review LLM` or go to link above, and select red button `Join`. Synchronize all library. Your evaluation library ready.

3. Set up test queries with expected DOIs in `evaluation/query_list.json` (ground truth or golden standard):
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

4. Run the evaluation script:
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
