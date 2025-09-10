# Zotero LLM Integration

This project provides a simple yet powerful integration between Zotero and LLMs (Large Language Models) to enhance research workflow.

## Features

- Connect to Zotero library
- Analyze research papers and their metadata
- Generate insights and summaries using LLM
- Interactive web interface for easy querying

## Setup

### 1. Standalone install

1. Run a Zotero Desktop. Go to Edit > Settings > Advanced and check "Allow other application on this computer communicate with Zotero".
**OR**
Create new or use existing Zotero account, go to [Settings](https://www.zotero.org/settings/security), scroll to section "Applications" and press "Create new private key" and make new one, then write it to `.env` file in `ZOTERO_API_KEY = "your key"`. Also write `ZOTERO_USER_ID` from numbers at the and of line "User ID: Your user ID for use in API calls is".

2. Create virtual environment in conda:
```bash
conda create -n zotero-llm python=3.12 pip
conda activate zotero-llm
```

3. Copy `.env.example` to `.env` file and edit with your credentials.

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Prepare LLM config file `zotero_llm/llm_config.json` with the following configuration sections:

- `answers_llm` - parameters for the primary LLM that generates answers
- `review_llm` - parameters for the LLM that reviews generated answers
- `judge_llm` - parameters for the LLM that makes final judgments

Each block have parameters:
- `model_name`: The name of the LLM model to be used (e.g., "mistral/mistral-medium-latest", "openrouter/deepseek/deepseek-r1-0528:free"). Model name must have a prefix with corresponding provider. For information look liteLLM documentation.
- `system_prompt`: Initial instructions/context provided to the LLM that defines its behavior and role
- `base_url`: API endpoint URL for the model
- `timeout`: Maximum time in seconds to wait between retries of the LLM response, in seconds (e.g., 5)
- `retries`: Number of retry attempts if the request fails

- `embedding_model` - parameters for the vector embeddings
    - **FastEmbed (default)**: Use existing format for backward compatibility
        - `embedding_model`: The model to use for generating embeddings (e.g., "jinaai/jina-embeddings-v2-base-en")
        - `embedding_model_size`: Dimension size of the embedding vectors (e.g., 768)
    - **External providers**: Use new format for models not supported by FastEmbed
        - `provider_type`: Set to "litellm" for external providers
        - `model_name`: The embedding model identifier (e.g., "text-embedding-3-large")
        - `base_url`: API endpoint URL (optional)
        - `api_key`: API key for authentication (optional, can use environment variables)
        - `timeout`: Request timeout in seconds (optional, default: 30)
        - `input_params`: Additional parameters for the embedding API (optional)
    - **HuggingFace local models**: Use local transformers models
        - `provider_type`: Set to "huggingface" for local HuggingFace models
        - `model_name`: The HuggingFace model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        - `device`: Device to run on ("cpu", "cuda", "auto", optional)
        - `max_length`: Maximum sequence length (optional, default: 512)
        - `pooling_strategy`: How to pool embeddings ("mean", "cls", "max", optional, default: "mean")
        - `trust_remote_code`: Whether to trust remote code (optional, default: false)
        - `model_kwargs`: Additional model arguments (optional)

    📖 **For detailed configuration examples and supported providers, see [EXTERNAL_EMBEDDINGS.md](EXTERNAL_EMBEDDINGS.md)**
    📋 **For comprehensive model examples and evaluation configurations, see [`evaluation/rag_list.json`](evaluation/rag_list.json)**

6. Install Grafana for visualization of metrics:
- Run Grafana docker image by command:
```bash
docker run -d -p 3000:3000 --name=grafana --volume "$PWD/grafana:/var/lib/grafana" grafana/grafana-enterprise
```
- Enter to Grafana CLI and install SQLite datasource plugin by commands
```bash
docker exec -it grafana grafana-cli plugins install frser-sqlite-datasource
```
and restart Grafana:
```bash
docker restart grafana
```
- Go to [http://localhost:3000/](Grafana Dashboards), change password at first time (default `admin/admin`).

### 2. Docker compose

Due Zotero Desktop local connection limitation, there isn't able to add it to `docker-compose.yaml`. Then, `docker-compose` only run main code and it's dependencies.

1. In root folder of cloned repository run
```bash
docker-compose up -d --build
```
then make a config of Grafana as mentioned in p.6 above.

## Usage

### 1. Standalone install

1. Run Zotero Desktop

2. Run Qdrant and Grafana Docker images:

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v "./qdrant_storage:/qdrant/storage:z" qdrant/qdrant
docker run -d -p 3000:3000 --name=grafana --volume "$PWD/grafana:/var/lib/grafana" grafana/grafana-enterprise
```

3. Run the main script:
```bash
python zotero_llm/main.py
```

4. Go to [http://0.0.0.0:7860](http://0.0.0.0:7860) to open web interface and start querying.


### 2. Docker compose

Simply run
```bash
docker-compose up -d
```

## Evaluation

The project includes two types of evaluation:

### RAG Evaluation

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

### Prompt Evaluation

To optimize and evaluate the system prompts for LLMs:

1. Make sure you have a test query list in `evaluation/query_list.json`:
```json
[
    {
        "query": "What are the main approaches to LLM evaluation?",
        "context_dois": [
            "10.xxxx/xxx.xxxx.xxxxx"
        ]
    }
]
```

2. Run the prompt evaluation script:
```bash
python evaluation/evaluation-prompt.py
```

The script will:
- Start with the current prompt from `llm_config.json`
- For each iteration:
  - Test the prompt with all queries
  - Calculate performance metrics
  - Analyze weaknesses and issues
  - Suggest prompt improvements
  - Allow you to accept/reject changes
- Save results to `evaluation/prompt_results/[model_name]/`:
  - `iteration_N/prompt.txt`: The prompt used in each iteration
  - `iteration_N/results.csv`: Detailed metrics
  - `iteration_N/responses.json`: All responses and their weaknesses
- Cache all results to avoid redundant API calls
- Update `llm_config.json` with improved prompts

The evaluation metrics include:

1. **Query Understanding Score** (0-1.0)
   - Measures how well the system interprets and addresses the user's question
   - Perfect score (1.0) indicates complete understanding and appropriate response focus

2. **Retrieval Quality** (0-1.0)
   - Assesses relevance of retrieved Zotero excerpts to the query
   - Score of 1.0 means ALL retrieved context is relevant
   - Lower scores indicate irrelevant or partially relevant context

3. **Generation Quality** (0-1.0)
   - Evaluates how effectively the answer uses the provided context
   - Perfect score (1.0) indicates optimal use of context without omissions
   - Considers accuracy, completeness, and coherence

4. **Error Detection Score** (0-1.0)
   - Measures system's ability to identify and handle:
     - Missing context
     - Contradictory information
     - Outdated or insufficient data
   - High scores indicate proper identification of limitations

5. **Citation Integrity** (0-1.0)
   - Evaluates accuracy of source attributions
   - Checks correctness of citations and references
   - Verifies proper linking between claims and sources

6. **Hallucination Index** (0-1.0)
   - Measures presence of unsupported claims
   - 0.0 means no hallucinations
   - Scores increase with number of unsupported statements
   - Value of 1.0 indicates multiple unsupported claims

The system is considered "Valid" when it:
- Correctly identifies missing/contradictory/outdated context
- Maintains high citation integrity
- Shows minimal hallucination
- Demonstrates good query understanding and context usage

## License

GPL v3.0
