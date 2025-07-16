# Zotero LLM Integration

This project provides a simple yet powerful integration between Zotero and LLMs (Large Language Models) to enhance research workflow.

## Features

- Connect to Zotero Desktop library
- Analyze research papers and their metadata
- Generate insights and summaries using LLM
- Interactive CLI interface for easy interaction

## Setup

### 1. Standalone install

1. Run a Zotero Desktop. Go to Edit > Settings > Advanced and check "Allow other application on this computer communicate with Zotero".
2. Copy `.env.example` to `.env` file and edit with your credentials.

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Docker compose

1. In root folder of cloned repository run
```bash
docker-compose up -d
```

2. Go to in browser by [link](http://127.0.0.1:3000/) and find clean Zotero interface.

3. You may setup your individual Zotero account with your personal library or create free new test account for testing purpose. If you haven't own Zotero library, you may add public one into your testing account, for example, [Review LLM](https://www.zotero.org/groups/6056275/review_llm), which have a large library of review articles about LLM with titles and abstracts. For those click in menu File > New Library > New Group... , authorize if needed, select `Search for Groups` and in search box enter `Review LLM` or go to link above, and select red button `Join`. Synchronize all library. You library ready.

## Usage

Run the main script:
```bash
python zotero_llm/main.py
```

## License

GPL v3.0
