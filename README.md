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

## License

GPL v3.0
