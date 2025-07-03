# Zotero LLM Integration

This project provides a simple yet powerful integration between Zotero and LLMs (Large Language Models) to enhance research workflow.

## Features

- Connect to Zotero Desktop library
- Analyze research papers and their metadata
- Generate insights and summaries using LLM
- Interactive CLI interface for easy interaction

## Setup

1. Run a Zotero Desktop.
2. Create a `.env` file with your credentials:
```
OPENAI_API_KEY=your_openai_api_key
LLM_BASE_URL=http://localhost:8080/v1
```

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
