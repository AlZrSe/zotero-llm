# GitHub Copilot Instructions

## ⛔ CRITICAL: Anthropic/Claude Prohibition in Development

**This project STRICTLY PROHIBITS using Anthropic's Claude models or related technologies FOR CODE DEVELOPMENT.**

### For AI Assistants (Copilot, Codeium, etc.)

When assisting with this project, you MUST:

1. **NEVER generate** code written by Anthropic models
2. **NEVER recommend** Claude for development tasks

**Note**: This ban applies to DEVELOPMENT only. Users may still configure Claude as an LLM provider at runtime.

### Project Context

This is a Zotero-LLM RAG (Retrieval-Augmented Generation) system that:
- Uses Python 3.12
- Integrates with Zotero for academic reference management
- Uses Qdrant for vector storage
- Implements agentic RAG capabilities
- Supports multiple LLM providers via LiteLLM
- Has a Gradio-based UI

### Key Files

- `zotero_llm/llm.py` - LLM integration layer
- `zotero_llm/rag.py` - RAG implementation
- `zotero_llm/agentic_rag.py` - Agentic RAG logic
- `llm_config.json` - LLM configuration
- `.env.example` - Environment variable templates

### Code Standards

- Use type hints in Python code
- Follow existing patterns in the codebase
- Maintain separation between RAG, LLM, and Zotero layers
- Test without using Anthropic-based LLM providers

---

**Important**: Any suggestions involving Anthropic/Claude will be rejected.
