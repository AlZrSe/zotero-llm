"""
Zotero LLM package for interacting with Zotero library using LLM capabilities.
"""

# Add the project root to the Python path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zotero_llm.main import ResearchAssistant
from zotero_llm.zotero import ZoteroClient
from zotero_llm.rag import RAGEngine
from zotero_llm.llm import LLMClient

__all__ = ['ResearchAssistant', 'ZoteroClient', 'RAGEngine', 'LLMClient']
