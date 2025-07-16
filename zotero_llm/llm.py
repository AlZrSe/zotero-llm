from litellm import completion
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Paper:
    """Data class to represent a research paper."""
    title: str
    abstract: str
    year: str
    authors: List[Dict] = None
    keywords: List[str] = None

class LLMClient:
    """Class to handle interactions with Language Learning Models."""
    
    def __init__(self, model_name: str):
        """Initialize LLM client with model configuration."""
        self.model_name = model_name
        self._system_prompt = """You are a research assistant analyzing academic papers.
        Based on the provided papers and query, provide useful thoughts, summary, insights
        and suggestions. Also, provide citations as numbers in square brackets in mentioned
        sentences with a reference list of the papers used at the end of your response."""

    def _format_papers_context(self, papers: List[Dict]) -> str:
        """Format papers into a string context for the LLM."""
        context_items = []
        for paper in papers:
            context = (
                f'Title: {paper.get("title", "")}\n'
                f'Abstract: {paper.get("abstract", "")}\n'
                f'Year: {paper.get("year", "")}'
            )
            if paper.get("keywords"):
                context += f'\nKeywords: {", ".join(paper.get("keywords", []))}'
            context_items.append(context)
        
        return "\n\n".join(context_items)

    def _create_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        """Create message list for LLM completion."""
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nPapers:\n{context}"}
        ]

    def ask_question(self, query: str, papers: List[Dict]) -> str:
        """
        Ask a research question with context from papers.
        
        Args:
            query: The research question to ask
            papers: List of paper dictionaries containing title, abstract, etc.
            
        Returns:
            str: LLM's analysis and response
        """
        try:
            # Format the context from papers
            context = self._format_papers_context(papers)
            
            # Create messages for LLM
            messages = self._create_messages(query, context)
            
            # Get completion from LLM
            response = completion(
                model=self.model_name,
                messages=messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return f"Failed to get analysis: {str(e)}"

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt used for LLM interactions."""
        self._system_prompt = new_prompt