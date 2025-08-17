from litellm import completion
from typing import List, Dict, Optional
from dataclasses import dataclass
from time import sleep
import json
import re

def extract_json_from_response(response_text: str) -> dict:
    """
    Извлекает JSON из текста ответа LLM, обрабатывая распространенные ошибки форматирования.
    
    Алгоритм:
    1. Пытается распарсить весь ответ как JSON (если вывод чистый)
    2. Ищет JSON-блок между ```json ... ``` (Markdown-форматирование)
    3. Ищет JSON-объект по первой '{' и последней '}' в тексте
    4. Использует "умный" поиск парных скобок для вложенных структур
    
    Возвращает: dict с данными JSON
    Выбрасывает ValueError если не удалось извлечь валидный JSON
    """
    # Попытка 1: Чистый JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Попытка 2: Markdown-форматированный JSON (```json ... ```)
    markdown_match = re.search(r'```json(.*?)```', response_text, re.DOTALL)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Попытка 3: Авто-обнаружение по скобкам
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}')
    
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        raise ValueError("JSON delimiters not found")
    
    # "Умный" поиск парных скобок для вложенных структур
    stack = []
    for i in range(start_idx, len(response_text)):
        char = response_text[i]
        if char == '{':
            stack.append(i)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:  # Нашли закрывающую скобку верхнего уровня
                    end_idx = i
                    break
    
    json_str = response_text[start_idx:end_idx+1]
    
    # Попытка распарсить извлеченную строку
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Расширенная диагностика ошибок
        error_msg = f"JSON decode error: {e}\n"
        error_msg += f"Extracted string: {json_str[:100]}..." if len(json_str) > 100 else f"Extracted string: {json_str}"
        raise ValueError(error_msg) from e

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

    def __init__(self, model_name: str, base_url: str = None, timeout: int = 5, retries: int = 5,
                 system_prompt: Optional[str] = None, input_params: Dict = {}, rewrite_prompt: Optional[str] = None):
        """Initialize LLM client with model configuration."""
        self.model_name = model_name
        self._system_prompt = system_prompt or """You are a research assistant analyzing academic papers.
        Based on the provided papers and query, provide useful thoughts, summary, insights
        and suggestions. Also, provide citations as numbers in square brackets in mentioned
        sentences with a reference list of the papers used at the end of your response."""
        self.base_url = base_url
        self.timeout = timeout
        self.retries = retries
        self.input_params = input_params
        self.rewrite_prompt = rewrite_prompt

    def format_papers_context(self, papers: List[Dict]) -> str:
        """Format papers into a string context for the LLM."""
        context_items = []
        for paper in papers:
            context = (
                f'Title: {paper.get("title", "")}\n'
                f'Authors: {", ".join(author.get("name", "") or author.get("lastName", "") for author in paper.get("authors", []))}\n'
                f'Abstract: {paper.get("abstract", "")}\n'
                f'Year: {paper.get("year", "")}'
            )
            if paper.get("keywords"):
                context += f'\nKeywords: {", ".join(paper.get("keywords", []))}'
            context_items.append(context)
        
        return "\n\n".join(context_items)

    def create_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Create message list for LLM completion."""
        return [
            {"role": "user", "content": self._system_prompt.format(**kwargs)}
        ]

    def ask_llm(self, messages: List[Dict[str, str]]) -> str:
        """Ask the LLM a question and return its response."""

        for _ in range(self.retries):
            try:
                if self.base_url:
                    return completion(
                        model=self.model_name,
                        messages=messages,
                        base_url=self.base_url,
                        **self.input_params
                    ).choices[0].message.content
                else:
                    return completion(
                        model=self.model_name,
                        messages=messages,
                        **self.input_params
                    ).choices[0].message.content
            except Exception as e:
                print(f"Error during LLM completion: {e}")
                sleep(self.timeout)  # Wait before retrying

        raise Exception("Failed to get a response from the LLM after retries.")
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite the user's query using the rewrite prompt."""
        if self.rewrite_prompt:
            messages = [{"role": "user", "content": self.rewrite_prompt.format(query=query)}]
            return self.ask_llm(messages)
        return query

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
            context = self.format_papers_context(papers)
            
            # Create messages for LLM
            messages = self.create_messages(query=query, context=context)

            # Get completion from LLM
            return self.ask_llm(messages)

        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return f"Failed to get analysis: {str(e)}"

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt used for LLM interactions."""
        self._system_prompt = new_prompt