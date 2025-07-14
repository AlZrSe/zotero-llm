from dotenv import load_dotenv
import os
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from typing import Dict, Optional
from zotero import ZoteroClient
from rag import RAGEngine
from llm import LLMClient

class ResearchAssistant:
    def __init__(self):
        """Initialize the Research Assistant with its components."""
        self.console = Console()
        self.log_file = "zotero_llm.log"
        self.credentials = self._setup_credentials()
        self.zotero = ZoteroClient()
        self.rag = RAGEngine()
        self.llm = LLMClient(self.credentials['llm_model'])

    def _setup_credentials(self) -> Dict[str, str]:
        """Load and validate credentials from .env file."""
        load_dotenv()
        
        required_vars = ['LLM_MODEL', 'EMBEDDING_MODEL']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            self.console.print(f"[red]Missing required environment variables: {', '.join(missing_vars)}")
            self.console.print("Please add them to your .env file")
            exit(1)
        
        return {
            'llm_base_url': os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1'),
            'llm_model': os.getenv('LLM_MODEL', 'mistral/mistral-large-latest'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'jinaai/jina-embeddings-v2-base-en:768')
        }

    def _initialize_connections(self) -> bool:
        """Initialize and test all necessary connections."""
        # Test Zotero connection
        if not self.zotero.test_connection():
            self.console.print("[red]Failed to connect to Zotero.[/red]")
            return False
        self.console.print("[green]Connected to local Zotero library successfully![/green]")

        # Test Qdrant connection
        if not self.rag.test_connection():
            self.console.print("[red]Failed to connect to Qdrant.[/red]")
            return False
        self.console.print("[green]Connected to local Qdrant server successfully![/green]")

        return True

    def _ensure_collection_exists(self) -> None:
        """Ensure the main Zotero collection exists in Qdrant."""
        collections = self.rag.get_collections()
        if self.rag.DEFAULT_COLLECTION not in collections:
            self.console.print(f"[yellow]Collection '{self.rag.DEFAULT_COLLECTION}' does not exist. Creating it...[/yellow]")
            documents = self.zotero.fetch_all_items()
            if documents:
                self.rag.upload_documents(documents)
                self.console.print(f"[green]Collection '{self.rag.DEFAULT_COLLECTION}' created successfully![/green]")
        else:
            self.console.print(f"[blue]Collection '{self.rag.DEFAULT_COLLECTION}' already exists.[/blue]")

    def _process_query(self, query: str) -> Optional[str]:
        """Process a research query and return analysis results."""
        try:
            context = self.rag.search_documents(query)
            analysis = self.llm.ask_question(query, context)
            
            # Log the interaction
            with open(self.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Query: {query}\nResponse: {analysis}\n\n\n")
                
            return analysis
        except Exception as e:
            self.console.print(f"[red]Error during analysis: {str(e)}")
            return None

    def run(self) -> None:
        """Main loop to run the Research Assistant."""
        self.console.print(Panel.fit("Welcome to Zotero-LLM Integration", title="🔍 Research Assistant"))
        
        if not self._initialize_connections():
            return

        self._ensure_collection_exists()
        
        while True:
            query = Prompt.ask("\n[cyan]Enter your research question[/cyan] (or 'exit' to quit)")
            
            if query.lower() == 'exit':
                break
            
            with self.console.status("[bold green]Analyzing papers..."):
                if analysis := self._process_query(query):
                    self.console.print(Panel(analysis, title="💡 LLM Insights"))
            
            self.console.print("\n---")

def main():
    """Entry point of the application."""
    assistant = ResearchAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
