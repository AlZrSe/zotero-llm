from dotenv import load_dotenv
import os
import requests
import openai
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from zotero import *
from rag import *
from llm import *

console = Console()

def setup_credentials():
    """Load credentials from .env file"""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'LLM_MODEL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        console.print(f"[red]Missing required environment variables: {', '.join(missing_vars)}")
        console.print("Please add them to your .env file")
        exit(1)
    
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'llm_base_url': os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1'),
        'llm_model': os.getenv('LLM_MODEL', 'gpt-4.1-mini'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-8B')
    }

def main():
    """Main function to run the Zotero-LLM integration"""
    console.print(Panel.fit("Welcome to Zotero-LLM Integration", title="🔍 Research Assistant"))
    
    # Setup credentials
    credentials = setup_credentials()
    
    # Initialize Zotero client
    zot = get_zotero_client()

    try:
        zot.count_items()  # Test connection
    except Exception as e:
        console.print("[red]Failed to connect to Zotero.[/red]")
        return
    console.print("[green]Connected to local Zotero library successfully![/green]")

    rag = get_qdrant_client()
    try:
        rag.get_collections()  # Test connection
    except Exception as e:
        console.print(f"[red]Failed to connect to Qdrant: {e}[/red]")
        return
    console.print("[green]Connected to local Qdrant server successfully![/green]")

    # Initialize LLM client
    try:
        llm_client = openai.OpenAI(api_key=credentials['openai_api_key'],
                                    base_url=credentials['llm_base_url'])
        # Test connection by fetching models
        llm_client.models.list()
    except Exception as e:
        console.print(f"[red]Failed to connect to LLM: {e}[/red]")
        return
    console.print("[green]Connected to LLM successfully![/green]")

    # Check if main zotero collection exists:
    collections = [collection.name for collection in rag.get_collections().collections]
    if collection_name_PR_zotero not in collections:
        console.print(f"[yellow]Collection '{collection_name_PR_zotero}' does not exist. Creating it...[/yellow]")
        upload_documents(rag, fetch_all_items(zot), collection_name_PR_zotero, 
                         embedding_model=credentials['embedding_model'])
        console.print(f"[green]Collection '{collection_name_PR_zotero}' created successfully![/green]")
    else:
        console.print(f"[blue]Collection '{collection_name_PR_zotero}' already exists.[/blue]")

    
    while True:
        query = Prompt.ask("\n[cyan]Enter your research question[/cyan] (or 'exit' to quit)")
        
        if query.lower() == 'exit':
            break
        
        with console.status("[bold green]Analyzing papers..."):
            try:
                # analysis = analyze_papers(zot, query, credentials)
                context = search_documents(rag, query, collection_name_PR_zotero)
                # console.print(Panel(context, title="📚 Analysis Results"))
                analysis = ask_llm(query, context, llm_client, credentials)
                console.print(Panel(analysis, title="💡 LLM Insights"))
            except Exception as e:
                console.print(f"[red]Error during analysis: {str(e)}")
        
        console.print("\n---")

if __name__ == "__main__":
    main()
