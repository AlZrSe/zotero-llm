from dotenv import load_dotenv
import os
import requests
import openai
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from zotero import *
from rag import *

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

def analyze_papers(zot, query, credentials):
    """Analyze papers from local Zotero library using LLM"""
    try:
        items = zot.items(limit=5, sort='dateModified', direction='desc')
    except Exception as e:
        console.print(f"[red]Failed to connect to local Zotero: {e}")
        return "Could not fetch items from local Zotero."

    # Prepare context for LLM
    papers_context = []
    for item in items:
        data = item.get('data', item)
        if data.get('title'):
            context = {
                'title': data.get('title', ''),
                'abstract': data.get('abstractNote', ''),
                'authors': data.get('creators', []),
                'year': data.get('date', '')
            }
            papers_context.append(context)
    
    # Prepare prompt for LLM
    system_prompt = """You are a research assistant analyzing academic papers.
    Based on the provided papers and query, provide insights and suggestions."""
    
    user_prompt = f"""Query: {query}
    
    Papers:
    {papers_context}
    
    Please provide:
    1. Key insights related to the query
    2. Connections between papers
    3. Suggestions for further research"""
    
    # Call OpenAI API
    client = openai.OpenAI(api_key=credentials['openai_api_key'],
                           base_url=credentials['llm_base_url'])
    response = client.chat.completions.create(
        model=credentials['llm_model'],
        extra_query={"provider": "OpenaiChat"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content


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

    # Check if collection exists
    collections = rag.get_collections()
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
                analysis = analyze_papers(zot, query, credentials)
                console.print(Panel(analysis, title="📚 Analysis Results"))
            except Exception as e:
                console.print(f"[red]Error during analysis: {str(e)}")
        
        console.print("\n---")

if __name__ == "__main__":
    main()
