from dotenv import load_dotenv
import os
import gradio as gr
from typing import Dict, Optional, Tuple
from zotero import ZoteroClient
from rag import RAGEngine
from llm import LLMClient

class ResearchAssistant:
    def __init__(self):
        """Initialize the Research Assistant with its components."""
        self.log_file = "zotero_llm.log"
        self.credentials = self._setup_credentials()
        self.zotero = ZoteroClient()
        self.rag = RAGEngine()
        self.llm = LLMClient(self.credentials['llm_model'])
        
        # Initialize status states
        self.zotero_status = gr.State(False)
        self.qdrant_status = gr.State(False)
        self.llm_status = gr.State(False)
        
        self._initialize_system()

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

    def _initialize_system(self) -> Tuple[bool, str]:
        """Initialize and test all necessary connections."""
        messages = []
        success = True

        # Test Zotero connection
        if not self.zotero.test_connection():
            messages.append("❌ Failed to connect to Zotero.")
            success = False
        else:
            messages.append("✅ Connected to local Zotero library successfully!")

        # Test Qdrant connection
        if not self.rag.test_connection():
            messages.append("❌ Failed to connect to Qdrant.")
            success = False
        else:
            messages.append("✅ Connected to local Qdrant server successfully!")

        # Ensure collection exists
        collections = self.rag.get_collections()
        if self.rag.DEFAULT_COLLECTION not in collections:
            messages.append(f"⚠️ Creating collection '{self.rag.DEFAULT_COLLECTION}'...")
            documents = self.zotero.fetch_all_items()
            if documents:
                self.rag.upload_documents(documents)
                messages.append(f"✅ Collection '{self.rag.DEFAULT_COLLECTION}' created successfully!")
        else:
            messages.append(f"✅ Collection '{self.rag.DEFAULT_COLLECTION}' exists.")

        return success, "\n".join(messages)

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

    def check_services_status(self) -> Tuple[bool, bool, bool]:
        """Check the status of all services."""
        zotero_ok = self.zotero.test_connection()
        qdrant_ok = self.rag.test_connection()
        llm_ok = True  # We'll assume LLM is OK until a query fails
        return zotero_ok, qdrant_ok, llm_ok
    
    def update_status_leds(self):
        """Update the status LEDs for all services."""
        zotero_ok, qdrant_ok, llm_ok = self.check_services_status()
        return [
            "🟢" if zotero_ok else "🔴",
            "🟢" if qdrant_ok else "🔴",
            "🟢" if llm_ok else "🔴"
        ]

    def process_query(self, query: str) -> str:
        """Process a research query and return analysis results."""
        try:
            context = self.rag.search_documents(query)
            analysis = self.llm.ask_question(query, context)
            
            # Log the interaction
            with open(self.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Query: {query}\nResponse: {analysis}\n\n\n")
                
            return analysis
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""
        success, status = self._initialize_system()
        if not success:
            raise RuntimeError(f"Failed to initialize system:\n{status}")

        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Zotero-LLM Research Assistant")
            gr.Markdown("Ask questions about your Zotero library and get AI-powered insights.")
            
            # Status indicators
            with gr.Row():
                with gr.Column(scale=1):
                    zotero_led = gr.Markdown("🔴", label="Zotero Status")
                with gr.Column(scale=1):
                    qdrant_led = gr.Markdown("🔴", label="Qdrant Status")
                with gr.Column(scale=1):
                    llm_led = gr.Markdown("🔴", label="LLM Status")
            
            # Service names
            with gr.Row():
                gr.Markdown("Zotero")
                gr.Markdown("Qdrant")
                gr.Markdown("LLM")
            
            gr.Markdown("---")
            
            # Query interface
            with gr.Row():
                with gr.Column(scale=8):
                    query_input = gr.Textbox(
                        lines=3,
                        placeholder="Enter your research question...",
                        label="Research Question"
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Ask", variant="primary")
            
            analysis_output = gr.Textbox(
                lines=10,
                label="LLM Analysis"
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["What are the main themes in my library about machine learning?"],
                    ["Summarize the recent papers about natural language processing."],
                    ["What are the key findings about deep learning architectures?"]
                ],
                inputs=query_input
            )
            
            # Set up event handlers
            submit_btn.click(
                fn=self.process_query,
                inputs=query_input,
                outputs=analysis_output
            )

            # Add Ctrl+Enter shortcut
            query_input.submit(
                fn=self.process_query,
                inputs=query_input,
                outputs=analysis_output,
                api_name=False  # Prevent API endpoint creation
            )#.then(
            #     None,  # No additional function to call
            #     _js="""() => {
            #         // Focus back on the input after submission
            #         document.querySelector('textarea').focus();
            #     }"""
            # )
            
            # Update status LEDs every 30 seconds
            interface.load(
                fn=self.update_status_leds,
                outputs=[zotero_led, qdrant_led, llm_led],
                # every=30  # Update every 30 seconds
            )
            
            # Initial status update
            self.update_status_leds()
            
            return interface

def main():
    """Entry point of the application."""
    try:
        assistant = ResearchAssistant()
        interface = assistant.create_interface()
        interface.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,        # Default Gradio port
            share=False              # Create a public link
        )
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
