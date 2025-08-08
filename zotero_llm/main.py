import json
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gradio as gr
from typing import Dict, Optional, Tuple
from zotero_llm.zotero import ZoteroClient
from zotero_llm.rag import RAGEngine
from zotero_llm.llm import LLMClient, extract_json_from_response

class ResearchAssistant:
    DEFAULT_COLLECTION = "zotero_llm_abstracts"

    def __init__(self, embedding_model = None, collection_name = None,
                 answers_llm: Optional[Dict] = None):
        """Initialize the Research Assistant with its components."""
        self.log_file = "zotero_llm.log"
        self.debug_messages = []
        self.llm_config = self._llm_config()
        self.zotero = ZoteroClient()
        embedding_model = embedding_model or self.llm_config.get('embedding_model', {})
        self.collection_name = collection_name or ResearchAssistant.DEFAULT_COLLECTION
        self.rag = RAGEngine(**embedding_model, collection_name=self.collection_name)

        answers_llm = answers_llm or self.llm_config.get('answers_llm', {})
        self.llm = LLMClient(**answers_llm)
        if 'review_llm' in self.llm_config:
            self.llm_review = LLMClient(**self.llm_config['review_llm'])
        else:
            self.llm_review = None

        # Initialize status states
        self.zotero_status = gr.State(False)
        self.qdrant_status = gr.State(False)
        self.llm_status = gr.State(False)
        
        # self._initialize_system()

    def debug_print(self, message: str) -> None:
        """Print a message to both console and debug output."""
        print(message)
        self.debug_messages.append(message)
        
    def get_debug_output(self) -> str:
        """Get all debug messages."""
        return "\n".join(self.debug_messages)

    def _llm_config(self) -> Dict[str, str]:
        """Load LLM configuration from `llm_config.json`."""
        with open("llm_config.json", "r") as f:
            config = json.load(f)
        return config

    def _initialize_system(self) -> Tuple[bool, str]:
        """Initialize and test all necessary connections."""
        messages = []
        success = True

        # Test Zotero connection
        if not self.zotero.test_connection():
            self.debug_print("❌ Failed to connect to Zotero.")
            success = False
        else:
            self.debug_print("✅ Connected to local Zotero library successfully!")

        # Test Qdrant connection
        if not self.rag.test_connection():
            self.debug_print("❌ Failed to connect to Qdrant.")
            success = False
        else:
            self.debug_print("✅ Connected to local Qdrant server successfully!")

        # Ensure collection exists
        if not self.rag.if_collection_exists(self.collection_name):
            self.debug_print(f"⚠️ Creating collection '{self.collection_name}'...")
            documents = self.zotero.fetch_all_items()
            if documents:
                self.rag.upload_documents(documents)
                self.debug_print(f"✅ Collection '{self.collection_name}' created successfully!")
        else:
            self.debug_print(f"✅ Collection '{self.collection_name}' exists.")

        return success, "\n".join(messages)
    
    def upload_documents(self, collection_name: Optional[str] = None) -> None:
        """Upload documents to the specified collection."""
        collection_name = collection_name or self.collection_name
        
        self.debug_print(f"INFO: Uploading documents to collection '{collection_name}'...")
        documents = self.zotero.fetch_all_items()
        if documents:
            self.rag.upload_documents(documents, collection_name)
            self.debug_print(f"SUCCESS: Uploaded {len(documents)} documents to '{collection_name}'.")
        else:
            self.debug_print("WARNING: No documents found to upload.")

    def _ensure_collection_exists(self) -> None:
        """Ensure the main Zotero collection exists in Qdrant."""

        res = self.rag.client.collection_exists(self.collection_name)
        if not res:
            self.debug_print(f"INFO: Collection '{self.collection_name}' does not exist. Creating it...")
            self.upload_documents(self.collection_name)
        else:
            self.debug_print(f"INFO: Collection '{self.collection_name}' already exists.")

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

    def process_query(self, query: str) -> Tuple[str, str]:
        """Process a research query and return analysis results."""
        try:
            self.debug_print(f"INFO: Processing query: {query}")
            context = self.rag.search_documents(query)
            context = sorted(context, key=lambda x: x['year'])
            analysis = self.llm.ask_question(query, context)
            
            # Log the interaction
            with open(self.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n\n\nQuery: {query}\nResponse: {analysis}")

            if self.llm_review:
                messages = self.llm_review.create_messages(query=query, context=context, response=analysis)
                review_analysis = self.llm_review.ask_llm(messages)
                review_json = extract_json_from_response(review_analysis)
                self.debug_print(f"INFO: Review analysis: {json.dumps(review_json, indent=4)}")

                with open(self.log_file, 'a', encoding='utf-8') as log_file:
                    log_file.write(f"\n\nReview analysis: {json.dumps(review_json, indent=4)}")

                return analysis, self.get_debug_output(), review_json, review_analysis

            self.debug_print("SUCCESS: Query processed successfully")
            return analysis, self.get_debug_output(), {}, ""
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.debug_print(f"ERROR: {error_msg}")
            return error_msg, self.get_debug_output(), {}, ""

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
            
            debug_output = gr.Textbox(
                lines=5,
                label="Debug Output",
                value=self.get_debug_output(),
                interactive=False,
                visible=False
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
            review_json = gr.JSON(visible=False)
            review_analysis = gr.Textbox(visible=False)
            
            submit_btn.click(
                fn=self.process_query,
                inputs=query_input,
                outputs=[analysis_output, debug_output, review_json, review_analysis]
            )

            # Add Ctrl+Enter shortcut
            query_input.submit(
                fn=self.process_query,
                inputs=query_input,
                outputs=[analysis_output, debug_output, review_json, review_analysis],
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
