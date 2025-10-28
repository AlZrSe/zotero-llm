import json
from datetime import datetime
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
import threading
from queue import Queue
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import gradio as gr
from typing import Dict, Optional, Tuple, Any, List
from zotero_llm.zotero import ZoteroClient
from zotero_llm.rag import RAGEngine
from zotero_llm.llm import LLMClient, extract_json_from_response, usage
from zotero_llm.models import init_db, get_session, Interaction

# Import agentic RAG functionality
try:
    from zotero_llm.agentic_rag import AgenticRAGEngine
    AGENTIC_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agentic RAG not available: {e}")
    AGENTIC_RAG_AVAILABLE = False

class ResearchAssistant:
    DEFAULT_COLLECTION = "zotero_llm_abstracts"

    def __init__(self, embedding_model = None, collection_name = None,
                 answers_llm: Optional[Dict] = None):
        """Initialize the Research Assistant with its components."""
        # Load environment variables early
        load_dotenv()

        self.log_file = "zotero_llm.log"
        self.debug_messages = []
        self.llm_config = self._llm_config()
        self.zotero = ZoteroClient(
            user_id=os.getenv("ZOTERO_USER_ID", None),
            api_key=os.getenv("ZOTERO_API_KEY", None)
        )
        
        # Process embedding model configuration
        embedding_config = embedding_model or self.llm_config.get('embedding_model', {})
        self.collection_name = collection_name or embedding_config.get('collection_name') or ResearchAssistant.DEFAULT_COLLECTION

        # Extract RAG-specific parameters
        rag_params = {
            'collection_name': self.collection_name,
            'server_url': embedding_config.get('server_url', f'http://{os.getenv("QDRANT_HOST", "localhost")}:{os.getenv("QDRANT_PORT", "6333")}'),
            'embedding_model': embedding_config.get('embedding_model', 'jinaai/jina-embeddings-v2-base-en'),
            'embedding_model_size': embedding_config.get('embedding_model_size', 768),
            'use_sentence_splitting': embedding_config.get('use_sentence_splitting', True)
        }

        self.rag = RAGEngine(**rag_params)

        answers_llm = answers_llm or self.llm_config.get('answers_llm', {})
        self.llm = LLMClient(**answers_llm)
        if 'review_llm' in self.llm_config:
            self.llm_review = LLMClient(**self.llm_config['review_llm'])
        else:
            self.llm_review = None

        # Initialize agentic RAG if available and enabled
        self.agentic_rag = None
        self.agentic_rag_enabled = False
        if AGENTIC_RAG_AVAILABLE and self.llm_config.get('agentic_rag', {}).get('enabled', False):
            try:
                agent_llm_config = self.llm_config.get('agentic_rag', {}).get('agent_llm', {})
                agent_llm = LLMClient(**agent_llm_config) if agent_llm_config else self.llm
                self.agentic_rag = AgenticRAGEngine(
                    rag_engine=self.rag,
                    document_resolver=self.get_document_by_key,
                    agent_llm_client=agent_llm
                )
                self.agentic_rag_enabled = True
                self.debug_print("✅ Agentic RAG initialized successfully!")
            except Exception as e:
                self.debug_print(f"⚠️ Failed to initialize agentic RAG: {e}")
                self.agentic_rag_enabled = False

        # Initialize database
        db_path = os.path.join(project_root, "grafana/metrics.db")
        self.engine = init_db(f"sqlite:///{db_path}")
        self.db_session = get_session(self.engine)

        # Initialize document cache
        self.document_cache: Dict[str, Dict] = {}

        # Initialize status states
        self.zotero_status = gr.State(False)
        self.qdrant_status = gr.State(False)
        self.llm_status = gr.State(False)
        
        # self.upload_task = None
        # self.upload_queue = Queue()
        # self._start_upload_worker()
        
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
            self.debug_print("✅ Connected to Zotero library successfully!")

        # Test Qdrant connection
        if not self.rag.test_connection():
            self.debug_print("❌ Failed to connect to Qdrant.")
            success = False
        else:
            self.debug_print("✅ Connected to local Qdrant server successfully!")

        # Check what count of documents is equal to Zotero items count
        zotero_documents = self.zotero.fetch_all_items() or []
        zotero_documents = [doc for doc in zotero_documents if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0]

        # Populate document cache
        self._populate_document_cache(zotero_documents)

        if not self.rag.if_collection_exists(self.collection_name):
            self.debug_print(f"⚠️ Creating collection '{self.collection_name}'...")
            if zotero_documents:
                self.rag.upload_documents(zotero_documents)
                self.debug_print(f"✅ Collection '{self.collection_name}' created successfully!")
        else:
            self.debug_print(f"⚠️ Updating collection '{self.collection_name}'...")
            self.update_collection(self.collection_name, zotero_documents)
            self.debug_print(f"✅ Collection '{self.collection_name}' updated successfully!")

        return success, "\n".join(messages)

    def _populate_document_cache(self, documents: List[Dict]) -> None:
        """Populate the document cache with Zotero documents.
        
        Args:
            documents: List of parsed Zotero documents
        """
        self.document_cache.clear()
        for doc in documents:
            zotero_key = doc.get('zotero_key')
            if zotero_key:
                self.document_cache[zotero_key] = doc

    def get_document_by_key(self, zotero_key: str) -> Optional[Dict]:
        """Retrieve full document information by Zotero key from cache.
        
        Args:
            zotero_key: The Zotero key of the document to retrieve
            
        Returns:
            Full document information or None if not found
        """
        return self.document_cache.get(zotero_key)

    def update_collection(self, collection_name: str, documents: List[Dict[str, Any]]) -> None:
        """Update the main Zotero collection in Qdrant."""

        def get_zotero_keys_qdrant(collection_name, batch_size=1000):
            zotero_keys = []
            qdrant_ids = []
            offset = None

            while True:
                try:
                    result = self.rag.client.scroll(collection_name=collection_name,
                                                    limit=batch_size,
                                                    offset=offset,
                                                    with_payload=True,
                                                    with_vectors=False
                    )
                    
                    points = result[0]
                    
                    # Извлекаем zotero_key
                    batch_keys = []
                    batch_ids = []
                    for point in points:
                        try:
                            if point.payload and 'zotero_key' in point.payload:
                                batch_keys.append(point.payload['zotero_key'])
                                batch_ids.append(point.id)
                        except Exception as e:
                            self.debug_print(f"Error processing point {point.id}: {e}")
                            continue
                    
                    zotero_keys.extend(batch_keys)
                    qdrant_ids.extend(batch_ids)
                    self.debug_print(f"Processed {len(points)} documents, founded {len(batch_keys)} keys.")

                    # Проверяем следующую страницу
                    next_offset = result[1]
                    if next_offset is None:
                        break
                        
                    offset = next_offset
                    
                except Exception as e:
                    self.debug_print(f"Error fetching Zotero keys: {e}")

            self.debug_print(f"Extracted {len(zotero_keys)} Zotero keys")
            return zotero_keys, qdrant_ids
        
        # Get all payloads from Qdrant
        qdrant_keys_list, qdrant_ids = get_zotero_keys_qdrant(collection_name)
        qdrant_keys = set(qdrant_keys_list)
        zotero_keys = set([doc['zotero_key'] for doc in documents if doc['title'] != '' or doc['abstract'] != '' or len(doc.get('keywords', [])) > 0])

        # Compare and update documents as needed
        need_upload = list(zotero_keys - qdrant_keys)
        need_delete = list(qdrant_keys - zotero_keys)

        if need_upload:
            upload_documents = [doc for doc in documents if doc['zotero_key'] in need_upload]
            self.rag.upload_documents(upload_documents, collection_name=collection_name, start_index=max([-1] + qdrant_ids) + 1)

        if need_delete:
            key_ids = [item[1] for item in zip(qdrant_keys_list, qdrant_ids) if item[0] in need_delete]
            self.rag.delete_documents(key_ids, collection_name=collection_name)

    def upload_documents(self, collection_name: Optional[str] = None) -> None:
        """Queue document upload to run in background."""
        collection_name = collection_name or self.collection_name
        self.debug_print(f"INFO: Uploading documents to collection '{collection_name}'...")
        documents = self.zotero.fetch_all_items()
        if documents:
            self._populate_document_cache(documents)
            self.rag.upload_documents(documents, collection_name)
        self.debug_print(f"INFO: Queued upload for collection '{collection_name}'")

    def is_upload_running(self) -> bool:
        """Check if document upload is currently in progress."""
        return self.rag.is_upload_running()

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

    def get_latest_metrics_markdown(self) -> str:
        """Return Markdown with LLM-as-a-Judge metrics for the latest interaction."""
        try:
            interaction = (
                self.db_session.query(Interaction)
                .order_by(Interaction.id.desc())
                .first()
            )
            if not interaction:
                return "### LLM-as-a-Judge\nNo metric data available. Ask a question to get an answer evaluation."

            def fmt(value):
                return "—" if value is None else (f"{value:.3f}" if isinstance(value, (int, float)) else str(value))

            # Check if review metrics are available
            has_review_metrics = any([
                interaction.summary is not None,
                interaction.verdict is not None,
                interaction.query_understanding_score is not None,
                interaction.retrieval_quality is not None,
                interaction.generation_quality is not None,
                interaction.error_detection_score is not None,
                interaction.citation_integrity is not None,
                interaction.hallucination_index is not None
            ])
            
            if not has_review_metrics:
                return "### LLM-as-a-Judge\nReview LLM failed after all retries. No evaluation metrics available."

            lines = [
                "### LLM-as-a-Judge",
                f"**Summary**: {fmt(interaction.summary)}\n",
                f"**Verdict**: {fmt(interaction.verdict)}",
                "",
                "**Metrics**:",
                f"- Query understanding: {fmt(interaction.query_understanding_score)}",
                f"- Retrieval quality: {fmt(interaction.retrieval_quality)}",
                f"- Generation quality: {fmt(interaction.generation_quality)}",
                f"- Error detection: {fmt(interaction.error_detection_score)}",
                f"- Citation integrity: {fmt(interaction.citation_integrity)}",
                f"- Hallucination index: {fmt(interaction.hallucination_index)}",
            ]
            return "\n".join(lines)
        except Exception as e:
            self.debug_print(f"ERROR: Failed to render metrics: {str(e)}")
            return "### LLM-as-a-Judge\nError rendering metrics."

    def save_user_feedback(self, rating: int) -> str:
        """Save user feedback (like/dislike) to the last interaction record."""
        try:
            # Get the last interaction from the database
            last_interaction = self.db_session.query(Interaction).order_by(Interaction.id.desc()).first()
            
            if last_interaction:
                # Update the user_rating field
                last_interaction.user_rating = rating
                self.db_session.commit()
                
                rating_text = "👍 Like" if rating == 1 else "👎 Dislike"
                return f"✅ Thank you for your feedback! Your review '{rating_text}' has been saved."
            else:
                return "❌ No records found to rate."
                
        except Exception as e:
            self.debug_print(f"ERROR: Failed to save user feedback: {str(e)}")
            return f"❌ Error saving feedback: {str(e)}"

    def process_agentic_query(self, message: str, history: Optional[List] = None) -> str:
        """Process a research query using agentic RAG and return analysis results."""
        if not self.agentic_rag_enabled or not self.agentic_rag:
            self.debug_print("WARNING: Agentic RAG not available, falling back to standard RAG")
            return self.process_query(message, history)
        
        try:
            usage.reset()
            self.debug_print(f"INFO: Processing agentic query: {message}")
            
            # Use unified agentic RAG for enhanced search
            agentic_results = self.agentic_rag.search(message)
            
            # Log the estimated limit for debugging
            estimated_limit = agentic_results.get('estimated_limit', 'unknown')
            limit_source = agentic_results.get('limit_source', 'unknown')
            user_requested = agentic_results.get('user_requested_limit')
            
            if limit_source == 'user_request':
                self.debug_print(f"INFO: Using user-requested limit: {user_requested} documents")
            elif limit_source == 'manual_override':
                self.debug_print(f"INFO: Using manually specified limit: {estimated_limit} documents")
            else:
                self.debug_print(f"INFO: Agentic RAG estimated optimal limit: {estimated_limit} documents")
            
            # Log accumulated irrelevant documents count
            irrelevant_count = len(agentic_results.get('irrelevant_documents_accumulated', []))
            self.debug_print(f"INFO: Accumulated {irrelevant_count} irrelevant documents during search")
            
            # Extract context from agentic results
            if agentic_results.get('fallback_used', False):
                self.debug_print("INFO: Agentic RAG used fallback, processing standard results")
                context = agentic_results.get('standard_results', [])
                query = self.llm.rewrite_query(message)  # Standard query rewriting
            else:
                self.debug_print("INFO: Agentic RAG completed successfully")
                # Extract documents from agentic results
                context = agentic_results.get('documents', [])
                # For now, use standard query rewriting - could be enhanced with agent insights
                query = self.llm.rewrite_query(message)
            
            # Enrich context with full document information from cache
            enriched_context = []
            for doc in context:
                zotero_key = doc.get('zotero_key', '')
                if zotero_key:
                    full_doc = self.get_document_by_key(zotero_key)
                    if full_doc:
                        # Merge the RAG result with full document info
                        enriched_doc = full_doc.copy()
                        enriched_doc.update(doc)  # RAG results take precedence for score, etc.
                        enriched_context.append(enriched_doc)
                    else:
                        # Fallback to RAG result if not in cache
                        enriched_context.append(doc)
                else:
                    # Fallback to RAG result if no zotero_key
                    enriched_context.append(doc)
            
            context = sorted(enriched_context, key=lambda x: x.get('year', 0))
            
            # Generate enhanced prompt with agentic insights
            enhanced_context = context
            if not agentic_results.get('fallback_used', False):
                # Add agentic insights to the analysis
                agentic_info = f"\n\nAgentic Analysis Insights:\n{agentic_results.get('agentic_results', 'No additional insights available')}"
                self.debug_print(f"INFO: Adding agentic insights to analysis")
            
            analysis = self.llm.ask_question(query, enhanced_context)
            
            # Log the interaction with agentic flag
            with open(self.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n\n\nAgentic Query: {query}\nAgentic Results: {json.dumps(agentic_results, indent=2)}\nResponse: {analysis}")

            usage_summary = usage.summarize()

            # Store interaction and usage in database with agentic flag
            interaction = Interaction(
                query=message,
                processed_query=query,
                response=analysis,
                used_documents=[doc.get('doi', '') for doc in context if doc.get('doi')],
                llm_model=f"{self.llm.model_name} (agentic)",
                llm_tokens_used=usage_summary['tokens_in'] + usage_summary['tokens_out'],
                llm_cost=usage_summary['cost_estimate'],
                llm_response_time=usage_summary['duration']
            )

            # Continue with review LLM processing (same as standard)
            if self.llm_review:
                try:
                    usage.reset()
                    messages = self.llm_review.create_messages(query=query, context=enhanced_context, response=analysis)
                    review_analysis = self.llm_review.ask_llm(messages)
                    review_json = extract_json_from_response(review_analysis)
                    self.debug_print(f"INFO: Review analysis: {json.dumps(review_json, indent=4)}")

                    usage_summary = usage.summarize()

                    interaction.review_llm_model = self.llm_review.model_name
                    interaction.review_llm_tokens_used = usage_summary['tokens_in'] + usage_summary['tokens_out']
                    interaction.review_llm_cost = usage_summary['cost_estimate']
                    interaction.review_llm_response_time = usage_summary['duration']

                    with open(self.log_file, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n\nReview analysis: {json.dumps(review_json, indent=4)}")
                    
                    # Store review metrics
                    if isinstance(review_json, dict):
                        interaction.summary = review_json.get('summary', None)
                        interaction.verdict = review_json.get('verdict', None)
                        if isinstance(review_json.get('metrics'), dict):
                            metrics_json = review_json['metrics']
                        else:
                            metrics_json = review_json
                        interaction.query_understanding_score = metrics_json.get('query_understanding_score', None)
                        interaction.retrieval_quality = metrics_json.get('retrieval_quality', None)
                        interaction.generation_quality = metrics_json.get('generation_quality', None)
                        interaction.error_detection_score = metrics_json.get('error_detection_score', None)
                        interaction.citation_integrity = metrics_json.get('citation_integrity', None)
                        interaction.hallucination_index = metrics_json.get('hallucination_index', None)
                        if isinstance(review_json.get('strengths'), list):
                            interaction.strengths = review_json['strengths']
                        if isinstance(review_json.get('weaknesses'), list):
                            interaction.weaknesses = review_json['weaknesses']
                            
                except Exception as e:
                    self.debug_print(f"WARNING: Review LLM failed after all retries: {str(e)}")
                    self.debug_print("INFO: Skipping review analysis and continuing with main response")
                    with open(self.log_file, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n\nReview LLM failed: {str(e)}")

            # Save to database
            self.db_session.add(interaction)
            self.db_session.commit()

            self.debug_print("SUCCESS: Agentic query processed successfully")
            return analysis
            
        except Exception as e:
            error_msg = f"Error during agentic analysis: {str(e)}"
            self.debug_print(f"ERROR: {error_msg}")
            self.debug_print("INFO: Falling back to standard RAG")
            return self.process_query(message, history)

    def process_query(self, message: str, history: Optional[List] = None, rag_mode: str = "Standard RAG") -> str:
        """Process a research query and return analysis results."""
        try:
            usage.reset()
            self.debug_print(f"INFO: Rewriting query: {message}")
            query = self.llm.rewrite_query(message)
            self.debug_print(f"INFO: Processed query: {query}")
            
            # Use either standard RAG or agentic RAG based on user selection
            if rag_mode == "Agentic RAG" and self.agentic_rag_enabled and self.agentic_rag:
                # Use agentic RAG search
                agentic_results = self.agentic_rag.agentic_search(query)
                context = agentic_results.get("documents", [])
            else:
                # Use standard RAG search
                context = self.rag.search_documents(query)
            
            # Enrich context with full document information from cache
            enriched_context = []
            for doc in context:
                zotero_key = doc.get('zotero_key', '')
                if zotero_key:
                    full_doc = self.get_document_by_key(zotero_key)
                    if full_doc:
                        # Merge the RAG result with full document info
                        enriched_doc = full_doc.copy()
                        enriched_doc.update(doc)  # RAG results take precedence for score, etc.
                        enriched_context.append(enriched_doc)
                    else:
                        # Fallback to RAG result if not in cache
                        enriched_context.append(doc)
                else:
                    # Fallback to RAG result if no zotero_key
                    enriched_context.append(doc)
            
            context = sorted(enriched_context, key=lambda x: x.get('year', 0))
            analysis = self.llm.ask_question(query, context)
            
            # Log the interaction
            with open(self.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n\n\nQuery: {query}\nResponse: {analysis}")

            usage_summary = usage.summarize()

            # Store interaction and usage in database 
            interaction = Interaction(
                query=message,
                processed_query=query,
                response=analysis,
                used_documents=[doc.get('doi', '') for doc in context if doc.get('doi')],
                llm_model = self.llm.model_name,
                llm_tokens_used = usage_summary['tokens_in'] + usage_summary['tokens_out'],
                llm_cost = usage_summary['cost_estimate'],
                llm_response_time = usage_summary['duration']
            )

            if self.llm_review:
                try:
                    usage.reset()
                    messages = self.llm_review.create_messages(query=query, context=context, response=analysis)
                    review_analysis = self.llm_review.ask_llm(messages)
                    review_json = extract_json_from_response(review_analysis)
                    self.debug_print(f"INFO: Review analysis: {json.dumps(review_json, indent=4)}")

                    usage_summary = usage.summarize()

                    interaction.review_llm_model = self.llm_review.model_name
                    interaction.review_llm_tokens_used = usage_summary['tokens_in'] + usage_summary['tokens_out']
                    interaction.review_llm_cost = usage_summary['cost_estimate']
                    interaction.review_llm_response_time = usage_summary['duration']

                    with open(self.log_file, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n\nReview analysis: {json.dumps(review_json, indent=4)}")
                    
                    # Store review metrics
                    if isinstance(review_json, dict):
                        interaction.summary = review_json.get('summary', None)
                        interaction.verdict = review_json.get('verdict', None)
                        if isinstance(review_json.get('metrics'), dict):
                            metrics_json = review_json['metrics']
                        else:
                            metrics_json = review_json
                        interaction.query_understanding_score = metrics_json.get('query_understanding_score', None)
                        interaction.retrieval_quality = metrics_json.get('retrieval_quality', None)
                        interaction.generation_quality = metrics_json.get('generation_quality', None)
                        interaction.error_detection_score = metrics_json.get('error_detection_score', None)
                        interaction.citation_integrity = metrics_json.get('citation_integrity', None)
                        interaction.hallucination_index = metrics_json.get('hallucination_index', None)
                        if isinstance(review_json.get('strengths'), list):
                            interaction.strengths = review_json['strengths']
                        if isinstance(review_json.get('weaknesses'), list):
                            interaction.weaknesses = review_json['weaknesses']
                            
                except Exception as e:
                    self.debug_print(f"WARNING: Review LLM failed after all retries: {str(e)}")
                    self.debug_print("INFO: Skipping review analysis and continuing with main response")
                    # Log the failure but continue processing
                    with open(self.log_file, 'a', encoding='utf-8') as log_file:
                        log_file.write(f"\n\nReview LLM failed: {str(e)}")

            # Save to database
            self.db_session.add(interaction)
            self.db_session.commit()

            self.debug_print("SUCCESS: Query processed successfully")
            return analysis
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.debug_print(f"ERROR: {error_msg}")
            return error_msg

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""
        # Try initialize, but don't block UI on failure
        success, status = self._initialize_system()
        if not success:
            self.debug_print(f"Startup warning: initialization failed, UI will still launch.\n{status}")

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
            
            # RAG Mode Selection
            with gr.Row():
                if self.agentic_rag_enabled:
                    rag_mode_toggle = gr.Radio(
                        choices=["Standard RAG", "Agentic RAG"],
                        value="Standard RAG",
                        label="RAG Mode",
                        info="Choose between standard retrieval or agentic multi-agent search"
                    )
                    rag_status = gr.Markdown("ℹ️ **Standard RAG**: Using traditional hybrid search")
                else:
                    rag_mode_toggle = gr.Radio(
                        choices=["Standard RAG"],
                        value="Standard RAG",
                        label="RAG Mode",
                        info="Agentic RAG not available - check dependencies and configuration",
                        interactive=False
                    )
                    rag_status = gr.Markdown("⚠️ **Agentic RAG unavailable**: Using standard RAG only")
            
            gr.Markdown("---")
            
            with gr.Row():
                left_col = gr.Column(scale=3)
                right_col = gr.Column(scale=1)

                # Right: Metrics panel (empty at startup)
                with right_col:
                    metrics_panel = gr.Markdown("")

                # Define wrapper that returns answer + metrics based on selected mode
                def process_query_and_metrics(message, history=None, rag_mode=None):
                    answer = self.process_query(message, history, rag_mode)
                    return answer, self.get_latest_metrics_markdown()
                
                # Define mode change handler
                def on_rag_mode_change(mode):
                    if mode == "Agentic RAG":
                        return gr.Markdown("🤖 **Agentic RAG**: Using AI agents for enhanced multi-strategy search")
                    else:
                        return gr.Markdown("ℹ️ **Standard RAG**: Using traditional hybrid search")
                
                # Connect mode change handler
                if self.agentic_rag_enabled:
                    rag_mode_toggle.change(
                        fn=on_rag_mode_change,
                        inputs=[rag_mode_toggle],
                        outputs=[rag_status]
                    )

                # Left: Chat interface
                with left_col:
                    chat = gr.ChatInterface(
                        fn=lambda message, history, rag_mode: process_query_and_metrics(
                            message, 
                            history,
                            rag_mode
                        ),
                        additional_inputs=[rag_mode_toggle],
                        examples=[
                            ["What are the main themes in my library about machine learning?"],
                            ["Summarize the recent papers about natural language processing."],
                            ["What are the key findings about deep learning architectures?"]
                        ],
                        title="Research Assistant Chat",
                        description="Ask questions about your Zotero library and get AI-powered insights.",
                        analytics_enabled=False,
                        autofocus=True,
                        type="messages",
                        additional_outputs=[metrics_panel]
                    )
                    # Feedback buttons row
                    with gr.Row():
                        gr.Markdown("**Rate the quality of the answer:**")
                    with gr.Row():
                        like_btn = gr.Button("👍 Like", variant="primary", size="sm")
                        dislike_btn = gr.Button("👎 Dislike", variant="secondary", size="sm")
                        feedback_status = gr.Markdown("", visible=False)

            # Feedback button handlers
            def handle_like():
                result = self.save_user_feedback(1)
                return gr.Markdown(result, visible=False)
            
            def handle_dislike():
                result = self.save_user_feedback(-1)
                return gr.Markdown(result, visible=False)
            
            like_btn.click(
                fn=handle_like,
                outputs=[feedback_status]
            )
            
            dislike_btn.click(
                fn=handle_dislike,
                outputs=[feedback_status]
            )
            
            # Update status LEDs every 30 seconds
            interface.load(
                fn=self.update_status_leds,
                outputs=[zotero_led, qdrant_led, llm_led]
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
        if assistant.rag.is_upload_running():
            assistant.rag.upload_task.join(timeout=0)
    except Exception as e:
        print(f"Error starting the application: {str(e)}")

if __name__ == "__main__":
    main()