"""
Agentic RAG implementation for iterative document search with relevance estimation.

This module provides a simplified agentic approach using CrewAI focused on:
1. Searching documents in Qdrant using a tool
2. Estimating optimal search limits
3. Iterative relevance checking using LLM
4. Supervisor agent that returns only relevant documents
"""

import json
import re
from typing import Dict, List, Optional, Any, Callable
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from zotero_llm.rag import RAGEngine
from zotero_llm.llm import LLMClient


# Search limit bounds
MIN_LIMIT = 3
MAX_LIMIT = 50

# --- Add Pydantic models for CrewAI Task output_json ----
class SearchDocumentModel(BaseModel):
    zotero_key: Optional[str] = None
    score: Optional[float] = None

class SearchResultsModel(BaseModel):
    documents: List[SearchDocumentModel] = []
    query: Optional[str] = None
    limit_used: Optional[int] = None

class LimitEstimationModel(BaseModel):
    estimated_limit: int
    source: str
    reasoning: Optional[str] = ""

class RelevanceEvalDoc(BaseModel):
    index: int
    relevance_score: float
    relevance_reason: Optional[str] = ""
    is_relevant: bool = Field(description="Whether the document is relevant (True/False)")

class RelevanceEvalModel(BaseModel):
    relevant_documents: List[RelevanceEvalDoc] = []
    irrelevant_documents: List[RelevanceEvalDoc] = []
    needs_more_search: bool = False
    refined_query: Optional[str] = None


class UnifiedSearchResult(BaseModel):
    documents: List[Dict] = []
    query: str
    total_found: int
    iterations_used: int = 1
    search_strategy: str = "unified agentic search"


class QdrantSearchTool(BaseTool):
    """CrewAI tool for searching documents in Qdrant using RAG engine's search_documents method."""
    name: str = "qdrant_search"
    description: str = """Search academic documents in Qdrant vector database using RAG engine.
    Input: query (str) and optional limit (int, default 10).
    Returns: JSON with documents list including title, abstract, authors, year, zotero_key, and score."""
    rag_engine: RAGEngine = Field(description="RAG engine for document search")
    document_resolver: Optional[Callable] = Field(description="Function to resolve Zotero keys to full metadata")
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, rag_engine: RAGEngine, document_resolver: Optional[Callable] = None):
        """Initialize search tool with RAG engine.
        
        Args:
            rag_engine: RAG engine that provides search_documents method
            document_resolver: Function to resolve Zotero keys to full metadata (optional)
        """
        super().__init__(rag_engine=rag_engine, document_resolver=document_resolver)
    
    def _run(self, query: str, limit: int = 10) -> str:
        """Execute search in Qdrant using RAG engine's search_documents method.
        
        This method delegates to rag_engine.search_documents() which handles:
        - Query embedding generation
        - Hybrid search (semantic + BM25)
        - Document deduplication
        - Score normalization
        
        Args:
            query: Search query string
            limit: Maximum number of documents to retrieve (default: 10)
            
        Returns:
            JSON string with documents and metadata
        """
        try:
            # Use RAG engine's search_documents method
            # This handles all the complexity: embedding, hybrid search, deduplication
            results = self.rag_engine.search_documents(
                query=query,
                limit=limit,
                deduplicate_documents=True
            )
            
            # Format results for agent consumption
            formatted_results = []
            for doc in results:
                # Always enrich with full metadata from Zotero using document_resolver
                zotero_key = doc.get("zotero_key")
                if zotero_key and self.document_resolver:
                    try:
                        full_metadata = self.document_resolver(zotero_key)
                        if full_metadata:
                            doc.update(full_metadata)
                    except Exception as e:
                        print(f"Warning: Failed to resolve metadata for {zotero_key}: {e}")
                
                formatted_results.append(doc)
            
            return json.dumps({
                "documents": formatted_results,
                "count": len(formatted_results),
                "query": query,
                "limit_used": limit
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "documents": [],
                "count": 0
            })


class AgenticRAGEngine:
    """
    Agentic RAG Engine with iterative search and relevance filtering using CrewAI.
    
    Uses:
    - QdrantSearchTool: For searching documents (only tool)
    - Limit Estimation Agent: Estimates optimal document count from query
    - Search Agent: Uses tool to search documents based on estimated limit
    - Relevance Agent: Evaluates relevance (uses answers_llm)
    - Supervisor Agent: Orchestrates iterative process
    """
    
    def __init__(self, rag_engine: RAGEngine, 
                 agentic_llm_client: LLMClient,
                 answers_llm_client: LLMClient,
                 document_resolver: Optional[Callable] = None):
        """Initialize the agentic RAG engine.
        
        Args:
            rag_engine: RAG engine for Qdrant search
            agentic_llm_client: LLM client for agentic operations (from agentic_rag config)
            answers_llm_client: LLM client for relevance evaluation (from answers_llm config)
            document_resolver: Function to resolve Zotero keys to full metadata
        """
        self.rag_engine = rag_engine
        self.agentic_llm = agentic_llm_client
        self.answers_llm = answers_llm_client
        self.document_resolver = document_resolver or (lambda key: {"zotero_key": key})
        
        # Initialize tool
        self.search_tool = QdrantSearchTool(rag_engine, document_resolver)
        
        # Initialize CrewAI agents
        self._setup_agents()
    
    def _setup_agents(self):
        """Set up CrewAI agents for agentic RAG."""
        
        # Limit Estimation Agent - estimates optimal document count or extracts from query
        self.limit_estimation_agent = Agent(
            role="Search Limit Estimator",
            goal="Determine the optimal number of documents needed - either from explicit query mention or by analyzing complexity",
            backstory="""You are an expert at analyzing research queries to determine how many 
            documents should be retrieved. 
            
            FIRST, check if the query explicitly mentions a number of documents (e.g., "retrieve 15 items", 
            "find 20 papers", "get 10 documents", "show me 5 articles"). If found, use that number.
            
            If NO explicit number is mentioned, estimate based on query complexity:
            - Query complexity and scope (broad vs. specific)
            - Research requirements (comparative, comprehensive, focused)
            - Technical depth and domain coverage needed
            
            Guidelines for estimation:
            - Broad queries (reviews, surveys, comparisons): 15-50 documents
            - Moderate queries (specific topics with multiple aspects): 8-15 documents
            - Focused queries (single concept, specific paper): 3-8 documents
            
            You provide estimates between {MIN_LIMIT} and {MAX_LIMIT} documents with reasoning.""",
            tools=[],  # No tools - uses LLM analysis only
            verbose=True,
            llm=self.agentic_llm.model_name
        )
        
        # Search Agent - uses Qdrant search tool
        self.search_agent = Agent(
            role="Document Search Specialist",
            goal="Search for relevant academic documents in the Qdrant database",
            backstory="""You are an expert at searching academic databases. You use the Qdrant 
            search tool to find documents based on queries. You understand how to formulate 
            effective search queries and retrieve comprehensive results. You work with the 
            limit estimator to determine how many documents to search for.""",
            tools=[self.search_tool],
            verbose=True,
            llm=self.agentic_llm.model_name,
            output_json=True,  # Output JSON for processing
        )
        
        # Relevance Evaluator Agent - uses answers_llm for consistency
        self.relevance_agent = Agent(
            role="Relevance Evaluator",
            goal="Evaluate document relevance and filter out irrelevant results",
            backstory="""You are an expert at evaluating the relevance of academic documents 
            to research queries. You analyze document titles, abstracts, and metadata to 
            determine if they truly address the user's research question. You provide 
            relevance scores and decide which documents should be included.""",
            tools=[],  # No tools - uses LLM only
            verbose=True,
            llm=self.answers_llm.model_name
        )
        
        # Supervisor Agent - orchestrates the process
        self.supervisor_agent = Agent(
            role="Research Supervisor",
            goal="Coordinate iterative search and ensure high-quality relevant results",
            backstory="""You are a research supervisor who coordinates the search process. 
            You work with the limit estimator to determine document needs, direct the search 
            specialist, work with the relevance evaluator, and decide when sufficient relevant 
            documents have been found. You can request refined searches if needed.""",
            tools=[],  # No tools - coordinates other agents
            verbose=True,
            llm=self.agentic_llm.model_name
        )
    
    def agentic_search(self, query: str, limit: Optional[int] = None,
                      max_iterations: int = 3) -> Dict[str, Any]:
        """
        Perform iterative agentic search with relevance filtering using CrewAI.
        
        Implements a cycle between Search and Relevance agents that:
        - Searches for documents
        - Evaluates relevance and keeps relevant ones
        - Continues searching until target reached or max iterations hit
        - Stops if new search returns only irrelevant documents
        
        Args:
            query: Search query
            limit: Optional manual limit override
            max_iterations: Maximum search iterations (default: 3)
            
        Returns:
            Dictionary with documents and metadata
        """
        try:
            print(f"\n{'='*60}")
            print(f"Starting CrewAI iterative agentic search for: '{query}'")
            print(f"{'='*60}\n")
            
            # Step 1: Estimate document limit (or extract from query)
            if limit is not None:
                target_limit = limit
                print(f"Using manual limit: {target_limit}")
            else:
                target_limit = self._estimate_limit_with_agent(query)
                print(f"Estimated target limit: {target_limit}")
            
            # Step 2: Iterative search-relevance cycle
            all_relevant_docs = []
            seen_keys = set()
            current_query = query
            
            for iteration in range(1, max_iterations + 1):
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}/{max_iterations}")
                print(f"Relevant docs so far: {len(all_relevant_docs)}/{target_limit}")
                print(f"{'='*60}\n")
                
                # Check if we have enough documents
                if len(all_relevant_docs) >= target_limit:
                    print("✓ Target limit reached!")
                    break
                
                # Search for documents
                search_limit = min((target_limit - len(all_relevant_docs)) * 2 + 5, MAX_LIMIT)
                print(f"Searching for {search_limit} documents with query: '{current_query}'")
                
                search_results = self._search_with_agent(current_query, search_limit)
                
                if not search_results:
                    print("✗ No documents found in this iteration")
                    break
                
                # Filter out already seen documents
                new_docs = [doc for doc in search_results 
                           if doc.get("zotero_key") not in seen_keys]
                
                if not new_docs:
                    print("✗ No new documents found (all already seen)")
                    break
                
                print(f"Found {len(new_docs)} new documents")
                
                # Evaluate relevance
                eval_result = self._evaluate_relevance_with_agent(
                    query=query,
                    documents=new_docs,
                    target_limit=target_limit,
                    current_relevant_count=len(all_relevant_docs)
                )
                
                relevant_docs = eval_result.get("relevant_documents", [])
                irrelevant_docs = eval_result.get("irrelevant_documents", [])
                
                print(f"Relevant: {len(relevant_docs)}, Irrelevant: {len(irrelevant_docs)}")
                
                # Stop if no relevant documents found in this iteration
                if not relevant_docs:
                    print("✗ No relevant documents found in this iteration, stopping")
                    # But don't break immediately - check if we have enough documents overall
                    if len(all_relevant_docs) >= target_limit:
                        break
                    # If we still need more documents, continue to next iteration to try different search strategy
                    if len(all_relevant_docs) < target_limit and iteration < max_iterations:
                        continue
                    else:
                        break
                
                # Add relevant documents
                for doc in relevant_docs:
                    zotero_key = doc.get("zotero_key")
                    if zotero_key and zotero_key not in seen_keys:
                        all_relevant_docs.append(doc)
                        seen_keys.add(zotero_key)
                
                # Mark all processed documents as seen
                for doc in new_docs:
                    if doc.get("zotero_key"):
                        seen_keys.add(doc.get("zotero_key"))
                
                # Check if we need more iterations
                if len(all_relevant_docs) >= target_limit:
                    print("✓ Target limit reached!")
                    break
                
                # Refine query for next iteration if suggested
                refined_query = eval_result.get("refined_query")
                if refined_query and iteration < max_iterations:
                    current_query = refined_query
                    print(f"→ Refining query for next iteration: '{current_query}'")
                elif iteration < max_iterations:
                    print("→ No query refinement suggested, continuing with same query")
            
            # Step 3: Finalize results
            final_docs = self._finalize_with_supervisor(
                all_relevant_docs=all_relevant_docs,
                target_limit=target_limit,
                query=query
            )
            
            print(f"\n{'='*60}")
            print(f"Search complete: {len(final_docs)} relevant documents")
            print(f"{'='*60}\n")
            
            return {
                "documents": final_docs,
                "query": query,
                "total_found": len(final_docs),
                "success": True,
                "iterations_used": iteration
            }
            
        except Exception as e:
            print(f"Agentic search failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to standard search
            try:
                fallback_docs = self.rag_engine.search_documents(query, limit=limit or 5)
                return {
                    "documents": fallback_docs,
                    "query": query,
                    "total_found": len(fallback_docs),
                    "success": False,
                    "error": str(e),
                    "fallback_used": True
                }
            except Exception as fallback_error:
                return {
                    "documents": [],
                    "query": query,
                    "total_found": 0,
                    "success": False,
                    "error": f"Agentic: {e}, Fallback: {fallback_error}"
                }
    
    def unified_agentic_search(self, query: str, limit: Optional[int] = None,
                              max_iterations: int = 3) -> Dict[str, Any]:
        """
        Perform agentic search using a single unified Crew with all agents working together.
        
        This method creates a single Crew with all agents and a unified task that handles
        the entire agentic RAG process in one go, making it more efficient and easier to call
        from main.py.
        
        Args:
            query: Search query
            limit: Optional manual limit override
            max_iterations: Maximum search iterations (default: 3)
            
        Returns:
            Dictionary with documents and metadata
        """
        try:
            print(f"\n{'='*60}")
            print(f"Starting unified CrewAI agentic search for: '{query}'")
            print(f"{'='*60}\n")
            
            # Step 1: Estimate document limit (or extract from query)
            if limit is not None:
                target_limit = limit
                print(f"Using manual limit: {target_limit}")
            else:
                target_limit = self._estimate_limit_with_agent(query)
                print(f"Estimated target limit: {target_limit}")
            
            # Step 2: Create a unified task that orchestrates the entire process
            unified_task = Task(
                description=f"""Perform a comprehensive agentic RAG search to find relevant academic documents for the query: "{query}"
                
                Follow these steps:
                1. FIRST, estimate the optimal number of documents needed (already done: {target_limit})
                2. Search for academic documents using the qdrant_search tool
                3. Evaluate document relevance using your expertise
                4. Continue searching iteratively if needed (up to {max_iterations} iterations)
                5. Return only the most relevant documents (target: {target_limit})
                
                Search Strategy:
                - Start with the original query: "{query}"
                - For subsequent iterations, refine the query based on relevance feedback
                - Search for more documents than needed to ensure quality filtering
                - Deduplicate documents across iterations
                - Stop when you have enough relevant documents or reach max iterations
                
                Relevance Evaluation Criteria:
                - Direct relevance to the research query
                - Quality of the research (peer-reviewed sources preferred)
                - Recency of publication (unless historical context is needed)
                - Comprehensiveness of coverage
                
                For each document, provide:
                1. Relevance assessment (highly relevant, somewhat relevant, not relevant)
                2. Brief reasoning for the assessment
                3. Relevance score (0.0-1.0)
                
                Final Output Requirements:
                Return a JSON object with this structure:
                {{
                  "documents": [list of relevant documents with full metadata and relevance scores],
                  "query": "{query}",
                  "total_found": <number of relevant documents>,
                  "iterations_used": <number of search iterations performed>,
                  "search_strategy": "brief description of the search approach used"
                }}
                
                Ensure that only highly relevant documents (relevance score > 0.5) are included in the final results.
                """,
                agent=self.supervisor_agent,
                expected_output="JSON with list of relevant documents and metadata",
                output_json=UnifiedSearchResult
            )
            
            # Step 3: Create a unified Crew with all agents
            unified_crew = Crew(
                agents=[
                    self.limit_estimation_agent,
                    self.search_agent,
                    self.relevance_agent,
                    self.supervisor_agent
                ],
                tasks=[unified_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Step 4: Execute the unified Crew
            result = unified_crew.kickoff()
            
            # Step 5: Process and return results
            try:
                # Handle Pydantic model output
                if hasattr(result, 'dict'):
                    # If result is a Pydantic model, convert to dict
                    result_data = result.dict()
                elif hasattr(result, 'json'):
                    # If result has json method, use it
                    result_data = result.json()
                    if isinstance(result_data, str):
                        result_data = json.loads(result_data)
                else:
                    # Try to parse as JSON string
                    result_data = json.loads(str(result))
                
                # Handle both dict and Pydantic model outputs
                if isinstance(result_data, dict):
                    documents = result_data.get("documents", [])
                    total_found = result_data.get("total_found", len(documents))
                    iterations_used = result_data.get("iterations_used", 1)
                    search_strategy = result_data.get("search_strategy", "unified agentic search")
                    
                    print(f"\n{'='*60}")
                    print(f"Unified search complete: {total_found} relevant documents")
                    print(f"Iterations used: {iterations_used}")
                    print(f"Search strategy: {search_strategy}")
                    print(f"{'='*60}\n")
                    
                    return {
                        "documents": documents,
                        "query": query,
                        "total_found": total_found,
                        "success": True,
                        "iterations_used": iterations_used,
                        "search_strategy": search_strategy
                    }
                elif hasattr(result_data, 'documents'):
                    # If it's a Pydantic model instance
                    documents = result_data.documents
                    total_found = result_data.total_found
                    iterations_used = result_data.iterations_used
                    search_strategy = result_data.search_strategy
                    
                    print(f"\n{'='*60}")
                    print(f"Unified search complete: {total_found} relevant documents")
                    print(f"Iterations used: {iterations_used}")
                    print(f"Search strategy: {search_strategy}")
                    print(f"{'='*60}\n")
                    
                    return {
                        "documents": documents,
                        "query": query,
                        "total_found": total_found,
                        "success": True,
                        "iterations_used": iterations_used,
                        "search_strategy": search_strategy
                    }
            except (json.JSONDecodeError, AttributeError):
                # If JSON parsing fails, return the raw result
                print(f"\n{'='*60}")
                print("Unified search completed with raw result")
                print(f"{'='*60}\n")
                
                return {
                    "documents": [],
                    "query": query,
                    "total_found": 0,
                    "success": False,
                    "raw_result": str(result),
                    "error": "Could not parse result as JSON"
                }
            
        except Exception as e:
            print(f"Unified agentic search failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to standard search
            try:
                fallback_docs = self.rag_engine.search_documents(query, limit=limit or 5)
                return {
                    "documents": fallback_docs,
                    "query": query,
                    "total_found": len(fallback_docs),
                    "success": False,
                    "error": str(e),
                    "fallback_used": True
                }
            except Exception as fallback_error:
                return {
                    "documents": [],
                    "query": query,
                    "total_found": 0,
                    "success": False,
                    "error": f"Agentic: {e}, Fallback: {fallback_error}"
                }

    def _estimate_limit_with_agent(self, query: str) -> int:
        """Use limit estimation agent to determine optimal document count."""
        try:
            task = Task(
                description=f"""Analyze this research query and determine the optimal number of documents needed:
                
                Query: "{query}"
                
                STEP 1: Check if the query explicitly mentions a number of documents
                Look for phrases like: "retrieve X items", "find X papers", "get X documents", "show me X articles", 
                "X papers about", "top X results", etc.
                
                If explicit number found: Use that number (ensure it's between {MIN_LIMIT} and {MAX_LIMIT})
                
                STEP 2: If NO explicit number, estimate based on query complexity
                
                Return JSON with:
                {{
                  "estimated_limit": <number between {MIN_LIMIT} and {MAX_LIMIT}>,
                  "source": "explicit" or "estimated",
                  "reasoning": "brief explanation"
                }}""",
                agent=self.limit_estimation_agent,
                expected_output="JSON with document limit"
            )
            
            crew = Crew(
                agents=[self.limit_estimation_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            result_str = str(result)
            
            # Extract JSON using a more robust approach
            start_match = re.search(r'\{\s*"estimated_limit"', result_str)
            if start_match:
                # Find the matching closing brace
                start_pos = start_match.start()
                brace_count = 0
                pos = start_pos
                json_str = None
                
                while pos < len(result_str):
                    if result_str[pos] == '{':
                        brace_count += 1
                    elif result_str[pos] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            json_str = result_str[start_pos:pos+1]
                            break
                    pos += 1
                
                if json_str:
                    try:
                        parsed = json.loads(json_str)
                        limit = parsed.get("estimated_limit", 10)
                        source = parsed.get("source", "estimated")
                        reasoning = parsed.get("reasoning", "")
                        print(f"Limit {source}: {limit} - {reasoning}")
                        return max(MIN_LIMIT, min(MAX_LIMIT, limit))
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing failed for limit estimation: {e}")
            
            return 10  # Default fallback
            
        except Exception as e:
            print(f"Limit estimation failed: {e}, using default 10")
            return 10
    
    def _search_with_agent(self, query: str, limit: int) -> List[Dict]:
        """Use search agent to find documents."""
        try:
            task = Task(
                description=f"""Search for academic documents related to this query: "{query}"
                
                Use the qdrant_search tool to find {limit} documents.
                
                Return the search results in JSON format.""",
                agent=self.search_agent,
                expected_output="JSON with list of documents",
                output_json=SearchResultsModel
            )
            
            crew = Crew(
                agents=[self.search_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            # Try to obtain a dict from the result object safely
            result_json = {}
            if hasattr(result, "json"):
                try:
                    r = result.json_dict
                    # result.json() might return a dict or a JSON string
                    if isinstance(r, str):
                        result_json = json.loads(r)
                    elif isinstance(r, dict):
                        result_json = r
                    else:
                        # If it's a Pydantic model, .dict() is available
                        if hasattr(r, "dict"):
                            result_json = r.dict()
                except Exception as e:
                    print(f"Error parsing search result JSON: {e}")
                    result_json = {}
            
            # Also try to parse from string representation
            if not result_json:
                result_str = str(result)
                json_match = re.search(r'\{.*"documents".*?\}', result_str, re.DOTALL)
                if json_match:
                    try:
                        result_json = json.loads(json_match.group(0))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing search result from string: {e}")
                        result_json = {}
            
            if result_json:
                docs = result_json.get("documents", [])
                
                return docs
            
            return []
            
        except Exception as e:
            print(f"Search with agent failed: {e}")
            return []
    
    def _evaluate_relevance_with_agent(self, query: str, documents: List[Dict],
                                      target_limit: int, current_relevant_count: int) -> Dict[str, Any]:
        """Use relevance agent to evaluate documents."""
        try:
            # Format documents for evaluation, preserving all metadata for mapping back
            formatted_docs = []
            for i, doc in enumerate(documents):
                formatted_doc = {
                    "index": i,
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", "")[:500],
                    "year": doc.get("year", ""),
                    "zotero_key": doc.get("zotero_key", "")
                }
                
                formatted_docs.append(formatted_doc)
            
            task = Task(
                description=f"""Evaluate the relevance of these documents for query: "{query}"
                
                Documents:
                {json.dumps(formatted_docs, indent=2)}
                
                Current status:
                - Target: {target_limit} relevant documents
                - Already found: {current_relevant_count} relevant documents
                - Need: {max(0, target_limit - current_relevant_count)} more
                
                For each document:
                1. Determine if it's relevant (true/false) in is_relevant field
                2. Assign relevance score (0.0-1.0)
                3. Provide brief reasoning
                
                Also determine if we need more search and suggest a refined query if needed.
                
                Return JSON with:
                {{
                  "relevant_documents": [list with is_relevant=true, relevance_score and relevance_reason],
                  "irrelevant_documents": [list with is_relevant=false],
                  "needs_more_search": boolean,
                  "refined_query": "optional refined query"
                }}""",
                agent=self.relevance_agent,
                expected_output="JSON with relevance evaluation"
            )
            
            crew = Crew(
                agents=[self.relevance_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            result_str = str(result)
            
            # Extract JSON using a more robust approach
            start_match = re.search(r'\{\s*"relevant_documents"', result_str)
            if start_match:
                # Find the matching closing brace
                start_pos = start_match.start()
                brace_count = 0
                pos = start_pos
                json_str = None
                
                while pos < len(result_str):
                    if result_str[pos] == '{':
                        brace_count += 1
                    elif result_str[pos] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            json_str = result_str[start_pos:pos+1]
                            break
                    pos += 1
                
                if json_str:
                    try:
                        parsed = json.loads(json_str)
                        
                        # Map back to original documents
                        relevant_docs = []
                        irrelevant_docs = []
                        
                        # Process relevant documents
                        for eval_doc in parsed.get("relevant_documents", []):
                            idx = eval_doc.get("index")
                            if idx is not None and 0 <= idx < len(documents):
                                # Check if the document is actually marked as relevant
                                is_relevant = eval_doc.get("is_relevant", False)
                                if is_relevant:
                                    # Preserve all original document metadata and add relevance information
                                    doc = documents[idx].copy()
                                    doc["relevance_score"] = eval_doc.get("relevance_score", 0.5)
                                    doc["relevance_reason"] = eval_doc.get("relevance_reason", "")
                                    doc["is_relevant"] = True
                                    relevant_docs.append(doc)
                        
                        # Process irrelevant documents
                        for eval_doc in parsed.get("irrelevant_documents", []):
                            idx = eval_doc.get("index")
                            if idx is not None and 0 <= idx < len(documents):
                                # Check if the document is actually marked as irrelevant
                                is_relevant = eval_doc.get("is_relevant", True)
                                if not is_relevant:
                                    # Preserve all original document metadata and mark as irrelevant
                                    doc = documents[idx].copy()
                                    doc["is_relevant"] = False
                                    irrelevant_docs.append(doc)

                        return {
                            "relevant_documents": relevant_docs,
                            "irrelevant_documents": irrelevant_docs,
                            "needs_more_search": parsed.get("needs_more_search", False),
                            "refined_query": parsed.get("refined_query")
                        }
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing failed: {e}")
            
            # Fallback: return all as potentially relevant
            return {
                "relevant_documents": documents,
                "irrelevant_documents": [],
                "needs_more_search": current_relevant_count + len(documents) < target_limit,
                "refined_query": None
            }
            
        except Exception as e:
            print(f"Relevance evaluation failed: {e}")
            return {
                "relevant_documents": documents,
                "irrelevant_documents": [],
                "needs_more_search": False,
                "refined_query": None
            }
    
    def _finalize_with_supervisor(self, all_relevant_docs: List[Dict],
                                  target_limit: int, query: str) -> List[Dict]:
        """Use supervisor agent to finalize and rank results."""
        try:
            # Sort by relevance score
            sorted_docs = sorted(all_relevant_docs,
                               key=lambda x: x.get("relevance_score", 0),
                               reverse=True)
            
            # Return top N documents
            return sorted_docs[:target_limit]
            
        except Exception as e:
            print(f"Finalization failed: {e}")
            return all_relevant_docs[:target_limit]
