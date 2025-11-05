"""
Simplified Agentic RAG implementation using CrewAI with a single agent
that replicates the standard RAG functionality for querying the Qdrant database.
"""

import json
from typing import Dict, List, Optional, Any, Callable
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class QdrantSearchTool(BaseTool):
    """Custom tool for searching documents in Qdrant database."""
    name: str = "qdrant_search"
    description: str = "Search academic documents in Qdrant database"
    rag_engine: Any = Field(description="RAG engine for document search")
    document_resolver: Optional[Callable] = Field(default=None, description="Function to resolve Zotero keys to full document metadata")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, rag_engine: Any, document_resolver: Optional[Callable] = None):
        super().__init__(rag_engine=rag_engine, document_resolver=document_resolver)
    
    def _run(self, query: str, limit: int) -> str:
        """Execute search in Qdrant database and return enriched, formatted results.

        This method performs a document search using the configured RAG engine, optionally
        enriches the results by resolving Zotero keys to full document metadata using the
        document resolver function, and formats the output into a standardized JSON structure
        suitable for agent consumption.

        The enrichment process attempts to merge RAG search results with complete document
        metadata (title, abstract, authors, etc.) when a document resolver is available.
        If enrichment fails for any document, the original RAG result is used as fallback.

        Args:
            query: The search query string to find relevant documents
            limit: Maximum number of documents to return from the search

        Returns:
            JSON string containing a dictionary with:
            - 'documents': List of formatted document objects with metadata
            - 'query': The original search query
            - 'limit': The requested limit parameter
            - 'total_results': Number of documents returned
            On error, returns JSON with 'error' key and empty documents list
        """
        try:
            # Use the RAG engine to search documents in Qdrant
            results = self.rag_engine.search_documents(query, limit=limit)
            
            # Resolve Zotero keys to full document metadata if resolver is available
            enriched_results = []
            for doc in results:
                zotero_key = doc.get("zotero_key")
                if zotero_key and self.document_resolver:
                    try:
                        full_doc = self.document_resolver(zotero_key)
                        if full_doc:
                            # Merge the RAG result with full document info
                            enriched_doc = full_doc.copy()
                            enriched_doc.update(doc)  # RAG results take precedence for score, etc.
                            enriched_results.append(enriched_doc)
                        else:
                            # If resolver fails, add the original document
                            enriched_results.append(doc)
                    except Exception:
                        # If resolver fails, add the original document
                        enriched_results.append(doc)
                else:
                    enriched_results.append(doc)
            
            # Format results for agent consumption
            formatted_results = []
            for doc in enriched_results:
                formatted_doc = {
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "authors": doc.get("authors", []),
                    "year": doc.get("year", ""),
                    "doi": doc.get("doi", ""),
                    "score": doc.get("score", 0.0),
                    "zotero_key": doc.get("zotero_key", "")
                }
                formatted_results.append(formatted_doc)
            
            return json.dumps({
                "documents": formatted_results,
                "query": query,
                "limit": limit,
                "total_results": len(formatted_results)
            })
        except Exception as e:
            return json.dumps({"error": str(e), "documents": []})


class AgenticRAGEngine:
    """
    Simplified Agentic RAG Engine that uses a single CrewAI agent
    to replicate standard RAG functionality for querying Qdrant database.
    """
    
    def __init__(self, rag_engine: Any, document_resolver: Optional[Callable] = None, 
                 agent_llm_client: Optional[Any] = None):
        """Initialize the simplified agentic RAG engine.
        
        Args:
            rag_engine: The RAG engine for document search
            document_resolver: Function to resolve Zotero keys to full document metadata
            agent_llm_client: LLM client for agent operations (uses agent_llm config from llm_config.json)
        """
        self.rag_engine = rag_engine
        self.document_resolver = document_resolver
        self.agent_llm_client = agent_llm_client
        self.qdrant_tool = QdrantSearchTool(rag_engine=rag_engine, document_resolver=document_resolver)
        self._setup_agent()
    
    def _setup_agent(self):
        """Set up a single agent that handles all RAG operations."""
        # Determine which LLM to use for the agent
        agent_llm = self.agent_llm_client.model_name if self.agent_llm_client else None
        
        self.query_agent = Agent(
            role="Intelligent Research Query Processor",
            goal="Process research queries, estimate document needs, filter for relevance, and ensure sufficient high-quality results",
            backstory="""You are an expert research assistant with advanced document retrieval capabilities.
            
            CRITICAL SEARCH RULES:
            - STRICTLY preserve exact terms, abbreviations, and chemical formulas from the original query
            - May expand abbreviations unless explicitly requested (e.g., search "ML" as "machine learning")
            - Do NOT modify chemical formulas (e.g., "H2O", "CO2", "C6H12O6" must remain exact)
            - Do NOT change technical terms or acronyms
            - You have EXACTLY 3 database searches maximum - use them wisely
            
            Your responsibilities include:
            1. ESTIMATE DOCUMENT COUNT: Analyze the query to determine how many documents are needed.
               - Broad queries ("overview", "survey", "what are") need 8-15 documents
               - Specific queries ("how does X work", specific method/technique) need 3-7 documents
               - Comparative queries ("compare X and Y") need 5-10 documents
               - IMPORTANT: Always specify a limit parameter when using the qdrant_search tool

            2. ASSESS RELEVANCE: After retrieving documents, critically evaluate each by:
               - Title relevance: Does the title match the query topic?
               - Abstract relevance: Does the abstract address the query?
               - Keywords relevance: Do keywords align with the query?
               - Keep all documents that are relevant

            3. ITERATIVE SEARCH (MAXIMUM 3 TOTAL SEARCHES):
               - First search: Use original query with appropriate limit
               - If you need more relevant documents and searches < 3:
                 * Second search: Try slight variations with appropriate limit
                 * Third search: Try broader formulation with appropriate limit
               - STOP after 3 searches regardless of results
               - Always specify limit parameter in qdrant_search calls
            
            You have access to a Qdrant search tool that retrieves documents with full metadata 
            including title, abstract, keywords, authors, year, DOI, and relevance scores.""",
            tools=[self.qdrant_tool],
            verbose=True,
            llm=agent_llm
        )
    
    def search(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Perform a search using the agentic approach with a single agent.
        
        Args:
            query: The research query
            limit: Maximum number of documents to return
            
        Returns:
            List of documents retrieved from the Qdrant database
        """
        # Create a task for the agent to process the query
        search_task = Task(
            description=f"""Process the following research query and retrieve relevant documents:
            Query: "{query}"
            {"Limit: " + str(limit) if limit else "Estimate the optimal number of documents needed"}
            
            Follow this workflow:
            
            STEP 1 - ESTIMATE DOCUMENT COUNT:
            - Analyze the query type and complexity
            - Determine how many documents are needed (3-15 range)
            - Consider: Is it broad/survey? Specific/technical? Comparative?
            - Always specify a limit parameter when using the qdrant_search tool

            STEP 2 - INITIAL SEARCH:
            - PRESERVE exact terms, abbreviations, and chemical formulas from the query
            - Use qdrant_search tool with an appropriate limit parameter
            - Retrieve documents with full metadata

            STEP 3 - ASSESS RELEVANCE:
            - For each document, evaluate:
              * Title: Does it match the query topic?
              * Abstract: Does it address the research question?
              * Keywords: Are they aligned with the query?
            - Keep all relevant documents and cut based on estimated count
            - Count how many relevant documents remain

            STEP 4 - ITERATIVE SEARCH (MAXIMUM 3 TOTAL SEARCHES):
            - If you need more relevant documents and searches < 3:
              * Rewrite the query (preserve exact terms/abbreviations/formulas)
              * Perform additional search with appropriate limit parameter
              * Assess new results for relevance
              * Repeat once more if still insufficient (max 3 searches total)
            - STOP after 3 searches regardless of results

            STEP 5 - RETURN RESULTS:
            - Return relevant documents up to the estimated count with full metadata
            - Include a summary of your process (estimated count, searches performed, relevance filtering)
            
            The qdrant_search tool will automatically resolve Zotero keys to full document metadata.
            """,
            agent=self.query_agent,
            expected_output="""A JSON object containing:
            - 'documents': List of relevant academic documents with full metadata
            - 'estimated_count': Number of documents you estimated were needed
            - 'searches_performed': Number of search iterations
            - 'query_variations': List of query variations used
            - 'relevance_summary': Brief summary of relevance filtering applied
            - 'total_retrieved': Total documents retrieved before filtering
            - 'total_relevant': Total relevant documents after filtering"""
        )
        
        # Create and execute the crew with a single agent
        crew = Crew(
            agents=[self.query_agent],
            tasks=[search_task],
            verbose=True
        )
        
        # Execute the crew and get results
        result = crew.kickoff()
        
        # Parse the results to extract documents
        try:
            if isinstance(result, str):
                result_data = json.loads(result)
                if "documents" in result_data:
                    return result_data["documents"]
            
            # If we can't parse the result, try to get documents directly from RAG engine
            # This is a fallback to ensure we always return results
            return self._get_enriched_documents(query, limit=limit)
        except Exception:
            # Final fallback to direct RAG search
            return self._get_enriched_documents(query, limit=limit)
    
    def _get_enriched_documents(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        """Get enriched documents by searching and resolving Zotero keys.

        Args:
            query: The search query
            limit: Maximum number of documents to return

        Returns:
            List of enriched documents with full metadata
        """
        # Use estimated limit if none provided (for fallback cases)
        if limit is None:
            limit = self.rag_engine._estimate_optimal_limit(query)

        # Use the RAG engine to search documents in Qdrant
        results = self.rag_engine.search_documents(query, limit=limit)
        
        # Resolve Zotero keys to full document metadata if resolver is available
        enriched_results = []
        for doc in results:
            zotero_key = doc.get("zotero_key")
            if zotero_key and self.document_resolver:
                try:
                    full_doc = self.document_resolver(zotero_key)
                    if full_doc:
                        # Merge the RAG result with full document info
                        enriched_doc = full_doc.copy()
                        enriched_doc.update(doc)  # RAG results take precedence for score, etc.
                        enriched_results.append(enriched_doc)
                    else:
                        # If resolver fails, add the original document
                        enriched_results.append(doc)
                except Exception:
                    # If resolver fails, add the original document
                    enriched_results.append(doc)
            else:
                enriched_results.append(doc)
        
        return enriched_results
    
    def agentic_search(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform agentic search using a single agent approach.
        
        Args:
            query: The research query
            limit: Maximum number of documents to return
            
        Returns:
            Dictionary containing search results and metadata
        """
        try:
            # Use the agentic search with the agent and tool
            results = self.search(query, limit)
            
            return {
                "documents": results,
                "query": query,
                "limit": limit,
                "method": "agentic_single_agent",
                "success": True
            }
            
        except Exception as e:
            # Fallback to standard RAG in case of any issues
            print(f"Agentic search failed, falling back to standard RAG: {e}")
            try:
                fallback_results = self._get_enriched_documents(query, limit=limit)
                return {
                    "documents": fallback_results,
                    "query": query,
                    "limit": limit,
                    "method": "standard_rag_fallback",
                    "success": True,
                    "error": str(e)
                }
            except Exception as fallback_error:
                return {
                    "documents": [],
                    "query": query,
                    "limit": limit,
                    "method": "failed",
                    "success": False,
                    "error": str(fallback_error)
                }
