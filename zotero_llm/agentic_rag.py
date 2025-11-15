"""
Simplified Agentic RAG implementation using CrewAI with a single agent
that replicates the standard RAG functionality for querying the Qdrant database.
"""

from crewai.crews.crew_output import CrewOutput
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Set up logging for agentic responses
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentic.log')
    ]
)
agentic_logger = logging.getLogger('agentic_rag')


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
            # The RAG engine now supports search state management for continuation
            results = self.rag_engine.search_documents(query, limit=limit)
            
            # Log the search results
            agentic_logger.info(f"QdrantSearchTool._run: Search returned {len(results)} results for query: '{query}' with limit: {limit}")
            
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
                    except Exception as e:
                        # If resolver fails, add the original document
                        agentic_logger.warning(f"QdrantSearchTool._run: Failed to resolve document {zotero_key}: {str(e)}")
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
            
            response = json.dumps({
                "documents": formatted_results,
                "query": query,
                "limit": limit,
                "total_results": len(formatted_results)
            })
            
            # Log the formatted response
            agentic_logger.info(f"QdrantSearchTool._run: Response formatted with {len(formatted_results)} documents")
            return response
        except Exception as e:
            error_response = json.dumps({"error": str(e), "documents": []})
            agentic_logger.error(f"QdrantSearchTool._run: Error occurred: {str(e)}")
            return error_response


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
            - ADAPTIVE SEARCH: If previous search results were highly relevant, continue with the same query; if not relevant enough, rewrite query for better results
            
            Your responsibilities include:
            1. ESTIMATE DOCUMENT COUNT: Analyze the query to determine how many documents are needed.
               - Broad queries ("overview", "survey", "what are") need 8-15 documents
               - Specific queries ("how does X work", specific method/technique) need 3-7 documents
               - Comparative queries ("compare X and Y") need 5-10 documents
               - IMPORTANT: Always specify a limit parameter when using the qdrant_search tool

            2. ASSESS RELEVANCE: After retrieving documents, critically evaluate each by:
               - Title relevance: Does the title match the query topic?
               - Abstract relevance: Does the abstract address the research question?
               - Keywords relevance: Do keywords align with the query?
               - Score relevance: Are scores above 0.7 for highly relevant results?
               - Keep all documents that are relevant

            3. ADAPTIVE SEARCH STRATEGY:
               - If current results are highly relevant (scores > 0.7), continue search with same query
               - If results are not relevant enough, rewrite query for better vector search while preserving key terms
               - Use your judgment based on document relevance to decide whether to continue or rewrite

            4. ITERATIVE SEARCH (MAXIMUM 3 TOTAL SEARCHES):
               - First search: Use original query with appropriate limit
               - If you need more relevant documents and searches < 3:
                 * Second search: If first results were relevant, continue with same query; if not, rewrite for better results
                 * Third search: If second results were relevant, continue with same query; if not, rewrite for better results
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

            STEP 3 - ASSESS RELEVANCE AND ADAPTIVE STRATEGY:
            - For each document, evaluate:
              * Title: Does it match the query topic?
              * Abstract: Does it address the research question?
              * Keywords: Are they aligned with the query?
              * Score: Are scores above 0.7 for highly relevant results?
            - JUDGE RELEVANCE: Decide if results are highly relevant or need improvement
            - COUNT RELEVANT DOCUMENTS: Keep track of how many documents you consider relevant
            - ADAPTIVE DECISION: 
              * If results are highly relevant, continue with same query to get more results
              * If results are not relevant enough, rewrite query for better vector search while preserving key terms

            STEP 4 - ITERATIVE SEARCH (MAXIMUM 3 TOTAL SEARCHES):
            - If you need more relevant documents and searches < 3:
              * Rewrite the query (preserve exact terms/abbreviations/formulas) for better vector search
              * Perform additional search with appropriate limit parameter
              * Assess new results for relevance using your judgment
              * COUNT RELEVANT DOCUMENTS: Keep track of how many documents you consider relevant
              * Repeat once more if still insufficient (max 3 searches total)
            - STOP after 3 searches regardless of results

            STEP 5 - RETURN RESULTS:
            - Return relevant documents up to the estimated count with full metadata
            - Include a summary of your process (estimated count, searches performed, relevance filtering)
            - ENSURE CONSISTENCY: The 'total_relevant' field should match the number of documents you actually include in your response
            - BE TRANSPARENT: If you filtered documents, explain why in the relevance_summary
            
            The qdrant_search tool will automatically resolve Zotero keys to full document metadata.
            Use your judgment to decide whether to continue with the same query or rewrite based on relevance.
            """,
            agent=self.query_agent,
            expected_output="""A JSON object containing:
            - 'documents': List of relevant academic documents with full metadata
            - 'estimated_count': Number of documents you estimated were needed
            - 'searches_performed': Number of search iterations
            - 'query_variations': List of query variations used
            - 'relevance_summary': Brief summary of relevance filtering applied
            - 'total_retrieved': Total documents retrieved before filtering
            - 'total_relevant': Total relevant documents after filtering (should match length of 'documents' array)"""
        )
        
        # Create and execute the crew with a single agent
        crew = Crew(
            agents=[self.query_agent],
            tasks=[search_task],
            verbose=True
        )
        
        # Execute the crew and get results
        result = crew.kickoff()
        
        # Log the agent's response
        agentic_logger.info(f"AgenticRAGEngine.search: Agent response for query '{query}': {result}")
        
        # Parse the results to extract documents
        try:
            if result.json_dict:
                result_data = result.json_dict
            elif isinstance(result, dict):
                result_data = result
            else:
                result_data = json.loads(result.raw)
            
            if "documents" in result_data:
                # Log the parsed results
                agentic_logger.info(f"AgenticRAGEngine.search: Parsed documents count: {len(result_data['documents'])}")
                agentic_logger.info(f"AgenticRAGEngine.search: Total relevant reported: {result_data.get('total_relevant', 'N/A')}")
                return result_data["documents"]
            else:
                # Handle case where result is JSON but doesn't contain documents
                agentic_logger.warning(f"AgenticRAGEngine.search: Agent response doesn't contain documents: {result}")
                return []
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            agentic_logger.error(f"AgenticRAGEngine.search: Failed to parse agent response as JSON: {str(e)}")
            return []
        except Exception as e:
            # Handle any other errors
            agentic_logger.error(f"AgenticRAGEngine.search: Unexpected error processing agent response: {str(e)}")
            return []
    
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
        # The RAG engine now supports search state management for continuation
        results = self.rag_engine.search_documents(query, limit=limit)
        
        # Log the RAG search results
        agentic_logger.info(f"AgenticRAGEngine._get_enriched_documents: RAG search returned {len(results)} documents for query: '{query}'")
        
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
                except Exception as e:
                    # If resolver fails, add the original document
                    agentic_logger.warning(f"AgenticRAGEngine._get_enriched_documents: Failed to resolve document {zotero_key}: {str(e)}")
                    enriched_results.append(doc)
            else:
                enriched_results.append(doc)
        
        agentic_logger.info(f"AgenticRAGEngine._get_enriched_documents: Enriched {len(enriched_results)} documents for query: '{query}'")
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
            
            response = {
                "documents": results,
                "query": query,
                "limit": limit,
                "method": "agentic_single_agent",
                "success": True
            }
            
            # Log the final response
            agentic_logger.info(f"AgenticRAGEngine.agentic_search: Agentic search completed for query '{query}' with {len(results)} documents")
            return response
            
        except Exception as e:
            # No fallback - return error response
            agentic_logger.error(f"AgenticRAGEngine.agentic_search: Agentic search failed for query '{query}': {e}")
            return {
                "documents": [],
                "query": query,
                "limit": limit,
                "method": "failed",
                "success": False,
                "error": str(e)
            }
