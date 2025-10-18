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
    
    def _run(self, query: str, limit: int = 5) -> str:
        """Execute search in Qdrant database.
        
        Args:
            query: The search query
            limit: Maximum number of documents to return
            
        Returns:
            JSON string with search results
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
            role="Research Query Processor",
            goal="Process research queries and retrieve relevant documents from the Qdrant database",
            backstory="""You are an expert research assistant that specializes in retrieving 
            academic documents from a vector database. You understand research queries and 
            can effectively search for relevant documents using semantic search techniques. 
            You have access to a Qdrant search tool that allows you to query the database directly 
            and retrieve full document metadata.""",
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
            {"Limit: " + str(limit) if limit else "Use default limit estimation"}
            
            Your task is to:
            1. Understand the research query
            2. Use the qdrant_search tool to find relevant documents in the Qdrant database
            3. Return the most relevant documents found with full metadata
            
            Use the qdrant_search tool with appropriate parameters to search for documents.
            The tool will automatically resolve Zotero keys to full document metadata.
            """,
            agent=self.query_agent,
            expected_output="A list of relevant academic documents with full metadata in JSON format"
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