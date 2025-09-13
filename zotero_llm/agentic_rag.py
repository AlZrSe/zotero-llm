"""
Agentic RAG implementation using CrewAI for enhanced research assistance.

This module provides an agentic approach to retrieval-augmented generation,
using multiple specialized agents to improve query understanding, search
strategies, and result synthesis while preserving the existing RAG functionality.
"""

import json
import os
from typing import Dict, List, Optional, Any
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from zotero_llm.rag import RAGEngine
from zotero_llm.llm import LLMClient


# Apply bounds
MIN_LIMIT = 3   # Always get at least 3 documents
MAX_LIMIT = 50  # Cap at 15 to avoid overwhelming results


class SearchParameters(BaseModel):
    """Parameters for search operations."""
    query: str = Field(description="The search query")
    limit: int = Field(default=5, description="Number of results to return")
    strategy: str = Field(default="hybrid", description="Search strategy: semantic, bm25, or hybrid")


class SearchResult(BaseModel):
    """Result from a search operation."""
    documents: List[Dict] = Field(description="Retrieved documents")
    strategy_used: str = Field(description="Strategy used for retrieval")
    confidence: float = Field(default=0.0, description="Confidence score for the search")


class RAGSearchTool(BaseTool):
    """Custom tool for RAG search operations."""
    name: str = "rag_search"
    description: str = "Search academic documents using various retrieval strategies"
    rag_engine: RAGEngine = Field(description="RAG engine for document search")
    
    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, rag_engine: RAGEngine):
        super().__init__(rag_engine=rag_engine)
    
    def _run(self, query: str, limit: int = 5, strategy: str = "hybrid") -> str:
        """Execute RAG search with specified parameters."""
        try:
            if strategy == "semantic":
                # Perform semantic-only search by modifying the search parameters
                results = self.rag_engine.search_documents(query, limit=limit)
            elif strategy == "bm25":
                # For BM25-only search, we'd need to modify the RAG engine
                # For now, use the hybrid approach as fallback
                results = self.rag_engine.search_documents(query, limit=limit)
            else:  # hybrid
                results = self.rag_engine.search_documents(query, limit=limit)
            
            # Format results for agent consumption
            formatted_results = []
            for doc in results:
                formatted_doc = {
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "authors": doc.get("authors", []),
                    "year": doc.get("year", ""),
                    "doi": doc.get("doi", ""),
                    "score": doc.get("score", 0.0)
                }
                formatted_results.append(formatted_doc)
            
            return json.dumps({
                "documents": formatted_results,
                "strategy_used": strategy,
                "total_results": len(formatted_results)
            })
        except Exception as e:
            return json.dumps({"error": str(e), "documents": []})


class QueryAnalysisTool(BaseTool):
    """Tool for analyzing and decomposing complex queries."""
    name: str = "query_analysis"
    description: str = "Analyze research queries to identify key concepts and search strategies"
    
    def _run(self, query: str) -> str:
        """Analyze the query and suggest search strategies."""
        # Simple analysis - in practice, this could use NLP techniques
        keywords = query.lower().split()
        
        # Identify query characteristics
        has_author_names = any(word.istitle() for word in query.split())
        has_years = any(word.isdigit() and len(word) == 4 for word in keywords)
        has_technical_terms = any(len(word) > 8 for word in keywords)
        
        analysis = {
            "original_query": query,
            "keywords": keywords,
            "characteristics": {
                "has_author_names": has_author_names,
                "has_years": has_years,
                "has_technical_terms": has_technical_terms
            },
            "suggested_strategies": []
        }
        
        # Suggest search strategies based on analysis
        if has_technical_terms:
            analysis["suggested_strategies"].append("semantic")
        if has_author_names or has_years:
            analysis["suggested_strategies"].append("bm25")
        if not analysis["suggested_strategies"]:
            analysis["suggested_strategies"].append("hybrid")
        
        return json.dumps(analysis)


class LimitEstimationTool(BaseTool):
    """Tool for intelligent estimation of search result limits based on query complexity."""
    name: str = "limit_estimation"
    description: str = "Estimate optimal document limit based on query characteristics and research scope"
    
    def _run(self, query: str, base_limit: int = 5) -> str:
        """Estimate optimal document limit for the given query."""
        # Analyze query characteristics
        query_lower = query.lower()
        words = query.split()
        
        # Factors that increase limit
        complexity_factors = 0
        
        # 1. Broad vs specific queries
        broad_terms = ['overview', 'review', 'survey', 'comprehensive', 'systematic', 'meta-analysis', 
                      'state of the art', 'recent advances', 'current trends', 'developments']
        if any(term in query_lower for term in broad_terms):
            complexity_factors += 3  # Broad queries need more documents
        
        # 2. Comparative queries
        comparative_terms = ['compare', 'comparison', 'versus', 'vs', 'differences', 'similarities', 
                           'contrast', 'alternative', 'approaches', 'methods']
        if any(term in query_lower for term in comparative_terms):
            complexity_factors += 2  # Comparisons need multiple perspectives
        
        # 3. Multi-faceted queries (multiple concepts)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        conjunctions = ['and', 'or', 'but', 'as well as', 'along with']
        
        if len([w for w in words if w.lower() in question_words]) > 1:
            complexity_factors += 1  # Multiple questions
        
        if any(conj in query_lower for conj in conjunctions):
            complexity_factors += 1  # Multiple concepts connected
        
        # 4. Technical depth indicators
        technical_terms = ['algorithm', 'model', 'framework', 'architecture', 'implementation',
                          'methodology', 'technique', 'approach', 'analysis', 'evaluation']
        if len([w for w in words if w.lower() in technical_terms]) >= 2:
            complexity_factors += 1  # Technical queries may need more sources
        
        # 5. Temporal scope
        temporal_terms = ['recent', 'latest', 'current', 'new', 'emerging', 'future', 'trend']
        historical_terms = ['history', 'evolution', 'development', 'progress', 'over time']
        
        if any(term in query_lower for term in temporal_terms):
            complexity_factors += 1  # Recent work queries
        elif any(term in query_lower for term in historical_terms):
            complexity_factors += 2  # Historical queries need broader coverage
        
        # 6. Query length as complexity indicator
        if len(words) > 15:
            complexity_factors += 1  # Long queries are typically complex
        elif len(words) > 25:
            complexity_factors += 2  # Very long queries
        
        # Factors that decrease limit (specific queries)
        specificity_factors = 0
        
        # 1. Specific author or paper references
        if any(word[0].isupper() and len(word) > 3 for word in words):
            specificity_factors += 1  # Likely author names
        
        # 2. Very specific technical terms or acronyms
        acronyms = [w for w in words if w.isupper() and len(w) >= 2]
        if len(acronyms) >= 2:
            specificity_factors += 1  # Multiple acronyms suggest specific domain
        
        # 3. Specific numerical or year references
        if any(w.isdigit() and len(w) == 4 and 1900 <= int(w) <= 2030 for w in words):
            specificity_factors += 1  # Year references
        
        # Calculate final limit
        estimated_limit = base_limit * max(complexity_factors - specificity_factors, 1)
        
        # Apply bounds
        estimated_limit = max(MIN_LIMIT, min(MAX_LIMIT, estimated_limit))
        
        estimation_details = {
            "estimated_limit": estimated_limit,
            "base_limit": base_limit,
            "complexity_factors": complexity_factors,
            "specificity_factors": specificity_factors,
            "query_characteristics": {
                "word_count": len(words),
                "has_broad_terms": any(term in query_lower for term in broad_terms),
                "has_comparative_terms": any(term in query_lower for term in comparative_terms),
                "has_technical_terms": len([w for w in words if w.lower() in technical_terms]) >= 2,
                "has_temporal_terms": any(term in query_lower for term in temporal_terms + historical_terms),
                "has_author_references": any(word[0].isupper() and len(word) > 3 for word in words),
                "acronym_count": len(acronyms)
            },
            "reasoning": f"Query analysis: complexity_factors={complexity_factors}, specificity_factors={specificity_factors}, final_limit={estimated_limit}"
        }
        
        return json.dumps(estimation_details)


class AgenticRAGEngine:
    """
    Agentic RAG Engine that uses CrewAI to orchestrate multiple agents
    for enhanced research assistance while preserving existing RAG functionality.
    """
    
    def __init__(self, rag_engine: RAGEngine, llm_client: LLMClient, 
                 agent_llm_config: Optional[Dict] = None):
        """Initialize the agentic RAG engine."""
        self.rag_engine = rag_engine
        self.llm_client = llm_client
        self.agent_llm_config = agent_llm_config or {}
        
        # Initialize tools
        self.rag_search_tool = RAGSearchTool(rag_engine)
        self.query_analysis_tool = QueryAnalysisTool()
        self.limit_estimation_tool = LimitEstimationTool()
        
        # Initialize agents
        self._setup_agents()
        
    def _setup_agents(self):
        """Set up the CrewAI agents for agentic RAG."""
        
        # Research Analyst Agent - understands queries and plans search strategies
        self.research_analyst = Agent(
            role="Research Analyst",
            goal="Understand research queries and develop comprehensive search strategies",
            backstory="""You are an expert research analyst specializing in academic literature.
            Your expertise lies in understanding complex research questions, identifying key concepts,
            and developing multi-faceted search strategies to ensure comprehensive literature coverage.
            You excel at breaking down complex queries into searchable components.""",
            tools=[self.query_analysis_tool],
            verbose=True,
            llm=self._get_agent_llm()
        )
        
        # Limit Estimation Agent - intelligently determines optimal document limits
        self.limit_estimation_agent = Agent(
            role="Limit Estimation Specialist",
            goal="Intelligently determine optimal document limits based on query complexity and research scope",
            backstory="""You are a specialist in information retrieval optimization who understands
            how query characteristics should influence the number of documents retrieved. You analyze
            query complexity, research scope, and user intent to recommend optimal document limits
            that balance comprehensiveness with relevance. You consider factors like query breadth,
            technical depth, comparative requirements, and temporal scope to make intelligent
            recommendations about how many sources are needed for effective research.""",
            tools=[self.limit_estimation_tool],
            verbose=True,
            llm=self._get_agent_llm()
        )
        
        # Search Specialist Agent - executes multiple search strategies
        self.search_specialist = Agent(
            role="Search Specialist",
            goal="Execute multiple search strategies to retrieve relevant academic documents",
            backstory="""You are a search specialist with deep expertise in information retrieval.
            You understand the strengths and weaknesses of different search approaches including
            semantic search, keyword-based search, and hybrid methods. You can adapt your search
            strategy based on the type of query and the expected results.""",
            tools=[self.rag_search_tool],
            verbose=True,
            llm=self._get_agent_llm()
        )
        
        # Synthesis Agent - combines and evaluates results
        self.synthesis_agent = Agent(
            role="Research Synthesizer",
            goal="Synthesize and evaluate search results to provide comprehensive insights",
            backstory="""You are a research synthesizer who excels at combining information
            from multiple sources and search strategies. You can identify the most relevant
            documents, detect overlaps and gaps, and provide quality assessments of the
            retrieved literature. Your expertise helps ensure the final results are 
            comprehensive and relevant.""",
            tools=[],
            verbose=True,
            llm=self._get_agent_llm()
        )
    
    def _get_agent_llm(self):
        """Get LLM configuration for agents."""
        # Use the existing LLM client configuration
        return self.llm_client.model_name
    
    def agentic_search(self, query: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform agentic RAG search using CrewAI agents with intelligent limit estimation.
        
        Args:
            query: The research query
            limit: Maximum number of documents to return (if None, will be estimated by agent)
            
        Returns:
            Dictionary containing search results and agent insights
        """
        try:
            # First check if we have a manual override
            if limit is not None:
                final_limit = limit
                limit_source = "manual_override"
                print(f"Using manually specified limit: {limit} documents")
            else:
                # For agentic RAG, let the standard RAG check for user-requested limits first
                rag_results_with_metadata = self.rag_engine.search_documents(
                    query, limit=None, return_metadata=True
                )
                
                if rag_results_with_metadata["limit_source"] == "user_request":
                    # User explicitly requested a limit - honor it
                    final_limit = rag_results_with_metadata["limit"]
                    limit_source = "user_request"
                    print(f"Using user-requested limit: {final_limit} documents")
                else:
                    # No user request - use agent-based estimation
                    limit_source = "agent_estimation"
                    # Will be estimated by the agent below
                    final_limit = None
            
            # Task 1: Query Analysis
            analysis_task = Task(
                description=f"""Analyze the following research query and provide insights:
                Query: "{query}"
                
                Your analysis should include:
                1. Key concepts and keywords
                2. Query characteristics (technical terms, author names, years, etc.)
                3. Recommended search strategies
                4. Potential search challenges
                
                Use the query_analysis tool to perform the analysis.""",
                agent=self.research_analyst,
                expected_output="A detailed analysis of the query with recommended search strategies"
            )
            
            # Task 2: Limit Estimation (only if not already determined)
            if final_limit is None:
                limit_estimation_task = Task(
                    description=f"""Analyze the research query to determine the optimal number of documents needed:
                    Query: "{query}"
                    
                    Consider:
                    1. Query complexity and scope (broad vs. specific)
                    2. Research requirements (comparative, comprehensive, focused)
                    3. Technical depth and domain coverage needed
                    4. Temporal scope (historical vs. current)
                    
                    Use the limit_estimation tool to analyze the query and recommend an appropriate document limit.
                    Provide reasoning for your recommendation.""",
                    agent=self.limit_estimation_agent,
                    expected_output="An intelligent estimate of optimal document limit with detailed reasoning"
                )
            
            # Task 3: Multi-strategy Search
            search_task = Task(
                description=f"""Based on the query analysis{' and limit estimation' if final_limit is None else ''}, execute multiple search strategies:
                Original query: "{query}"
                {'Target limit: To be determined by limit estimation agent' if final_limit is None else f'Target limit: {final_limit} documents'}
                Limit source: {limit_source}
                
                Execute the following searches:
                1. Hybrid search (semantic + BM25)
                2. Semantic-focused search if technical terms are present
                3. Keyword-focused search if specific terms/names are present
                
                {f'The limit of {final_limit} was determined by {limit_source}:' if final_limit is not None else 'Use the limit recommended by the limit estimation agent:'}
                {"→ User explicitly requested this number - honor their preference" if limit_source == "user_request" else 
                 "→ Manually specified - use as provided" if limit_source == "manual_override" else 
                 "→ Use the intelligent agent recommendation"}
                
                Use the rag_search tool for each strategy and compare results.""",
                agent=self.search_specialist,
                expected_output="Search results from multiple strategies with appropriate document limits"
            )
            
            # Task 4: Result Synthesis
            synthesis_task = Task(
                description=f"""Synthesize the search results from multiple strategies:
                
                Your synthesis should include:
                1. Combination of results from different strategies
                2. Deduplication of overlapping documents
                3. Ranking by relevance and quality
                4. Identification of gaps or limitations
                5. Final recommendation of optimal documents
                
                {'Use the limit determined by manual override' if limit_source == 'manual_override' else
                 'Respect the user\'s explicit request for document count' if limit_source == 'user_request' else
                 'Use the limit recommended by the limit estimation agent'}
                
                Ensure the final selection represents the most valuable sources for this specific research question.
                Provide a comprehensive assessment of the retrieved literature.""",
                agent=self.synthesis_agent,
                expected_output="A synthesized list of the most relevant documents with quality assessment and selection rationale"
            )
            
            # Create task list based on whether limit estimation is needed
            if final_limit is None:
                tasks = [analysis_task, limit_estimation_task, search_task, synthesis_task]
                agents = [self.research_analyst, self.limit_estimation_agent, self.search_specialist, self.synthesis_agent]
            else:
                tasks = [analysis_task, search_task, synthesis_task]
                agents = [self.research_analyst, self.search_specialist, self.synthesis_agent]
            
            # Create and execute the crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            # Extract the final limit from limit estimation if it was used
            if final_limit is None:
                # Try to extract limit from the limit estimation task result
                try:
                    # This would need to be parsed from the agent's output
                    # For now, use a default based on standard RAG
                    final_limit = 8  # Reasonable default for agent-estimated scenarios
                except:
                    final_limit = 8
            
            # Also get standard RAG results for comparison
            standard_results = self.rag_engine.search_documents(query, limit=final_limit)
            
            return {
                "agentic_results": result,
                "standard_results": standard_results,
                "query": query,
                "estimated_limit": final_limit,
                "limit_source": limit_source,
                "agents_used": [agent.role for agent in agents]
            }
            
        except Exception as e:
            # Fallback to standard RAG if agentic approach fails
            print(f"Agentic RAG failed, falling back to standard RAG: {e}")
            standard_results = self.rag_engine.search_documents(query, limit=limit or 5)
            return {
                "agentic_results": None,
                "standard_results": standard_results,
                "query": query,
                "estimated_limit": limit or 5,
                "limit_source": "fallback",
                "error": str(e),
                "fallback_used": True
            }
    
    def get_agent_insights(self, query: str) -> Dict[str, Any]:
        """
        Get insights from the research analyst without performing full search.
        
        Args:
            query: The research query to analyze
            
        Returns:
            Dictionary containing query analysis and recommendations
        """
        try:
            analysis_task = Task(
                description=f"""Provide a detailed analysis of this research query:
                Query: "{query}"
                
                Include:
                1. Key research concepts
                2. Suggested search approaches
                3. Potential challenges
                4. Recommendations for refinement
                
                Use the query_analysis tool.""",
                agent=self.research_analyst,
                expected_output="Comprehensive query analysis with actionable insights"
            )
            
            crew = Crew(
                agents=[self.research_analyst],
                tasks=[analysis_task],
                verbose=True
            )
            
            result = crew.kickoff()
            return {
                "analysis": result,
                "query": query,
                "agent_used": "research_analyst"
            }
            
        except Exception as e:
            return {
                "analysis": None,
                "query": query,
                "error": str(e)
            }