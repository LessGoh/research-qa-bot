"""
Research Query Engine for specialized research tasks
"""
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.schema import QueryBundle
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.response.schema import Response

from models import ResearchQuery
from utils import get_logger
from prompts import (
    analysis_prompts,
    comparison_prompts, 
    extraction_prompts,
    summary_prompts
)


class ResearchQueryEngine:
    """
    Specialized query engine for research tasks
    Provides different query methods for various research modes
    """
    
    def __init__(
        self, 
        index: VectorStoreIndex,
        llm: BaseLLM,
        similarity_top_k: int = 5,
        response_mode: str = "compact"
    ):
        """
        Initialize Research Query Engine
        
        Args:
            index: LlamaIndex vector store index
            llm: Language model instance
            similarity_top_k: Number of similar documents to retrieve
            response_mode: Response mode for query engine
        """
        self.logger = get_logger("ResearchQueryEngine")
        self.index = index
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        self.response_mode = response_mode
        
        # Create base query engine
        self._base_engine = self._create_base_engine()
        
        self.logger.info("Research Query Engine initialized")
    
    def _create_base_engine(self) -> BaseQueryEngine:
        """Create base query engine with standard settings"""
        return self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=self.similarity_top_k,
            response_mode=self.response_mode,
            verbose=True
        )
    
    def query(self, research_query: ResearchQuery) -> Response:
        """
        General query method
        
        Args:
            research_query: Research query object
            
        Returns:
            Query response
        """
        try:
            self.logger.info(f"Processing general query: {research_query.text[:100]}...")
            
            # Use base engine for general queries
            response = self._base_engine.query(research_query.text)
            
            self.logger.info("General query completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in general query: {e}")
            raise
    
    def extract_facts(self, research_query: ResearchQuery) -> Response:
        """
        Extract key facts and definitions from the knowledge base
        
        Args:
            research_query: Research query object
            
        Returns:
            Response with extracted facts
        """
        try:
            self.logger.info(f"Extracting facts for: {research_query.text[:100]}...")
            
            # Create specialized prompt for fact extraction
            fact_prompt = extraction_prompts.create_fact_extraction_prompt(
                research_query.text,
                context=research_query.context
            )
            
            # Query with specialized prompt
            response = self._base_engine.query(fact_prompt)
            
            self.logger.info("Fact extraction completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in fact extraction: {e}")
            raise
    
    def compare_sources(self, research_query: ResearchQuery) -> Response:
        """
        Perform comparative analysis of different sources
        
        Args:
            research_query: Research query object
            
        Returns:
            Response with comparative analysis
        """
        try:
            self.logger.info(f"Performing comparative analysis: {research_query.text[:100]}...")
            
            # Create specialized prompt for comparison
            comparison_prompt = comparison_prompts.create_comparison_prompt(
                research_query.text,
                context=research_query.context
            )
            
            # Use higher similarity_top_k for comparison to get more sources
            comparison_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=min(self.similarity_top_k * 2, 10),  # Get more sources
                response_mode=self.response_mode,
                verbose=True
            )
            
            response = comparison_engine.query(comparison_prompt)
            
            self.logger.info("Comparative analysis completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in comparative analysis: {e}")
            raise
    
    def deep_analysis(self, research_query: ResearchQuery) -> Response:
        """
        Perform deep analysis of research topics
        
        Args:
            research_query: Research query object
            
        Returns:
            Response with deep analysis
        """
        try:
            self.logger.info(f"Performing deep analysis: {research_query.text[:100]}...")
            
            # Create specialized prompt for analysis
            analysis_prompt = analysis_prompts.create_analysis_prompt(
                research_query.text,
                context=research_query.context
            )
            
            # Use higher token limit for detailed analysis
            original_max_tokens = self.llm.max_tokens
            try:
                self.llm.max_tokens = 3000  # Increase for detailed analysis
                
                response = self._base_engine.query(analysis_prompt)
                
            finally:
                # Restore original max_tokens
                self.llm.max_tokens = original_max_tokens
            
            self.logger.info("Deep analysis completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in deep analysis: {e}")
            raise
    
    def summarize(self, research_query: ResearchQuery) -> Response:
        """
        Summarize large volumes of documentation
        
        Args:
            research_query: Research query object
            
        Returns:
            Response with summary
        """
        try:
            self.logger.info(f"Summarizing content: {research_query.text[:100]}...")
            
            # Create specialized prompt for summarization
            summary_prompt = summary_prompts.create_summary_prompt(
                research_query.text,
                context=research_query.context
            )
            
            # Use more sources for comprehensive summarization
            summary_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=min(self.similarity_top_k * 3, 15),  # Get many sources
                response_mode="tree_summarize",  # Better for summarization
                verbose=True
            )
            
            response = summary_engine.query(summary_prompt)
            
            self.logger.info("Summarization completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            raise
    
    def search_definitions(self, terms: List[str]) -> Response:
        """
        Search for specific term definitions
        
        Args:
            terms: List of terms to define
            
        Returns:
            Response with definitions
        """
        try:
            self.logger.info(f"Searching definitions for: {terms}")
            
            # Create query for definitions
            terms_str = ", ".join(terms)
            definition_query = extraction_prompts.create_definition_prompt(terms_str)
            
            response = self._base_engine.query(definition_query)
            
            self.logger.info("Definition search completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in definition search: {e}")
            raise
    
    def find_related_topics(self, topic: str) -> Response:
        """
        Find topics related to the given topic
        
        Args:
            topic: Main topic to find relations for
            
        Returns:
            Response with related topics
        """
        try:
            self.logger.info(f"Finding topics related to: {topic}")
            
            # Create query for related topics
            related_query = f"""
            Find and list topics, concepts, and research areas that are closely related to "{topic}".
            Include:
            1. Subtopics and specialized areas within {topic}
            2. Complementary fields and interdisciplinary connections
            3. Current research trends and emerging areas
            4. Key methodologies and approaches used in {topic}
            
            Provide a comprehensive overview of the research landscape around {topic}.
            """
            
            response = self._base_engine.query(related_query)
            
            self.logger.info("Related topics search completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in related topics search: {e}")
            raise
    
    def get_source_preview(self, query_text: str, max_sources: int = 3) -> List[Dict[str, Any]]:
        """
        Get preview of sources that would be used for a query
        
        Args:
            query_text: Query text
            max_sources: Maximum number of sources to preview
            
        Returns:
            List of source previews
        """
        try:
            self.logger.info(f"Getting source preview for: {query_text[:50]}...")
            
            # Create retriever to get source nodes
            retriever = self.index.as_retriever(similarity_top_k=max_sources)
            nodes = retriever.retrieve(query_text)
            
            sources = []
            for i, node in enumerate(nodes[:max_sources]):
                source_info = {
                    "rank": i + 1,
                    "score": getattr(node, 'score', 0.0),
                    "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                }
                sources.append(source_info)
            
            self.logger.info(f"Found {len(sources)} source previews")
            return sources
            
        except Exception as e:
            self.logger.error(f"Error getting source preview: {e}")
            return []
    
    def update_settings(
        self, 
        similarity_top_k: Optional[int] = None,
        response_mode: Optional[str] = None
    ):
        """
        Update query engine settings
        
        Args:
            similarity_top_k: New similarity top k value
            response_mode: New response mode
        """
        if similarity_top_k is not None:
            self.similarity_top_k = similarity_top_k
            
        if response_mode is not None:
            self.response_mode = response_mode
        
        # Recreate base engine with new settings
        self._base_engine = self._create_base_engine()
        
        self.logger.info(f"Updated settings: top_k={self.similarity_top_k}, mode={self.response_mode}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query engine statistics"""
        return {
            "similarity_top_k": self.similarity_top_k,
            "response_mode": self.response_mode,
            "index_type": type(self.index).__name__,
            "llm_model": getattr(self.llm, 'model', 'unknown')
        }
