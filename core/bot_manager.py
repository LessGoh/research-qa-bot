"""
Main bot manager for Research Q&A Bot
Central coordinator for all bot functionality with LlamaCloud support
"""
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

try:
    from models import (
        ResearchQuery, ResearchMode, ChatHistory, QueryResponse, 
        ErrorResponse, ResearchModeType
    )
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports
    from models.research_models import ResearchQuery, ResearchMode, ChatHistory, ResearchModeType
    from models.response_schemas import QueryResponse, ErrorResponse
from utils import config, get_logger, log_query, log_error, measure_time
from .query_engine import ResearchQueryEngine
from .chat_engine import ResearchChatEngine
from .response_formatter import ResponseFormatter


class LlamaCloudRetriever:
    """Simple LlamaCloud retriever for research queries"""
    
    def __init__(self, api_key: str, pipeline_id: str, api_base: str = "https://api.cloud.llamaindex.ai/api/v1"):
        self.api_key = api_key
        self.pipeline_id = pipeline_id
        self.api_base = api_base
        self.logger = get_logger("LlamaCloudRetriever")
        
    def retrieve(self, query: str, similarity_top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents from LlamaCloud pipeline"""
        try:
            url = f"{self.api_base}/pipelines/{self.pipeline_id}/retrieve"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "similarity_top_k": similarity_top_k
            }
            
            self.logger.info(f"Retrieving from LlamaCloud: {query[:50]}...")
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract nodes from response
            nodes = data.get("nodes", [])
            self.logger.info(f"Retrieved {len(nodes)} nodes from LlamaCloud")
            
            return nodes
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"LlamaCloud API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving from LlamaCloud: {e}")
            raise
    
    def format_context(self, nodes: List[Dict[str, Any]]) -> str:
        """Format retrieved nodes into context string"""
        context_parts = []
        
        for i, node in enumerate(nodes, 1):
            text = node.get("text", "")
            metadata = node.get("metadata", {})
            score = node.get("score", 0.0)
            
            # Create context entry
            context_entry = f"[Source {i}] (Relevance: {score:.3f})\n"
            
            # Add metadata if available
            if metadata:
                title = metadata.get("title", metadata.get("file_name", ""))
                if title:
                    context_entry += f"Title: {title}\n"
                
                page = metadata.get("page", metadata.get("page_number", ""))
                if page:
                    context_entry += f"Page: {page}\n"
            
            context_entry += f"Content: {text}\n"
            context_parts.append(context_entry)
        
        return "\n---\n".join(context_parts)


class ResearchBot:
    """
    Main Research Q&A Bot manager with LlamaCloud support
    Coordinates all bot functionality and maintains state
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize the Research Bot
        
        Args:
            index_path: Path to the LlamaIndex storage directory (for local index)
        """
        self.logger = get_logger("ResearchBot")
        self.index_path = index_path or config.get("llamaindex.index_path", "./data/index")
        
        # Initialize components
        self._index = None
        self._llm = None
        self._retriever = None  # For LlamaCloud
        self._query_engine = None
        self._chat_engine = None
        self._response_formatter = None
        self._chat_history = ChatHistory()
        
        # Cloud configuration
        self.use_cloud = config.get("llamacloud.use_cloud", False)
        
        # Load configuration
        self.research_modes = self._load_research_modes()
        
        # Initialize bot
        self._initialize_bot()
    
    def _load_research_modes(self) -> Dict[str, ResearchMode]:
        """Load research modes from configuration"""
        modes = {}
        for mode_config in config.research_modes:
            try:
                mode = ResearchMode(**mode_config)
                modes[mode.name] = mode
            except Exception as e:
                self.logger.error(f"Failed to load research mode {mode_config.get('name')}: {e}")
        
        return modes
    
    def _initialize_bot(self):
        """Initialize all bot components"""
        try:
            self.logger.info("Initializing Research Bot...")
            
            # Initialize LLM first
            self._initialize_llm()
            
            # Load index or setup cloud retriever
            if self.use_cloud:
                self._setup_llamacloud()
            else:
                self._load_local_index()
            
            # Initialize engines
            self._initialize_engines()
            
            # Initialize response formatter
            self._response_formatter = ResponseFormatter()
            
            self.logger.info("Research Bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize OpenAI LLM"""
        try:
            self._llm = OpenAI(
                api_key=config.openai_api_key,
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.get("llm.max_tokens", 2000)
            )
            
            self.logger.info(f"LLM initialized: {config.llm_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _setup_llamacloud(self):
        """Setup LlamaCloud retriever"""
        try:
            # Get credentials
            api_key = config.get("llamacloud.api_key") or os.getenv("LLAMA_CLOUD_API_KEY")
            pipeline_id = config.get("llamacloud.pipeline_id") or os.getenv("LLAMA_CLOUD_PIPELINE_ID")
            api_base = config.get("llamacloud.api_base", "https://api.cloud.llamaindex.ai/api/v1")
            
            if not api_key:
                raise ValueError("LlamaCloud API key not found. Set LLAMA_CLOUD_API_KEY environment variable.")
            
            if not pipeline_id:
                raise ValueError("LlamaCloud pipeline ID not found. Set LLAMA_CLOUD_PIPELINE_ID environment variable.")
            
            self.logger.info(f"Setting up LlamaCloud retriever with pipeline: {pipeline_id}")
            
            # Initialize retriever
            self._retriever = LlamaCloudRetriever(
                api_key=api_key,
                pipeline_id=pipeline_id,
                api_base=api_base
            )
            
            # Test connection
            test_nodes = self._retriever.retrieve("test query", similarity_top_k=1)
            self.logger.info(f"LlamaCloud connection successful, test returned {len(test_nodes)} nodes")
            
        except Exception as e:
            self.logger.error(f"Failed to setup LlamaCloud: {e}")
            raise
    
    def _load_local_index(self):
        """Load LlamaIndex from local storage"""
        try:
            index_path = Path(self.index_path)
            
            if not index_path.exists():
                raise FileNotFoundError(f"Index not found at: {index_path}")
            
            self.logger.info(f"Loading local index from: {index_path}")
            
            # Load index from storage
            storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
            self._index = load_index_from_storage(storage_context)
            
            self.logger.info("Local index loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load local index: {e}")
            raise
    
    def _initialize_engines(self):
        """Initialize query and chat engines"""
        try:
            if self.use_cloud:
                # For cloud mode, we'll handle queries differently
                self._query_engine = None  # Will use direct LLM calls with retrieved context
                self._chat_engine = None   # Will implement simple chat with context
                
                self.logger.info("Cloud mode: engines will use LlamaCloud retriever")
            else:
                # Local mode with standard LlamaIndex engines
                self._query_engine = ResearchQueryEngine(
                    index=self._index,
                    llm=self._llm,
                    similarity_top_k=config.similarity_top_k
                )
                
                # Chat Engine
                memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
                self._chat_engine = ResearchChatEngine(
                    index=self._index,
                    llm=self._llm,
                    memory=memory
                )
                
                self.logger.info("Local mode: standard LlamaIndex engines initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize engines: {e}")
            raise
    
    def _query_with_cloud(self, query_text: str, mode: str) -> str:
        """Query using LlamaCloud retriever and direct LLM call"""
        try:
            # Get similarity_top_k based on mode
            similarity_k = config.get("llamacloud.similarity_top_k", 5)
            
            # Adjust based on mode
            if mode == "summary":
                similarity_k = min(similarity_k * 2, 10)  # More sources for summary
            elif mode == "comparison":
                similarity_k = min(similarity_k * 2, 8)   # More sources for comparison
            
            # Retrieve relevant documents
            nodes = self._retriever.retrieve(query_text, similarity_top_k=similarity_k)
            
            if not nodes:
                return "No relevant information found in the knowledge base."
            
            # Format context
            context = self._retriever.format_context(nodes)
            
            # Get appropriate prompt based on mode
            from prompts import get_prompt, get_recommended_prompt
            
            prompt_category, prompt_type = get_recommended_prompt(mode)
            
            try:
                from prompts import get_prompt
                mode_prompt = get_prompt(prompt_category, prompt_type, query_text)
            except:
                # Fallback to simple prompt
                mode_prompt = f"Please answer the following research question based on the provided context:\n\nQuestion: {query_text}"
            
            # Combine prompt with context
            full_prompt = f"""{mode_prompt}

Context from research documents:
{context}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""
            
            # Get response from LLM
            response = self._llm.complete(full_prompt)
            
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error in cloud query: {e}")
            raise
    
    @measure_time
    def process_query(self, query_text: str, mode: str = "analysis") -> QueryResponse:
        """
        Process a research query
        
        Args:
            query_text: The research question
            mode: Research mode to use
            
        Returns:
            QueryResponse with structured results
        """
        start_time = time.time()
        
        try:
            # Validate mode
            if mode not in self.research_modes:
                raise ValueError(f"Unknown research mode: {mode}")
            
            # Create query object
            query = ResearchQuery(text=query_text, mode=mode)
            
            # Get mode configuration
            mode_config = self.research_modes[mode]
            
            # Update LLM settings for this mode
            self._update_llm_for_mode(mode_config)
            
            # Process query based on available method
            if self.use_cloud:
                # Use LlamaCloud retriever
                response_text = self._query_with_cloud(query_text, mode)
                
                # Create a mock response object for formatting
                class MockResponse:
                    def __init__(self, text, source_nodes=None):
                        self.response = text
                        self.source_nodes = source_nodes or []
                    
                    def __str__(self):
                        return self.response
                
                response = MockResponse(response_text)
                
            else:
                # Use local index with specialized query methods
                if mode == ResearchModeType.FACTS:
                    response = self._query_engine.extract_facts(query)
                elif mode == ResearchModeType.COMPARISON:
                    response = self._query_engine.compare_sources(query)
                elif mode == ResearchModeType.ANALYSIS:
                    response = self._query_engine.deep_analysis(query)
                elif mode == ResearchModeType.SUMMARY:
                    response = self._query_engine.summarize(query)
                else:
                    response = self._query_engine.query(query)
            
            # Format response
            structured_response = self._response_formatter.format_response(
                response, mode, query_text
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            from models.response_schemas import ResponseMetadata
            metadata = ResponseMetadata(
                processing_time=processing_time,
                mode=mode,
                query_keywords=self._extract_keywords(query_text),
                sources_count=len(getattr(response, 'source_nodes', []))
            )
            
            # Create final response
            result = QueryResponse(
                success=True,
                query=query_text,
                mode=mode,
                structured_response=structured_response,
                metadata=metadata
            )
            
            # Log the query
            log_query(query_text, processing_time, mode)
            
            return result
            
        except Exception as e:
            log_error(e, f"process_query(mode={mode})")
            
            error_response = ErrorResponse(
                error_type=type(e).__name__,
                message=str(e),
                suggestion="Please try rephrasing your question or using a different research mode."
            )
            
            processing_time = time.time() - start_time
            metadata = ResponseMetadata(
                processing_time=processing_time,
                mode=mode,
                query_keywords=self._extract_keywords(query_text),
                sources_count=0
            )
            
            return QueryResponse(
                success=False,
                query=query_text,
                mode=mode,
                error=error_response,
                metadata=metadata
            )
    
    def chat(self, message: str, mode: str = "analysis") -> str:
        """
        Chat with the bot (maintains conversation history)
        
        Args:
            message: User message
            mode: Research mode for context
            
        Returns:
            Bot response
        """
        try:
            # Update LLM for mode
            if mode in self.research_modes:
                self._update_llm_for_mode(self.research_modes[mode])
            
            # Add user message to history
            self._chat_history.add_message("user", message)
            
            if self.use_cloud:
                # Simple cloud-based chat
                response_text = self._chat_with_cloud(message, mode)
            else:
                # Use local chat engine
                self._chat_engine.set_research_mode(mode)
                response = self._chat_engine.chat(message)
                response_text = str(response)
            
            # Add bot response to history
            self._chat_history.add_message("assistant", response_text)
            
            return response_text
            
        except Exception as e:
            log_error(e, "chat")
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            self._chat_history.add_message("assistant", error_msg)
            return error_msg
    
    def _chat_with_cloud(self, message: str, mode: str) -> str:
        """Simple chat implementation using LlamaCloud"""
        try:
            # Get recent chat history for context
            recent_messages = self._chat_history.get_recent_messages(limit=5)
            
            # Build conversation context
            conversation_context = ""
            for msg in recent_messages[:-1]:  # Exclude the current message
                conversation_context += f"{msg.role.title()}: {msg.content}\n"
            
            # Retrieve relevant documents
            nodes = self._retriever.retrieve(message, similarity_top_k=3)
            document_context = self._retriever.format_context(nodes) if nodes else ""
            
            # Create chat prompt
            chat_prompt = f"""You are a research assistant. Please provide a helpful response based on the conversation history and available documents.

Conversation History:
{conversation_context}

Current Question: {message}

Relevant Documents:
{document_context}

Please provide a conversational response that:
1. Acknowledges the conversation context
2. Uses information from the documents when relevant
3. Maintains a helpful and professional tone
4. Focuses on research and analysis"""
            
            # Get response
            response = self._llm.complete(chat_prompt)
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error in cloud chat: {e}")
            return f"I apologize, but I encountered an error processing your message: {str(e)}"
    
    def _update_llm_for_mode(self, mode_config: ResearchMode):
        """Update LLM settings for specific research mode"""
        try:
            # Update temperature and max_tokens for the mode
            self._llm.temperature = mode_config.temperature
            self._llm.max_tokens = mode_config.max_tokens
            
        except Exception as e:
            self.logger.warning(f"Failed to update LLM settings: {e}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from query text"""
        from utils.helpers import extract_keywords
        return extract_keywords(text, max_keywords=5)
    
    def get_chat_history(self) -> ChatHistory:
        """Get current chat history"""
        return self._chat_history
    
    def clear_chat_history(self):
        """Clear chat history"""
        self._chat_history.clear()
        if self._chat_engine:
            self._chat_engine.reset()
    
    def get_available_modes(self) -> Dict[str, Dict[str, str]]:
        """Get available research modes"""
        return {
            name: {
                "display_name": mode.display_name,
                "description": mode.description
            }
            for name, mode in self.research_modes.items()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components"""
        status = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        try:
            # Check LLM
            status["components"]["llm"] = {
                "status": "ok" if self._llm else "error",
                "model": config.llm_model
            }
            
            if self.use_cloud:
                # Check LlamaCloud connection
                try:
                    test_nodes = self._retriever.retrieve("test", similarity_top_k=1)
                    status["components"]["llamacloud"] = {
                        "status": "ok",
                        "pipeline_id": config.get("llamacloud.pipeline_id", "unknown")
                    }
                except Exception as e:
                    status["components"]["llamacloud"] = {
                        "status": "error",
                        "error": str(e)
                    }
            else:
                # Check local index
                status["components"]["index"] = {
                    "status": "ok" if self._index else "error",
                    "path": self.index_path
                }
                
                status["components"]["query_engine"] = {
                    "status": "ok" if self._query_engine else "error"
                }
                
                status["components"]["chat_engine"] = {
                    "status": "ok" if self._chat_engine else "error"
                }
            
            # Check if any component failed
            if any(comp["status"] == "error" for comp in status["components"].values()):
                status["status"] = "degraded"
            
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics"""
        stats = {
            "chat_messages": self._chat_history.message_count,
            "available_modes": list(self.research_modes.keys()),
            "llm_model": config.llm_model,
            "mode": "cloud" if self.use_cloud else "local"
        }
        
        if self.use_cloud:
            stats["pipeline_id"] = config.get("llamacloud.pipeline_id", "unknown")
        else:
            stats["index_path"] = self.index_path
        
        return stats
