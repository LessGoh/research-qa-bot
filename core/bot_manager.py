"""
Main bot manager for Research Q&A Bot
Central coordinator for all bot functionality with LlamaCloud support
"""
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

# Safe imports with fallbacks
try:
    from llama_index.core import StorageContext, load_index_from_storage
except ImportError:
    StorageContext = None
    load_index_from_storage = None

try:
    from llama_index.core.memory import ChatMemoryBuffer
except ImportError:
    ChatMemoryBuffer = None

try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    OpenAI = None

# Local imports with fallbacks
try:
    from models.research_models import (
        ResearchQuery, ResearchMode, ChatHistory, ResearchModeType
    )
    from models.response_schemas import QueryResponse, ErrorResponse, ResponseMetadata
except ImportError as e:
    print(f"Models import error: {e}")
    # Create minimal fallback classes
    class ResearchQuery:
        def __init__(self, text, mode, **kwargs):
            self.text = text
            self.mode = mode
    
    class ResearchMode:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ChatHistory:
        def __init__(self):
            self.messages = []
            self.message_count = 0
        
        def add_message(self, role, content, metadata=None):
            self.messages.append({"role": role, "content": content})
            self.message_count += 1
        
        def get_recent_messages(self, limit=10):
            return self.messages[-limit:] if self.messages else []
        
        def clear(self):
            self.messages.clear()
            self.message_count = 0
    
    class QueryResponse:
        def __init__(self, success=True, query="", mode="", **kwargs):
            self.success = success
            self.query = query
            self.mode = mode
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ErrorResponse:
        def __init__(self, error_type="", message="", **kwargs):
            self.error_type = error_type
            self.message = message
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ResponseMetadata:
        def __init__(self, processing_time=0.0, mode="", **kwargs):
            self.processing_time = processing_time
            self.mode = mode
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    ResearchModeType = type('ResearchModeType', (), {
        'ANALYSIS': 'analysis',
        'FACTS': 'facts', 
        'COMPARISON': 'comparison',
        'SUMMARY': 'summary'
    })()

try:
    from utils import config, get_logger, log_query, log_error, measure_time
except ImportError as e:
    print(f"Utils import error: {e}")
    # Minimal fallbacks
    class Config:
        def get(self, key, default=None):
            return os.getenv(key.replace('.', '_').upper(), default)
        
        @property
        def openai_api_key(self):
            return os.getenv('OPENAI_API_KEY')
        
        @property
        def llm_model(self):
            return os.getenv('OPENAI_MODEL', 'gpt-4')
        
        @property
        def llm_temperature(self):
            return float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
        
        @property
        def similarity_top_k(self):
            return int(os.getenv('SIMILARITY_TOP_K', '5'))
        
        @property
        def research_modes(self):
            return [
                {"name": "analysis", "display_name": "📊 Deep Analysis", "description": "Comprehensive analysis", "temperature": 0.2, "max_tokens": 3000},
                {"name": "facts", "display_name": "🔍 Fact Extraction", "description": "Extract key facts", "temperature": 0.1, "max_tokens": 1500},
                {"name": "comparison", "display_name": "⚖️ Comparative Analysis", "description": "Compare sources", "temperature": 0.15, "max_tokens": 2500},
                {"name": "summary", "display_name": "📝 Summarization", "description": "Summarize content", "temperature": 0.2, "max_tokens": 2000}
            ]
    
    config = Config()
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)
    
    def log_query(query, time, mode):
        print(f"Query: {query[:50]}... Mode: {mode} Time: {time:.2f}s")
    
    def log_error(error, context):
        print(f"Error in {context}: {error}")
    
    def measure_time(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

# Import engines with fallbacks
try:
    from .query_engine import ResearchQueryEngine
except ImportError:
    ResearchQueryEngine = None

try:
    from .chat_engine import ResearchChatEngine  
except ImportError:
    ResearchChatEngine = None

try:
    from .response_formatter import ResponseFormatter
except ImportError:
    ResponseFormatter = None


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
        self.use_cloud = self._should_use_cloud()
        
        # Load configuration
        self.research_modes = self._load_research_modes()
        
        # Initialize bot
        self._initialize_bot()
    
    def _should_use_cloud(self) -> bool:
        """Determine if should use cloud based on available credentials"""
        # Check for LlamaCloud credentials
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        pipeline_id = os.getenv("LLAMA_CLOUD_PIPELINE_ID")
        
        # Check Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                api_key = api_key or st.secrets.get("LLAMA_CLOUD_API_KEY")
                pipeline_id = pipeline_id or st.secrets.get("LLAMA_CLOUD_PIPELINE_ID")
        except:
            pass
        
        return bool(api_key and pipeline_id)
    
    def _load_research_modes(self) -> Dict[str, ResearchMode]:
        """Load research modes from configuration"""
        modes = {}
        try:
            for mode_config in config.research_modes:
                mode = ResearchMode(**mode_config)
                modes[mode.name] = mode
        except Exception as e:
            self.logger.error(f"Failed to load research modes: {e}")
            # Fallback modes
            modes = {
                "analysis": ResearchMode(name="analysis", display_name="📊 Deep Analysis", description="Comprehensive analysis", temperature=0.2, max_tokens=3000),
                "facts": ResearchMode(name="facts", display_name="🔍 Fact Extraction", description="Extract key facts", temperature=0.1, max_tokens=1500),
                "comparison": ResearchMode(name="comparison", display_name="⚖️ Comparative Analysis", description="Compare sources", temperature=0.15, max_tokens=2500),
                "summary": ResearchMode(name="summary", display_name="📝 Summarization", description="Summarize content", temperature=0.2, max_tokens=2000)
            }
        
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
            
            # Initialize engines (if available)
            self._initialize_engines()
            
            # Initialize response formatter (if available)
            if ResponseFormatter:
                self._response_formatter = ResponseFormatter()
            
            self.logger.info("Research Bot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bot: {e}")
            # Don't raise - allow partial initialization
            print(f"Warning: Bot initialization failed: {e}")
    
    def _initialize_llm(self):
        """Initialize OpenAI LLM"""
        try:
            if not OpenAI:
                raise ImportError("OpenAI not available")
            
            # Get API key
            api_key = os.getenv("OPENAI_API_KEY")
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and not api_key:
                    api_key = st.secrets.get("OPENAI_API_KEY")
            except:
                pass
            
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            self._llm = OpenAI(
                api_key=api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
                max_tokens=2000
            )
            
            self.logger.info(f"LLM initialized: {os.getenv('OPENAI_MODEL', 'gpt-4')}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            # Create a mock LLM for testing
            class MockLLM:
                def __init__(self):
                    self.temperature = 0.1
                    self.max_tokens = 2000
                    self.model = "mock-llm"
                
                def complete(self, prompt):
                    return f"Mock response to: {prompt[:100]}..."
            
            self._llm = MockLLM()
            print("Warning: Using mock LLM - OpenAI integration failed")
    
    def _setup_llamacloud(self):
        """Setup LlamaCloud retriever"""
        try:
            # Get credentials
            api_key = os.getenv("LLAMA_CLOUD_API_KEY")
            pipeline_id = os.getenv("LLAMA_CLOUD_PIPELINE_ID")
            
            # Try Streamlit secrets
            try:
                import streamlit as st
                if hasattr(st, 'secrets'):
                    api_key = api_key or st.secrets.get("LLAMA_CLOUD_API_KEY")
                    pipeline_id = pipeline_id or st.secrets.get("LLAMA_CLOUD_PIPELINE_ID")
            except:
                pass
            
            if not api_key:
                raise ValueError("LlamaCloud API key not found. Set LLAMA_CLOUD_API_KEY.")
            
            if not pipeline_id:
                raise ValueError("LlamaCloud pipeline ID not found. Set LLAMA_CLOUD_PIPELINE_ID.")
            
            self.logger.info(f"Setting up LlamaCloud retriever with pipeline: {pipeline_id}")
            
            # Initialize retriever
            self._retriever = LlamaCloudRetriever(
                api_key=api_key,
                pipeline_id=pipeline_id,
                api_base="https://api.cloud.llamaindex.ai/api/v1"
            )
            
            # Test connection
            test_nodes = self._retriever.retrieve("test query", similarity_top_k=1)
            self.logger.info(f"LlamaCloud connection successful, test returned {len(test_nodes)} nodes")
            
        except Exception as e:
            self.logger.error(f"Failed to setup LlamaCloud: {e}")
            self._retriever = None
            print(f"Warning: LlamaCloud setup failed: {e}")
    
    def _load_local_index(self):
        """Load LlamaIndex from local storage"""
        try:
            if not (StorageContext and load_index_from_storage):
                raise ImportError("LlamaIndex core not available")
            
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
            self._index = None
            print(f"Warning: Local index loading failed: {e}")
    
    def _initialize_engines(self):
        """Initialize query and chat engines"""
        try:
            if self.use_cloud or not self._index:
                # For cloud mode or when local index failed
                self._query_engine = None
                self._chat_engine = None
                self.logger.info("Using cloud mode or engines unavailable")
            else:
                # Local mode with standard LlamaIndex engines
                if ResearchQueryEngine:
                    self._query_engine = ResearchQueryEngine(
                        index=self._index,
                        llm=self._llm,
                        similarity_top_k=config.similarity_top_k
                    )
                
                # Chat Engine
                if ChatMemoryBuffer and ResearchChatEngine:
                    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
                    self._chat_engine = ResearchChatEngine(
                        index=self._index,
                        llm=self._llm,
                        memory=memory
                    )
                
                self.logger.info("Local mode: standard LlamaIndex engines initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize engines: {e}")
            self._query_engine = None
            self._chat_engine = None
            print(f"Warning: Engine initialization failed: {e}")
    
    def _query_with_cloud(self, query_text: str, mode: str) -> str:
        """Query using LlamaCloud retriever and direct LLM call"""
        try:
            if not self._retriever:
                return "LlamaCloud retriever not available. Please check your configuration."
            
            # Get similarity_top_k based on mode
            similarity_k = 5
            
            # Adjust based on mode
            if mode == "summary":
                similarity_k = 8
            elif mode == "comparison":
                similarity_k = 6
            
            # Retrieve relevant documents
            nodes = self._retriever.retrieve(query_text, similarity_top_k=similarity_k)
            
            if not nodes:
                return "No relevant information found in the knowledge base."
            
            # Format context
            context = self._retriever.format_context(nodes)
            
            # Create simple prompt based on mode
            if mode == "analysis":
                mode_instruction = "Provide a comprehensive analysis of the research question based on the provided context."
            elif mode == "facts":
                mode_instruction = "Extract key facts and definitions from the provided context."
            elif mode == "comparison":
                mode_instruction = "Compare different approaches, findings, or perspectives mentioned in the provided context."
            elif mode == "summary":
                mode_instruction = "Summarize the main findings and conclusions from the provided context."
            else:
                mode_instruction = "Please answer the research question based on the provided context."
            
            # Combine prompt with context
            full_prompt = f"""{mode_instruction}

Research Question: {query_text}

Context from research documents:
{context}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""
            
            # Get response from LLM
            response = self._llm.complete(full_prompt)
            
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Error in cloud query: {e}")
            return f"I apologize, but I encountered an error processing your query: {str(e)}"
    
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
                available_modes = list(self.research_modes.keys())
                if available_modes:
                    mode = available_modes[0]  # Use first available mode as fallback
                else:
                    mode = "analysis"  # Ultimate fallback
            
            # Create query object
            query = ResearchQuery(text=query_text, mode=mode)
            
            # Get mode configuration
            mode_config = self.research_modes.get(mode)
            if mode_config:
                self._update_llm_for_mode(mode_config)
            
            # Process query based on available method
            if self.use_cloud and self._retriever:
                # Use LlamaCloud retriever
                response_text = self._query_with_cloud(query_text, mode)
            elif self._query_engine:
                # Use local index with specialized query methods
                if mode == "facts" and hasattr(self._query_engine, 'extract_facts'):
                    response = self._query_engine.extract_facts(query)
                elif mode == "comparison" and hasattr(self._query_engine, 'compare_sources'):
                    response = self._query_engine.compare_sources(query)
                elif mode == "analysis" and hasattr(self._query_engine, 'deep_analysis'):
                    response = self._query_engine.deep_analysis(query)
                elif mode == "summary" and hasattr(self._query_engine, 'summarize'):
                    response = self._query_engine.summarize(query)
                else:
                    response = self._query_engine.query(query)
                response_text = str(response)
            else:
                # Fallback to simple LLM completion
                response_text = str(self._llm.complete(f"Please provide a {mode} for the following research question: {query_text}"))
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = ResponseMetadata(
                processing_time=processing_time,
                mode=mode,
                query_keywords=self._extract_keywords(query_text),
                sources_count=0
            )
            
            # Create response (simplified without complex formatting for now)
            result = QueryResponse(
                success=True,
                query=query_text,
                mode=mode,
                structured_response=None,  # Simplified for stability
                metadata=metadata
            )
            
            # Store raw response for display
            result.raw_response = response_text
            
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
            
            if self.use_cloud and self._retriever:
                # Simple cloud-based chat
                response_text = self._chat_with_cloud(message, mode)
            elif self._chat_engine:
                # Use local chat engine
                if hasattr(self._chat_engine, 'set_research_mode'):
                    self._chat_engine.set_research_mode(mode)
                response = self._chat_engine.chat(message)
                response_text = str(response)
            else:
                # Fallback to simple LLM chat
                response_text = str(self._llm.complete(f"As a research assistant, please respond to: {message}"))
            
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
                conversation_context += f"{msg.get('role', 'unknown').title()}: {msg.get('content', '')}\n"
            
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
    
    def _update_llm_for_mode(self, mode_config):
        """Update LLM settings for specific research mode"""
        try:
            if hasattr(self._llm, 'temperature'):
                self._llm.temperature = getattr(mode_config, 'temperature', 0.1)
            if hasattr(self._llm, 'max_tokens'):
                self._llm.max_tokens = getattr(mode_config, 'max_tokens', 2000)
        except Exception as e:
            self.logger.warning(f"Failed to update LLM settings: {e}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from query text"""
        try:
            from utils.helpers import extract_keywords
            return extract_keywords(text, max_keywords=5)
        except:
            # Simple fallback keyword extraction
            words = text.lower().split()
            return [word for word in words if len(word) > 3][:5]
    
    def get_chat_history(self) -> ChatHistory:
        """Get current chat history"""
        return self._chat_history
    
    def clear_chat_history(self):
        """Clear chat history"""
        self._chat_history.clear()
        if self._chat_engine and hasattr(self._chat_engine, 'reset'):
            self._chat_engine.reset()
    
    def get_available_modes(self) -> Dict[str, Dict[str, str]]:
        """Get available research modes"""
        return {
            name: {
                "display_name": getattr(mode, 'display_name', name.title()),
                "description": getattr(mode, 'description', f"{name} mode")
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
                "model": getattr(self._llm, 'model', 'unknown')
            }
            
            if self.use_cloud:
                # Check LlamaCloud connection
                status["components"]["llamacloud"] = {
                    "status": "ok" if self._retriever else "error",
                    "pipeline_id": os.getenv("LLAMA_CLOUD_PIPELINE_ID", "unknown")
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
            
            # Check if any critical component failed
            critical_components = ["llm"]
            if self.use_cloud:
                critical_components.append("llamacloud")
            else:
                critical_components.extend(["index"])
            
            failed_critical = any(
                status["components"].get(comp, {}).get("status") == "error" 
                for comp in critical_components
            )
            
            if failed_critical:
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
            "llm_model": getattr(self._llm, 'model', 'unknown'),
            "mode": "cloud" if self.use_cloud else "local"
        }
        
        if self.use_cloud:
            stats["pipeline_id"] = os.getenv("LLAMA_CLOUD_PIPELINE_ID", "unknown")
        else:
            stats["index_path"] = self.index_path
        
        return stats
