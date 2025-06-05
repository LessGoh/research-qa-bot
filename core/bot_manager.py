"""
Main bot manager for Research Q&A Bot
Fixed version with LlamaCloud debugging
"""
import os
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

# Minimal fallback classes
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

# Simple OpenAI client
class SimpleOpenAI:
    def __init__(self, api_key, model="gpt-4", temperature=0.1, max_tokens=2000):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def complete(self, prompt):
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Logger fallback
def get_logger(name):
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

def log_query(query, time, mode):
    print(f"Query: {query[:50]}... Mode: {mode} Time: {time:.2f}s")

def log_error(error, context):
    print(f"Error in {context}: {error}")

def measure_time(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class LlamaCloudRetriever:
    """Simple LlamaCloud retriever for research queries"""
    
    def __init__(self, api_key: str, pipeline_id: str, api_base: str = "https://api.cloud.llamaindex.ai/api/v1"):
        self.api_key = api_key
        self.pipeline_id = pipeline_id
        self.api_base = api_base
        self.logger = get_logger("LlamaCloudRetriever")
        
        # Test connection immediately
        self.connection_tested = False
        self.connection_error = None
        self._test_connection()
        
    def _test_connection(self):
        """Test connection to LlamaCloud"""
        try:
            self.logger.info(f"Testing LlamaCloud connection to pipeline: {self.pipeline_id}")
            
            # Try a simple test query
            test_nodes = self.retrieve("test connection", similarity_top_k=1)
            self.connection_tested = True
            self.logger.info(f"LlamaCloud connection successful - retrieved {len(test_nodes)} test nodes")
            
        except Exception as e:
            self.connection_error = str(e)
            self.logger.error(f"LlamaCloud connection test failed: {e}")
            print(f"DEBUG: LlamaCloud connection failed - {e}")
        
    def retrieve(self, query: str, similarity_top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents from LlamaCloud pipeline"""
        try:
            # Try different endpoint formats
            endpoints_to_try = [
                f"{self.api_base}/pipelines/{self.pipeline_id}/retrieve",
                f"https://api.cloud.llamaindex.ai/api/v1/pipelines/{self.pipeline_id}/retrieve",
                f"https://cloud.llamaindex.ai/api/v1/pipelines/{self.pipeline_id}/retrieve"
            ]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "research-qa-bot/1.0"
            }
            
            payload = {
                "query": query,
                "similarity_top_k": similarity_top_k
            }
            
            self.logger.info(f"Retrieving from LlamaCloud: {query[:50]}...")
            
            last_error = None
            
            for i, url in enumerate(endpoints_to_try):
                try:
                    print(f"DEBUG: Trying endpoint {i+1}/{len(endpoints_to_try)}: {url}")
                    print(f"DEBUG: Headers: Authorization=Bearer {self.api_key[:10]}...")
                    print(f"DEBUG: Payload: {payload}")
                    
                    # Увеличиваем таймаут и добавляем retry logic
                    response = requests.post(
                        url, 
                        json=payload, 
                        headers=headers, 
                        timeout=60,  # Увеличили таймаут до 60 секунд
                        verify=True  # Проверяем SSL сертификаты
                    )
                    
                    print(f"DEBUG: Response status: {response.status_code}")
                    print(f"DEBUG: Response headers: {dict(response.headers)}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"DEBUG: Response data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        
                        # Extract nodes from response
                        nodes = data.get("nodes", [])
                        if "retrieval_results" in data:
                            nodes = data["retrieval_results"]
                        elif "results" in data:
                            nodes = data["results"]
                        
                        self.logger.info(f"Retrieved {len(nodes)} nodes from LlamaCloud")
                        return nodes
                    else:
                        print(f"DEBUG: HTTP {response.status_code}: {response.text}")
                        if i == len(endpoints_to_try) - 1:  # Last attempt
                            response.raise_for_status()
                        
                except requests.exceptions.Timeout as e:
                    last_error = f"Timeout error on endpoint {i+1}: {e}"
                    print(f"DEBUG: {last_error}")
                    if i == len(endpoints_to_try) - 1:
                        raise
                    continue
                    
                except requests.exceptions.ConnectionError as e:
                    last_error = f"Connection error on endpoint {i+1}: {e}"
                    print(f"DEBUG: {last_error}")
                    if i == len(endpoints_to_try) - 1:
                        raise
                    continue
                    
                except requests.exceptions.RequestException as e:
                    last_error = f"Request error on endpoint {i+1}: {e}"
                    print(f"DEBUG: {last_error}")
                    if i == len(endpoints_to_try) - 1:
                        raise
                    continue
            
            # If we get here, all endpoints failed
            raise Exception(f"All endpoints failed. Last error: {last_error}")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"LlamaCloud API error: {e}"
            self.logger.error(error_msg)
            print(f"DEBUG: {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Error retrieving from LlamaCloud: {e}"
            self.logger.error(error_msg)
            print(f"DEBUG: {error_msg}")
            raise
    
    def format_context(self, nodes: List[Dict[str, Any]]) -> str:
        """Format retrieved nodes into context string"""
        if not nodes:
            return "No relevant documents found."
            
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
    Fixed version with debugging
    """
    
    def __init__(self, index_path: Optional[str] = None):
        """Initialize the Research Bot"""
        self.logger = get_logger("ResearchBot")
        
        # Initialize components
        self._llm = None
        self._retriever = None
        self._chat_history = ChatHistory()
        
        # Debug info
        self.debug_info = {
            "initialization_time": time.time(),
            "llm_status": "not_initialized",
            "retriever_status": "not_initialized",
            "errors": []
        }
        
        # Load configuration
        self.research_modes = self._load_research_modes()
        
        # Initialize bot
        self._initialize_bot()
    
    def _load_research_modes(self) -> Dict[str, ResearchMode]:
        """Load research modes from configuration"""
        modes = {
            "analysis": ResearchMode(
                name="analysis", 
                display_name="📊 Deep Analysis", 
                description="Comprehensive analysis of research topics", 
                temperature=0.2, 
                max_tokens=3000
            ),
            "facts": ResearchMode(
                name="facts", 
                display_name="🔍 Fact Extraction", 
                description="Extract key facts and definitions", 
                temperature=0.1, 
                max_tokens=1500
            ),
            "comparison": ResearchMode(
                name="comparison", 
                display_name="⚖️ Comparative Analysis", 
                description="Compare different sources and findings", 
                temperature=0.15, 
                max_tokens=2500
            ),
            "summary": ResearchMode(
                name="summary", 
                display_name="📝 Summarization", 
                description="Summarize large volumes of documentation", 
                temperature=0.2, 
                max_tokens=2000
            )
        }
        return modes
    
    def _initialize_bot(self):
        """Initialize all bot components"""
        try:
            self.logger.info("Initializing Research Bot...")
            print("DEBUG: Starting bot initialization")
            
            # Initialize LLM
            self._initialize_llm()
            
            # Setup LlamaCloud retriever
            self._setup_llamacloud()
            
            self.logger.info("Research Bot initialized successfully")
            print("DEBUG: Bot initialization completed")
            
        except Exception as e:
            error_msg = f"Failed to initialize bot: {e}"
            self.logger.error(error_msg)
            self.debug_info["errors"].append(error_msg)
            print(f"DEBUG: {error_msg}")
    
    def _get_credential(self, key_name: str) -> Optional[str]:
        """Get credential from Streamlit secrets or environment"""
        value = None
        
        # Try Streamlit secrets first
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and key_name in st.secrets:
                value = st.secrets[key_name]
                print(f"DEBUG: Got {key_name} from Streamlit secrets")
        except Exception as e:
            print(f"DEBUG: Failed to get {key_name} from Streamlit secrets: {e}")
        
        # Fall back to environment variable
        if not value:
            value = os.getenv(key_name)
            if value:
                print(f"DEBUG: Got {key_name} from environment variable")
            else:
                print(f"DEBUG: {key_name} not found in environment")
        
        return value
    
    def _initialize_llm(self):
        """Initialize OpenAI LLM"""
        try:
            print("DEBUG: Initializing LLM...")
            
            # Get API key
            api_key = self._get_credential("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            # Get model settings
            model = self._get_credential("OPENAI_MODEL") or "gpt-4"
            temperature_str = self._get_credential("OPENAI_TEMPERATURE") or "0.1"
            
            try:
                temperature = float(temperature_str)
            except:
                temperature = 0.1
            
            print(f"DEBUG: Creating OpenAI client with model: {model}, temperature: {temperature}")
            
            self._llm = SimpleOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=2000
            )
            
            self.debug_info["llm_status"] = "initialized"
            self.logger.info(f"LLM initialized: {model}")
            print(f"DEBUG: LLM initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {e}"
            self.logger.error(error_msg)
            self.debug_info["llm_status"] = f"error: {e}"
            self.debug_info["errors"].append(error_msg)
            print(f"DEBUG: {error_msg}")
            
            # Create a mock LLM
            class MockLLM:
                def __init__(self):
                    self.temperature = 0.1
                    self.max_tokens = 2000
                    self.model = "mock-llm"
                
                def complete(self, prompt):
                    return f"Mock response: LLM not available - {prompt[:50]}..."
            
            self._llm = MockLLM()
            print("DEBUG: Using mock LLM")
    
    def _setup_llamacloud(self):
        """Setup LlamaCloud retriever"""
        try:
            print("DEBUG: Setting up LlamaCloud...")
            
            # Get credentials
            api_key = self._get_credential("LLAMA_CLOUD_API_KEY")
            pipeline_id = self._get_credential("LLAMA_CLOUD_PIPELINE_ID")
            
            print(f"DEBUG: API key present: {bool(api_key)}")
            print(f"DEBUG: Pipeline ID: {pipeline_id}")
            
            if not api_key:
                raise ValueError("LlamaCloud API key not found")
            
            if not pipeline_id:
                raise ValueError("LlamaCloud pipeline ID not found")
            
            self.logger.info(f"Setting up LlamaCloud retriever with pipeline: {pipeline_id}")
            
            # Initialize retriever
            self._retriever = LlamaCloudRetriever(
                api_key=api_key,
                pipeline_id=pipeline_id,
                api_base="https://api.cloud.llamaindex.ai/api/v1"
            )
            
            # Check if connection test passed
            if self._retriever.connection_tested:
                self.debug_info["retriever_status"] = "connected"
                print("DEBUG: LlamaCloud retriever connected successfully")
            else:
                self.debug_info["retriever_status"] = f"connection_failed: {self._retriever.connection_error}"
                print(f"DEBUG: LlamaCloud connection test failed: {self._retriever.connection_error}")
            
        except Exception as e:
            error_msg = f"Failed to setup LlamaCloud: {e}"
            self.logger.error(error_msg)
            self.debug_info["retriever_status"] = f"error: {e}"
            self.debug_info["errors"].append(error_msg)
            self._retriever = None
            print(f"DEBUG: {error_msg}")
    
    def _query_with_cloud(self, query_text: str, mode: str) -> str:
        """Query using LlamaCloud retriever and direct LLM call"""
        try:
            print(f"DEBUG: Processing query with cloud: {query_text[:50]}...")
            
            if not self._retriever:
                return "LlamaCloud retriever not available. Please check your configuration."
            
            if not self._retriever.connection_tested:
                return f"LlamaCloud connection failed: {self._retriever.connection_error}"
            
            # Get similarity_top_k based on mode
            similarity_k = 5
            if mode == "summary":
                similarity_k = 8
            elif mode == "comparison":
                similarity_k = 6
            
            print(f"DEBUG: Retrieving {similarity_k} nodes for mode {mode}")
            
            # Retrieve relevant documents
            nodes = self._retriever.retrieve(query_text, similarity_top_k=similarity_k)
            
            if not nodes:
                return "No relevant information found in the knowledge base."
            
            print(f"DEBUG: Retrieved {len(nodes)} nodes")
            
            # Format context
            context = self._retriever.format_context(nodes)
            
            # Create simple prompt based on mode
            mode_instructions = {
                "analysis": "Provide a comprehensive analysis of the research question based on the provided context.",
                "facts": "Extract key facts and definitions from the provided context.",
                "comparison": "Compare different approaches, findings, or perspectives mentioned in the provided context.",
                "summary": "Summarize the main findings and conclusions from the provided context."
            }
            
            mode_instruction = mode_instructions.get(mode, "Please answer the research question based on the provided context.")
            
            # Combine prompt with context
            full_prompt = f"""{mode_instruction}

Research Question: {query_text}

Context from research documents:
{context}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""
            
            print(f"DEBUG: Sending prompt to LLM (length: {len(full_prompt)} chars)")
            
            # Get response from LLM
            response = self._llm.complete(full_prompt)
            
            print(f"DEBUG: Got response from LLM (length: {len(str(response))} chars)")
            
            return str(response)
            
        except Exception as e:
            error_msg = f"Error in cloud query: {e}"
            self.logger.error(error_msg)
            print(f"DEBUG: {error_msg}")
            return f"I apologize, but I encountered an error processing your query: {str(e)}"
    
    @measure_time
    def process_query(self, query_text: str, mode: str = "analysis") -> QueryResponse:
        """Process a research query"""
        start_time = time.time()
        
        try:
            print(f"DEBUG: Processing query: {query_text[:50]}... in mode: {mode}")
            
            # Validate mode
            if mode not in self.research_modes:
                mode = "analysis"  # Fallback
            
            # Create query object
            query = ResearchQuery(text=query_text, mode=mode)
            
            # Get mode configuration and update LLM
            mode_config = self.research_modes.get(mode)
            if mode_config:
                self._update_llm_for_mode(mode_config)
            
            # Process query using LlamaCloud
            response_text = self._query_with_cloud(query_text, mode)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create metadata
            metadata = ResponseMetadata(
                processing_time=processing_time,
                mode=mode,
                query_keywords=self._extract_keywords(query_text),
                sources_count=0
            )
            
            # Create response
            result = QueryResponse(
                success=True,
                query=query_text,
                mode=mode,
                metadata=metadata
            )
            
            # Store raw response for display
            result.raw_response = response_text
            
            # Log the query
            log_query(query_text, processing_time, mode)
            
            print(f"DEBUG: Query processed successfully in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"process_query error: {e}"
            log_error(e, f"process_query(mode={mode})")
            print(f"DEBUG: {error_msg}")
            
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
        """Chat with the bot"""
        try:
            print(f"DEBUG: Processing chat message: {message[:50]}...")
            
            # Update LLM for mode
            if mode in self.research_modes:
                self._update_llm_for_mode(self.research_modes[mode])
            
            # Add user message to history
            self._chat_history.add_message("user", message)
            
            # Simple cloud-based chat
            response_text = self._chat_with_cloud(message, mode)
            
            # Add bot response to history
            self._chat_history.add_message("assistant", response_text)
            
            return response_text
            
        except Exception as e:
            error_msg = f"Chat error: {e}"
            log_error(e, "chat")
            print(f"DEBUG: {error_msg}")
            error_response = f"Sorry, I encountered an error: {str(e)}"
            self._chat_history.add_message("assistant", error_response)
            return error_response
    
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
            if self._retriever and self._retriever.connection_tested:
                nodes = self._retriever.retrieve(message, similarity_top_k=3)
                document_context = self._retriever.format_context(nodes) if nodes else ""
            else:
                document_context = "Document retrieval not available."
            
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
            error_msg = f"Error in cloud chat: {e}"
            self.logger.error(error_msg)
            print(f"DEBUG: {error_msg}")
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
        # Simple fallback keyword extraction
        words = text.lower().split()
        return [word for word in words if len(word) > 3][:5]
    
    def get_chat_history(self) -> ChatHistory:
        """Get current chat history"""
        return self._chat_history
    
    def clear_chat_history(self):
        """Clear chat history"""
        self._chat_history.clear()
    
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
            "debug_info": self.debug_info,
            "timestamp": time.time()
        }
        
        try:
            # Check LLM
            status["components"]["llm"] = {
                "status": "ok" if self._llm and self.debug_info["llm_status"] == "initialized" else "error",
                "model": getattr(self._llm, 'model', 'unknown'),
                "details": self.debug_info["llm_status"]
            }
            
            # Check LlamaCloud connection
            if self._retriever:
                if self._retriever.connection_tested:
                    status["components"]["llamacloud"] = {
                        "status": "ok",
                        "pipeline_id": self._retriever.pipeline_id[:8] + "...",
                        "details": "connected"
                    }
                else:
                    status["components"]["llamacloud"] = {
                        "status": "error",
                        "pipeline_id": self._retriever.pipeline_id[:8] + "...",
                        "details": self._retriever.connection_error
                    }
            else:
                status["components"]["llamacloud"] = {
                    "status": "error",
                    "details": "not_initialized"
                }
            
            # Check if any critical component failed
            if (not self._llm or 
                self.debug_info["llm_status"] != "initialized" or
                not self._retriever or 
                not self._retriever.connection_tested):
                status["status"] = "degraded"
            
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        
        return status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bot statistics"""
        return {
            "chat_messages": self._chat_history.message_count,
            "available_modes": list(self.research_modes.keys()),
            "llm_model": getattr(self._llm, 'model', 'unknown'),
            "mode": "cloud",
            "debug_info": self.debug_info
        }
