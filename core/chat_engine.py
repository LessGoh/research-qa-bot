"""
Research Chat Engine for conversational interactions
"""
from typing import Optional, List, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.memory import BaseMemory
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.core.schema import ChatMessage
from llama_index.core.response.schema import StreamingResponse, Response

from models import ChatHistory
from utils import get_logger, config


class ResearchChatEngine:
    """
    Specialized chat engine for research conversations
    Maintains context and provides research-focused responses
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        llm: BaseLLM,
        memory: BaseMemory,
        chat_mode: str = "condense_plus_context",
        similarity_top_k: int = 3
    ):
        """
        Initialize Research Chat Engine
        
        Args:
            index: LlamaIndex vector store index
            llm: Language model instance
            memory: Chat memory instance
            chat_mode: Chat mode to use
            similarity_top_k: Number of similar documents to retrieve
        """
        self.logger = get_logger("ResearchChatEngine")
        self.index = index
        self.llm = llm
        self.memory = memory
        self.similarity_top_k = similarity_top_k
        
        # Create chat engine based on mode
        self._chat_engine = self._create_chat_engine(chat_mode)
        
        # Research context
        self._current_research_mode = None
        self._research_context = {}
        
        self.logger.info(f"Research Chat Engine initialized with mode: {chat_mode}")
    
    def _create_chat_engine(self, chat_mode: str) -> BaseChatEngine:
        """Create chat engine based on specified mode"""
        try:
            if chat_mode == "condense_plus_context":
                return CondensePlusContextChatEngine.from_defaults(
                    index=self.index,
                    llm=self.llm,
                    memory=self.memory,
                    context_template=(
                        "You are a research assistant specialized in analyzing scientific literature and documents. "
                        "Use the following context information to provide accurate, detailed, and well-sourced responses. "
                        "When discussing research findings, always consider:\n"
                        "1. The credibility and recency of sources\n"
                        "2. Methodological considerations\n"
                        "3. Limitations and potential biases\n"
                        "4. Connections to broader research areas\n\n"
                        "Context information:\n"
                        "{context_str}\n\n"
                        "Based on this context, please respond to the user's question."
                    ),
                    verbose=True
                )
            
            elif chat_mode == "context":
                return self.index.as_chat_engine(
                    llm=self.llm,
                    memory=self.memory,
                    chat_mode=ChatMode.CONTEXT,
                    verbose=True
                )
            
            elif chat_mode == "react":
                return self.index.as_chat_engine(
                    llm=self.llm,
                    memory=self.memory,
                    chat_mode=ChatMode.REACT,
                    verbose=True
                )
            
            else:
                # Default to simple context mode
                return self.index.as_chat_engine(
                    llm=self.llm,
                    memory=self.memory,
                    chat_mode=ChatMode.CONTEXT,
                    verbose=True
                )
                
        except Exception as e:
            self.logger.error(f"Error creating chat engine: {e}")
            raise
    
    def chat(self, message: str) -> Response:
        """
        Chat with the research assistant
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        try:
            self.logger.info(f"Processing chat message: {message[:100]}...")
            
            # Add research context if available
            enhanced_message = self._enhance_message_with_context(message)
            
            # Get response from chat engine
            response = self._chat_engine.chat(enhanced_message)
            
            # Update research context based on conversation
            self._update_research_context(message, str(response))
            
            self.logger.info("Chat response generated successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise
    
    def stream_chat(self, message: str) -> StreamingResponse:
        """
        Stream chat response
        
        Args:
            message: User message
            
        Returns:
            Streaming response
        """
        try:
            self.logger.info(f"Processing streaming chat: {message[:100]}...")
            
            enhanced_message = self._enhance_message_with_context(message)
            
            # Check if chat engine supports streaming
            if hasattr(self._chat_engine, 'stream_chat'):
                response = self._chat_engine.stream_chat(enhanced_message)
            else:
                # Fallback to regular chat
                regular_response = self._chat_engine.chat(enhanced_message)
                response = self._create_mock_stream(str(regular_response))
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in stream chat: {e}")
            raise
    
    def _enhance_message_with_context(self, message: str) -> str:
        """Enhance message with research context"""
        if not self._current_research_mode:
            return message
        
        # Add mode-specific context
        mode_context = {
            "analysis": "Please provide a detailed analytical response with supporting evidence.",
            "facts": "Focus on extracting specific facts and definitions from the available sources.",
            "comparison": "Compare different perspectives, findings, or approaches mentioned in the sources.",
            "summary": "Provide a comprehensive summary that captures the main points and conclusions."
        }
        
        context_instruction = mode_context.get(self._current_research_mode, "")
        
        if context_instruction:
            enhanced = f"{context_instruction}\n\nUser question: {message}"
        else:
            enhanced = message
        
        return enhanced
    
    def _update_research_context(self, user_message: str, bot_response: str):
        """Update research context based on conversation"""
        # Extract key topics from conversation
        key_terms = self._extract_key_terms(user_message + " " + bot_response)
        
        # Update context
        self._research_context.update({
            "recent_topics": key_terms[:5],  # Keep top 5 recent topics
            "last_interaction": {
                "user": user_message[:200],
                "bot": bot_response[:200]
            }
        })
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        from utils.helpers import extract_keywords
        return extract_keywords(text, max_keywords=10)
    
    def _create_mock_stream(self, text: str) -> StreamingResponse:
        """Create a mock streaming response for engines that don't support streaming"""
        def generate():
            words = text.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield " " + word
        
        return StreamingResponse(generate())
    
    def set_research_mode(self, mode: str):
        """
        Set current research mode for context
        
        Args:
            mode: Research mode (analysis, facts, comparison, summary)
        """
        self._current_research_mode = mode
        self.logger.info(f"Research mode set to: {mode}")
    
    def get_research_mode(self) -> Optional[str]:
        """Get current research mode"""
        return self._current_research_mode
    
    def add_system_message(self, message: str):
        """
        Add a system message to guide the conversation
        
        Args:
            message: System message content
        """
        try:
            system_msg = ChatMessage(role="system", content=message)
            if hasattr(self.memory, 'put'):
                self.memory.put(system_msg)
            
            self.logger.info("System message added to conversation")
            
        except Exception as e:
            self.logger.warning(f"Could not add system message: {e}")
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        try:
            if hasattr(self.memory, 'get_all'):
                messages = self.memory.get_all()
                if messages:
                    # Create summary prompt
                    conversation_text = "\n".join([
                        f"{msg.role}: {msg.content}" 
                        for msg in messages[-10:]  # Last 10 messages
                    ])
                    
                    summary_prompt = f"""
                    Summarize the following research conversation, highlighting:
                    1. Main topics discussed
                    2. Key findings or insights
                    3. Outstanding questions or areas for further investigation
                    
                    Conversation:
                    {conversation_text}
                    """
                    
                    # Use LLM to generate summary
                    summary_response = self.llm.complete(summary_prompt)
                    return str(summary_response)
            
            return "No conversation history available."
            
        except Exception as e:
            self.logger.error(f"Error creating conversation summary: {e}")
            return "Error creating summary."
    
    def suggest_follow_up_questions(self, last_response: str) -> List[str]:
        """
        Suggest follow-up questions based on the last response
        
        Args:
            last_response: The bot's last response
            
        Returns:
            List of suggested follow-up questions
        """
        try:
            suggestion_prompt = f"""
            Based on the following research response, suggest 3-5 relevant follow-up questions 
            that would help deepen the research investigation:
            
            Response: {last_response[:500]}...
            
            Provide questions that would:
            1. Explore specific aspects in more detail
            2. Compare with alternative approaches
            3. Investigate practical applications
            4. Examine limitations or challenges
            
            Format as a simple list, one question per line.
            """
            
            suggestions_response = self.llm.complete(suggestion_prompt)
            
            # Parse suggestions into list
            suggestions = []
            for line in str(suggestions_response).split('\n'):
                line = line.strip()
                if line and not line.startswith('1.') and not line.startswith('-'):
                    # Clean up the line
                    line = line.lstrip('0123456789.- ')
                    if len(line) > 10:  # Reasonable question length
                        suggestions.append(line)
            
            return suggestions[:5]  # Return max 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating follow-up suggestions: {e}")
            return []
    
    def reset(self):
        """Reset chat engine and clear memory"""
        try:
            if hasattr(self.memory, 'clear'):
                self.memory.clear()
            
            # Clear research context
            self._research_context.clear()
            self._current_research_mode = None
            
            self.logger.info("Chat engine reset successfully")
            
        except Exception as e:
            self.logger.error(f"Error resetting chat engine: {e}")
    
    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get chat engine statistics"""
        try:
            message_count = 0
            if hasattr(self.memory, 'get_all'):
                messages = self.memory.get_all()
                message_count = len(messages) if messages else 0
            
            return {
                "message_count": message_count,
                "current_mode": self._current_research_mode,
                "research_context": self._research_context,
                "similarity_top_k": self.similarity_top_k
            }
            
        except Exception as e:
            self.logger.error(f"Error getting chat statistics: {e}")
            return {"error": str(e)}
    
    def export_conversation(self, format_type: str = "text") -> str:
        """
        Export conversation in specified format
        
        Args:
            format_type: Export format (text, json, markdown)
            
        Returns:
            Formatted conversation
        """
        try:
            if not hasattr(self.memory, 'get_all'):
                return "Conversation export not available for this memory type."
            
            messages = self.memory.get_all()
            if not messages:
                return "No conversation to export."
            
            if format_type == "json":
                import json
                conversation_data = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": getattr(msg, 'timestamp', None)
                    }
                    for msg in messages
                ]
                return json.dumps(conversation_data, indent=2)
            
            elif format_type == "markdown":
                markdown = "# Research Conversation\n\n"
                for msg in messages:
                    role_display = "**User:**" if msg.role == "user" else "**Assistant:**"
                    markdown += f"{role_display} {msg.content}\n\n"
                return markdown
            
            else:  # text format
                text = "Research Conversation\n" + "="*50 + "\n\n"
                for msg in messages:
                    role_display = "User:" if msg.role == "user" else "Assistant:"
                    text += f"{role_display} {msg.content}\n\n"
                return text
            
        except Exception as e:
            self.logger.error(f"Error exporting conversation: {e}")
            return f"Error exporting conversation: {str(e)}"
