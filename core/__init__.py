"""
Core package for Research Q&A Bot
Contains the main business logic and engine components
"""

from .bot_manager import ResearchBot
from .query_engine import ResearchQueryEngine
from .chat_engine import ResearchChatEngine
from .response_formatter import ResponseFormatter

__all__ = [
    "ResearchBot",
    "ResearchQueryEngine", 
    "ResearchChatEngine",
    "ResponseFormatter"
]
