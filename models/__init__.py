"""
Models package for Research Q&A Bot
Contains Pydantic models for data validation and structured responses
"""

from .research_models import (
    ResearchQuery,
    ResearchMode,
    QueryMetadata,
    ChatMessage,
    ChatHistory
)

from .response_schemas import (
    FactExtraction,
    KeyFact,
    Definition,
    ComparisonAnalysis,
    ComparisonItem,
    DeepAnalysis,
    AnalysisSection,
    SummaryResponse,
    SummaryPoint,
    StructuredResponse,
    ResponseMetadata,
    SourceReference
)

__all__ = [
    # Research models
    "ResearchQuery",
    "ResearchMode", 
    "QueryMetadata",
    "ChatMessage",
    "ChatHistory",
    
    # Response schemas
    "FactExtraction",
    "KeyFact",
    "Definition",
    "ComparisonAnalysis", 
    "ComparisonItem",
    "DeepAnalysis",
    "AnalysisSection",
    "SummaryResponse",
    "SummaryPoint",
    "StructuredResponse",
    "ResponseMetadata",
    "SourceReference"
]
