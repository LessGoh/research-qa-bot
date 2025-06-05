"""
Pydantic models for research-related data structures (Fixed for Pydantic v2)
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class ResearchModeType(str, Enum):
    """Research mode types"""
    ANALYSIS = "analysis"
    FACTS = "facts"
    COMPARISON = "comparison"
    SUMMARY = "summary"


class ResearchMode(BaseModel):
    """Research mode configuration"""
    name: ResearchModeType
    display_name: str
    description: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0, le=4000)
    
    class Config:
        use_enum_values = True


class QueryMetadata(BaseModel):
    """Metadata for research queries"""
    timestamp: datetime = Field(default_factory=datetime.now)
    mode: ResearchModeType
    keywords: List[str] = Field(default_factory=list)
    estimated_complexity: Optional[str] = None
    language: str = "en"
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Ensure keywords are not empty strings"""
        return [kw.strip() for kw in v if kw.strip()]


class ResearchQuery(BaseModel):
    """Research query model"""
    text: str = Field(min_length=3, max_length=2000)
    mode: ResearchModeType
    metadata: Optional[QueryMetadata] = None
    context: Optional[str] = None  # Additional context for the query
    follow_up: bool = False  # Is this a follow-up question?
    
    @validator('text')
    def validate_text(cls, v):
        """Clean and validate query text"""
        text = v.strip()
        if not text:
            raise ValueError("Query text cannot be empty")
        return text
    
    class Config:
        use_enum_values = True


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(pattern="^(user|assistant|system)$")  # Fixed: regex -> pattern
    content: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChatHistory(BaseModel):
    """Chat history model"""
    messages: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a new message to history"""
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages"""
        return self.messages[-limit:] if self.messages else []
    
    def clear(self):
        """Clear chat history"""
        self.messages.clear()
        self.updated_at = datetime.now()
    
    @property
    def message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QueryStatistics(BaseModel):
    """Query statistics model"""
    total_queries: int = 0
    queries_by_mode: Dict[ResearchModeType, int] = Field(default_factory=dict)
    average_response_time: float = 0.0
    most_common_keywords: List[str] = Field(default_factory=list)
    last_query_time: Optional[datetime] = None
    
    def update_stats(
        self,
        mode: ResearchModeType,
        response_time: float,
        keywords: List[str]
    ):
        """Update statistics with new query data"""
        self.total_queries += 1
        self.queries_by_mode[mode] = self.queries_by_mode.get(mode, 0) + 1
        
        # Update average response time
        if self.total_queries == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_queries - 1) + response_time)
                / self.total_queries
            )
        
        # Update keywords (simple frequency tracking)
        keyword_freq = {}
        for kw in self.most_common_keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        for kw in keywords:
            keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
        
        # Keep top 10 most common keywords
        sorted_keywords = sorted(
            keyword_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.most_common_keywords = [kw for kw, _ in sorted_keywords[:10]]
        
        self.last_query_time = datetime.now()
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserSession(BaseModel):
    """User session model"""
    session_id: str
    chat_history: ChatHistory = Field(default_factory=ChatHistory)
    statistics: QueryStatistics = Field(default_factory=QueryStatistics)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_active(self, timeout_minutes: int = 30) -> bool:
        """Check if session is still active"""
        timeout = datetime.now().timestamp() - (timeout_minutes * 60)
        return self.last_activity.timestamp() > timeout
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
