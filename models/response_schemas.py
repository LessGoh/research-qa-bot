"""
Pydantic models for structured response schemas
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class SourceReference(BaseModel):
    """Source reference information"""
    title: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    excerpt: Optional[str] = None
    
    class Config:
        extra = "allow"


class ResponseMetadata(BaseModel):
    """Metadata for responses"""
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float = Field(gt=0)
    mode: str
    query_keywords: List[str] = Field(default_factory=list)
    sources_count: int = Field(ge=0)
    confidence_level: Optional[str] = None  # "high", "medium", "low"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class KeyFact(BaseModel):
    """Individual key fact"""
    fact: str = Field(min_length=5)
    category: Optional[str] = None
    importance: Optional[str] = Field(None, regex="^(high|medium|low)$")
    source_reference: Optional[SourceReference] = None
    
    @validator('fact')
    def validate_fact(cls, v):
        """Ensure fact is meaningful"""
        fact = v.strip()
        if len(fact) < 5:
            raise ValueError("Fact must be at least 5 characters long")
        return fact


class Definition(BaseModel):
    """Term definition"""
    term: str = Field(min_length=1)
    definition: str = Field(min_length=10)
    category: Optional[str] = None
    context: Optional[str] = None
    source_reference: Optional[SourceReference] = None
    
    @validator('term')
    def validate_term(cls, v):
        """Clean term"""
        return v.strip().title()


class FactExtraction(BaseModel):
    """Structured response for fact extraction"""
    key_facts: List[KeyFact] = Field(default_factory=list)
    definitions: List[Definition] = Field(default_factory=list)
    summary: str = Field(min_length=20)
    sources: List[SourceReference] = Field(default_factory=list)
    metadata: ResponseMetadata
    
    @validator('key_facts')
    def validate_facts_not_empty(cls, v):
        """Ensure at least one fact is extracted"""
        if not v:
            raise ValueError("At least one key fact must be extracted")
        return v


class ComparisonItem(BaseModel):
    """Item being compared"""
    name: str
    description: str
    key_points: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    source_reference: Optional[SourceReference] = None


class ComparisonAnalysis(BaseModel):
    """Structured response for comparative analysis"""
    comparison_criteria: List[str] = Field(default_factory=list)
    items: List[ComparisonItem] = Field(min_items=2)
    similarities: List[str] = Field(default_factory=list)
    differences: List[str] = Field(default_factory=list)
    conclusion: str = Field(min_length=50)
    recommendation: Optional[str] = None
    sources: List[SourceReference] = Field(default_factory=list)
    metadata: ResponseMetadata
    
    @validator('items')
    def validate_comparison_items(cls, v):
        """Ensure at least 2 items for comparison"""
        if len(v) < 2:
            raise ValueError("At least 2 items required for comparison")
        return v


class AnalysisSection(BaseModel):
    """Section of deep analysis"""
    title: str
    content: str = Field(min_length=50)
    key_insights: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    source_references: List[SourceReference] = Field(default_factory=list)


class DeepAnalysis(BaseModel):
    """Structured response for deep analysis"""
    executive_summary: str = Field(min_length=100)
    sections: List[AnalysisSection] = Field(min_items=1)
    key_findings: List[str] = Field(default_factory=list)
    implications: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    methodology: Optional[str] = None
    limitations: Optional[str] = None
    sources: List[SourceReference] = Field(default_factory=list)
    metadata: ResponseMetadata


class SummaryPoint(BaseModel):
    """Individual summary point"""
    point: str = Field(min_length=10)
    importance: str = Field(regex="^(high|medium|low)$")
    details: Optional[str] = None
    source_reference: Optional[SourceReference] = None


class SummaryResponse(BaseModel):
    """Structured response for summarization"""
    overview: str = Field(min_length=50)
    main_points: List[SummaryPoint] = Field(min_items=1)
    key_themes: List[str] = Field(default_factory=list)
    conclusion: str = Field(min_length=30)
    scope: Optional[str] = None  # What was summarized
    coverage: Optional[str] = None  # How comprehensive
    sources: List[SourceReference] = Field(default_factory=list)
    metadata: ResponseMetadata


class StructuredResponse(BaseModel):
    """Generic structured response wrapper"""
    response_type: str = Field(regex="^(facts|comparison|analysis|summary)$")
    content: Union[FactExtraction, ComparisonAnalysis, DeepAnalysis, SummaryResponse]
    raw_response: Optional[str] = None  # Original unstructured response
    processing_notes: Optional[str] = None
    
    class Config:
        discriminator = 'response_type'


class ErrorResponse(BaseModel):
    """Error response model"""
    error_type: str
    message: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    suggestion: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QueryResponse(BaseModel):
    """Complete query response"""
    success: bool = True
    query: str
    mode: str
    structured_response: Optional[StructuredResponse] = None
    error: Optional[ErrorResponse] = None
    metadata: ResponseMetadata
    export_formats: List[str] = Field(default=["json", "markdown", "text"])
    
    @validator('structured_response', always=True)
    def validate_response_or_error(cls, v, values):
        """Ensure either structured_response or error is present"""
        success = values.get('success', True)
        error = values.get('error')
        
        if success and not v:
            raise ValueError("structured_response required when success=True")
        if not success and not error:
            raise ValueError("error required when success=False")
        if success and error:
            raise ValueError("Cannot have both success=True and error")
            
        return v
    
    def to_markdown(self) -> str:
        """Convert response to markdown format"""
        if not self.success or not self.structured_response:
            return f"# Error\n\n{self.error.message if self.error else 'Unknown error'}"
        
        content = self.structured_response.content
        md = f"# Query Response\n\n**Query:** {self.query}\n\n"
        md += f"**Mode:** {self.mode}\n\n"
        
        if isinstance(content, FactExtraction):
            md += "## Key Facts\n\n"
            for fact in content.key_facts:
                md += f"- {fact.fact}\n"
            
            if content.definitions:
                md += "\n## Definitions\n\n"
                for defn in content.definitions:
                    md += f"**{defn.term}:** {defn.definition}\n\n"
        
        # Add more formatting for other types as needed
        
        return md
    
    def to_text(self) -> str:
        """Convert response to plain text format"""
        if not self.success:
            return f"Error: {self.error.message if self.error else 'Unknown error'}"
        
        # Simple text conversion
        if self.structured_response and self.structured_response.raw_response:
            return self.structured_response.raw_response
        
        return "Response content not available in text format"
