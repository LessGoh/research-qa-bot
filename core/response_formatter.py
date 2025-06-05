"""
Response Formatter for structuring LLM responses using OpenAI
"""
import json
import re
from typing import Dict, Any, List, Optional, Union
from llama_index.core.response.schema import Response
from llama_index.llms.openai import OpenAI
from pydantic import ValidationError

from models.response_schemas import (
    FactExtraction, KeyFact, Definition, SourceReference,
    ComparisonAnalysis, ComparisonItem, 
    DeepAnalysis, AnalysisSection,
    SummaryResponse, SummaryPoint,
    StructuredResponse, ResponseMetadata
)
from utils import get_logger, config


class ResponseFormatter:
    """
    Formats raw LLM responses into structured formats using OpenAI
    """
    
    def __init__(self):
        """Initialize Response Formatter"""
        self.logger = get_logger("ResponseFormatter")
        
        # Initialize OpenAI for structured formatting
        self._formatting_llm = OpenAI(
            api_key=config.openai_api_key,
            model=config.get("formatting.model", "gpt-4"),
            temperature=config.get("formatting.temperature", 0.1),
            max_tokens=config.get("formatting.max_tokens", 2000)
        )
        
        self.logger.info("Response Formatter initialized")
    
    def format_response(
        self, 
        raw_response: Response, 
        mode: str, 
        original_query: str
    ) -> StructuredResponse:
        """
        Format response based on research mode
        
        Args:
            raw_response: Raw response from query engine
            mode: Research mode used
            original_query: Original user query
            
        Returns:
            Structured response object
        """
        try:
            self.logger.info(f"Formatting response for mode: {mode}")
            
            # Extract source references
            sources = self._extract_source_references(raw_response)
            
            # Format based on mode
            if mode == "facts":
                content = self._format_fact_extraction(raw_response, sources, original_query)
            elif mode == "comparison":
                content = self._format_comparison_analysis(raw_response, sources, original_query)
            elif mode == "analysis":
                content = self._format_deep_analysis(raw_response, sources, original_query)
            elif mode == "summary":
                content = self._format_summary(raw_response, sources, original_query)
            else:
                # Fallback to fact extraction for unknown modes
                content = self._format_fact_extraction(raw_response, sources, original_query)
            
            # Create structured response
            structured_response = StructuredResponse(
                response_type=mode,
                content=content,
                raw_response=str(raw_response)
            )
            
            self.logger.info("Response formatting completed successfully")
            return structured_response
            
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            # Return a fallback structured response
            return self._create_fallback_response(raw_response, mode, str(e))
    
    def _extract_source_references(self, response: Response) -> List[SourceReference]:
        """Extract source references from response"""
        sources = []
        
        try:
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes):
                    source = SourceReference(
                        title=self._extract_title_from_node(node),
                        relevance_score=getattr(node, 'score', None),
                        excerpt=self._create_excerpt(node.text),
                        page_number=self._extract_page_number(node),
                        section=self._extract_section(node)
                    )
                    sources.append(source)
            
        except Exception as e:
            self.logger.warning(f"Error extracting source references: {e}")
        
        return sources
    
    def _extract_title_from_node(self, node) -> str:
        """Extract title from source node"""
        if hasattr(node, 'metadata') and node.metadata:
            # Try different metadata keys for title
            for key in ['title', 'file_name', 'source', 'filename']:
                if key in node.metadata:
                    return str(node.metadata[key])
        
        # Fallback: use first line of text or generic title
        if hasattr(node, 'text') and node.text:
            first_line = node.text.split('\n')[0].strip()
            if len(first_line) < 100:  # Reasonable title length
                return first_line
        
        return "Source Document"
    
    def _create_excerpt(self, text: str, max_length: int = 150) -> str:
        """Create excerpt from text"""
        if len(text) <= max_length:
            return text
        
        # Try to cut at sentence boundary
        excerpt = text[:max_length]
        last_period = excerpt.rfind('.')
        if last_period > max_length * 0.7:  # If we can cut at a reasonable sentence boundary
            return excerpt[:last_period + 1]
        
        return excerpt + "..."
    
    def _extract_page_number(self, node) -> Optional[int]:
        """Extract page number from node metadata"""
        if hasattr(node, 'metadata') and node.metadata:
            for key in ['page', 'page_number', 'page_num']:
                if key in node.metadata:
                    try:
                        return int(node.metadata[key])
                    except (ValueError, TypeError):
                        pass
        return None
    
    def _extract_section(self, node) -> Optional[str]:
        """Extract section from node metadata"""
        if hasattr(node, 'metadata') and node.metadata:
            for key in ['section', 'chapter', 'heading']:
                if key in node.metadata:
                    return str(node.metadata[key])
        return None
    
    def _format_fact_extraction(
        self, 
        response: Response, 
        sources: List[SourceReference], 
        original_query: str
    ) -> FactExtraction:
        """Format response as fact extraction"""
        
        formatting_prompt = f"""
        Extract and structure key facts and definitions from the following research response.
        
        Original Query: {original_query}
        Response: {str(response)}
        
        Please structure this into a JSON format with the following schema:
        {{
            "key_facts": [
                {{
                    "fact": "Clear, concise statement of fact",
                    "category": "optional category",
                    "importance": "high|medium|low"
                }}
            ],
            "definitions": [
                {{
                    "term": "Technical term or concept",
                    "definition": "Clear definition",
                    "category": "optional category",
                    "context": "optional context"
                }}
            ],
            "summary": "Brief summary of main findings"
        }}
        
        Focus on:
        - Extracting concrete, verifiable facts
        - Identifying key technical terms and their definitions
        - Categorizing information by importance
        - Providing a concise summary
        
        Return only valid JSON.
        """
        
        try:
            structured_data = self._get_structured_response(formatting_prompt)
            
            # Parse key facts
            key_facts = []
            for fact_data in structured_data.get("key_facts", []):
                try:
                    fact = KeyFact(**fact_data)
                    key_facts.append(fact)
                except ValidationError as e:
                    self.logger.warning(f"Invalid fact data: {e}")
            
            # Parse definitions
            definitions = []
            for def_data in structured_data.get("definitions", []):
                try:
                    definition = Definition(**def_data)
                    definitions.append(definition)
                except ValidationError as e:
                    self.logger.warning(f"Invalid definition data: {e}")
            
            # Create metadata
            metadata = ResponseMetadata(
                processing_time=0.0,  # Will be set by caller
                mode="facts",
                sources_count=len(sources)
            )
            
            return FactExtraction(
                key_facts=key_facts,
                definitions=definitions,
                summary=structured_data.get("summary", str(response)[:200]),
                sources=sources,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in fact extraction formatting: {e}")
            return self._create_fallback_fact_extraction(response, sources)
    
    def _format_comparison_analysis(
        self, 
        response: Response, 
        sources: List[SourceReference], 
        original_query: str
    ) -> ComparisonAnalysis:
        """Format response as comparison analysis"""
        
        formatting_prompt = f"""
        Structure the following research response into a comparative analysis format.
        
        Original Query: {original_query}
        Response: {str(response)}
        
        Please structure this into JSON with the following schema:
        {{
            "comparison_criteria": ["criterion1", "criterion2"],
            "items": [
                {{
                    "name": "Item/approach/source being compared",
                    "description": "Brief description",
                    "key_points": ["point1", "point2"],
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1", "weakness2"]
                }}
            ],
            "similarities": ["similarity1", "similarity2"],
            "differences": ["difference1", "difference2"],
            "conclusion": "Overall comparative conclusion",
            "recommendation": "optional recommendation"
        }}
        
        Focus on:
        - Identifying what is being compared
        - Clear criteria for comparison
        - Balanced analysis of strengths and weaknesses
        - Highlighting key similarities and differences
        
        Return only valid JSON.
        """
        
        try:
            structured_data = self._get_structured_response(formatting_prompt)
            
            # Parse comparison items
            items = []
            for item_data in structured_data.get("items", []):
                try:
                    item = ComparisonItem(**item_data)
                    items.append(item)
                except ValidationError as e:
                    self.logger.warning(f"Invalid comparison item: {e}")
            
            # Create metadata
            metadata = ResponseMetadata(
                processing_time=0.0,
                mode="comparison",
                sources_count=len(sources)
            )
            
            return ComparisonAnalysis(
                comparison_criteria=structured_data.get("comparison_criteria", []),
                items=items,
                similarities=structured_data.get("similarities", []),
                differences=structured_data.get("differences", []),
                conclusion=structured_data.get("conclusion", str(response)[:200]),
                recommendation=structured_data.get("recommendation"),
                sources=sources,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in comparison formatting: {e}")
            return self._create_fallback_comparison(response, sources)
    
    def _format_deep_analysis(
        self, 
        response: Response, 
        sources: List[SourceReference], 
        original_query: str
    ) -> DeepAnalysis:
        """Format response as deep analysis"""
        
        formatting_prompt = f"""
        Structure the following research response into a comprehensive analytical format.
        
        Original Query: {original_query}
        Response: {str(response)}
        
        Please structure this into JSON with the following schema:
        {{
            "executive_summary": "Concise overview of main findings",
            "sections": [
                {{
                    "title": "Section title",
                    "content": "Detailed section content",
                    "key_insights": ["insight1", "insight2"],
                    "supporting_evidence": ["evidence1", "evidence2"]
                }}
            ],
            "key_findings": ["finding1", "finding2"],
            "implications": ["implication1", "implication2"],
            "recommendations": ["recommendation1", "recommendation2"],
            "methodology": "optional: research methodology used",
            "limitations": "optional: limitations of the analysis"
        }}
        
        Focus on:
        - Comprehensive analysis with multiple sections
        - Clear key findings and their implications
        - Evidence-based insights
        - Practical recommendations
        
        Return only valid JSON.
        """
        
        try:
            structured_data = self._get_structured_response(formatting_prompt)
            
            # Parse analysis sections
            sections = []
            for section_data in structured_data.get("sections", []):
                try:
                    section = AnalysisSection(**section_data)
                    sections.append(section)
                except ValidationError as e:
                    self.logger.warning(f"Invalid analysis section: {e}")
            
            # Create metadata
            metadata = ResponseMetadata(
                processing_time=0.0,
                mode="analysis",
                sources_count=len(sources)
            )
            
            return DeepAnalysis(
                executive_summary=structured_data.get("executive_summary", str(response)[:300]),
                sections=sections,
                key_findings=structured_data.get("key_findings", []),
                implications=structured_data.get("implications", []),
                recommendations=structured_data.get("recommendations", []),
                methodology=structured_data.get("methodology"),
                limitations=structured_data.get("limitations"),
                sources=sources,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in analysis formatting: {e}")
            return self._create_fallback_analysis(response, sources)
    
    def _format_summary(
        self, 
        response: Response, 
        sources: List[SourceReference], 
        original_query: str
    ) -> SummaryResponse:
        """Format response as summary"""
        
        formatting_prompt = f"""
        Structure the following research response into a comprehensive summary format.
        
        Original Query: {original_query}
        Response: {str(response)}
        
        Please structure this into JSON with the following schema:
        {{
            "overview": "High-level overview of the content",
            "main_points": [
                {{
                    "point": "Main point or finding",
                    "importance": "high|medium|low",
                    "details": "optional additional details"
                }}
            ],
            "key_themes": ["theme1", "theme2"],
            "conclusion": "Overall conclusion",
            "scope": "What was summarized",
            "coverage": "How comprehensive the summary is"
        }}
        
        Focus on:
        - Hierarchical organization of information
        - Importance-ranked main points
        - Identification of key themes
        - Clear scope and coverage assessment
        
        Return only valid JSON.
        """
        
        try:
            structured_data = self._get_structured_response(formatting_prompt)
            
            # Parse summary points
            main_points = []
            for point_data in structured_data.get("main_points", []):
                try:
                    point = SummaryPoint(**point_data)
                    main_points.append(point)
                except ValidationError as e:
                    self.logger.warning(f"Invalid summary point: {e}")
            
            # Create metadata
            metadata = ResponseMetadata(
                processing_time=0.0,
                mode="summary",
                sources_count=len(sources)
            )
            
            return SummaryResponse(
                overview=structured_data.get("overview", str(response)[:200]),
                main_points=main_points,
                key_themes=structured_data.get("key_themes", []),
                conclusion=structured_data.get("conclusion", "No conclusion available"),
                scope=structured_data.get("scope"),
                coverage=structured_data.get("coverage"),
                sources=sources,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in summary formatting: {e}")
            return self._create_fallback_summary(response, sources)
    
    def _get_structured_response(self, prompt: str) -> Dict[str, Any]:
        """Get structured response from OpenAI"""
        try:
            response = self._formatting_llm.complete(prompt)
            response_text = str(response).strip()
            
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                return json.loads(response_text)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error getting structured response: {e}")
            raise
    
    def _create_fallback_response(self, response: Response, mode: str, error: str) -> StructuredResponse:
        """Create fallback response when formatting fails"""
        self.logger.warning(f"Creating fallback response for mode {mode}")
        
        # Create basic metadata
        metadata = ResponseMetadata(
            processing_time=0.0,
            mode=mode,
            sources_count=0
        )
        
        # Create minimal structured content based on mode
        if mode == "facts":
            content = self._create_fallback_fact_extraction(response, [])
        elif mode == "comparison":
            content = self._create_fallback_comparison(response, [])
        elif mode == "analysis":
            content = self._create_fallback_analysis(response, [])
        else:
            content = self._create_fallback_summary(response, [])
        
        return StructuredResponse(
            response_type=mode,
            content=content,
            raw_response=str(response),
            processing_notes=f"Fallback formatting used due to error: {error}"
        )
    
    def _create_fallback_fact_extraction(self, response: Response, sources: List[SourceReference]) -> FactExtraction:
        """Create fallback fact extraction"""
        metadata = ResponseMetadata(processing_time=0.0, mode="facts", sources_count=len(sources))
        
        return FactExtraction(
            key_facts=[KeyFact(fact=str(response)[:200], importance="medium")],
            definitions=[],
            summary=str(response)[:300],
            sources=sources,
            metadata=metadata
        )
    
    def _create_fallback_comparison(self, response: Response, sources: List[SourceReference]) -> ComparisonAnalysis:
        """Create fallback comparison analysis"""
        metadata = ResponseMetadata(processing_time=0.0, mode="comparison", sources_count=len(sources))
        
        return ComparisonAnalysis(
            comparison_criteria=[],
            items=[ComparisonItem(name="Analysis", description=str(response)[:200])],
            similarities=[],
            differences=[],
            conclusion=str(response)[:300],
            sources=sources,
            metadata=metadata
        )
    
    def _create_fallback_analysis(self, response: Response, sources: List[SourceReference]) -> DeepAnalysis:
        """Create fallback deep analysis"""
        metadata = ResponseMetadata(processing_time=0.0, mode="analysis", sources_count=len(sources))
        
        return DeepAnalysis(
            executive_summary=str(response)[:300],
            sections=[AnalysisSection(title="Analysis", content=str(response))],
            key_findings=[],
            implications=[],
            recommendations=[],
            sources=sources,
            metadata=metadata
        )
    
    def _create_fallback_summary(self, response: Response, sources: List[SourceReference]) -> SummaryResponse:
        """Create fallback summary"""
        metadata = ResponseMetadata(processing_time=0.0, mode="summary", sources_count=len(sources))
        
        return SummaryResponse(
            overview=str(response)[:200],
            main_points=[SummaryPoint(point=str(response)[:200], importance="medium")],
            key_themes=[],
            conclusion=str(response)[:300],
            sources=sources,
            metadata=metadata
        )
