"""
Prompts package for Research Q&A Bot
Contains specialized prompts for different research tasks
"""

from . import analysis_prompts
from . import extraction_prompts
from . import comparison_prompts
from . import summary_prompts

# Import main prompt creation functions
from .analysis_prompts import (
    create_analysis_prompt,
    create_critical_analysis_prompt,
    create_comparative_analysis_prompt,
    create_trend_analysis_prompt,
    create_methodological_analysis_prompt,
    get_analysis_prompt
)

from .extraction_prompts import (
    create_fact_extraction_prompt,
    create_definition_prompt,
    create_data_extraction_prompt,
    create_evidence_extraction_prompt,
    create_concept_extraction_prompt,
    get_extraction_prompt,
    create_multi_source_extraction_prompt
)

from .comparison_prompts import (
    create_comparison_prompt,
    create_methodology_comparison_prompt,
    create_intervention_comparison_prompt,
    create_theory_comparison_prompt,
    create_source_comparison_prompt,
    create_cross_cultural_comparison_prompt,
    get_comparison_prompt,
    create_multi_criteria_comparison_prompt
)

from .summary_prompts import (
    create_summary_prompt,
    create_literature_review_summary_prompt,
    create_executive_summary_prompt,
    create_evidence_synthesis_prompt,
    create_research_brief_summary_prompt,
    create_thematic_summary_prompt,
    get_summary_prompt,
    create_multi_document_summary_prompt
)

__all__ = [
    # Analysis prompts
    "create_analysis_prompt",
    "create_critical_analysis_prompt", 
    "create_comparative_analysis_prompt",
    "create_trend_analysis_prompt",
    "create_methodological_analysis_prompt",
    "get_analysis_prompt",
    
    # Extraction prompts
    "create_fact_extraction_prompt",
    "create_definition_prompt",
    "create_data_extraction_prompt",
    "create_evidence_extraction_prompt",
    "create_concept_extraction_prompt",
    "get_extraction_prompt",
    "create_multi_source_extraction_prompt",
    
    # Comparison prompts
    "create_comparison_prompt",
    "create_methodology_comparison_prompt",
    "create_intervention_comparison_prompt",
    "create_theory_comparison_prompt",
    "create_source_comparison_prompt",
    "create_cross_cultural_comparison_prompt",
    "get_comparison_prompt",
    "create_multi_criteria_comparison_prompt",
    
    # Summary prompts
    "create_summary_prompt",
    "create_literature_review_summary_prompt",
    "create_executive_summary_prompt",
    "create_evidence_synthesis_prompt",
    "create_research_brief_summary_prompt",
    "create_thematic_summary_prompt",
    "get_summary_prompt",
    "create_multi_document_summary_prompt",
    
    # Module references
    "analysis_prompts",
    "extraction_prompts",
    "comparison_prompts",
    "summary_prompts"
]


# Prompt registry for easy access
PROMPT_REGISTRY = {
    "analysis": {
        "comprehensive": analysis_prompts.create_analysis_prompt,
        "critical": analysis_prompts.create_critical_analysis_prompt,
        "comparative": analysis_prompts.create_comparative_analysis_prompt,
        "trend": analysis_prompts.create_trend_analysis_prompt,
        "methodological": analysis_prompts.create_methodological_analysis_prompt
    },
    "extraction": {
        "facts": extraction_prompts.create_fact_extraction_prompt,
        "definitions": extraction_prompts.create_definition_prompt,
        "data": extraction_prompts.create_data_extraction_prompt,
        "evidence": extraction_prompts.create_evidence_extraction_prompt,
        "concepts": extraction_prompts.create_concept_extraction_prompt
    },
    "comparison": {
        "general": comparison_prompts.create_comparison_prompt,
        "methodology": comparison_prompts.create_methodology_comparison_prompt,
        "intervention": comparison_prompts.create_intervention_comparison_prompt,
        "theory": comparison_prompts.create_theory_comparison_prompt,
        "sources": comparison_prompts.create_source_comparison_prompt,
        "cross_cultural": comparison_prompts.create_cross_cultural_comparison_prompt
    },
    "summary": {
        "general": summary_prompts.create_summary_prompt,
        "literature_review": summary_prompts.create_literature_review_summary_prompt,
        "executive": summary_prompts.create_executive_summary_prompt,
        "evidence_synthesis": summary_prompts.create_evidence_synthesis_prompt,
        "research_brief": summary_prompts.create_research_brief_summary_prompt
    }
}


def get_prompt(category: str, prompt_type: str, query: str, context: str = None) -> str:
    """
    Get a prompt by category and type
    
    Args:
        category: Prompt category (analysis, extraction, comparison, summary)
        prompt_type: Specific prompt type within category
        query: Research question or topic
        context: Additional context (optional)
        
    Returns:
        Formatted prompt string
        
    Raises:
        ValueError: If category or prompt_type not found
    """
    
    if category not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt category: {category}. Available: {list(PROMPT_REGISTRY.keys())}")
    
    category_prompts = PROMPT_REGISTRY[category]
    
    if prompt_type not in category_prompts:
        raise ValueError(f"Unknown prompt type '{prompt_type}' for category '{category}'. Available: {list(category_prompts.keys())}")
    
    prompt_function = category_prompts[prompt_type]
    return prompt_function(query, context)


def list_available_prompts() -> dict:
    """
    List all available prompts organized by category
    
    Returns:
        Dictionary with categories and their available prompt types
    """
    
    return {
        category: list(prompts.keys()) 
        for category, prompts in PROMPT_REGISTRY.items()
    }


def get_prompt_description(category: str, prompt_type: str) -> str:
    """
    Get description of what a specific prompt does
    
    Args:
        category: Prompt category
        prompt_type: Specific prompt type
        
    Returns:
        Description string
    """
    
    descriptions = {
        "analysis": {
            "comprehensive": "Provides thorough, evidence-based analysis with multiple sections covering findings, implications, and future directions",
            "critical": "Evaluates and critiques research literature with emphasis on methodological quality and evidence assessment",
            "comparative": "Compares different approaches, theories, or perspectives systematically",
            "trend": "Analyzes historical development, current trends, and future directions in research",
            "methodological": "Focuses on research methodologies, their strengths, limitations, and best practices"
        },
        "extraction": {
            "facts": "Extracts key facts, findings, and verifiable information from research literature",
            "definitions": "Provides comprehensive definitions of technical terms and concepts",
            "data": "Extracts quantitative data, statistics, and numerical findings",
            "evidence": "Assesses evidence quality, study design, and methodological rigor",
            "concepts": "Extracts key concepts, theoretical frameworks, and conceptual models"
        },
        "comparison": {
            "general": "Conducts systematic comparison across multiple dimensions with balanced evaluation",
            "methodology": "Compares research methodologies focusing on validity, reliability, and practical implementation",
            "intervention": "Compares interventions, treatments, or approaches based on effectiveness and implementation",
            "theory": "Compares theories or theoretical frameworks on explanatory power and empirical support",
            "sources": "Compares information sources, databases, or research repositories",
            "cross_cultural": "Conducts cross-cultural or cross-national comparisons considering cultural contexts"
        },
        "summary": {
            "general": "Creates comprehensive summary synthesizing information from multiple sources",
            "literature_review": "Provides systematic literature review summary following academic standards",
            "executive": "Creates executive-style summary for decision-makers with actionable insights",
            "evidence_synthesis": "Conducts evidence synthesis combining and evaluating research findings systematically",
            "research_brief": "Creates accessible research brief for broad audience including practitioners"
        }
    }
    
    if category not in descriptions:
        return "Description not available"
    
    return descriptions[category].get(prompt_type, "Description not available")


def get_recommended_prompt(research_mode: str) -> tuple:
    """
    Get recommended prompt category and type based on research mode
    
    Args:
        research_mode: Research mode (analysis, facts, comparison, summary)
        
    Returns:
        Tuple of (category, prompt_type)
    """
    
    recommendations = {
        "analysis": ("analysis", "comprehensive"),
        "facts": ("extraction", "facts"),
        "comparison": ("comparison", "general"),
        "summary": ("summary", "general")
    }
    
    return recommendations.get(research_mode, ("analysis", "comprehensive"))
