"""
Analysis prompts for deep research analysis tasks
"""
from typing import Optional


def create_analysis_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for deep analysis tasks
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted analysis prompt
    """
    
    base_prompt = f"""
You are an expert research analyst conducting a comprehensive analysis. Your task is to provide a thorough, 
evidence-based analysis of the given research question using the available academic and scientific literature.

Research Question: {query}

Please provide a comprehensive analysis that includes:

1. **Executive Summary**
   - Brief overview of the main findings and conclusions
   - Key insights that directly address the research question

2. **Detailed Analysis Sections**
   For each major aspect of the topic, provide:
   - Clear explanation of the concept or finding
   - Supporting evidence from the literature
   - Critical evaluation of the evidence quality
   - Identification of any conflicting viewpoints or debates

3. **Key Findings**
   - Most significant discoveries or insights
   - Novel contributions to understanding
   - Established facts vs. emerging theories

4. **Research Implications**
   - Theoretical implications for the field
   - Practical applications and real-world relevance
   - Impact on current understanding or practice

5. **Future Research Directions**
   - Identified gaps in current knowledge
   - Suggested areas for further investigation
   - Methodological improvements needed

6. **Limitations and Considerations**
   - Limitations of current research
   - Potential biases or methodological concerns
   - Scope and applicability of findings

Analysis Guidelines:
- Maintain academic rigor and objectivity
- Distinguish between well-established facts and emerging theories
- Consider multiple perspectives and approaches
- Provide specific examples and evidence where possible
- Address the research question comprehensively
- Consider interdisciplinary connections where relevant
"""

    if context:
        base_prompt += f"""

Additional Context: {context}
Please incorporate this context into your analysis where relevant.
"""

    base_prompt += """

Format your analysis in a clear, structured manner with appropriate headings and subheadings. 
Ensure that each section flows logically to the next, building a comprehensive understanding of the topic.
"""

    return base_prompt


def create_critical_analysis_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for critical analysis with emphasis on evaluation
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted critical analysis prompt
    """
    
    prompt = f"""
You are conducting a critical analysis of research literature. Your goal is to evaluate, synthesize, 
and critique the available evidence on the given topic with scholarly rigor.

Research Topic: {query}

Conduct a critical analysis that addresses:

1. **Literature Evaluation**
   - Quality and credibility of sources
   - Methodological strengths and weaknesses
   - Sample sizes, study designs, and validity
   - Potential sources of bias or limitation

2. **Evidence Synthesis**
   - Integration of findings across studies
   - Identification of consistent patterns
   - Resolution of conflicting results
   - Strength of overall evidence base

3. **Critical Assessment**
   - Gaps in current understanding
   - Methodological limitations across the field
   - Theoretical frameworks and their adequacy
   - Alternative explanations or interpretations

4. **Scholarly Debate**
   - Major controversies or disagreements
   - Competing theories or models
   - Ongoing discussions in the field
   - Areas of consensus vs. dispute

5. **Research Quality Indicators**
   - Peer review status of sources
   - Replication of findings
   - Statistical significance and effect sizes
   - External validity and generalizability

Maintain a critical but balanced perspective, acknowledging both strengths and limitations 
of the research base. Support all assessments with specific evidence and reasoning.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_comparative_analysis_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for comparative analysis between different approaches or theories
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted comparative analysis prompt
    """
    
    prompt = f"""
Conduct a comparative analysis examining different approaches, theories, or perspectives related to the research topic.

Research Focus: {query}

Your comparative analysis should include:

1. **Identification of Approaches**
   - Map the major approaches, theories, or perspectives
   - Define key characteristics of each approach
   - Identify the theoretical foundations

2. **Systematic Comparison**
   - Core assumptions and principles
   - Methodological approaches used
   - Scope and applicability
   - Empirical support and evidence base

3. **Strengths and Limitations Analysis**
   For each approach:
   - Key advantages and contributions
   - Limitations and criticisms
   - Context-specific applicability
   - Predictive power and explanatory value

4. **Integration and Synthesis**
   - Areas of convergence and divergence
   - Complementary aspects
   - Potential for theoretical integration
   - Hybrid or combined approaches

5. **Practical Implications**
   - Real-world applications of each approach
   - Effectiveness in different contexts
   - Implementation considerations
   - Cost-benefit analyses where applicable

6. **Future Directions**
   - Emerging approaches or modifications
   - Potential for methodological advancement
   - Areas requiring further development

Present your analysis in a balanced manner, avoiding bias toward any particular approach 
while providing clear evaluative criteria for comparison.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_trend_analysis_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for analyzing trends and developments over time
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted trend analysis prompt
    """
    
    prompt = f"""
Analyze the historical development, current trends, and future directions in the research area.

Research Area: {query}

Provide a comprehensive trend analysis covering:

1. **Historical Development**
   - Key milestones and breakthrough studies
   - Evolution of theoretical understanding
   - Paradigm shifts and major developments
   - Influential researchers and contributions

2. **Current State of Research**
   - Recent major findings and discoveries
   - Current research priorities and focus areas
   - Active research groups and institutions
   - Emerging methodologies and technologies

3. **Trend Identification**
   - Patterns in research direction and focus
   - Increasing or decreasing research interest
   - Methodological trends and innovations
   - Interdisciplinary connections and influences

4. **Driving Forces**
   - Technological advances enabling new research
   - Societal needs and practical applications
   - Funding priorities and research policies
   - Global events or challenges influencing the field

5. **Future Projections**
   - Anticipated developments and breakthroughs
   - Emerging research questions and challenges
   - Potential technological or methodological advances
   - Long-term research priorities

6. **Impact Assessment**
   - Influence on related fields
   - Practical applications and implementations
   - Societal and economic implications
   - Knowledge transfer and translation

Analyze trends with attention to both continuity and change, identifying underlying 
patterns and potential future trajectories based on current evidence.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_methodological_analysis_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for analyzing research methodologies and approaches
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted methodological analysis prompt
    """
    
    prompt = f"""
Conduct a detailed analysis of the research methodologies used to investigate the given topic.

Research Topic: {query}

Your methodological analysis should examine:

1. **Methodological Landscape**
   - Primary research methodologies employed
   - Quantitative vs. qualitative approaches
   - Experimental vs. observational designs
   - Cross-sectional vs. longitudinal studies

2. **Method-Specific Analysis**
   For each major methodology:
   - Typical study designs and protocols
   - Data collection techniques
   - Analytical approaches and statistical methods
   - Sample characteristics and recruitment strategies

3. **Methodological Strengths**
   - Advantages of different approaches
   - Situations where each method excels
   - Quality of evidence generated
   - Validity and reliability considerations

4. **Methodological Limitations**
   - Common limitations and challenges
   - Potential sources of bias
   - Generalizability issues
   - Technical or practical constraints

5. **Innovation and Development**
   - Recent methodological advances
   - Novel approaches or techniques
   - Technology-enabled improvements
   - Interdisciplinary methodological borrowing

6. **Best Practices and Recommendations**
   - Methodological standards and guidelines
   - Quality assessment criteria
   - Recommendations for future research
   - Areas needing methodological development

Focus on how methodological choices impact research findings and their interpretation, 
and consider the appropriateness of different methods for addressing specific research questions.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


# Standard analysis configurations
ANALYSIS_TYPES = {
    "comprehensive": create_analysis_prompt,
    "critical": create_critical_analysis_prompt,
    "comparative": create_comparative_analysis_prompt,
    "trend": create_trend_analysis_prompt,
    "methodological": create_methodological_analysis_prompt
}


def get_analysis_prompt(analysis_type: str, query: str, context: Optional[str] = None) -> str:
    """
    Get analysis prompt by type
    
    Args:
        analysis_type: Type of analysis (comprehensive, critical, comparative, trend, methodological)
        query: Research question
        context: Additional context
        
    Returns:
        Formatted prompt
    """
    
    if analysis_type in ANALYSIS_TYPES:
        return ANALYSIS_TYPES[analysis_type](query, context)
    else:
        # Default to comprehensive analysis
        return create_analysis_prompt(query, context)
