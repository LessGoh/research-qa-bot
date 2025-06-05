"""
Summary prompts for document summarization and synthesis tasks
"""
from typing import Optional, List


def create_summary_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for general document summarization
    
    Args:
        query: Research question or topic to summarize
        context: Additional context
        
    Returns:
        Formatted summary prompt
    """
    
    base_prompt = f"""
You are a research analyst tasked with creating a comprehensive summary of available literature 
and research findings related to the given topic. Your goal is to synthesize information from 
multiple sources into a coherent, well-structured summary.

Research Topic: {query}

Create a comprehensive summary that includes:

1. **Executive Overview**
   - High-level summary of the current state of research
   - Key themes and main areas of focus
   - Overall conclusions that can be drawn from the literature

2. **Main Research Findings**
   - Most significant discoveries and results
   - Consistent findings across multiple studies
   - Novel or breakthrough research outcomes
   - Quantitative results and key statistics where applicable

3. **Key Themes and Patterns**
   - Recurring themes across the literature
   - Common methodological approaches
   - Shared theoretical frameworks
   - Consistent patterns in findings

4. **Research Landscape**
   - Major research groups and institutions involved
   - Geographic distribution of research efforts
   - Historical progression and timeline of developments
   - Current research priorities and focus areas

5. **Methodological Overview**
   - Primary research methods being used
   - Quality of evidence available
   - Strengths and limitations of current research approaches
   - Innovation in research methodologies

6. **Gaps and Future Directions**
   - Identified gaps in current knowledge
   - Areas needing further investigation
   - Emerging research questions
   - Recommended future research priorities

7. **Practical Implications**
   - Real-world applications of research findings
   - Policy implications and recommendations
   - Implementation considerations
   - Impact on practice and decision-making

Summary Guidelines:
- Synthesize information from multiple sources
- Maintain objectivity and balance
- Distinguish between well-established findings and preliminary results
- Include both positive and negative findings
- Provide appropriate context for all claims
- Use clear, accessible language while maintaining scientific accuracy
- Structure information hierarchically from general to specific

Present the summary in a logical flow that builds understanding progressively, 
ensuring that readers can grasp both the breadth and depth of current knowledge on the topic.
"""

    if context:
        base_prompt += f"""

Additional Context: {context}
Incorporate this context to focus the summary on the most relevant aspects.
"""

    return base_prompt


def create_literature_review_summary_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for literature review style summarization
    
    Args:
        query: Research question or topic
        context: Additional context
        
    Returns:
        Formatted literature review summary prompt
    """
    
    prompt = f"""
Conduct a systematic literature review summary on the given research topic, following 
academic standards for comprehensiveness and critical evaluation.

Research Topic: {query}

Structure your literature review summary as follows:

1. **Introduction and Scope**
   - Definition and scope of the research area
   - Importance and relevance of the topic
   - Objectives of the current literature review
   - Inclusion and exclusion criteria for relevant studies

2. **Methodological Overview**
   - Search strategy and databases used
   - Types of studies included
   - Quality assessment criteria
   - Data extraction and synthesis methods

3. **Thematic Organization of Findings**
   
   **By Research Question/Hypothesis:**
   - Studies addressing similar research questions
   - Consistency of findings across studies
   - Conflicting results and potential explanations
   
   **By Methodology:**
   - Experimental vs. observational studies
   - Quantitative vs. qualitative approaches
   - Longitudinal vs. cross-sectional designs
   
   **By Population/Setting:**
   - Different target populations studied
   - Variations by geographic region
   - Clinical vs. community-based research

4. **Critical Analysis and Synthesis**
   - Quality of evidence assessment
   - Risk of bias considerations
   - Strength of conclusions supported by evidence
   - Areas of consensus and disagreement

5. **Knowledge Gaps and Limitations**
   - Methodological limitations in existing studies
   - Underrepresented populations or contexts
   - Unanswered research questions
   - Inconsistencies requiring resolution

6. **Future Research Directions**
   - Priority areas for future investigation
   - Methodological improvements needed
   - Emerging technologies or approaches
   - Interdisciplinary research opportunities

7. **Conclusions and Implications**
   - Summary of key findings and their significance
   - Implications for theory and practice
   - Policy recommendations where appropriate
   - Clinical or practical applications

Literature Review Guidelines:
- Maintain systematic and comprehensive coverage
- Use critical analysis throughout
- Distinguish between high and low-quality evidence
- Identify patterns and themes across studies
- Address contradictory findings constructively
- Consider multiple perspectives and interpretations
- Follow established reporting standards (PRISMA-like approach)

Ensure the summary provides both breadth of coverage and depth of analysis, 
suitable for researchers and practitioners in the field.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_executive_summary_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for executive-style summary for decision makers
    
    Args:
        query: Research question or topic
        context: Additional context
        
    Returns:
        Formatted executive summary prompt
    """
    
    prompt = f"""
Create an executive summary of research findings designed for decision-makers, policymakers, 
and practitioners who need actionable insights based on current evidence.

Research Topic: {query}

Structure your executive summary to include:

1. **Key Messages (Bullet Points)**
   - 3-5 most important takeaways
   - Clear, actionable statements
   - Evidence-based conclusions
   - Bottom-line implications

2. **Background and Context**
   - Why this research matters
   - Current challenges or problems addressed
   - Relevance to decision-making
   - Scope of the evidence reviewed

3. **Major Findings**
   - Most significant research results
   - Consistent patterns across studies
   - Strength of evidence supporting findings
   - Practical significance of results

4. **Evidence Quality Assessment**
   - Overall quality of available research
   - Confidence levels in key findings
   - Limitations and uncertainties
   - Areas where evidence is strongest/weakest

5. **Implications and Recommendations**
   - Practical applications of findings
   - Policy implications and recommendations
   - Implementation considerations
   - Resource requirements and feasibility

6. **Risk and Benefit Analysis**
   - Potential benefits of implementing findings
   - Risks and potential negative consequences
   - Cost-benefit considerations
   - Mitigation strategies for identified risks

7. **Action Items and Next Steps**
   - Immediate actions that can be taken
   - Longer-term strategic considerations
   - Areas requiring further investigation
   - Monitoring and evaluation needs

Executive Summary Guidelines:
- Keep language clear and non-technical
- Focus on actionable insights
- Quantify benefits and risks where possible
- Provide specific recommendations
- Include confidence levels for key claims
- Consider resource and implementation constraints
- Address potential objections or concerns

Target length: Comprehensive but concise, suitable for busy decision-makers 
who need to understand key implications without extensive technical detail.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_evidence_synthesis_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for evidence synthesis and meta-analysis style summary
    
    Args:
        query: Research question or topic
        context: Additional context
        
    Returns:
        Formatted evidence synthesis prompt
    """
    
    prompt = f"""
Conduct an evidence synthesis that systematically combines and evaluates research findings 
to provide the most reliable conclusions possible based on available evidence.

Research Question: {query}

Structure your evidence synthesis as follows:

1. **Research Question and Objectives**
   - Primary research question being addressed
   - Secondary questions or subgroup analyses
   - Specific objectives of the synthesis
   - PICO framework (Population, Intervention, Comparison, Outcome) where applicable

2. **Study Characteristics**
   - Number and types of studies included
   - Study designs and methodological approaches
   - Sample sizes and population characteristics
   - Geographic and temporal distribution

3. **Quality Assessment**
   - Overall quality of included studies
   - Risk of bias assessment
   - Methodological strengths and limitations
   - Heterogeneity between studies

4. **Quantitative Synthesis**
   - Pooled estimates and effect sizes
   - Confidence intervals and significance tests
   - Heterogeneity statistics and explanations
   - Subgroup analyses and moderator effects

5. **Qualitative Synthesis**
   - Narrative synthesis of findings
   - Thematic analysis across studies
   - Contextual factors affecting results
   - Mechanisms and theoretical explanations

6. **Consistency and Reliability**
   - Consistency of findings across studies
   - Dose-response relationships
   - Temporal patterns and trends
   - Replication and reproducibility

7. **Strength of Evidence**
   - GRADE assessment or similar framework
   - Quality of evidence for each outcome
   - Certainty of conclusions
   - Areas where evidence is robust vs. limited

8. **Clinical/Practical Significance**
   - Magnitude of effects observed
   - Number needed to treat/harm
   - Minimal important differences
   - Real-world applicability

9. **Limitations and Bias Assessment**
   - Publication bias considerations
   - Selection and reporting bias
   - Confounding factors
   - Generalizability limitations

Evidence Synthesis Guidelines:
- Use systematic and transparent methods
- Assess and report on study quality
- Quantify uncertainty and confidence
- Consider both statistical and clinical significance
- Address heterogeneity and its sources
- Evaluate the overall strength of evidence
- Provide balanced conclusions based on totality of evidence

Present findings in a format suitable for evidence-based decision making, 
clearly distinguishing between strong and weak evidence.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_research_brief_summary_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for research brief style summary
    
    Args:
        query: Research question or topic
        context: Additional context
        
    Returns:
        Formatted research brief summary prompt
    """
    
    prompt = f"""
Create a research brief that provides a concise, accessible summary of key research findings 
for a broad audience including researchers, practitioners, and informed general readers.

Research Topic: {query}

Structure your research brief as follows:

1. **Research Snapshot**
   - One-sentence summary of the main finding
   - Key numbers or statistics
   - Time frame and scope of research
   - Geographic or population focus

2. **What We Know**
   - Established facts and findings
   - Areas of scientific consensus
   - Strength of current evidence
   - Most reliable conclusions

3. **What's New**
   - Recent developments and discoveries
   - Emerging trends and patterns
   - Novel methodological approaches
   - Breakthrough findings or innovations

4. **What's Uncertain**
   - Areas of ongoing debate
   - Conflicting findings and explanations
   - Methodological limitations
   - Gaps in current knowledge

5. **Why It Matters**
   - Practical implications and applications
   - Relevance to current challenges
   - Impact on policy and practice
   - Broader societal significance

6. **Looking Forward**
   - Promising research directions
   - Expected developments
   - Key questions to be answered
   - Timeline for potential advances

7. **Key Takeaways**
   - 3-5 main points for readers to remember
   - Action items or recommendations
   - Bottom-line conclusions
   - Implications for different stakeholder groups

Research Brief Guidelines:
- Use accessible language avoiding excessive jargon
- Include specific examples and case studies
- Quantify findings where possible
- Balance optimism with realistic assessment
- Consider multiple perspectives and uses
- Make connections to broader issues
- Include visual elements or infographic-style information where helpful

Target audience: Informed readers who need a comprehensive but accessible 
overview of current research status and implications.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_thematic_summary_prompt(query: str, themes: List[str], context: Optional[str] = None) -> str:
    """
    Create prompt for thematic organization of summary content
    
    Args:
        query: Research question or topic
        themes: List of themes to organize summary around
        context: Additional context
        
    Returns:
        Formatted thematic summary prompt
    """
    
    themes_str = "\n".join([f"- {theme}" for theme in themes])
    
    prompt = f"""
Create a thematically organized summary of research findings, structuring content 
around the specified themes to provide comprehensive coverage of the research area.

Research Topic: {query}

Organizing Themes:
{themes_str}

For each theme, provide:

1. **Theme Overview**
   - Definition and scope of the theme
   - Relevance to the overall research area
   - Connection to other themes
   - Importance in current research

2. **Key Findings by Theme**
   - Major research results and discoveries
   - Consistent patterns across studies
   - Quantitative findings and effect sizes
   - Qualitative insights and explanations

3. **Evidence Quality for Theme**
   - Strength of research supporting findings
   - Number and quality of relevant studies
   - Methodological considerations
   - Confidence in conclusions

4. **Practical Applications by Theme**
   - Real-world implications and uses
   - Implementation examples
   - Success stories and case studies
   - Barriers and facilitators

5. **Theme-Specific Challenges**
   - Methodological difficulties
   - Measurement and assessment issues
   - Conflicting findings or debates
   - Areas needing further research

6. **Cross-Theme Integration**
   - Connections between themes
   - Overlapping findings and implications
   - Synergistic effects or interactions
   - Holistic understanding across themes

7. **Future Directions by Theme**
   - Priority research questions
   - Emerging developments
   - Technological advances
   - Policy and practice needs

Thematic Summary Guidelines:
- Ensure comprehensive coverage of each theme
- Maintain balance across themes
- Highlight connections and interactions
- Avoid unnecessary repetition between themes
- Provide theme-specific depth while maintaining overall coherence
- Consider the relative importance of different themes

Present the summary in a way that allows readers to focus on specific themes 
of interest while understanding the broader research landscape.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


# Standard summary configurations
SUMMARY_TYPES = {
    "general": create_summary_prompt,
    "literature_review": create_literature_review_summary_prompt,
    "executive": create_executive_summary_prompt,
    "evidence_synthesis": create_evidence_synthesis_prompt,
    "research_brief": create_research_brief_summary_prompt,
    "thematic": lambda q, c=None: create_thematic_summary_prompt(q, [], c)
}


def get_summary_prompt(summary_type: str, query: str, context: Optional[str] = None) -> str:
    """
    Get summary prompt by type
    
    Args:
        summary_type: Type of summary (general, literature_review, executive, evidence_synthesis, research_brief, thematic)
        query: Research question or topic
        context: Additional context
        
    Returns:
        Formatted prompt
    """
    
    if summary_type in SUMMARY_TYPES:
        return SUMMARY_TYPES[summary_type](query, context)
    else:
        # Default to general summary
        return create_summary_prompt(query, context)


def create_multi_document_summary_prompt(
    query: str, 
    document_types: List[str], 
    context: Optional[str] = None
) -> str:
    """
    Create prompt for summarizing across multiple document types
    
    Args:
        query: Research question or topic
        document_types: Types of documents being summarized
        context: Additional context
        
    Returns:
        Formatted multi-document summary prompt
    """
    
    doc_types_str = ", ".join(document_types)
    
    prompt = f"""
Create a comprehensive summary that synthesizes information from multiple types of documents 
and sources to provide a complete picture of the research area.

Research Topic: {query}
Document Types: {doc_types_str}

Structure your multi-document summary to include:

1. **Source Integration Overview**
   - Types of documents and sources analyzed
   - Complementary information from different source types
   - Consistency and conflicts between source types
   - Relative strengths of different document types

2. **Synthesized Findings**
   - Conclusions supported across multiple document types
   - Unique contributions from each source type
   - Integrated understanding from combined sources
   - Resolution of conflicting information

3. **Document-Specific Insights**
   For each document type:
   - Key contributions and unique information
   - Limitations and biases specific to that type
   - Quality and reliability considerations
   - Optimal use cases and applications

4. **Cross-Document Validation**
   - Findings confirmed by multiple source types
   - Areas where only one source type provides information
   - Triangulation of evidence across sources
   - Confidence levels for different types of findings

5. **Comprehensive Conclusions**
   - Most reliable conclusions from the synthesis
   - Areas where more research is needed
   - Practical implications of combined findings
   - Recommendations based on totality of evidence

Present the summary in a way that leverages the unique strengths of each document type 
while providing an integrated understanding of the research area.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt
