"""
Extraction prompts for fact and definition extraction tasks
"""
from typing import Optional, List


def create_fact_extraction_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for extracting key facts
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted fact extraction prompt
    """
    
    base_prompt = f"""
You are a research assistant specializing in fact extraction from scientific literature. 
Your task is to identify and extract key facts, data points, and findings related to the given research question.

Research Question: {query}

Extract the following types of information:

1. **Key Facts and Findings**
   - Specific research findings and results
   - Statistical data, measurements, and quantitative results
   - Established facts and verified information
   - Significant discoveries or breakthroughs
   - Consensus findings across multiple studies

2. **Technical Definitions**
   - Important terms and concepts
   - Technical terminology and jargon
   - Operational definitions used in research
   - Classification systems and taxonomies
   - Standard measurements and units

3. **Empirical Data**
   - Numerical results and statistics
   - Prevalence rates, percentages, and proportions
   - Effect sizes and confidence intervals
   - Sample sizes and study populations
   - Time periods and durations

4. **Methodological Facts**
   - Standard research procedures
   - Common measurement techniques
   - Established protocols and guidelines
   - Validation criteria and benchmarks
   - Quality assessment standards

5. **Historical Facts**
   - Key dates and timeline information
   - Important publications and milestones
   - Development history of concepts
   - Chronological progression of research

Extraction Guidelines:
- Focus on verifiable, concrete information
- Distinguish between established facts and preliminary findings
- Include source information when available
- Prioritize recent and well-validated information
- Be precise with numerical data and measurements
- Avoid speculation or interpretation beyond the facts
- Include both positive and negative findings

Format each fact clearly and concisely, categorizing by type (finding, definition, data, etc.).
"""

    if context:
        base_prompt += f"""

Additional Context: {context}
Use this context to focus your fact extraction on the most relevant aspects.
"""

    return base_prompt


def create_definition_prompt(terms: str, context: Optional[str] = None) -> str:
    """
    Create prompt for extracting definitions of specific terms
    
    Args:
        terms: Terms to define
        context: Additional context
        
    Returns:
        Formatted definition prompt
    """
    
    prompt = f"""
Provide comprehensive definitions for the following terms from the available research literature:

Terms to Define: {terms}

For each term, provide:

1. **Primary Definition**
   - Clear, precise definition
   - Core meaning and essential characteristics
   - Scope and boundaries of the concept

2. **Technical Context**
   - Field-specific usage and meaning
   - How the term is used in research literature
   - Any variations in definition across disciplines

3. **Related Concepts**
   - Synonyms and related terms
   - Broader and narrower concepts
   - Connections to other important terms

4. **Operational Definitions**
   - How the term is measured or operationalized
   - Criteria for identification or classification
   - Practical applications and examples

5. **Evolution and Usage**
   - Historical development of the term
   - Changes in meaning over time
   - Current standard usage in the field

Definition Guidelines:
- Use authoritative sources and standard definitions
- Distinguish between formal and colloquial usage
- Include multiple perspectives if definitions vary
- Provide examples to illustrate usage
- Note any controversies or debates about definitions
- Be precise and avoid circular definitions

Present definitions in a clear, hierarchical format with the most important 
information first, followed by additional context and details.
"""

    if context:
        prompt += f"\n\nContext for Definitions: {context}"

    return prompt


def create_data_extraction_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for extracting quantitative data and statistics
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted data extraction prompt
    """
    
    prompt = f"""
Extract quantitative data, statistics, and numerical findings related to the research question.

Research Focus: {query}

Extract the following types of quantitative information:

1. **Research Results**
   - Statistical test results (p-values, effect sizes, confidence intervals)
   - Descriptive statistics (means, medians, standard deviations)
   - Correlation coefficients and regression results
   - Experimental outcomes and measurements

2. **Sample Characteristics**
   - Sample sizes and response rates
   - Demographic information and distributions
   - Inclusion and exclusion criteria numbers
   - Attrition and dropout rates

3. **Measurement Data**
   - Scale scores and psychometric properties
   - Reliability coefficients (Cronbach's alpha, test-retest)
   - Validity measures and factor loadings
   - Cut-off scores and diagnostic criteria

4. **Comparative Data**
   - Between-group differences and comparisons
   - Pre-post intervention effects
   - Dose-response relationships
   - Time-series and longitudinal changes

5. **Meta-analytic Data**
   - Pooled effect sizes and summary statistics
   - Heterogeneity measures (I², Q-statistic)
   - Number of studies and total sample sizes
   - Subgroup analysis results

6. **Prevalence and Incidence**
   - Population prevalence rates
   - Incidence and occurrence data
   - Geographic and demographic variations
   - Temporal trends and changes

Data Extraction Guidelines:
- Include exact numerical values with units
- Note statistical significance levels
- Specify measurement scales and instruments
- Include confidence intervals where available
- Note study design and methodology context
- Distinguish between raw data and adjusted results
- Include information about data quality and limitations

Present data in a clear, organized format with appropriate categorization 
and context for interpretation.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_evidence_extraction_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for extracting evidence levels and study quality information
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted evidence extraction prompt
    """
    
    prompt = f"""
Extract and assess the quality of evidence related to the research question, focusing on 
study design, methodology, and strength of evidence.

Research Question: {query}

Extract evidence-related information:

1. **Study Design Classification**
   - Types of studies (RCT, cohort, case-control, cross-sectional)
   - Level of evidence (systematic review, meta-analysis, primary studies)
   - Sample characteristics and recruitment methods
   - Study settings and populations

2. **Methodological Quality**
   - Randomization and blinding procedures
   - Control group characteristics
   - Outcome measurement approaches
   - Follow-up periods and completion rates

3. **Evidence Strength Indicators**
   - Consistency of findings across studies
   - Magnitude of effects observed
   - Precision of estimates (confidence intervals)
   - Dose-response relationships

4. **Risk of Bias Assessment**
   - Selection bias indicators
   - Performance and detection bias
   - Attrition and reporting bias
   - Conflicts of interest and funding sources

5. **Reproducibility and Replication**
   - Independent replication of findings
   - Consistency across different populations
   - Robustness of results to different methods
   - Publication bias considerations

6. **Clinical or Practical Significance**
   - Real-world applicability
   - Meaningful effect sizes
   - Number needed to treat/harm
   - Cost-effectiveness considerations

Evidence Assessment Guidelines:
- Use established quality assessment frameworks
- Distinguish between statistical and practical significance
- Consider the totality of evidence, not just individual studies
- Note limitations and potential sources of bias
- Assess generalizability to different populations
- Consider the hierarchy of evidence types

Provide a comprehensive assessment of the evidence base, highlighting both 
strengths and limitations of the available research.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_concept_extraction_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for extracting key concepts and theoretical frameworks
    
    Args:
        query: Research question
        context: Additional context
        
    Returns:
        Formatted concept extraction prompt
    """
    
    prompt = f"""
Extract key concepts, theoretical frameworks, and conceptual models related to the research area.

Research Area: {query}

Extract conceptual information including:

1. **Core Concepts**
   - Fundamental concepts and constructs
   - Key variables and factors
   - Important phenomena and processes
   - Central themes and topics

2. **Theoretical Frameworks**
   - Major theories and models
   - Conceptual frameworks used
   - Theoretical perspectives and approaches
   - Paradigms and schools of thought

3. **Relationships and Connections**
   - Causal relationships and pathways
   - Mediating and moderating factors
   - Interactions and dependencies
   - Network connections and associations

4. **Measurement and Operationalization**
   - How concepts are defined operationally
   - Measurement approaches and instruments
   - Indicators and proxy measures
   - Validation and reliability evidence

5. **Conceptual Debates**
   - Competing theoretical perspectives
   - Unresolved conceptual issues
   - Alternative definitions and interpretations
   - Emerging theoretical developments

6. **Applications and Implications**
   - Practical applications of concepts
   - Policy and intervention implications
   - Cross-disciplinary connections
   - Future research directions

Concept Extraction Guidelines:
- Focus on well-established and important concepts
- Include both traditional and emerging frameworks
- Note relationships between different concepts
- Consider multiple disciplinary perspectives
- Include operational definitions where available
- Distinguish between theoretical and empirical concepts

Organize concepts hierarchically from broad frameworks to specific constructs, 
showing relationships and connections between different levels of abstraction.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


# Standard extraction configurations
EXTRACTION_TYPES = {
    "facts": create_fact_extraction_prompt,
    "definitions": create_definition_prompt,
    "data": create_data_extraction_prompt,
    "evidence": create_evidence_extraction_prompt,
    "concepts": create_concept_extraction_prompt
}


def get_extraction_prompt(extraction_type: str, query: str, context: Optional[str] = None) -> str:
    """
    Get extraction prompt by type
    
    Args:
        extraction_type: Type of extraction (facts, definitions, data, evidence, concepts)
        query: Research question or terms
        context: Additional context
        
    Returns:
        Formatted prompt
    """
    
    if extraction_type in EXTRACTION_TYPES:
        return EXTRACTION_TYPES[extraction_type](query, context)
    else:
        # Default to fact extraction
        return create_fact_extraction_prompt(query, context)


def create_multi_source_extraction_prompt(query: str, focus_areas: List[str], context: Optional[str] = None) -> str:
    """
    Create prompt for extracting information from multiple sources with specific focus areas
    
    Args:
        query: Research question
        focus_areas: Specific areas to focus extraction on
        context: Additional context
        
    Returns:
        Formatted multi-source extraction prompt
    """
    
    focus_str = ", ".join(focus_areas)
    
    prompt = f"""
Conduct a comprehensive extraction of information from multiple sources, focusing on specific areas of interest.

Research Question: {query}
Focus Areas: {focus_str}

For each focus area, extract:

1. **Key Information**
   - Most important facts and findings
   - Core concepts and definitions
   - Significant data points and statistics

2. **Source Comparison**
   - Consistent findings across sources
   - Conflicting or contradictory information
   - Unique contributions from different sources

3. **Quality Assessment**
   - Reliability and credibility of information
   - Recency and relevance of sources
   - Depth and comprehensiveness of coverage

4. **Integration Opportunities**
   - Complementary information that can be combined
   - Gaps that could be filled by other sources
   - Synthesis potential across different perspectives

Present extracted information organized by focus area, with clear indication of 
source reliability and consistency of findings across multiple sources.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt
