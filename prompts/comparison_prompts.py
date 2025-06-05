"""
Comparison prompts for comparative analysis tasks
"""
from typing import Optional, List


def create_comparison_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for general comparative analysis
    
    Args:
        query: Research question involving comparison
        context: Additional context
        
    Returns:
        Formatted comparison prompt
    """
    
    base_prompt = f"""
You are a research analyst conducting a systematic comparative analysis. Your task is to identify, 
compare, and contrast different approaches, findings, theories, or methodologies related to the research question.

Research Question: {query}

Conduct a comprehensive comparative analysis that includes:

1. **Identification of Elements to Compare**
   - Identify the main approaches, theories, methods, or findings to compare
   - Define clear criteria for comparison
   - Establish the scope and boundaries of the comparison

2. **Systematic Comparison Framework**
   - Create consistent evaluation criteria
   - Apply the same standards across all items being compared
   - Use both qualitative and quantitative measures where applicable

3. **Detailed Comparison Across Key Dimensions**
   
   **Theoretical Foundations:**
   - Underlying assumptions and principles
   - Conceptual frameworks and models
   - Historical development and evolution
   
   **Methodological Approaches:**
   - Research designs and methods used
   - Data collection and analysis techniques
   - Validation and verification procedures
   
   **Empirical Evidence:**
   - Quality and quantity of supporting evidence
   - Consistency of findings across studies
   - Replication and validation status
   
   **Practical Applications:**
   - Real-world implementation and use cases
   - Effectiveness in different contexts
   - Scalability and feasibility considerations
   
   **Strengths and Advantages:**
   - Unique benefits and contributions
   - Superior performance in specific areas
   - Innovation and methodological advances
   
   **Limitations and Weaknesses:**
   - Known limitations and constraints
   - Areas of poor performance or applicability
   - Methodological or theoretical shortcomings

4. **Synthesis and Integration**
   - Areas of convergence and agreement
   - Key differences and points of divergence
   - Complementary aspects that could be combined
   - Potential for hybrid or integrated approaches

5. **Contextual Considerations**
   - Performance in different settings or populations
   - Cultural, geographic, or temporal factors
   - Resource requirements and constraints
   - Ethical and practical considerations

6. **Recommendations and Conclusions**
   - Best use cases for each approach
   - Situational recommendations
   - Future research directions
   - Areas needing further comparative study

Comparison Guidelines:
- Maintain objectivity and avoid bias toward any particular approach
- Use evidence-based evaluation criteria
- Consider both absolute and relative performance
- Address potential confounding factors
- Acknowledge limitations in the comparison process
- Provide balanced assessment of all alternatives

Present your analysis in a clear, structured format that allows readers to understand 
both the similarities and differences between the compared elements.
"""

    if context:
        base_prompt += f"""

Additional Context: {context}
Incorporate this context into your comparative analysis where relevant.
"""

    return base_prompt


def create_methodology_comparison_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for comparing research methodologies
    
    Args:
        query: Research question about methodologies
        context: Additional context
        
    Returns:
        Formatted methodology comparison prompt
    """
    
    prompt = f"""
Compare different research methodologies used to investigate the given research area, 
evaluating their relative strengths, limitations, and appropriate applications.

Research Area: {query}

Compare methodologies across the following dimensions:

1. **Methodological Characteristics**
   - Study design and structure
   - Data collection procedures
   - Sample selection and recruitment
   - Measurement approaches and instruments

2. **Validity and Reliability**
   - Internal validity considerations
   - External validity and generalizability
   - Reliability of measurements and procedures
   - Construct validity and theoretical alignment

3. **Practical Implementation**
   - Resource requirements (time, money, personnel)
   - Technical complexity and expertise needed
   - Accessibility and feasibility
   - Scalability for larger studies

4. **Data Quality and Rigor**
   - Precision and accuracy of measurements
   - Completeness and representativeness of data
   - Control of confounding variables
   - Statistical power and sensitivity

5. **Ethical and Practical Considerations**
   - Ethical requirements and constraints
   - Participant burden and safety
   - Privacy and confidentiality issues
   - Regulatory and approval processes

6. **Outcome Quality and Utility**
   - Type and quality of evidence generated
   - Relevance to research questions
   - Policy and practice implications
   - Contribution to theoretical understanding

For each methodology, evaluate:
- Optimal use cases and research questions
- Situations where the method excels
- Contexts where limitations become problematic
- Complementary methods that could enhance the approach

Provide recommendations for methodology selection based on:
- Research objectives and questions
- Available resources and constraints
- Target population characteristics
- Desired evidence quality and type
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_intervention_comparison_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for comparing interventions, treatments, or approaches
    
    Args:
        query: Research question about interventions
        context: Additional context
        
    Returns:
        Formatted intervention comparison prompt
    """
    
    prompt = f"""
Compare different interventions, treatments, or approaches related to the research question, 
evaluating their effectiveness, implementation, and practical considerations.

Research Focus: {query}

Compare interventions across these key areas:

1. **Intervention Characteristics**
   - Theoretical basis and mechanisms of action
   - Components and implementation procedures
   - Duration, intensity, and dosage
   - Target population and inclusion criteria

2. **Effectiveness and Outcomes**
   - Primary outcome measures and results
   - Secondary outcomes and side effects
   - Short-term vs. long-term effectiveness
   - Effect sizes and clinical significance

3. **Evidence Base Quality**
   - Number and quality of supporting studies
   - Study designs and methodological rigor
   - Consistency of findings across trials
   - Risk of bias and limitations

4. **Implementation Factors**
   - Training and expertise requirements
   - Resource needs and costs
   - Infrastructure and setting requirements
   - Scalability and sustainability

5. **Safety and Tolerability**
   - Adverse events and contraindications
   - Dropout rates and acceptability
   - Safety monitoring requirements
   - Risk-benefit profiles

6. **Practical Considerations**
   - Accessibility and availability
   - Cost-effectiveness and economic impact
   - Integration with existing systems
   - Cultural and contextual appropriateness

7. **Comparative Effectiveness**
   - Head-to-head comparison studies
   - Network meta-analyses and indirect comparisons
   - Relative ranking and positioning
   - Optimal sequencing and combination strategies

For each intervention, assess:
- Patient/client populations most likely to benefit
- Optimal timing and context for implementation
- Barriers and facilitators to adoption
- Potential for personalization or adaptation

Provide evidence-based recommendations for:
- First-line vs. alternative interventions
- Combination or sequential approaches
- Resource allocation decisions
- Areas needing further comparative research
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_theory_comparison_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for comparing theories or theoretical frameworks
    
    Args:
        query: Research question about theories
        context: Additional context
        
    Returns:
        Formatted theory comparison prompt
    """
    
    prompt = f"""
Compare different theories or theoretical frameworks related to the research area, 
evaluating their explanatory power, empirical support, and practical utility.

Research Area: {query}

Compare theories across these dimensions:

1. **Theoretical Foundations**
   - Core assumptions and propositions
   - Conceptual definitions and constructs
   - Logical structure and coherence
   - Historical development and evolution

2. **Explanatory Power**
   - Scope of phenomena explained
   - Depth and detail of explanations
   - Predictive capabilities
   - Integration of multiple factors

3. **Empirical Support**
   - Quality and quantity of supporting evidence
   - Replication across different contexts
   - Falsifiability and testability
   - Consistency with established findings

4. **Practical Applications**
   - Real-world utility and relevance
   - Intervention and policy implications
   - Guidance for practice and decision-making
   - Problem-solving capabilities

5. **Theoretical Sophistication**
   - Complexity and nuance
   - Handling of contradictory evidence
   - Integration with other theories
   - Adaptability and refinement potential

6. **Research Productivity**
   - Generation of testable hypotheses
   - Stimulation of research activity
   - Methodological innovations inspired
   - Cross-disciplinary influence

For each theory, evaluate:
- Strengths and unique contributions
- Limitations and criticisms
- Optimal application contexts
- Potential for integration with other theories

Assess theoretical debates and controversies:
- Points of agreement and disagreement
- Complementary vs. competing perspectives
- Synthesis opportunities
- Future theoretical developments

Provide analysis of:
- Current theoretical consensus and debates
- Emerging theoretical perspectives
- Areas needing theoretical development
- Implications for future research directions
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_source_comparison_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for comparing different information sources or databases
    
    Args:
        query: Research question about sources
        context: Additional context
        
    Returns:
        Formatted source comparison prompt
    """
    
    prompt = f"""
Compare different information sources, databases, or research repositories related to the research area,
evaluating their coverage, quality, and utility for research purposes.

Research Area: {query}

Compare sources across these criteria:

1. **Coverage and Scope**
   - Breadth of topic coverage
   - Depth of information provided
   - Geographic and temporal coverage
   - Language and cultural diversity

2. **Content Quality**
   - Accuracy and reliability of information
   - Currency and update frequency
   - Peer review and quality control processes
   - Editorial standards and guidelines

3. **Accessibility and Usability**
   - Ease of access and navigation
   - Search capabilities and functionality
   - Download and export options
   - User interface and experience

4. **Technical Features**
   - Metadata quality and standardization
   - Indexing and classification systems
   - Integration with other databases
   - API availability and functionality

5. **Research Utility**
   - Relevance to specific research questions
   - Support for systematic reviews and meta-analyses
   - Citation tracking and metrics
   - Historical and longitudinal data availability

6. **Cost and Licensing**
   - Subscription costs and pricing models
   - Institutional vs. individual access
   - Copyright and usage restrictions
   - Open access availability

For each source, assess:
- Optimal use cases and research applications
- Strengths and unique features
- Limitations and gaps in coverage
- Complementary sources for comprehensive research

Provide recommendations for:
- Primary vs. supplementary source selection
- Search strategy optimization
- Quality assessment and validation
- Integration of multiple sources
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


def create_cross_cultural_comparison_prompt(query: str, context: Optional[str] = None) -> str:
    """
    Create prompt for cross-cultural or cross-national comparisons
    
    Args:
        query: Research question involving cultural comparison
        context: Additional context
        
    Returns:
        Formatted cross-cultural comparison prompt
    """
    
    prompt = f"""
Conduct a cross-cultural or cross-national comparative analysis of research findings, 
considering cultural, social, and contextual factors that may influence results.

Research Question: {query}

Compare findings across different cultural or national contexts:

1. **Cultural Context Analysis**
   - Cultural values and belief systems
   - Social norms and behavioral expectations
   - Historical and political influences
   - Economic and social development factors

2. **Research Methodology Considerations**
   - Measurement equivalence across cultures
   - Translation and adaptation issues
   - Sampling and recruitment differences
   - Cultural appropriateness of methods

3. **Findings Comparison**
   - Similarities across cultural contexts
   - Significant cultural differences
   - Patterns and trends by region or culture
   - Universal vs. culture-specific phenomena

4. **Explanatory Factors**
   - Cultural factors explaining differences
   - Socioeconomic influences
   - Educational and institutional factors
   - Historical and political contexts

5. **Methodological Considerations**
   - Challenges in cross-cultural research
   - Validity and reliability across cultures
   - Bias and confounding factors
   - Standardization vs. cultural adaptation

6. **Implications and Applications**
   - Generalizability of findings
   - Cultural adaptation requirements
   - Policy and intervention implications
   - Cross-cultural learning opportunities

Address important considerations:
- Avoiding cultural stereotyping and bias
- Recognizing within-culture diversity
- Understanding historical and contextual factors
- Balancing universal and specific perspectives

Provide insights on:
- Cultural factors that consistently influence outcomes
- Areas where cultural differences are most pronounced
- Opportunities for cross-cultural learning and adaptation
- Future directions for cross-cultural research
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt


# Standard comparison configurations
COMPARISON_TYPES = {
    "general": create_comparison_prompt,
    "methodology": create_methodology_comparison_prompt,
    "intervention": create_intervention_comparison_prompt,
    "theory": create_theory_comparison_prompt,
    "sources": create_source_comparison_prompt,
    "cross_cultural": create_cross_cultural_comparison_prompt
}


def get_comparison_prompt(comparison_type: str, query: str, context: Optional[str] = None) -> str:
    """
    Get comparison prompt by type
    
    Args:
        comparison_type: Type of comparison (general, methodology, intervention, theory, sources, cross_cultural)
        query: Research question
        context: Additional context
        
    Returns:
        Formatted prompt
    """
    
    if comparison_type in COMPARISON_TYPES:
        return COMPARISON_TYPES[comparison_type](query, context)
    else:
        # Default to general comparison
        return create_comparison_prompt(query, context)


def create_multi_criteria_comparison_prompt(
    query: str, 
    criteria: List[str], 
    context: Optional[str] = None
) -> str:
    """
    Create prompt for comparison using specific criteria
    
    Args:
        query: Research question
        criteria: List of specific comparison criteria
        context: Additional context
        
    Returns:
        Formatted multi-criteria comparison prompt
    """
    
    criteria_str = "\n".join([f"- {criterion}" for criterion in criteria])
    
    prompt = f"""
Conduct a systematic comparison using the specified criteria to evaluate different approaches, 
methods, or findings related to the research question.

Research Question: {query}

Comparison Criteria:
{criteria_str}

For each criterion, provide:

1. **Criterion Definition**
   - Clear explanation of what the criterion measures
   - Why this criterion is important for comparison
   - How it should be evaluated

2. **Comparative Assessment**
   - Evaluation of each approach/method against the criterion
   - Scoring or ranking where appropriate
   - Evidence supporting the assessment

3. **Relative Performance**
   - Which approaches perform best on this criterion
   - Significant differences between approaches
   - Trade-offs and compromises involved

4. **Weighting and Importance**
   - Relative importance of this criterion
   - Context-dependent variations in importance
   - Impact on overall recommendations

Provide an integrated analysis that:
- Synthesizes performance across all criteria
- Identifies optimal choices for different contexts
- Highlights areas where no clear winner emerges
- Suggests ways to improve performance on key criteria

Present results in a clear, systematic format that allows for easy comparison 
and decision-making based on the specified criteria.
"""

    if context:
        prompt += f"\n\nAdditional Context: {context}"

    return prompt
