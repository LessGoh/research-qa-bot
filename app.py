"""
Main Streamlit application for Research Q&A Bot
"""
import streamlit as st
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Local imports
from core import ResearchBot
from models import ResearchModeType, QueryResponse
from utils import config, get_logger, ensure_directory_exists
from prompts import list_available_prompts, get_prompt_description

# Configure Streamlit page
st.set_page_config(
    page_title=config.app_title,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .mode-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .source-reference {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #28a745;
        font-size: 0.9rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f1f8e9;
        margin-right: 2rem;
    }
    
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize logger
logger = get_logger("StreamlitApp")

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "analysis"
    if 'initialization_attempted' not in st.session_state:
        st.session_state.initialization_attempted = False

@st.cache_resource
def initialize_bot() -> Optional[ResearchBot]:
    """Initialize the research bot (cached)"""
    try:
        logger.info("Initializing Research Bot...")
        
        # Check if index exists
        index_path = Path(config.index_path)
        if not index_path.exists():
            st.error(f"❌ Index not found at: {index_path}")
            st.info("Please ensure your LlamaIndex is built and saved at the specified path.")
            return None
        
        # Initialize bot
        bot = ResearchBot(index_path=str(index_path))
        logger.info("Research Bot initialized successfully")
        return bot
        
    except Exception as e:
        logger.error(f"Failed to initialize bot: {e}")
        st.error(f"❌ Failed to initialize bot: {str(e)}")
        return None

def display_bot_status():
    """Display bot initialization status"""
    if st.session_state.bot is None and not st.session_state.initialization_attempted:
        with st.spinner("🤖 Initializing Research Bot..."):
            st.session_state.bot = initialize_bot()
            st.session_state.initialization_attempted = True
    
    if st.session_state.bot is None:
        st.error("❌ Bot initialization failed. Please check your configuration and index path.")
        st.stop()
    else:
        st.success("✅ Research Bot is ready!")

def display_sidebar():
    """Display sidebar with configuration and controls"""
    with st.sidebar:
        st.title("🔬 Research Assistant")
        
        # Bot status
        if st.session_state.bot:
            st.success("🤖 Bot: Online")
            
            # Health check button
            if st.button("🔍 Health Check"):
                health = st.session_state.bot.health_check()
                if health["status"] == "healthy":
                    st.success("✅ All systems operational")
                else:
                    st.warning(f"⚠️ Status: {health['status']}")
                    st.json(health)
        else:
            st.error("🤖 Bot: Offline")
        
        st.divider()
        
        # Research mode selection
        st.subheader("🎯 Research Mode")
        
        if st.session_state.bot:
            available_modes = st.session_state.bot.get_available_modes()
            mode_options = {
                mode_data["display_name"]: mode_name 
                for mode_name, mode_data in available_modes.items()
            }
            
            selected_display = st.selectbox(
                "Select research mode:",
                options=list(mode_options.keys()),
                index=list(mode_options.values()).index(st.session_state.current_mode)
            )
            
            st.session_state.current_mode = mode_options[selected_display]
            
            # Display mode description
            current_mode_data = available_modes[st.session_state.current_mode]
            st.info(f"ℹ️ {current_mode_data['description']}")
        
        st.divider()
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.bot:
                st.session_state.bot.clear_chat_history()
            st.session_state.chat_history.clear()
            st.rerun()
        
        # Show chat statistics
        if st.session_state.bot:
            stats = st.session_state.bot.get_stats()
            st.metric("💬 Messages", stats.get("chat_messages", 0))
        
        st.divider()
        
        # Export options
        st.subheader("📥 Export")
        
        if st.session_state.query_history:
            export_format = st.selectbox(
                "Export format:",
                ["JSON", "Markdown", "Text"]
            )
            
            if st.button(f"📄 Export as {export_format}"):
                export_data(export_format.lower())

def export_data(format_type: str):
    """Export query history in specified format"""
    try:
        if format_type == "json":
            data = json.dumps(st.session_state.query_history, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON",
                data=data,
                file_name=f"research_queries_{int(time.time())}.json",
                mime="application/json"
            )
        
        elif format_type == "markdown":
            markdown = "# Research Query History\n\n"
            for i, query_data in enumerate(st.session_state.query_history, 1):
                markdown += f"## Query {i}\n"
                markdown += f"**Mode:** {query_data.get('mode', 'Unknown')}\n\n"
                markdown += f"**Query:** {query_data.get('query', 'N/A')}\n\n"
                if 'response' in query_data:
                    markdown += f"**Response:** {query_data['response'][:500]}...\n\n"
                markdown += "---\n\n"
            
            st.download_button(
                label="💾 Download Markdown",
                data=markdown,
                file_name=f"research_queries_{int(time.time())}.md",
                mime="text/markdown"
            )
        
        elif format_type == "text":
            text = "Research Query History\n" + "="*50 + "\n\n"
            for i, query_data in enumerate(st.session_state.query_history, 1):
                text += f"Query {i}\n"
                text += f"Mode: {query_data.get('mode', 'Unknown')}\n"
                text += f"Query: {query_data.get('query', 'N/A')}\n"
                if 'response' in query_data:
                    text += f"Response: {query_data['response'][:500]}...\n"
                text += "\n" + "-"*50 + "\n\n"
            
            st.download_button(
                label="💾 Download Text",
                data=text,
                file_name=f"research_queries_{int(time.time())}.txt",
                mime="text/plain"
            )
    
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def display_research_interface():
    """Display main research interface"""
    
    # Main header
    st.markdown('<div class="main-header">🔬 Research Q&A Assistant</div>', unsafe_allow_html=True)
    
    # Create tabs for different interfaces
    tab1, tab2, tab3 = st.tabs(["🔍 Query Research", "💬 Chat Mode", "📊 Analytics"])
    
    with tab1:
        display_query_interface()
    
    with tab2:
        display_chat_interface()
    
    with tab3:
        display_analytics()

def display_query_interface():
    """Display query interface for structured research"""
    
    st.subheader(f"🎯 {st.session_state.current_mode.title()} Mode")
    
    # Query input
    query_text = st.text_area(
        "Enter your research question:",
        height=100,
        placeholder="What would you like to research? Be specific and detailed for best results.",
        help="Provide a clear, focused research question. The more specific you are, the better the results will be."
    )
    
    # Additional context (optional)
    with st.expander("➕ Additional Context (Optional)"):
        context_text = st.text_area(
            "Provide additional context or specific aspects to focus on:",
            height=80,
            placeholder="Any specific aspects, time periods, populations, or methodologies to focus on..."
        )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            similarity_k = st.slider(
                "Number of sources to retrieve:",
                min_value=3,
                max_value=15,
                value=5,
                help="More sources provide comprehensive coverage but may include less relevant information"
            )
        
        with col2:
            show_sources = st.checkbox(
                "Show source details",
                value=True,
                help="Display detailed information about sources used in the response"
            )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button(
            f"🔬 Conduct {st.session_state.current_mode.title()} Research",
            type="primary",
            use_container_width=True
        )
    
    # Process query
    if submit_button and query_text.strip():
        process_research_query(query_text, context_text, similarity_k, show_sources)
    elif submit_button:
        st.warning("⚠️ Please enter a research question.")

def process_research_query(query_text: str, context: str, similarity_k: int, show_sources: bool):
    """Process research query and display results"""
    
    with st.spinner(f"🔬 Conducting {st.session_state.current_mode} research..."):
        try:
            # Update bot settings
            if hasattr(st.session_state.bot, '_query_engine'):
                st.session_state.bot._query_engine.update_settings(similarity_top_k=similarity_k)
            
            # Process query
            start_time = time.time()
            result = st.session_state.bot.process_query(
                query_text=query_text,
                mode=st.session_state.current_mode
            )
            processing_time = time.time() - start_time
            
            # Display results
            display_query_results(result, show_sources, processing_time)
            
            # Save to history
            save_query_to_history(query_text, result, context)
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            st.error(f"❌ Error processing query: {str(e)}")

def display_query_results(result: QueryResponse, show_sources: bool, processing_time: float):
    """Display formatted query results"""
    
    if result.success:
        st.success("✅ Research completed successfully!")
        
        # Show processing time
        st.caption(f"⏱️ Processed in {processing_time:.2f} seconds")
        
        # Display structured response
        if result.structured_response:
            display_structured_response(result.structured_response, show_sources)
        
        # Raw response as fallback
        if result.structured_response and result.structured_response.raw_response:
            with st.expander("📝 Raw Response"):
                st.write(result.structured_response.raw_response)
    
    else:
        st.error("❌ Research failed")
        if result.error:
            st.error(f"Error: {result.error.message}")
            if result.error.suggestion:
                st.info(f"💡 Suggestion: {result.error.suggestion}")

def display_structured_response(structured_response, show_sources: bool):
    """Display structured response based on type"""
    
    response_type = structured_response.response_type
    content = structured_response.content
    
    if response_type == "facts":
        display_fact_extraction(content, show_sources)
    elif response_type == "comparison":
        display_comparison_analysis(content, show_sources)
    elif response_type == "analysis":
        display_deep_analysis(content, show_sources)
    elif response_type == "summary":
        display_summary_response(content, show_sources)
    else:
        st.write(content)

def display_fact_extraction(content, show_sources: bool):
    """Display fact extraction results"""
    
    st.subheader("🔍 Key Facts and Findings")
    
    # Summary
    st.write(content.summary)
    
    # Key facts
    if content.key_facts:
        st.subheader("📋 Key Facts")
        for i, fact in enumerate(content.key_facts, 1):
            importance_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            importance_icon = importance_color.get(fact.importance, "⚪")
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>{importance_icon} Fact {i}:</strong> {fact.fact}
                {f'<br><em>Category: {fact.category}</em>' if fact.category else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Definitions
    if content.definitions:
        st.subheader("📚 Definitions")
        for definition in content.definitions:
            st.markdown(f"""
            <div class="metric-card">
                <strong>📖 {definition.term}</strong><br>
                {definition.definition}
                {f'<br><em>Context: {definition.context}</em>' if definition.context else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Sources
    if show_sources and content.sources:
        display_sources(content.sources)

def display_comparison_analysis(content, show_sources: bool):
    """Display comparison analysis results"""
    
    st.subheader("⚖️ Comparative Analysis")
    
    # Conclusion
    st.write(content.conclusion)
    
    # Comparison items
    if content.items:
        st.subheader("🔍 Items Compared")
        for item in content.items:
            with st.expander(f"📊 {item.name}"):
                st.write(f"**Description:** {item.description}")
                
                if item.key_points:
                    st.write("**Key Points:**")
                    for point in item.key_points:
                        st.write(f"• {point}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if item.strengths:
                        st.write("**Strengths:**")
                        for strength in item.strengths:
                            st.write(f"✅ {strength}")
                
                with col2:
                    if item.weaknesses:
                        st.write("**Weaknesses:**")
                        for weakness in item.weaknesses:
                            st.write(f"❌ {weakness}")
    
    # Similarities and differences
    col1, col2 = st.columns(2)
    
    with col1:
        if content.similarities:
            st.subheader("🤝 Similarities")
            for similarity in content.similarities:
                st.write(f"• {similarity}")
    
    with col2:
        if content.differences:
            st.subheader("🔄 Differences")
            for difference in content.differences:
                st.write(f"• {difference}")
    
    # Recommendation
    if content.recommendation:
        st.info(f"💡 **Recommendation:** {content.recommendation}")
    
    # Sources
    if show_sources and content.sources:
        display_sources(content.sources)

def display_deep_analysis(content, show_sources: bool):
    """Display deep analysis results"""
    
    st.subheader("🧠 Deep Analysis")
    
    # Executive summary
    st.markdown(f"**Executive Summary:** {content.executive_summary}")
    
    # Analysis sections
    if content.sections:
        for section in content.sections:
            st.subheader(f"📑 {section.title}")
            st.write(section.content)
            
            if section.key_insights:
                st.write("**Key Insights:**")
                for insight in section.key_insights:
                    st.write(f"💡 {insight}")
    
    # Key findings, implications, recommendations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if content.key_findings:
            st.subheader("🔑 Key Findings")
            for finding in content.key_findings:
                st.write(f"• {finding}")
    
    with col2:
        if content.implications:
            st.subheader("📈 Implications")
            for implication in content.implications:
                st.write(f"• {implication}")
    
    with col3:
        if content.recommendations:
            st.subheader("💡 Recommendations")
            for recommendation in content.recommendations:
                st.write(f"• {recommendation}")
    
    # Methodology and limitations
    if content.methodology or content.limitations:
        col1, col2 = st.columns(2)
        
        with col1:
            if content.methodology:
                st.subheader("🔬 Methodology")
                st.write(content.methodology)
        
        with col2:
            if content.limitations:
                st.subheader("⚠️ Limitations")
                st.write(content.limitations)
    
    # Sources
    if show_sources and content.sources:
        display_sources(content.sources)

def display_summary_response(content, show_sources: bool):
    """Display summary results"""
    
    st.subheader("📝 Research Summary")
    
    # Overview
    st.write(content.overview)
    
    # Main points
    if content.main_points:
        st.subheader("🎯 Main Points")
        for point in content.main_points:
            importance_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            importance_icon = importance_color.get(point.importance, "⚪")
            
            st.markdown(f"""
            <div class="metric-card">
                {importance_icon} {point.point}
                {f'<br><em>{point.details}</em>' if point.details else ''}
            </div>
            """, unsafe_allow_html=True)
    
    # Key themes
    if content.key_themes:
        st.subheader("🏷️ Key Themes")
        theme_cols = st.columns(min(len(content.key_themes), 4))
        for i, theme in enumerate(content.key_themes):
            with theme_cols[i % len(theme_cols)]:
                st.metric(label="Theme", value=theme)
    
    # Conclusion
    st.subheader("🎯 Conclusion")
    st.write(content.conclusion)
    
    # Scope and coverage
    if content.scope or content.coverage:
        col1, col2 = st.columns(2)
        
        with col1:
            if content.scope:
                st.write(f"**Scope:** {content.scope}")
        
        with col2:
            if content.coverage:
                st.write(f"**Coverage:** {content.coverage}")
    
    # Sources
    if show_sources and content.sources:
        display_sources(content.sources)

def display_sources(sources):
    """Display source references"""
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        relevance_score = f" (Relevance: {source.relevance_score:.2f})" if source.relevance_score else ""
        page_info = f", Page {source.page_number}" if source.page_number else ""
        section_info = f", Section: {source.section}" if source.section else ""
        
        st.markdown(f"""
        <div class="source-reference">
            <strong>📄 Source {i}: {source.title}</strong>{relevance_score}<br>
            {source.excerpt if source.excerpt else 'No excerpt available'}<br>
            <em>Reference: {source.title}{page_info}{section_info}</em>
        </div>
        """, unsafe_allow_html=True)

def display_chat_interface():
    """Display chat interface"""
    
    st.subheader("💬 Research Chat")
    st.caption("Have a conversation with the research assistant. Context is maintained across messages.")
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            role_class = "user-message" if message["role"] == "user" else "assistant-message"
            role_icon = "👤" if message["role"] == "user" else "🤖"
            
            st.markdown(f"""
            <div class="chat-message {role_class}">
                <strong>{role_icon} {message["role"].title()}:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    chat_input = st.chat_input("Ask a research question...")
    
    if chat_input:
        process_chat_message(chat_input)

def process_chat_message(message: str):
    """Process chat message"""
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": message})
    
    try:
        with st.spinner("🤖 Thinking..."):
            # Set research mode for chat context
            st.session_state.bot._chat_engine.set_research_mode(st.session_state.current_mode)
            
            # Get response
            response = st.session_state.bot.chat(message, mode=st.session_state.current_mode)
            
            # Add bot response to history
            st.session_state.chat_history.append({"role": "assistant", "content": str(response)})
            
            # Rerun to update display
            st.rerun()
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        st.rerun()

def display_analytics():
    """Display analytics and statistics"""
    
    st.subheader("📊 Research Analytics")
    
    if not st.session_state.query_history:
        st.info("📝 No research queries yet. Start by asking questions in the Query Research tab!")
        return
    
    # Query statistics
    total_queries = len(st.session_state.query_history)
    mode_counts = {}
    
    for query in st.session_state.query_history:
        mode = query.get('mode', 'unknown')
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", total_queries)
    
    with col2:
        most_used_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "None"
        st.metric("Most Used Mode", most_used_mode)
    
    with col3:
        avg_length = sum(len(q.get('query', '')) for q in st.session_state.query_history) / total_queries if total_queries > 0 else 0
        st.metric("Avg Query Length", f"{avg_length:.0f} chars")
    
    with col4:
        if st.session_state.bot:
            chat_stats = st.session_state.bot.get_stats()
            st.metric("Chat Messages", chat_stats.get("chat_messages", 0))
    
    # Mode distribution
    if mode_counts:
        st.subheader("📈 Research Mode Distribution")
        
        # Create simple bar chart data
        modes = list(mode_counts.keys())
        counts = list(mode_counts.values())
        
        # Display as columns
        for mode, count in mode_counts.items():
            percentage = (count / total_queries) * 100
            st.write(f"**{mode.title()}**: {count} queries ({percentage:.1f}%)")
            st.progress(percentage / 100)
    
    # Recent queries
    st.subheader("🕒 Recent Research Queries")
    
    recent_queries = st.session_state.query_history[-5:]  # Last 5 queries
    for i, query in enumerate(reversed(recent_queries), 1):
        with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {query.get('query', 'N/A')[:50]}..."):
            st.write(f"**Mode:** {query.get('mode', 'Unknown')}")
            st.write(f"**Query:** {query.get('query', 'N/A')}")
            if 'timestamp' in query:
                st.write(f"**Time:** {query['timestamp']}")

def save_query_to_history(query: str, result: QueryResponse, context: str = ""):
    """Save query to history"""
    
    query_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "context": context,
        "mode": result.mode,
        "success": result.success,
        "response": str(result.structured_response.raw_response) if result.structured_response else "",
        "processing_time": result.metadata.processing_time if result.metadata else 0
    }
    
    st.session_state.query_history.append(query_data)
    
    # Keep only last 100 queries to prevent memory issues
    if len(st.session_state.query_history) > 100:
        st.session_state.query_history = st.session_state.query_history[-100:]

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Check bot status and display main interface
    display_bot_status()
    display_research_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        f"🔬 {config.app_title} | Powered by LlamaIndex & OpenAI"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()