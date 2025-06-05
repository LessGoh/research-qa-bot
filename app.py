"""
Main Streamlit application for Research Q&A Bot (Standalone version)
"""
import streamlit as st
import time
import os

# Configure Streamlit page first
st.set_page_config(
    page_title="🔬 Research Q&A Bot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get app title from environment or secrets
def get_app_title():
    """Get app title from environment or secrets"""
    try:
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and 'APP_TITLE' in st.secrets:
            return st.secrets["APP_TITLE"]
    except:
        pass
    
    # Fall back to environment variable or default
    return os.getenv("APP_TITLE", "🔬 Research Q&A Bot")

APP_TITLE = get_app_title()

# Safe import of core components
try:
    from core import ResearchBot
    CORE_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Core modules not available: {e}")
    CORE_AVAILABLE = False
    
    # Create a minimal fallback
    class ResearchBot:
        def __init__(self, *args, **kwargs):
            self.available = False
        
        def process_query(self, query_text, mode="analysis"):
            return type('obj', (object,), {
                'success': False,
                'error': type('obj', (object,), {'message': 'Bot not available'})(),
                'query': query_text,
                'mode': mode
            })()
        
        def chat(self, message, mode="analysis"):
            return "Bot not available - please check configuration"
        
        def get_available_modes(self):
            return {"analysis": {"display_name": "📊 Analysis", "description": "Basic analysis"}}
        
        def health_check(self):
            return {"status": "error", "components": {}}
        
        def get_stats(self):
            return {"chat_messages": 0}
        
        def clear_chat_history(self):
            pass

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
    
    .error-message {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .config-info {
        background-color: #f5f5f5;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #2196f3;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

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
def initialize_bot():
    """Initialize the research bot (cached)"""
    try:
        if not CORE_AVAILABLE:
            return None
            
        st.info("🤖 Initializing Research Bot...")
        
        # Initialize bot
        bot = ResearchBot()
        
        # Test bot functionality
        health = bot.health_check()
        if health["status"] == "error":
            st.warning(f"⚠️ Bot health check failed")
            return bot  # Return anyway, might still be partially functional
        
        st.success("✅ Research Bot initialized successfully")
        return bot
        
    except Exception as e:
        st.error(f"❌ Failed to initialize bot: {str(e)}")
        return None

def check_configuration():
    """Check and display configuration status"""
    config_status = {}
    
    # Check OpenAI API key
    openai_key = None
    try:
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            openai_key = st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")
    
    config_status["openai"] = bool(openai_key)
    
    # Check LlamaCloud credentials
    llama_key = None
    llama_pipeline = None
    
    try:
        if hasattr(st, 'secrets'):
            llama_key = st.secrets.get("LLAMA_CLOUD_API_KEY")
            llama_pipeline = st.secrets.get("LLAMA_CLOUD_PIPELINE_ID")
    except:
        pass
    
    if not llama_key:
        llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_pipeline:
        llama_pipeline = os.getenv("LLAMA_CLOUD_PIPELINE_ID")
    
    config_status["llamacloud"] = bool(llama_key and llama_pipeline)
    config_status["pipeline_id"] = llama_pipeline[:8] + "..." if llama_pipeline else "Missing"
    
    return config_status

def display_sidebar():
    """Display sidebar with configuration and controls"""
    with st.sidebar:
        st.title("🔬 Research Assistant")
        
        # Configuration status
        config_status = check_configuration()
        
        st.subheader("⚙️ Configuration Status")
        
        if config_status["openai"]:
            st.markdown('<div class="config-info">🔑 <strong>OpenAI:</strong> ✅ Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">🔑 <strong>OpenAI:</strong> ❌ Missing API Key</div>', unsafe_allow_html=True)
        
        if config_status["llamacloud"]:
            st.markdown('<div class="config-info">☁️ <strong>LlamaCloud:</strong> ✅ Connected</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="config-info">📋 <strong>Pipeline:</strong> {config_status["pipeline_id"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">☁️ <strong>LlamaCloud:</strong> ❌ Missing Credentials</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Bot status
        if st.session_state.bot and CORE_AVAILABLE:
            st.success("🤖 Bot: Online")
            
            # Health check button
            if st.button("🔍 Health Check"):
                health = st.session_state.bot.health_check()
                if health["status"] == "healthy":
                    st.success("✅ All systems operational")
                elif health["status"] == "degraded":
                    st.warning("⚠️ Some components have issues")
                    with st.expander("Details"):
                        st.json(health.get("components", {}))
                else:
                    st.error(f"❌ Status: {health['status']}")
                    with st.expander("Details"):
                        st.json(health)
        else:
            st.error("🤖 Bot: Offline")
        
        st.divider()
        
        # Research mode selection
        st.subheader("🎯 Research Mode")
        
        if st.session_state.bot and CORE_AVAILABLE:
            try:
                available_modes = st.session_state.bot.get_available_modes()
                mode_options = {
                    mode_data["display_name"]: mode_name 
                    for mode_name, mode_data in available_modes.items()
                }
                
                if mode_options:
                    selected_display = st.selectbox(
                        "Select research mode:",
                        options=list(mode_options.keys()),
                        index=0
                    )
                    
                    st.session_state.current_mode = mode_options[selected_display]
                    
                    # Display mode description
                    current_mode_data = available_modes[st.session_state.current_mode]
                    st.info(f"ℹ️ {current_mode_data['description']}")
                else:
                    st.warning("No research modes available")
            except Exception as e:
                st.error(f"Error loading modes: {e}")
        else:
            # Fallback mode selection
            mode_options = {
                "📊 Deep Analysis": "analysis",
                "🔍 Fact Extraction": "facts", 
                "⚖️ Comparative Analysis": "comparison",
                "📝 Summarization": "summary"
            }
            
            selected_display = st.selectbox(
                "Select research mode:",
                options=list(mode_options.keys()),
                index=0
            )
            
            st.session_state.current_mode = mode_options[selected_display]
        
        st.divider()
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.bot:
                st.session_state.bot.clear_chat_history()
            st.session_state.chat_history.clear()
            st.rerun()
        
        # Show statistics
        if st.session_state.bot:
            try:
                stats = st.session_state.bot.get_stats()
                st.metric("💬 Messages", stats.get("chat_messages", 0))
                st.metric("🤖 Mode", stats.get("mode", "unknown"))
            except Exception as e:
                st.caption(f"Stats unavailable: {e}")
        
        st.divider()
        
        # Troubleshooting
        if not CORE_AVAILABLE or not config_status["openai"] or not config_status["llamacloud"]:
            st.subheader("🛠️ Troubleshooting")
            
            if not CORE_AVAILABLE:
                st.error("❌ Core modules not available")
                st.info("Check your installation and dependencies")
            
            if not config_status["openai"]:
                st.warning("⚠️ OpenAI API key missing")
                st.info("Add OPENAI_API_KEY to Streamlit secrets")
            
            if not config_status["llamacloud"]:
                st.warning("⚠️ LlamaCloud credentials missing")
                st.info("Add LLAMA_CLOUD_API_KEY and LLAMA_CLOUD_PIPELINE_ID to secrets")

def display_bot_status():
    """Display bot initialization status"""
    config_status = check_configuration()
    
    if not CORE_AVAILABLE:
        st.error("❌ Core components not available. Please check your installation.")
        st.info("""
        **Troubleshooting steps:**
        1. Check that all required packages are installed
        2. Verify your dependencies in requirements.txt
        3. Check the app logs for import errors
        """)
        return False
    
    if not config_status["openai"] or not config_status["llamacloud"]:
        st.error("❌ Missing required credentials.")
        st.info("""
        **Required Streamlit Secrets:**
        - OPENAI_API_KEY
        - LLAMA_CLOUD_API_KEY  
        - LLAMA_CLOUD_PIPELINE_ID
        """)
        return False
    
    if st.session_state.bot is None and not st.session_state.initialization_attempted:
        with st.spinner("🤖 Initializing Research Bot..."):
            st.session_state.bot = initialize_bot()
            st.session_state.initialization_attempted = True
    
    if st.session_state.bot is None:
        st.error("❌ Bot initialization failed. Please check the sidebar for details.")
        return False
    else:
        return True

def display_main_interface():
    """Display main research interface"""
    
    # Main header
    st.markdown(f'<div class="main-header">{APP_TITLE}</div>', unsafe_allow_html=True)
    
    # Check if bot is available
    if not display_bot_status():
        st.stop()
    
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
        process_research_query(query_text)
    elif submit_button:
        st.warning("⚠️ Please enter a research question.")

def process_research_query(query_text: str):
    """Process research query and display results"""
    
    if not st.session_state.bot:
        st.error("❌ Bot not available")
        return
    
    with st.spinner(f"🔬 Conducting {st.session_state.current_mode} research..."):
        try:
            start_time = time.time()
            result = st.session_state.bot.process_query(
                query_text=query_text,
                mode=st.session_state.current_mode
            )
            processing_time = time.time() - start_time
            
            # Display results
            display_query_results(result, processing_time)
            
            # Save to history
            save_query_to_history(query_text, result)
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

def display_query_results(result, processing_time: float):
    """Display query results"""
    
    if hasattr(result, 'success') and result.success:
        st.markdown(f"""
        <div class="success-message">
            ✅ <strong>Research completed successfully!</strong><br>
            ⏱️ Processed in {processing_time:.2f} seconds
        </div>
        """, unsafe_allow_html=True)
        
        # Display response
        if hasattr(result, 'raw_response') and result.raw_response:
            st.markdown("### 📄 Response")
            st.write(result.raw_response)
        else:
            st.info("Response generated but content not available for display.")
    
    else:
        st.markdown("""
        <div class="error-message">
            ❌ <strong>Research failed</strong>
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(result, 'error') and result.error:
            error_msg = result.error.message if hasattr(result.error, 'message') else str(result.error)
            st.error(f"Error: {error_msg}")

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
    
    if not st.session_state.bot:
        st.error("❌ Bot not available")
        return
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": message})
    
    try:
        with st.spinner("🤖 Thinking..."):
            # Get response
            response = st.session_state.bot.chat(message, mode=st.session_state.current_mode)
            
            # Add bot response to history
            st.session_state.chat_history.append({"role": "assistant", "content": str(response)})
            
            # Rerun to update display
            st.rerun()
    
    except Exception as e:
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
        st.metric("Chat Messages", len(st.session_state.chat_history))
    
    # Mode distribution
    if mode_counts:
        st.subheader("📈 Research Mode Distribution")
        
        for mode, count in mode_counts.items():
            percentage = (count / total_queries) * 100
            st.write(f"**{mode.title()}**: {count} queries ({percentage:.1f}%)")
            st.progress(percentage / 100)

def save_query_to_history(query: str, result):
    """Save query to history"""
    
    query_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "mode": st.session_state.current_mode,
        "success": getattr(result, 'success', False),
        "response": getattr(result, 'raw_response', str(result))[:200] + "..."
    }
    
    st.session_state.query_history.append(query_data)
    
    # Keep only last 50 queries to prevent memory issues
    if len(st.session_state.query_history) > 50:
        st.session_state.query_history = st.session_state.query_history[-50:]

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display sidebar
    display_sidebar()
    
    # Display main interface
    display_main_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        f"🔬 {APP_TITLE} | Powered by LlamaIndex & OpenAI"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
