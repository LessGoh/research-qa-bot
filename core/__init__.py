"""
Core package for Research Q&A Bot (Minimal version)
"""

try:
    from .bot_manager import ResearchBot
except ImportError as e:
    print(f"Warning: Could not import ResearchBot: {e}")
    
    # Create a fallback class
    class ResearchBot:
        def __init__(self, *args, **kwargs):
            self.available = False
            print("Warning: Using fallback ResearchBot")
        
        def process_query(self, query_text, mode="analysis"):
            return {
                "success": False, 
                "error": {"message": "ResearchBot not available - import failed"},
                "query": query_text,
                "mode": mode
            }
        
        def chat(self, message, mode="analysis"):
            return "ResearchBot not available - import failed"
        
        def get_available_modes(self):
            return {"analysis": {"display_name": "Analysis", "description": "Basic analysis"}}
        
        def health_check(self):
            return {"status": "error", "components": {"import": {"status": "failed"}}}
        
        def get_stats(self):
            return {"chat_messages": 0, "mode": "fallback"}
        
        def clear_chat_history(self):
            pass

__all__ = ["ResearchBot"]
