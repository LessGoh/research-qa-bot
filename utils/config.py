"""
Configuration loader for Research Q&A Bot with Streamlit Cloud support
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration manager for the application with Streamlit Cloud support"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load environment variables first
        load_dotenv()
        
        # Load YAML configuration
        self.config_path = Path(config_path)
        self._config = self._load_yaml_config()
        
        # Override with environment variables
        self._apply_env_overrides()
        
        # Override with Streamlit secrets (для cloud деплоя)
        self._apply_streamlit_secrets()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            # Create default config if not exists
            default_config = self._get_default_config()
            return default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "app": {
                "title": "🔬 Research Q&A Bot",
                "description": "AI-powered assistant for scientific research and document analysis",
                "version": "1.0.0"
            },
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
                "streaming": True
            },
            "llamacloud": {
                "use_cloud": True,
                "api_base": "https://api.cloud.llamaindex.ai/api/v1",
                "similarity_top_k": 5
            },
            "llamaindex": {
                "index_path": "./data/index",
                "similarity_top_k": 5,
                "response_mode": "compact"
            },
            "research": {
                "modes": [
                    {
                        "name": "analysis",
                        "display_name": "📊 Deep Analysis",
                        "description": "Comprehensive analysis of research topics",
                        "temperature": 0.2,
                        "max_tokens": 3000
                    },
                    {
                        "name": "facts",
                        "display_name": "🔍 Fact Extraction",
                        "description": "Extract key facts and definitions",
                        "temperature": 0.1,
                        "max_tokens": 1500
                    },
                    {
                        "name": "comparison",
                        "display_name": "⚖️ Comparative Analysis",
                        "description": "Compare different sources and findings",
                        "temperature": 0.15,
                        "max_tokens": 2500
                    },
                    {
                        "name": "summary",
                        "display_name": "📝 Summarization",
                        "description": "Summarize large volumes of documentation",
                        "temperature": 0.2,
                        "max_tokens": 2000
                    }
                ]
            },
            "chat": {
                "max_history": 50,
                "memory_type": "buffer",
                "save_history": False
            },
            "output": {
                "structured_responses": True,
                "include_sources": True,
                "include_confidence": False,
                "export_formats": ["json", "markdown", "text"]
            },
            "ui": {
                "theme": "light",
                "sidebar_expanded": True,
                "show_query_stats": True,
                "show_source_preview": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _apply_env_overrides(self):
        """Override configuration with environment variables"""
        # OpenAI settings
        if os.getenv("OPENAI_API_KEY"):
            self._config.setdefault("openai", {})["api_key"] = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("OPENAI_MODEL"):
            self._config.setdefault("llm", {})["model"] = os.getenv("OPENAI_MODEL")
        
        if os.getenv("OPENAI_TEMPERATURE"):
            try:
                temp = float(os.getenv("OPENAI_TEMPERATURE"))
                self._config.setdefault("llm", {})["temperature"] = temp
            except (ValueError, TypeError):
                pass
        
        # LlamaCloud settings
        if os.getenv("LLAMA_CLOUD_API_KEY"):
            self._config.setdefault("llamacloud", {})["api_key"] = os.getenv("LLAMA_CLOUD_API_KEY")
        
        if os.getenv("LLAMA_CLOUD_PIPELINE_ID"):
            self._config.setdefault("llamacloud", {})["pipeline_id"] = os.getenv("LLAMA_CLOUD_PIPELINE_ID")
        
        if os.getenv("LLAMA_CLOUD_USE_CLOUD"):
            use_cloud = os.getenv("LLAMA_CLOUD_USE_CLOUD").lower() in ('true', '1', 'yes', 'on')
            self._config.setdefault("llamacloud", {})["use_cloud"] = use_cloud
        
        # LlamaIndex local settings
        if os.getenv("LLAMA_INDEX_PATH"):
            self._config.setdefault("llamaindex", {})["index_path"] = os.getenv("LLAMA_INDEX_PATH")
        
        if os.getenv("SIMILARITY_TOP_K"):
            try:
                top_k = int(os.getenv("SIMILARITY_TOP_K"))
                self._config.setdefault("llamaindex", {})["similarity_top_k"] = top_k
                self._config.setdefault("llamacloud", {})["similarity_top_k"] = top_k
            except (ValueError, TypeError):
                pass
        
        # App settings
        if os.getenv("APP_TITLE"):
            self._config.setdefault("app", {})["title"] = os.getenv("APP_TITLE")
        
        if os.getenv("LOG_LEVEL"):
            self._config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
    
    def _apply_streamlit_secrets(self):
        """Apply Streamlit secrets if available (for cloud deployment)"""
        try:
            import streamlit as st
            
            # Check if we're in a Streamlit environment and secrets are available
            if hasattr(st, 'secrets') and st.secrets:
                # OpenAI settings
                if 'OPENAI_API_KEY' in st.secrets:
                    self._config.setdefault("openai", {})["api_key"] = st.secrets["OPENAI_API_KEY"]
                
                if 'OPENAI_MODEL' in st.secrets:
                    self._config.setdefault("llm", {})["model"] = st.secrets["OPENAI_MODEL"]
                
                if 'OPENAI_TEMPERATURE' in st.secrets:
                    try:
                        temp = float(st.secrets["OPENAI_TEMPERATURE"])
                        self._config.setdefault("llm", {})["temperature"] = temp
                    except (ValueError, TypeError):
                        pass
                
                # LlamaCloud settings
                if 'LLAMA_CLOUD_API_KEY' in st.secrets:
                    self._config.setdefault("llamacloud", {})["api_key"] = st.secrets["LLAMA_CLOUD_API_KEY"]
                
                if 'LLAMA_CLOUD_PIPELINE_ID' in st.secrets:
                    self._config.setdefault("llamacloud", {})["pipeline_id"] = st.secrets["LLAMA_CLOUD_PIPELINE_ID"]
                
                if 'LLAMA_CLOUD_USE_CLOUD' in st.secrets:
                    use_cloud_str = str(st.secrets["LLAMA_CLOUD_USE_CLOUD"]).lower()
                    use_cloud = use_cloud_str in ('true', '1', 'yes', 'on')
                    self._config.setdefault("llamacloud", {})["use_cloud"] = use_cloud
                
                # App settings
                if 'APP_TITLE' in st.secrets:
                    self._config.setdefault("app", {})["title"] = st.secrets["APP_TITLE"]
                
                if 'LOG_LEVEL' in st.secrets:
                    self._config.setdefault("logging", {})["level"] = st.secrets["LOG_LEVEL"]
                
                # Similarity settings
                if 'SIMILARITY_TOP_K' in st.secrets:
                    try:
                        top_k = int(st.secrets["SIMILARITY_TOP_K"])
                        self._config.setdefault("llamaindex", {})["similarity_top_k"] = top_k
                        self._config.setdefault("llamacloud", {})["similarity_top_k"] = top_k
                    except (ValueError, TypeError):
                        pass
                
                # Auto-enable cloud mode if we have cloud credentials
                if (self._config.get("llamacloud", {}).get("api_key") and 
                    self._config.get("llamacloud", {}).get("pipeline_id")):
                    self._config.setdefault("llamacloud", {})["use_cloud"] = True
                
        except ImportError:
            # Streamlit не доступен (локальная разработка)
            pass
        except Exception as e:
            # Ошибка доступа к secrets - это нормально для локальной разработки
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (supports dot notation like 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name
            
        Returns:
            Dictionary with section configuration
        """
        return self._config.get(section, {})
    
    # Convenience properties for common settings
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key"""
        key = (os.getenv("OPENAI_API_KEY") or 
               self.get("openai.api_key") or
               self._get_streamlit_secret("OPENAI_API_KEY"))
        
        if not key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or Streamlit secret.")
        return key
    
    @property
    def llm_model(self) -> str:
        """Get LLM model name"""
        return (os.getenv("OPENAI_MODEL") or 
                self._get_streamlit_secret("OPENAI_MODEL") or
                self.get("llm.model", "gpt-4"))
    
    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature"""
        # Try environment variable first
        env_temp = os.getenv("OPENAI_TEMPERATURE")
        if env_temp:
            try:
                return float(env_temp)
            except (ValueError, TypeError):
                pass
        
        # Try Streamlit secrets
        secret_temp = self._get_streamlit_secret("OPENAI_TEMPERATURE")
        if secret_temp:
            try:
                return float(secret_temp)
            except (ValueError, TypeError):
                pass
        
        # Fall back to config
        return self.get("llm.temperature", 0.1)
    
    @property
    def use_llamacloud(self) -> bool:
        """Check if should use LlamaCloud"""
        # Check environment variable
        env_cloud = os.getenv("LLAMA_CLOUD_USE_CLOUD")
        if env_cloud:
            return env_cloud.lower() in ('true', '1', 'yes', 'on')
        
        # Check Streamlit secrets
        secret_cloud = self._get_streamlit_secret("LLAMA_CLOUD_USE_CLOUD")
        if secret_cloud:
            return str(secret_cloud).lower() in ('true', '1', 'yes', 'on')
        
        # Check if we have cloud credentials - auto-enable if available
        if (self.get("llamacloud.api_key") and self.get("llamacloud.pipeline_id")):
            return True
        
        # Fall back to config
        return self.get("llamacloud.use_cloud", False)
    
    @property
    def llamacloud_api_key(self) -> Optional[str]:
        """Get LlamaCloud API key (optional, returns None if not found)"""
        return (os.getenv("LLAMA_CLOUD_API_KEY") or
                self._get_streamlit_secret("LLAMA_CLOUD_API_KEY") or
                self.get("llamacloud.api_key"))
    
    @property
    def llamacloud_pipeline_id(self) -> Optional[str]:
        """Get LlamaCloud pipeline ID (optional, returns None if not found)"""
        return (os.getenv("LLAMA_CLOUD_PIPELINE_ID") or
                self._get_streamlit_secret("LLAMA_CLOUD_PIPELINE_ID") or
                self.get("llamacloud.pipeline_id"))
    
    @property
    def index_path(self) -> str:
        """Get LlamaIndex local path"""
        return (os.getenv("LLAMA_INDEX_PATH") or
                self.get("llamaindex.index_path", "./data/index"))
    
    @property
    def similarity_top_k(self) -> int:
        """Get similarity top k"""
        # Try environment variable first
        env_k = os.getenv("SIMILARITY_TOP_K")
        if env_k:
            try:
                return int(env_k)
            except (ValueError, TypeError):
                pass
        
        # Try Streamlit secrets
        secret_k = self._get_streamlit_secret("SIMILARITY_TOP_K")
        if secret_k:
            try:
                return int(secret_k)
            except (ValueError, TypeError):
                pass
        
        # Use cloud or local config based on mode
        if self.use_llamacloud:
            return self.get("llamacloud.similarity_top_k", 5)
        else:
            return self.get("llamaindex.similarity_top_k", 5)
    
    @property
    def app_title(self) -> str:
        """Get application title"""
        return (os.getenv("APP_TITLE") or
                self._get_streamlit_secret("APP_TITLE") or
                self.get("app.title", "Research Q&A Bot"))
    
    @property
    def research_modes(self) -> list:
        """Get research modes configuration"""
        return self.get("research.modes", [])
    
    @property
    def max_chat_history(self) -> int:
        """Get max chat history"""
        return self.get("chat.max_history", 50)
    
    @property
    def log_level(self) -> str:
        """Get logging level"""
        return (os.getenv("LOG_LEVEL") or
                self._get_streamlit_secret("LOG_LEVEL") or
                self.get("logging.level", "INFO"))
    
    @property
    def log_file(self) -> str:
        """Get log file path"""
        return self.get("logging.file", "./logs/app.log")
    
    def _get_streamlit_secret(self, key: str) -> Optional[str]:
        """Helper to safely get Streamlit secret"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except (ImportError, Exception):
            pass
        return None
    
    def update(self, key: str, value: Any):
        """
        Update configuration value
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """
        Save configuration to YAML file
        
        Args:
            path: Optional path to save config (defaults to original path)
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self._config, file, default_flow_style=False, indent=2)
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment configuration info for debugging"""
        info = {
            "deployment_mode": "cloud" if self.use_llamacloud else "local",
            "has_openai_key": bool(self.get("openai.api_key")),
            "has_llamacloud_key": bool(self.get("llamacloud.api_key")),
            "has_llamacloud_pipeline": bool(self.get("llamacloud.pipeline_id")),
            "config_sources": []
        }
        
        # Check config sources
        if self.config_path.exists():
            info["config_sources"].append("yaml_file")
        
        if any(os.getenv(key) for key in ["OPENAI_API_KEY", "LLAMA_CLOUD_API_KEY"]):
            info["config_sources"].append("environment_variables")
        
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and st.secrets:
                info["config_sources"].append("streamlit_secrets")
        except:
            pass
        
        return info
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required OpenAI settings
        try:
            self.openai_api_key
        except ValueError as e:
            validation["valid"] = False
            validation["errors"].append(str(e))
        
        # Check cloud configuration if cloud mode is enabled
        if self.use_llamacloud:
            if not self.llamacloud_api_key:
                validation["valid"] = False
                validation["errors"].append("LlamaCloud API key not found but cloud mode is enabled")
            
            if not self.llamacloud_pipeline_id:
                validation["valid"] = False
                validation["errors"].append("LlamaCloud pipeline ID not found but cloud mode is enabled")
        else:
            # Check local index path
            if not Path(self.index_path).exists():
                validation["warnings"].append(f"Local index path does not exist: {self.index_path}")
        
        # Check research modes
        if not self.research_modes:
            validation["warnings"].append("No research modes configured")
        
        return validation


# Global configuration instance
config = Config()
