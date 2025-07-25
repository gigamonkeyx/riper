"""
OpenRouter API Client for RIPER-Ω System
Provides integration with Qwen3-Coder-480B-A35B-Instruct via OpenRouter
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterConfig:
    """OpenRouter configuration settings"""

    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "qwen/qwen-2.5-coder-32b-instruct"  # Updated to available model
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30


@dataclass
class OpenRouterResponse:
    """OpenRouter API response structure"""

    success: bool
    content: str
    usage: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    model_used: Optional[str] = None


class OpenRouterClient:
    """
    OpenRouter API client for Qwen3 model integration
    Handles authentication, rate limiting, and error recovery
    """

    def __init__(self, config: Optional[OpenRouterConfig] = None):
        # Enforce D: drive compliance FIRST
        self._enforce_d_drive_compliance()

        self.config = config or self._load_config()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://riper-omega.local",  # Required by OpenRouter
                "X-Title": "RIPER-Omega Multi-Agent System",
            }
        )

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

        logger.info(f"OpenRouter client initialized for model: {self.config.model}")

    def _enforce_d_drive_compliance(self):
        """Enforce D: drive storage compliance for all operations"""
        import tempfile

        # Load from .env if available
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        # Set all cache directories to D: drive
        cache_vars = {
            "HF_HOME": "D:/huggingface_cache",
            "TRANSFORMERS_CACHE": "D:/transformers_cache",
            "HF_DATASETS_CACHE": "D:/datasets_cache",
            "TORCH_HOME": "D:/torch_cache",
            "XDG_CACHE_HOME": "D:/cache",
            "TMPDIR": "D:/temp",
            "TEMP": "D:/temp",
            "TMP": "D:/temp",
        }

        for var, path in cache_vars.items():
            if var not in os.environ:  # Don't override if already set
                os.environ[var] = path

        # Set Python tempfile directory
        tempfile.tempdir = "D:/temp"

        # Create directories if they don't exist
        cache_dirs = [
            "D:/huggingface_cache",
            "D:/transformers_cache",
            "D:/datasets_cache",
            "D:/torch_cache",
            "D:/cache",
            "D:/temp",
        ]

        for cache_dir in cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)

    def _load_config(self) -> OpenRouterConfig:
        """Load configuration from environment or defaults"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("OPENROUTER_API_KEY not set - using placeholder")
            api_key = "your-api-key-here"

        return OpenRouterConfig(
            api_key=api_key,
            model=os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct"),
        )

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> OpenRouterResponse:
        """
        Send chat completion request to OpenRouter

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API call
        """
        start_time = time.time()

        # Rate limiting
        self._rate_limit()

        # Prepare messages
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Prepare request payload
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": False,
        }

        try:
            response = self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=self.config.timeout,
            )

            execution_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                return OpenRouterResponse(
                    success=True,
                    content=data["choices"][0]["message"]["content"],
                    usage=data.get("usage", {}),
                    execution_time=execution_time,
                    model_used=data.get("model", self.config.model),
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"OpenRouter API error: {error_msg}")

                return OpenRouterResponse(
                    success=False,
                    content="",
                    usage={},
                    execution_time=execution_time,
                    error_message=error_msg,
                )

        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {self.config.timeout}s"
            logger.error(error_msg)

            return OpenRouterResponse(
                success=False,
                content="",
                usage={},
                execution_time=time.time() - start_time,
                error_message=error_msg,
            )

        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)

            return OpenRouterResponse(
                success=False,
                content="",
                usage={},
                execution_time=time.time() - start_time,
                error_message=error_msg,
            )

    def qwen3_code_generation(
        self, prompt: str, context: Optional[str] = None, language: str = "python"
    ) -> OpenRouterResponse:
        """
        Specialized method for code generation using Qwen3-Coder

        Args:
            prompt: Code generation prompt
            context: Optional context or existing code
            language: Programming language (default: python)
        """
        system_prompt = f"""You are Qwen3-Coder, an expert programming assistant specializing in {language}.
        
        Your role in the RIPER-Ω system:
        - Generate high-quality, optimized code
        - Follow evolutionary algorithm best practices
        - Optimize for RTX 3080 GPU performance when applicable
        - Maintain >70% fitness standards in code quality
        - Ensure local focus (no cloud dependencies in generated code)
        
        Provide clean, well-commented code with explanations."""

        messages = []

        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nRequest:\n{prompt}",
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        return self.chat_completion(messages, system_prompt)

    def qwen3_fitness_analysis(
        self, fitness_data: Dict[str, Any], generation: int = 0
    ) -> OpenRouterResponse:
        """
        Specialized method for evolutionary fitness analysis

        Args:
            fitness_data: Dictionary containing fitness metrics
            generation: Current generation number
        """
        system_prompt = """You are Qwen3-Coder analyzing evolutionary algorithm fitness for the RIPER-Ω system.
        
        Your analysis should:
        - Evaluate fitness trends and patterns
        - Suggest optimization strategies
        - Identify potential improvements
        - Recommend parameter adjustments
        - Ensure >70% fitness threshold achievement
        
        Provide actionable insights in JSON format."""

        prompt = f"""Analyze the following evolutionary fitness data for generation {generation}:

        Fitness Data: {json.dumps(fitness_data, indent=2)}
        
        Provide analysis including:
        1. Current fitness assessment (0.0-1.0 scale)
        2. Improvement recommendations
        3. Parameter optimization suggestions
        4. Next generation strategy
        
        Format response as JSON with clear recommendations."""

        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, system_prompt)

    def qwen3_agent_coordination(
        self, coordination_request: Dict[str, Any]
    ) -> OpenRouterResponse:
        """
        Specialized method for multi-agent coordination decisions

        Args:
            coordination_request: Dictionary containing coordination parameters
        """
        system_prompt = """You are Qwen3-Coder coordinating multi-agent systems in RIPER-Ω.
        
        Your coordination role:
        - Optimize agent task distribution
        - Resolve coordination conflicts
        - Suggest communication improvements
        - Enhance A2A protocol efficiency
        - Maintain system coherence
        
        Provide clear coordination decisions and rationale."""

        prompt = f"""Multi-agent coordination request:

        {json.dumps(coordination_request, indent=2)}
        
        Provide coordination decision including:
        1. Task assignment recommendations
        2. Communication protocol optimizations
        3. Resource allocation suggestions
        4. Conflict resolution strategies
        
        Focus on maximizing system efficiency and fitness."""

        messages = [{"role": "user", "content": prompt}]
        return self.chat_completion(messages, system_prompt)

    def test_connection(self) -> bool:
        """Test OpenRouter API connection"""
        try:
            response = self.chat_completion(
                [
                    {
                        "role": "user",
                        "content": "Hello, please respond with 'OpenRouter connection successful'",
                    }
                ]
            )

            if response.success and "successful" in response.content.lower():
                logger.info("✅ OpenRouter connection test successful")
                return True
            else:
                logger.error(
                    f"❌ OpenRouter connection test failed: {response.error_message}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ OpenRouter connection test error: {e}")
            return False


# Utility functions for easy integration
def get_openrouter_client() -> OpenRouterClient:
    """Get configured OpenRouter client instance"""
    return OpenRouterClient()


def qwen3_generate_code(prompt: str, context: Optional[str] = None) -> str:
    """Quick code generation using Qwen3"""
    client = get_openrouter_client()
    response = client.qwen3_code_generation(prompt, context)

    if response.success:
        return response.content
    else:
        logger.error(f"Code generation failed: {response.error_message}")
        return f"# Code generation failed: {response.error_message}"


def qwen3_analyze_fitness(fitness_data: Dict[str, Any]) -> Dict[str, Any]:
    """Quick fitness analysis using Qwen3"""
    client = get_openrouter_client()
    response = client.qwen3_fitness_analysis(fitness_data)

    if response.success:
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"analysis": response.content, "error": "JSON parsing failed"}
    else:
        return {"error": response.error_message}


# Configuration helper
def setup_openrouter_config(api_key: str, model: Optional[str] = None):
    """Setup OpenRouter configuration"""
    os.environ["OPENROUTER_API_KEY"] = api_key
    if model:
        os.environ["OPENROUTER_MODEL"] = model

    logger.info("OpenRouter configuration updated")
