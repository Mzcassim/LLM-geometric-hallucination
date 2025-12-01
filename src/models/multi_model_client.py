"""Multi-model client abstraction.

Unified interface for OpenAI, Anthropic, and Together AI models.
"""

import os
from typing import Optional, Dict, Any
import anthropic
from openai import OpenAI


class MultiModelClient:
    """Unified client for multiple LLM providers."""
    
    def __init__(self, provider: str, model_name: str, **kwargs):
        """
        Initialize multi-model client.
        
        Args:
            provider: 'openai', 'anthropic', or 'together'
            model_name: Model identifier
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.kwargs = kwargs
        
        if self.provider == 'openai':
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        elif self.provider == 'anthropic':
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        elif self.provider == 'together':
            # Together uses OpenAI-compatible API
            self.client = OpenAI(
                api_key=os.getenv('TOGETHER_API_KEY'),
                base_url="https://api.together.xyz/v1"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> str:
        """Generate text completion."""
        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                # OpenAI and Together use same API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating with {self.provider}/{self.model_name}: {e}")
            raise
    
    def __repr__(self):
        return f"MultiModelClient(provider='{self.provider}', model='{self.model_name}')"


# Model configurations
MODEL_CONFIGS = {
    # Frontier models (current SOTA)
    'gpt-4o': {'provider': 'openai', 'model': 'gpt-4o-2024-08-06'},
    'gpt-4o-mini': {'provider': 'openai', 'model': 'gpt-4o-mini'},
    'claude-3-5-sonnet': {'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'},
    'claude-3-5-haiku': {'provider': 'anthropic', 'model': 'claude-3-5-haiku-20241022'},
    
    # Legacy models (older but still used)
    'gpt-3.5-turbo': {'provider': 'openai', 'model': 'gpt-3.5-turbo'},
    
    # Lesser-known models (Together AI)
    'llama-3.1-70b': {'provider': 'together', 'model': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'},
    'llama-3.1-8b': {'provider': 'together', 'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'},
    'mixtral-8x7b': {'provider': 'together', 'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1'},
    'qwen-2.5-72b': {'provider': 'together', 'model': 'Qwen/Qwen2.5-72B-Instruct-Turbo'},
}


def get_model_client(model_key: str) -> MultiModelClient:
    """Get a configured model client by key."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_key]
    return MultiModelClient(provider=config['provider'], model_name=config['model'])


if __name__ == "__main__":
    # Test different providers
    import sys
    
    test_prompt = "What is 2+2?"
    
    print("Testing multi-model clients...\n")
    
    # Test a few models if API keys are available
    test_models = ['gpt-4o-mini']  # Safe default
    
    if os.getenv('ANTHROPIC_API_KEY'):
        test_models.append('claude-3-5-haiku')
    
    if os.getenv('TOGETHER_API_KEY'):
        test_models.append('llama-3.1-8b')
    
    for model_key in test_models:
        try:
            print(f"Testing {model_key}...")
            client = get_model_client(model_key)
            response = client.generate(test_prompt, max_tokens=50)
            print(f"  Response: {response[:100]}...")
            print(f"  ✓ Success\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
