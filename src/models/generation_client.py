"""Generation client for obtaining model responses."""

import time
from typing import Any, Optional
from openai import OpenAI


class GenerationClient:
    """Client for generating text responses from language models."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        client: Any = None,
        max_retries: int = 3,
        timeout: int = 60,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize generation client.
        
        Args:
            model_name: Name of the generation model
            client: OpenAI client instance (or None to create default)
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.client = client if client is not None else OpenAI()
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_system_prompt = "You are a helpful assistant that answers clearly and directly."
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt (uses default if None)
            
        Returns:
            Generated text response
        """
        system = system_prompt if system_prompt is not None else self.default_system_prompt
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Error generating response (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to generate response after {self.max_retries} attempts")
                    raise
        
        return ""  # Should never reach here
    
    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None
    ) -> list[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt for all requests
            
        Returns:
            List of generated responses
        """
        responses = []
        for i, prompt in enumerate(prompts):
            print(f"Generating response {i + 1}/{len(prompts)}...")
            response = self.generate(prompt, system_prompt)
            responses.append(response)
        
        return responses
