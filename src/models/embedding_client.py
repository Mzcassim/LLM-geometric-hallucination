"""Embedding client for obtaining text embeddings from OpenAI or other providers."""

import time
from typing import Any
import numpy as np
from openai import OpenAI


class EmbeddingClient:
    """Client for generating text embeddings."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        client: Any = None,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: int = 60
    ):
        """
        Initialize embedding client.
        
        Args:
            model_name: Name of the embedding model
            client: OpenAI client instance (or None to create default)
            batch_size: Number of texts to embed in each API call
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.client = client if client is not None else OpenAI()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
    
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            2D NumPy array of shape (n_texts, embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a single batch of texts with retry logic.
        
        Args:
            texts: List of texts to embed (must be <= batch_size)
            
        Returns:
            List of embedding vectors
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    timeout=self.timeout
                )
                
                # Extract embeddings in the correct order
                embeddings = [item.embedding for item in response.data]
                return embeddings
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    print(f"Error embedding batch (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to embed batch after {self.max_retries} attempts")
                    raise
        
        return []  # Should never reach here
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            1D NumPy array of embedding values
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]
