from typing import List
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from config import NVIDIA_EMBEDDING_KEY

class NvidiaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """NVIDIA Embedding Function using OpenAI-compatible API."""

    def __init__(self, model: str = "nvidia/nv-embed-v1", batch_size: int = 32):
        api_key = os.getenv("NVIDIA_EMBEDDING_KEY") or NVIDIA_EMBEDDING_KEY
        if not api_key:
            raise ValueError("NVIDIA_EMBEDDING_KEY is not set in environment variables.")
        
        self.client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")
        self.model = model
        self.batch_size = batch_size
        self.embedding_dim = 1024

    def __call__(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        
        texts = [text.replace("\n", " ").strip() for text in texts]
        texts = [text for text in texts if text]

        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    extra_body={"input_type": input_type}
                )
                for data in response.data:
                    embeddings.append(data.embedding)

            except Exception as e:
                print(f"[NvidiaEmbeddingFunction Error] {e}")
                # Add placeholder vectors for failed batch to maintain alignment
                embeddings.extend([[0.0]*self.embedding_dim for _ in batch])  # adjust 1024 to your embedding dim

        return embeddings
