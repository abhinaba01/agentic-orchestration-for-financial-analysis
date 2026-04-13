from typing import List

import openai
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import Config


class TextEmbedder:
    """Embed text using OpenAI embeddings with retry logic."""

    def __init__(self, config: Config) -> None:
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.EMBEDDING_MODEL

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in a single OpenAI API call."""
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        return self._call_api(cleaned_texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        cleaned_query = query.replace("\n", " ")
        embeddings = self._call_api([cleaned_query])
        return embeddings[0]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.APIConnectionError)),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    )
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API and return ordered embeddings."""
        response = self.client.embeddings.create(input=texts, model=self.model)
        embeddings = [None] * len(texts)
        for item in response.data:
            index = item.index
            embeddings[index] = item.embedding
        return embeddings
