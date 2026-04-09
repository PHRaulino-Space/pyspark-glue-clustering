"""
Abstract base class for embedding API clients.
Concrete implementations define URL, model, authentication, and request format.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingClient(ABC):
    """
    Abstraction over any embedding API.

    The concrete subclass is responsible for:
    - API URL and authentication headers
    - Model selection
    - Request/response serialisation
    - Rate limit handling

    This base class defines only the contract that the EmbeddingService depends on.
    """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: Non-empty list of plain-text strings.

        Returns:
            List of float vectors in the same order as `texts`.
            Each vector has length == self.vector_dimensions.

        Raises:
            Should raise on unrecoverable errors after internal retries.
            Retry logic and rate-limit back-off are handled here.
        """

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text string.

        Args:
            text: Plain-text string.

        Returns:
            Float vector of length == self.vector_dimensions.
        """

    @property
    @abstractmethod
    def vector_dimensions(self) -> int:
        """
        Number of dimensions in each embedding vector produced by this client.
        Must match EmbeddingConfig.vector_dimensions used in the pipeline.
        """
