"""
Abstract base class for embedding API clients.
Concrete implementations define URL, model, authentication, and request format.
"""

from abc import ABC, abstractmethod


class BaseEmbeddingClient(ABC):
    """
    Abstraction over any embedding API (e.g. Edi Bins).

    The concrete subclass is responsible for:
    - API URL and authentication headers
    - Model selection
    - Request/response serialisation
    - Rate limit handling
    - Token lifecycle management

    This base class defines only the contract that the EmbeddingService depends on.
    """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: Non-empty list of plain-text strings.
                   Max length is determined by the API limit (e.g. 500 for Edi Bins).

        Returns:
            List of float vectors in the same order as `texts`.
            Each vector has length == self.vector_dimensions.

        Raises:
            Should raise on unrecoverable errors after internal retries.
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

    @abstractmethod
    async def refresh_token(self) -> None:
        """
        Renew the authentication token used for API calls.

        Called by EmbeddingService before every batch request and between
        parallel calls to ensure the token is always valid.

        The concrete implementation must:
        - Request a new token from the auth endpoint
        - Store it internally so the next embed_batch() call uses it
        - Be safe to call concurrently (EmbeddingService serialises calls
          via an asyncio.Lock, but the implementation should be idempotent
          in case it is called from multiple contexts)

        Raises:
            Should raise if the token cannot be refreshed after retries,
            so the calling service can abort rather than send invalid requests.
        """

    @property
    @abstractmethod
    def vector_dimensions(self) -> int:
        """
        Number of dimensions in each embedding vector produced by this client.
        Must match EmbeddingConfig.vector_dimensions used in the pipeline.
        """
