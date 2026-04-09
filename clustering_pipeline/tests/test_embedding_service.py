"""
Unit tests for EmbeddingService using a mock embedding client.

Runs without any real API calls or Spark cluster.
"""

import asyncio
import pytest

from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

from clustering_pipeline.config.settings import EmbeddingConfig
from clustering_pipeline.core.base_embedding_client import BaseEmbeddingClient
from clustering_pipeline.services.embedding_service import EmbeddingService
from clustering_pipeline.tests.fixtures.sample_embeddings import (
    generate_small_synthetic_embeddings,
)
from clustering_pipeline.utils.logger import PipelineLogger
from clustering_pipeline.utils.spark_utils import create_local_spark_session


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class MockEmbeddingClient(BaseEmbeddingClient):
    """
    Deterministic mock that returns pre-seeded vectors without any HTTP calls.
    Supports simulated failures to exercise retry logic.
    Tracks refresh_token() calls for assertion in tests.
    """

    def __init__(self, dims: int = 32, fail_first_n: int = 0):
        self._dims = dims
        self._fail_count = fail_first_n
        self._call_count = 0
        self.refresh_count = 0  # exposed for test assertions

    async def embed_batch(self, texts: list) -> list:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Simulated failure #{self._call_count}")
        return [[float(i % 10) / 10 for i in range(self._dims)] for _ in texts]

    async def embed_single(self, text: str) -> list:
        return [0.0] * self._dims

    async def refresh_token(self) -> None:
        """No-op for tests; tracks call count."""
        self.refresh_count += 1

    @property
    def vector_dimensions(self) -> int:
        return self._dims


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    session = create_local_spark_session("test-embedding-service")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def embedding_config():
    return EmbeddingConfig(
        batch_size=50,
        max_concurrent=5,
        retry_attempts=3,
        retry_delay_seconds=0.0,   # no delay in tests
        vector_dimensions=32,
    )


@pytest.fixture(scope="module")
def phrase_df(spark):
    """Small DataFrame with 120 synthetic phrases."""
    records = generate_small_synthetic_embeddings(n_samples=120, n_dims=32, seed=99)
    schema = StructType(
        [
            StructField("id", StringType()),
            StructField("frase", StringType()),
            StructField("embedding", ArrayType(FloatType())),
        ]
    )
    data = [(r["id"], r["frase"]) for r in records]
    schema_minimal = StructType(
        [
            StructField("id", StringType()),
            StructField("frase", StringType()),
        ]
    )
    return spark.createDataFrame(data, schema=schema_minimal)


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------

class TestEmbeddingServiceHappyPath:
    def test_returns_correct_count(self, spark, embedding_config, phrase_df):
        """process_dataframe must return one row per input phrase."""
        client = MockEmbeddingClient(dims=32)
        svc = EmbeddingService(
            client=client,
            config=embedding_config,
            logger=PipelineLogger("test", "EmbeddingService"),
        )
        result = asyncio.run(
            svc.process_dataframe(df=phrase_df, text_col="frase", id_col="id", spark=spark)
        )
        assert result.count() == 120

    def test_output_has_embedding_column(self, spark, embedding_config, phrase_df):
        """Result must have an 'embedding' column."""
        client = MockEmbeddingClient(dims=32)
        svc = EmbeddingService(
            client=client,
            config=embedding_config,
            logger=PipelineLogger("test", "EmbeddingService"),
        )
        result = asyncio.run(
            svc.process_dataframe(df=phrase_df, text_col="frase", id_col="id", spark=spark)
        )
        assert "embedding" in result.columns

    def test_embedding_dimensions(self, spark, embedding_config, phrase_df):
        """Each embedding vector must have exactly vector_dimensions elements."""
        client = MockEmbeddingClient(dims=32)
        svc = EmbeddingService(
            client=client,
            config=embedding_config,
            logger=PipelineLogger("test", "EmbeddingService"),
        )
        result = asyncio.run(
            svc.process_dataframe(df=phrase_df, text_col="frase", id_col="id", spark=spark)
        )
        from pyspark.sql import functions as F

        sample = result.select(F.size("embedding").alias("dim")).distinct().collect()
        dims = {row["dim"] for row in sample}
        assert dims == {32}, f"Expected all embeddings to have dim=32, got {dims}"

    def test_no_null_embeddings(self, spark, embedding_config, phrase_df):
        """No row in the result should have a null embedding."""
        client = MockEmbeddingClient(dims=32)
        svc = EmbeddingService(
            client=client,
            config=embedding_config,
            logger=PipelineLogger("test", "EmbeddingService"),
        )
        result = asyncio.run(
            svc.process_dataframe(df=phrase_df, text_col="frase", id_col="id", spark=spark)
        )
        from pyspark.sql import functions as F

        nulls = result.filter(F.col("embedding").isNull()).count()
        assert nulls == 0, f"{nulls} rows have null embedding"

    def test_refresh_token_called_per_batch(self, spark, embedding_config, phrase_df):
        """
        refresh_token() must be called once per batch chunk (120 phrases / 50 per batch = 3 chunks).
        Verifies the Edi Bins token refresh contract.
        """
        client = MockEmbeddingClient(dims=32)
        svc = EmbeddingService(
            client=client,
            config=embedding_config,
            logger=PipelineLogger("test", "EmbeddingService"),
        )
        asyncio.run(
            svc.process_dataframe(df=phrase_df, text_col="frase", id_col="id", spark=spark)
        )
        # 120 phrases / batch_size=50 → 3 chunks → 3 token refreshes
        expected_chunks = -(-120 // embedding_config.batch_size)  # ceil division
        assert client.refresh_count == expected_chunks, (
            f"Expected {expected_chunks} refresh_token() calls, got {client.refresh_count}"
        )


# ---------------------------------------------------------------------------
# Tests — retry logic
# ---------------------------------------------------------------------------

class TestEmbeddingServiceRetry:
    def test_succeeds_after_transient_failures(self, spark, phrase_df):
        """
        Service must succeed when the first 2 calls fail (retry_attempts=3).
        """
        config = EmbeddingConfig(
            batch_size=50,
            max_concurrent=2,
            retry_attempts=3,
            retry_delay_seconds=0.0,
            vector_dimensions=32,
        )
        # fail_first_n=2 means first 2 batch calls raise; 3rd succeeds
        client = MockEmbeddingClient(dims=32, fail_first_n=2)
        svc = EmbeddingService(
            client=client,
            config=config,
            logger=PipelineLogger("test", "EmbeddingService"),
        )
        result = asyncio.run(
            svc.process_dataframe(df=phrase_df, text_col="frase", id_col="id", spark=spark)
        )
        # At least some results should come through (first batch retried, others fine)
        assert result.count() > 0

    def test_raises_after_all_retries_exhausted(self, spark, phrase_df):
        """
        Service must propagate the exception when all retries are exhausted.
        fail_first_n > retry_attempts forces permanent failure on the first batch.
        """
        config = EmbeddingConfig(
            batch_size=120,  # one batch = all records
            max_concurrent=1,
            retry_attempts=2,
            retry_delay_seconds=0.0,
            vector_dimensions=32,
        )
        client = MockEmbeddingClient(dims=32, fail_first_n=99)
        svc = EmbeddingService(
            client=client,
            config=config,
            logger=PipelineLogger("test", "EmbeddingService"),
        )
        with pytest.raises(RuntimeError):
            asyncio.run(
                svc.process_dataframe(df=phrase_df, text_col="frase", id_col="id", spark=spark)
            )
