"""
EmbeddingService — orchestrates async batch embedding generation for the Edi Bins API.

Edi Bins limits enforced via EmbeddingConfig:
    - max_concurrent = 20   (parallel requests)
    - batch_size     = 500  (items per request input array)

Token refresh strategy:
    - A shared asyncio.Lock serialises token refreshes so that concurrent
      requests don't trigger simultaneous renewals.
    - refresh_token() is called once per batch slot, right before embed_batch(),
      ensuring every request uses a fresh token regardless of concurrency.
"""

import asyncio
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType

from clustering_pipeline.config.settings import EmbeddingConfig
from clustering_pipeline.core.base_embedding_client import BaseEmbeddingClient
from clustering_pipeline.utils.logger import PipelineLogger


class EmbeddingService:
    """
    Orchestrates batch embedding generation with async concurrency control
    and per-call token refresh.

    Token refresh flow:
        Before every embed_batch() call the service acquires a shared
        asyncio.Lock and asks the client to refresh_token().  This guarantees:
        1. Only one refresh happens at a time (no race conditions).
        2. Every batch — both new and retry attempts — gets a fresh token.
        3. Parallel slots wait for the current refresh before proceeding.
    """

    def __init__(
        self,
        client: BaseEmbeddingClient,
        config: EmbeddingConfig,
        logger: PipelineLogger,
    ):
        self._client = client
        self._config = config
        self._logger = logger
        # Shared lock: serialises token refreshes across all concurrent tasks
        self._token_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_dataframe(
        self,
        df: DataFrame,
        text_col: str,
        id_col: str,
        spark: SparkSession,
    ) -> DataFrame:
        """
        Generate embeddings for all texts in the DataFrame.

        Flow:
            1. Collect only (id, text) columns from Spark to the driver.
            2. Split into chunks of config.batch_size (≤ 500 for Edi Bins).
            3. Process chunks asynchronously with Semaphore(config.max_concurrent).
            4. Before each chunk call, refresh the auth token via the shared Lock.
            5. Return a new DataFrame: id | text | embedding (array<float>).

        Args:
            df:       Source DataFrame containing id_col and text_col.
            text_col: Column name holding the raw text.
            id_col:   Column name holding the unique record identifier.
            spark:    Active SparkSession used to build the result DataFrame.

        Returns:
            DataFrame with columns [id_col, text_col, "embedding"].
        """
        # Initialise the token lock fresh for each process_dataframe call
        # so the service is re-entrant if called multiple times in one event loop.
        self._token_lock = asyncio.Lock()

        rows = df.select(id_col, text_col).collect()
        total = len(rows)
        batch_size = self._config.batch_size
        chunks = [rows[i : i + batch_size] for i in range(0, total, batch_size)]
        total_chunks = len(chunks)

        self._logger.info(
            "Iniciando geração de embeddings",
            total_frases=total,
            total_batches=total_chunks,
            batch_size=batch_size,
            max_concurrent=self._config.max_concurrent,
        )

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        tasks = [
            self._process_chunk_with_retry(
                chunk_id=idx,
                total_chunks=total_chunks,
                ids=[r[id_col] for r in chunk],
                texts=[r[text_col] for r in chunk],
                semaphore=semaphore,
            )
            for idx, chunk in enumerate(chunks)
        ]

        global_start = time.perf_counter()
        results_nested = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - global_start

        # Flatten results
        records: list[dict] = [item for batch in results_nested for item in batch]

        rate = total / total_time if total_time > 0 else 0
        self._logger.info(
            "Embeddings gerados",
            total_processado=len(records),
            tempo_total=f"{total_time:.2f}s",
            taxa=f"{rate:.1f} frases/s",
        )

        # Build result DataFrame
        schema = StructType(
            [
                StructField(id_col, StringType(), nullable=False),
                StructField(text_col, StringType(), nullable=False),
                StructField("embedding", ArrayType(FloatType()), nullable=False),
            ]
        )
        data = [(r["id"], r["text"], r["embedding"]) for r in records]
        return spark.createDataFrame(data, schema=schema).toDF(
            id_col, text_col, "embedding"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _refresh_token_safe(self, batch_label: str) -> None:
        """
        Refresh the auth token under the shared lock.

        Acquiring the lock means:
        - Only one concurrent task refreshes at a time.
        - Other tasks queue behind it and use the fresh token when released.
        - Between parallel calls the token is always up-to-date.
        """
        async with self._token_lock:
            self._logger.debug("Renovando token", batch=batch_label)
            await self._client.refresh_token()
            self._logger.debug("Token renovado", batch=batch_label)

    async def _process_chunk_with_retry(
        self,
        chunk_id: int,
        total_chunks: int,
        ids: list,
        texts: list,
        semaphore: asyncio.Semaphore,
    ) -> list[dict]:
        """
        Process a single chunk with exponential back-off retry.

        Token is refreshed on every attempt (including retries) before the
        actual embed_batch() call, so expired tokens are automatically healed
        by the retry loop without any extra logic.

        Args:
            chunk_id:     Zero-based index of this chunk.
            total_chunks: Total number of chunks for log context.
            ids:          List of record identifiers.
            texts:        List of text strings aligned with ids.
            semaphore:    Shared concurrency limiter (max_concurrent slots).

        Returns:
            List of dicts: {"id": ..., "text": ..., "embedding": [...]}.
        """
        batch_label = f"{chunk_id + 1}/{total_chunks}"
        delay = self._config.retry_delay_seconds

        async with semaphore:
            for attempt in range(1, self._config.retry_attempts + 1):
                # Refresh token before every call (and on every retry)
                await self._refresh_token_safe(batch_label)

                chunk_start = time.perf_counter()
                try:
                    embeddings = await self._client.embed_batch(texts)
                    elapsed = time.perf_counter() - chunk_start
                    self._logger.info(
                        "Batch processado",
                        batch=batch_label,
                        frases=len(texts),
                        tempo=f"{elapsed:.2f}s",
                    )
                    return [
                        {"id": id_, "text": text, "embedding": emb}
                        for id_, text, emb in zip(ids, texts, embeddings)
                    ]
                except Exception as exc:
                    self._logger.error(
                        "Falha no batch",
                        batch=batch_label,
                        tentativa=f"{attempt}/{self._config.retry_attempts}",
                        erro=type(exc).__name__,
                        detalhe=str(exc),
                    )
                    if attempt < self._config.retry_attempts:
                        await asyncio.sleep(delay)
                        delay *= 2  # exponential back-off
                    else:
                        raise
