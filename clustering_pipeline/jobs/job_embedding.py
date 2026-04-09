"""
Job Embedding — Dedicated embedding generation job.

Reads raw phrases from the frases_raw table, identifies which ones do not yet
have an embedding (anti-join against the embeddings table), generates embeddings
via the Edi Bins API (async, up to 20 parallel requests, 500 items per batch),
and persists the results to the embeddings Iceberg table.

Designed to run BEFORE job_training or job_assignment.
Idempotent: re-running does not duplicate records already in the embeddings table.
"""

import asyncio
from datetime import datetime, timezone

from pyspark.sql import functions as F

from clustering_pipeline.config.settings import PipelineSettings
from clustering_pipeline.core.base_job import BaseJob
from clustering_pipeline.services.embedding_service import EmbeddingService
from clustering_pipeline.services.storage_service import IcebergStorageService
from clustering_pipeline.utils.logger import PipelineLogger
from clustering_pipeline.utils.validators import InputValidator, ValidationError


class JobEmbedding(BaseJob):
    """
    Embedding-only job: reads raw phrases, generates embeddings, saves to Iceberg.

    Responsible for NOTHING else — no clustering, no model loading, no assignment.

    Phases:
        1. Read existing embedding IDs (anti-join optimisation)
        2. Read frases_raw
        3. Anti-join → new phrases only
        4. Generate embeddings via Edi Bins (async, token refresh per batch)
        5. Persist new embeddings to the embeddings table
    """

    def __init__(
        self,
        settings: PipelineSettings,
        embedding_client,
    ):
        """
        Args:
            settings:         Pipeline-wide configuration.
            embedding_client: Concrete BaseEmbeddingClient implementation
                              (must implement embed_batch + refresh_token).
        """
        super().__init__(settings)
        self._embedding_client = embedding_client
        self.logger = PipelineLogger(job_name=self.job_name, service_name="JobEmbedding")

    @property
    def job_name(self) -> str:
        return "job_embedding"

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate_inputs(self) -> bool:
        """
        Validate pre-conditions:
        - frases_raw table exists and has at least one record.
        - embedding_client was provided.
        """
        if self._embedding_client is None:
            raise ValidationError(
                "embedding_client é obrigatório para o JobEmbedding."
            )

        validator = InputValidator(
            PipelineLogger(self.job_name, "InputValidator")
        )

        validator.assert_table_exists(self.spark, self.settings.storage.table_frases_raw)

        df_raw = self.spark.table(self.settings.storage.table_frases_raw)
        validator.assert_not_empty(df_raw, "frases_raw")

        return True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all embedding phases."""
        cfg = self.settings
        created_at = datetime.now(tz=timezone.utc)

        storage_svc = IcebergStorageService(
            spark=self.spark,
            config=cfg.storage,
            logger=PipelineLogger(self.job_name, "IcebergStorageService"),
        )
        emb_svc = EmbeddingService(
            client=self._embedding_client,
            config=cfg.embedding,
            logger=PipelineLogger(self.job_name, "EmbeddingService"),
        )

        # ── PHASE 1: Load existing embedding IDs ───────────────────────
        self.logger.info("FASE 1 — Carregando IDs de embeddings existentes")

        embeddings_table_exists = storage_svc.table_exists(cfg.storage.table_embeddings)
        if embeddings_table_exists:
            df_existing_ids = storage_svc.get_existing_ids(
                cfg.storage.table_embeddings, "id"
            )
        else:
            self.logger.warning(
                "Tabela de embeddings ainda não existe — todos os registros serão processados"
            )
            from pyspark.sql.types import StringType, StructField, StructType
            df_existing_ids = self.spark.createDataFrame(
                [], schema=StructType([StructField("id", StringType())])
            )

        existing_count = df_existing_ids.count()

        # ── PHASE 2: Read frases_raw ───────────────────────────────────
        self.logger.info("FASE 2 — Lendo frases_raw")
        df_frases_raw = storage_svc.read_table(cfg.storage.table_frases_raw)
        total_raw = df_frases_raw.count()

        self.logger.info(
            "Fase 1-2 concluída",
            frases_raw=total_raw,
            embeddings_existentes=existing_count,
        )

        # ── PHASE 3: Anti-join → new phrases only ─────────────────────
        self.logger.info("FASE 3 — Filtrando frases sem embedding (anti-join)")
        df_novas = df_frases_raw.join(df_existing_ids, on="id", how="left_anti")
        novas_count = df_novas.count()
        ja_processadas = total_raw - novas_count

        self.logger.info(
            "Filtragem concluída",
            frases_novas=novas_count,
            ja_tinham_embedding=ja_processadas,
        )

        if novas_count == 0:
            self.logger.info(
                "Nenhuma frase nova — todas já possuem embedding. Job encerrado."
            )
            return

        # ── PHASE 4: Generate embeddings ──────────────────────────────
        self.logger.info(
            "FASE 4 — Gerando embeddings via Edi Bins",
            frases=novas_count,
            batch_size=cfg.embedding.batch_size,
            max_concurrent=cfg.embedding.max_concurrent,
        )
        df_embeddings = asyncio.run(
            emb_svc.process_dataframe(
                df=df_novas,
                text_col="frase",
                id_col="id",
                spark=self.spark,
            )
        )

        # ── PHASE 5: Persist new embeddings ───────────────────────────
        self.logger.info("FASE 5 — Persistindo embeddings no Iceberg")
        df_to_save = df_embeddings.withColumn(
            "created_at", F.lit(created_at).cast("timestamp")
        )

        storage_svc.write_table(
            df=df_to_save,
            table_name=cfg.storage.table_embeddings,
            mode="append",
        )

        self.logger.info(
            "RESUMO — JobEmbedding concluído",
            embeddings_gerados=novas_count,
            ja_existiam=ja_processadas,
            total_acumulado=existing_count + novas_count,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # embedding_client must be injected by the Glue job script
    # (concrete implementation with Edi Bins credentials)
    raise RuntimeError(
        "JobEmbedding requer um embedding_client concreto. "
        "Instancie o job a partir do script Glue passando o cliente configurado."
    )
