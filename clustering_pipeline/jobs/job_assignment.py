"""
Job 2 — Daily Assignment Job.

Processes new phrases arriving daily, generates embeddings for those not
yet embedded, assigns them to existing clusters, and writes the final
enriched output to Iceberg.

Designed to be idempotent: running twice on the same day produces no
duplicate records thanks to the anti-join in Phase 2.
"""

import asyncio
from datetime import datetime, timezone

from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import NormalizerModel, PCAModel
from pyspark.sql import functions as F

from clustering_pipeline.config.settings import PipelineSettings
from clustering_pipeline.core.base_job import BaseJob
from clustering_pipeline.services.clustering_service import ClusteringService, SparkKMeansStrategy
from clustering_pipeline.services.embedding_service import EmbeddingService
from clustering_pipeline.services.storage_service import IcebergStorageService
from clustering_pipeline.utils.logger import PipelineLogger
from clustering_pipeline.utils.spark_utils import array_to_vector_column
from clustering_pipeline.utils.validators import InputValidator, ValidationError


class JobAssignment(BaseJob):
    """
    Daily job that embeds new phrases and assigns them to pre-trained clusters.

    Phases:
        1. Read existing IDs, today's raw phrases, cluster mapping, cluster names, models
        2. Anti-join → identify new phrases
        3. Generate embeddings for new phrases (async)
        4. Build full DataFrame (new + existing day embeddings)
        5. Assign clusters via pre-trained KMeans N1 + mapping join
        6. Enrich with cluster names and save to clusters_output
    """

    def __init__(
        self,
        settings: PipelineSettings,
        embedding_client=None,
    ):
        super().__init__(settings)
        self._embedding_client = embedding_client
        self.logger = PipelineLogger(job_name=self.job_name, service_name="JobAssignment")

    @property
    def job_name(self) -> str:
        return "job_assignment"

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate_inputs(self) -> bool:
        """
        Validate pre-conditions before processing any data:
        - All required models exist on S3.
        - Cluster mapping and names tables exist.
        - Today's raw phrases table has records.
        """
        validator = InputValidator(
            PipelineLogger(self.job_name, "InputValidator")
        )
        cfg = self.settings.storage

        # S3 models
        for path in [
            cfg.model_normalizer_path,
            cfg.model_pca_path,
            cfg.model_kmeans_n1_path,
        ]:
            validator.assert_s3_path_accessible(self.spark, path)

        # Required Iceberg tables
        for table in [cfg.table_cluster_mapping, cfg.table_frases_raw]:
            validator.assert_table_exists(self.spark, table)

        # Today's raw phrases must not be empty
        df_raw = self.spark.table(cfg.table_frases_raw)
        validator.assert_not_empty(df_raw, "frases_raw do dia")

        return True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all assignment phases."""
        cfg = self.settings
        processed_at = datetime.now(tz=timezone.utc)

        storage_svc = IcebergStorageService(
            spark=self.spark,
            config=cfg.storage,
            logger=PipelineLogger(self.job_name, "IcebergStorageService"),
        )
        clustering_svc = ClusteringService(
            spark=self.spark,
            strategy=SparkKMeansStrategy(cfg.clustering),
            config=cfg.clustering,
            logger=PipelineLogger(self.job_name, "ClusteringService"),
        )

        # ── PHASE 1: Parallel reads + model loading ─────────────────────
        self.logger.info("FASE 1 — Leitura paralela de tabelas e modelos")

        df_existing_ids = storage_svc.get_existing_ids(
            cfg.storage.table_embeddings, "id"
        )
        df_frases_raw = storage_svc.read_table(cfg.storage.table_frases_raw)
        df_cluster_mapping = storage_svc.read_table(cfg.storage.table_cluster_mapping)

        cluster_names_exists = storage_svc.table_exists(cfg.storage.table_cluster_names)
        df_cluster_names = (
            storage_svc.read_table(cfg.storage.table_cluster_names)
            if cluster_names_exists
            else None
        )

        self.logger.info(
            "Carregando modelos do S3",
            normalizer=cfg.storage.model_normalizer_path,
            pca=cfg.storage.model_pca_path,
            kmeans_n1=cfg.storage.model_kmeans_n1_path,
        )
        normalizer_model = NormalizerModel.load(cfg.storage.model_normalizer_path)
        pca_model = PCAModel.load(cfg.storage.model_pca_path)
        kmeans_n1_model = KMeansModel.load(cfg.storage.model_kmeans_n1_path)

        total_raw = df_frases_raw.count()
        existing_count = df_existing_ids.count()
        self.logger.info(
            "Fase 1 concluída",
            frases_raw=total_raw,
            embeddings_existentes=existing_count,
        )

        # ── PHASE 2: Filter new phrases ────────────────────────────────
        self.logger.info("FASE 2 — Filtrando frases novas (anti-join)")
        df_novas = df_frases_raw.join(
            df_existing_ids, on="id", how="left_anti"
        )
        novas_count = df_novas.count()
        ja_processadas = total_raw - novas_count

        self.logger.info(
            "Filtragem concluída",
            frases_novas=novas_count,
            ja_tinham_embedding=ja_processadas,
        )

        # ── PHASE 3: Generate embeddings for new phrases ───────────────
        df_novos_embeddings = None
        if novas_count == 0:
            self.logger.warning(
                "FASE 3 — Nenhuma frase nova para processar; pulando geração de embeddings"
            )
        else:
            self.logger.info(
                "FASE 3 — Gerando embeddings para frases novas",
                frases=novas_count,
            )
            if self._embedding_client is None:
                raise ValidationError(
                    "embedding_client é obrigatório quando há frases novas para processar."
                )

            emb_svc = EmbeddingService(
                client=self._embedding_client,
                config=cfg.embedding,
                logger=PipelineLogger(self.job_name, "EmbeddingService"),
            )
            df_novos_embeddings = asyncio.run(
                emb_svc.process_dataframe(
                    df=df_novas,
                    text_col="frase",
                    id_col="id",
                    spark=self.spark,
                )
            )

            # Add created_at and persist
            df_novos_embeddings = df_novos_embeddings.withColumn(
                "created_at", F.lit(processed_at).cast("timestamp")
            )
            storage_svc.write_table(
                df=df_novos_embeddings,
                table_name=cfg.storage.table_embeddings,
                mode="append",
            )
            self.logger.info(
                "FASE 3 concluída",
                embeddings_gerados=novas_count,
            )

        # ── PHASE 4: Full day DataFrame ────────────────────────────────
        self.logger.info("FASE 4 — Montando DataFrame completo do dia")
        df_existing_day = storage_svc.read_table_with_filter(
            cfg.storage.table_embeddings,
            f"date(created_at) = date('{processed_at.date()}')",
        )

        if df_novos_embeddings is not None:
            df_all = df_existing_day.union(
                df_novos_embeddings.select(df_existing_day.columns)
            )
        else:
            df_all = df_existing_day

        total_all = df_all.count()
        self.logger.info("DataFrame completo montado", registros=total_all)

        # ── PHASE 5: Assign clusters ───────────────────────────────────
        self.logger.info("FASE 5 — Atribuindo clusters")
        df_clustered = clustering_svc.transform_new_embeddings(
            df=df_all,
            normalizer_model=normalizer_model,
            pca_model=pca_model,
            kmeans_n1_model=kmeans_n1_model,
            cluster_mapping=df_cluster_mapping,
            features_col="embedding",
        )
        self.logger.info("Atribuição de clusters concluída")

        # ── PHASE 5b: Enrich with cluster names (optional) ─────────────
        if df_cluster_names is not None:
            self.logger.info("Enriquecendo com nomes de clusters")
            # Pivot names to wide format per level
            n1_names = df_cluster_names.filter(F.col("nivel") == 1).select(
                F.col("cluster_id").alias("cluster_n1"),
                F.col("nome").alias("nome_n1"),
                F.col("dor").alias("dor_n1"),
            )
            n2_names = df_cluster_names.filter(F.col("nivel") == 2).select(
                F.col("cluster_id").alias("cluster_n2"),
                F.col("nome").alias("nome_n2"),
                F.col("dor").alias("dor_n2"),
            )
            n3_names = df_cluster_names.filter(F.col("nivel") == 3).select(
                F.col("cluster_id").alias("cluster_n3"),
                F.col("nome").alias("nome_n3"),
                F.col("dor").alias("dor_n3"),
            )
            df_clustered = (
                df_clustered
                .join(n1_names, on="cluster_n1", how="left")
                .join(n2_names, on="cluster_n2", how="left")
                .join(n3_names, on="cluster_n3", how="left")
            )
            self.logger.info("Nomes de clusters adicionados")
        else:
            self.logger.warning(
                "Tabela cluster_names não encontrada — output sem nomes de clusters"
            )

        # ── PHASE 6: Final output ──────────────────────────────────────
        self.logger.info("FASE 6 — Salvando output final")
        df_output = df_clustered.withColumn(
            "processed_at", F.lit(processed_at).cast("timestamp")
        )

        storage_svc.write_table(
            df=df_output,
            table_name=cfg.storage.table_clusters_output,
            mode="append",
        )

        self.logger.info(
            "RESUMO — Job de atribuição concluído",
            registros_salvos=total_all,
            frases_novas=novas_count,
            colunas=df_output.columns,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = PipelineSettings.from_env()
    job = JobAssignment(settings)
    job.execute()
