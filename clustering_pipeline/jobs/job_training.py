"""
Job 1 — Training Job.

Trains the 3-level hierarchical KMeans models on historical embeddings,
optionally evaluates k candidates, names every cluster via LLM,
and persists all artifacts to S3 / Iceberg.

Executed on demand (not daily).
"""

import asyncio
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F

from clustering_pipeline.config.settings import PipelineSettings
from clustering_pipeline.core.base_job import BaseJob
from clustering_pipeline.services.clustering_service import ClusteringService, SparkKMeansStrategy
from clustering_pipeline.services.evaluation_service import EvaluationService
from clustering_pipeline.services.storage_service import IcebergStorageService
from clustering_pipeline.utils.logger import PipelineLogger
from clustering_pipeline.utils.validators import InputValidator, ValidationError


class JobTraining(BaseJob):
    """
    Trains KMeans models at 3 hierarchy levels on historical embedding data.

    Phases:
        1. Read historical embeddings
        2. (Optional) Evaluate k candidates — Elbow + Silhouette
        3. Train 3-level pipeline
        4. Persist models (S3) and cluster mapping (Iceberg)
        5. Name clusters via LLM
        6. Save cluster names and training metrics
    """

    def __init__(
        self,
        settings: PipelineSettings,
        naming_client=None,
    ):
        super().__init__(settings)
        self._naming_client = naming_client
        self.logger = PipelineLogger(job_name=self.job_name, service_name="JobTraining")

    @property
    def job_name(self) -> str:
        return "job_training"

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate_inputs(self) -> bool:
        """
        Validate pre-conditions:
        - Embeddings table exists and is non-empty.
        - k hierarchy is sane (k1 > k2 > k3).
        - S3 model base path is accessible.
        """
        validator = InputValidator(
            PipelineLogger(self.job_name, "InputValidator")
        )

        validator.assert_clustering_hierarchy(self.settings.clustering)

        storage = self.settings.storage
        validator.assert_table_exists(self.spark, storage.table_embeddings)

        df_check = self.spark.table(storage.table_embeddings)
        validator.assert_not_empty(df_check, "embeddings históricos")

        return True

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute all training phases."""
        cfg = self.settings
        run_id = str(uuid.uuid4())
        trained_at = datetime.now(tz=timezone.utc)

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

        # ── PHASE 1: Read embeddings ────────────────────────────────────
        self.logger.info("FASE 1 — Lendo embeddings históricos")
        df_embeddings = storage_svc.read_table(cfg.storage.table_embeddings)
        record_count = df_embeddings.count()
        self.logger.info(
            "Embeddings carregados",
            registros=record_count,
            colunas=df_embeddings.columns,
        )

        # ── PHASE 2: Evaluate k (optional) ────────────────────────────
        if cfg.clustering.run_evaluation:
            self.logger.info("FASE 2 — Avaliando k candidatos")
            eval_svc = EvaluationService(
                spark=self.spark,
                config=cfg.clustering,
                logger=PipelineLogger(self.job_name, "EvaluationService"),
            )
            eval_result = eval_svc.find_optimal_k(
                df=df_embeddings,
                features_col="embedding",
                k_candidates=cfg.clustering.evaluation_k_candidates,
            )
            self.logger.info(
                "Avaliação concluída",
                k_recomendado=eval_result["recommended_k"],
                motivo=eval_result["recommendation_reason"],
            )
            # Save evaluation metrics
            self._save_eval_metrics(
                storage_svc=storage_svc,
                eval_result=eval_result,
                run_id=run_id,
                trained_at=trained_at,
            )
        else:
            self.logger.info("FASE 2 — Avaliação de k desativada (run_evaluation=False)")

        # ── PHASE 3: Train 3-level pipeline ───────────────────────────
        self.logger.info("FASE 3 — Treinando pipeline hierárquico")
        pipeline_result = clustering_svc.train_pipeline(df_embeddings)

        # ── PHASE 4: Persist models + cluster mapping ──────────────────
        self.logger.info("FASE 4 — Persistindo modelos no S3")
        self._save_models(pipeline_result, cfg.storage)

        self.logger.info("FASE 4 — Salvando cluster_mapping no Iceberg")
        storage_svc.write_table(
            df=pipeline_result.cluster_mapping,
            table_name=cfg.storage.table_cluster_mapping,
            mode="overwrite",
        )

        # ── PHASE 5: Name clusters via LLM ────────────────────────────
        if self._naming_client is not None:
            self.logger.info("FASE 5 — Nomeando clusters via LLM")
            from clustering_pipeline.services.naming_service import NamingService

            naming_svc = NamingService(
                client=self._naming_client,
                config=cfg.naming,
                logger=PipelineLogger(self.job_name, "NamingService"),
            )

            # We need cluster_n1/n2/n3 in df_embeddings — apply transform first
            df_assigned = clustering_svc.transform_new_embeddings(
                df=df_embeddings,
                normalizer_model=pipeline_result.normalizer_model,
                pca_model=pipeline_result.pca_model,
                kmeans_n1_model=pipeline_result.kmeans_n1,
                cluster_mapping=pipeline_result.cluster_mapping,
                features_col="embedding",
            )

            text_col = "frase"
            names_df = asyncio.run(
                naming_svc.name_all_clusters(
                    df_clustered=df_assigned,
                    text_col=text_col,
                    cluster_mapping=pipeline_result.cluster_mapping,
                )
            )

            storage_svc.write_table(
                df=names_df,
                table_name=cfg.storage.table_cluster_names,
                mode="overwrite",
            )
        else:
            self.logger.warning(
                "FASE 5 — naming_client não fornecido; nomeação de clusters ignorada"
            )

        # ── PHASE 6: Final training metrics ───────────────────────────
        self.logger.info("FASE 6 — Salvando métricas finais do treino")
        self._save_training_metrics(
            storage_svc=storage_svc,
            metrics=pipeline_result.metrics,
            run_id=run_id,
            trained_at=trained_at,
            record_count=record_count,
        )

        self.logger.info(
            "RESUMO — Treinamento concluído",
            run_id=run_id,
            registros=record_count,
            k_n1=cfg.clustering.k_level_1,
            k_n2=cfg.clustering.k_level_2,
            k_n3=cfg.clustering.k_level_3,
            silhouette_n1=pipeline_result.metrics["n1"].get("silhouette"),
            silhouette_n2=pipeline_result.metrics["n2"].get("silhouette"),
            silhouette_n3=pipeline_result.metrics["n3"].get("silhouette"),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_models(self, pipeline_result, storage_cfg) -> None:
        """Persist all trained models to S3."""
        from pyspark.ml.feature import NormalizerModel, PCAModel

        pipeline_result.normalizer_model.write().overwrite().save(
            storage_cfg.model_normalizer_path
        )
        self.logger.info("Normalizer salvo", path=storage_cfg.model_normalizer_path)

        pipeline_result.pca_model.write().overwrite().save(
            storage_cfg.model_pca_path
        )
        self.logger.info("PCA salvo", path=storage_cfg.model_pca_path)

        pipeline_result.kmeans_n1.write().overwrite().save(
            storage_cfg.model_kmeans_n1_path
        )
        self.logger.info("KMeans N1 salvo", path=storage_cfg.model_kmeans_n1_path)

        pipeline_result.kmeans_n2.write().overwrite().save(
            storage_cfg.model_kmeans_n2_path
        )
        self.logger.info("KMeans N2 salvo", path=storage_cfg.model_kmeans_n2_path)

        pipeline_result.kmeans_n3.write().overwrite().save(
            storage_cfg.model_kmeans_n3_path
        )
        self.logger.info("KMeans N3 salvo", path=storage_cfg.model_kmeans_n3_path)

    def _save_eval_metrics(
        self,
        storage_svc: IcebergStorageService,
        eval_result: dict,
        run_id: str,
        trained_at,
    ) -> None:
        """Save evaluation (Elbow/Silhouette) results per k candidate."""
        from pyspark.sql.types import (
            FloatType,
            IntegerType,
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        schema = StructType(
            [
                StructField("run_id", StringType()),
                StructField("trained_at", TimestampType()),
                StructField("k_n1", IntegerType()),
                StructField("silhouette_n1", FloatType()),
                StructField("inertia_n1", FloatType()),
            ]
        )
        rows = [
            (
                run_id,
                trained_at,
                r["k"],
                float(r["silhouette"]),
                float(r["inertia"]),
            )
            for r in eval_result["results"]
        ]
        df = self.spark.createDataFrame(rows, schema=schema)
        storage_svc.write_table(df, self.settings.storage.table_cluster_metrics)

    def _save_training_metrics(
        self,
        storage_svc: IcebergStorageService,
        metrics: dict,
        run_id: str,
        trained_at,
        record_count: int,
    ) -> None:
        """Save final training metrics (one row per run)."""
        from pyspark.sql.types import (
            FloatType,
            IntegerType,
            LongType,
            StringType,
            StructField,
            StructType,
            TimestampType,
        )
        from clustering_pipeline.utils.spark_utils import SCHEMA_CLUSTER_METRICS

        cfg = self.settings.clustering

        row = (
            run_id,
            trained_at,
            cfg.k_level_1,
            cfg.k_level_2,
            cfg.k_level_3,
            _safe_float(metrics["n1"].get("silhouette")),
            _safe_float(metrics["n2"].get("silhouette")),
            _safe_float(metrics["n3"].get("silhouette")),
            _safe_float(metrics["n1"].get("inertia")),
            _safe_float(metrics["n2"].get("inertia")),
            _safe_float(metrics["n3"].get("inertia")),
            record_count,
            cfg.pca_output_dims,
        )
        df = self.spark.createDataFrame([row], schema=SCHEMA_CLUSTER_METRICS)
        storage_svc.write_table(df, self.settings.storage.table_cluster_metrics)


def _safe_float(value) -> float | None:
    return float(value) if value is not None else None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = PipelineSettings.from_env()
    job = JobTraining(settings)
    job.execute()
