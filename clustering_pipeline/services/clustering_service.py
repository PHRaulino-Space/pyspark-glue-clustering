"""
ClusteringService — orchestrates hierarchical 3-level KMeans clustering.

Pipeline:
    1. L2-normalise embedding vectors
    2. Reduce dimensions via PCA (vector_dims → pca_output_dims)
    3. KMeans N1 on normalised/reduced vectors
    4. KMeans N2 on N1 centroids
    5. KMeans N3 on N2 centroids
    6. Build N1→N2→N3 cluster mapping
"""

import time
from dataclasses import dataclass
from typing import Any

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Normalizer, PCA
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, IntegerType

from clustering_pipeline.config.settings import ClusteringConfig
from clustering_pipeline.core.base_clustering import BaseClusteringStrategy
from clustering_pipeline.utils.logger import PipelineLogger
from clustering_pipeline.utils.spark_utils import array_to_vector_column


@dataclass
class ClusteringPipelineResult:
    """Contains all trained models and artifacts produced during training."""

    normalizer_model: Any
    pca_model: Any
    kmeans_n1: Any
    kmeans_n2: Any
    kmeans_n3: Any
    cluster_mapping: DataFrame   # columns: cluster_n1, cluster_n2, cluster_n3
    metrics: dict                # inertia and silhouette per level


class SparkKMeansStrategy(BaseClusteringStrategy):
    """
    Concrete clustering strategy wrapping Spark ML KMeans.
    Used as the default strategy in ClusteringService.
    """

    def __init__(self, config: ClusteringConfig):
        self._config = config

    def fit(self, df: DataFrame, features_col: str, k: int):
        kmeans = KMeans(
            featuresCol=features_col,
            predictionCol="prediction",
            k=k,
            initMode=self._config.init_mode,
            initSteps=self._config.init_steps,
            maxIter=self._config.max_iter,
            seed=self._config.seed,
            tol=self._config.tolerance,
        )
        return kmeans.fit(df)

    def transform(self, model, df: DataFrame, features_col: str) -> DataFrame:
        return model.transform(df)

    def save_model(self, model, path: str) -> None:
        model.write().overwrite().save(path)

    def load_model(self, path: str):
        from pyspark.ml.clustering import KMeansModel
        return KMeansModel.load(path)


class ClusteringService:
    """
    Orchestrates three-level hierarchical KMeans clustering.

    Level 1 (N1): fine-grained segments   (k = k_level_1, e.g. 1000)
    Level 2 (N2): mid-level groups        (k = k_level_2, e.g.  200)
    Level 3 (N3): macro categories        (k = k_level_3, e.g.   50)
    """

    def __init__(
        self,
        spark: SparkSession,
        strategy: BaseClusteringStrategy,
        config: ClusteringConfig,
        logger: PipelineLogger,
    ):
        self._spark = spark
        self._strategy = strategy
        self._config = config
        self._logger = logger
        self._evaluator = ClusteringEvaluator(
            featuresCol="features_pca",
            predictionCol="prediction",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_pipeline(self, df: DataFrame) -> ClusteringPipelineResult:
        """
        Train the full 3-level hierarchical KMeans pipeline.

        Args:
            df: DataFrame with an "embedding" column (array<float>).

        Returns:
            ClusteringPipelineResult with all models, mapping, and metrics.
        """
        total_start = time.perf_counter()

        # ── Step 1: convert array → vector ─────────────────────────────
        df = array_to_vector_column(df, "embedding", "features_raw")

        # ── Step 2: L2 normalisation ────────────────────────────────────
        self._logger.info("Normalizando vetores L2")
        normalizer = Normalizer(inputCol="features_raw", outputCol="features_norm", p=2.0)
        normalizer_model = normalizer.fit(df)
        df_norm = normalizer_model.transform(df)

        # ── Step 3: PCA dimensionality reduction ────────────────────────
        pca_dims = self._config.pca_output_dims
        self._logger.info("Reduzindo dimensões via PCA", output_dims=pca_dims)
        pca = PCA(k=pca_dims, inputCol="features_norm", outputCol="features_pca")
        pca_model = pca.fit(df_norm)
        df_pca = pca_model.transform(df_norm).cache()

        record_count = df_pca.count()

        # ── Step 4: KMeans N1 ───────────────────────────────────────────
        kmeans_n1, metrics_n1, df_n1 = self._fit_level(
            df=df_pca,
            features_col="features_pca",
            k=self._config.k_level_1,
            level=1,
            record_count=record_count,
        )

        # ── Step 5: centroids N1 → KMeans N2 ───────────────────────────
        centroids_n1_df = self._centroids_as_df(kmeans_n1, level=1)
        kmeans_n2, metrics_n2, _ = self._fit_level(
            df=centroids_n1_df,
            features_col="centroid",
            k=self._config.k_level_2,
            level=2,
            record_count=self._config.k_level_1,
        )

        # ── Step 6: centroids N2 → KMeans N3 ───────────────────────────
        centroids_n2_df = self._centroids_as_df(kmeans_n2, level=2)
        kmeans_n3, metrics_n3, _ = self._fit_level(
            df=centroids_n2_df,
            features_col="centroid",
            k=self._config.k_level_3,
            level=3,
            record_count=self._config.k_level_2,
        )

        df_pca.unpersist()

        # ── Step 7: build N1→N2→N3 mapping ─────────────────────────────
        cluster_mapping = self.build_cluster_mapping(kmeans_n1, kmeans_n2, kmeans_n3)

        total_elapsed = time.perf_counter() - total_start
        self._logger.info(
            "Pipeline de clustering concluído",
            tempo_total=f"{total_elapsed:.2f}s",
            k_n1=self._config.k_level_1,
            k_n2=self._config.k_level_2,
            k_n3=self._config.k_level_3,
        )

        metrics = {
            "n1": metrics_n1,
            "n2": metrics_n2,
            "n3": metrics_n3,
        }

        return ClusteringPipelineResult(
            normalizer_model=normalizer_model,
            pca_model=pca_model,
            kmeans_n1=kmeans_n1,
            kmeans_n2=kmeans_n2,
            kmeans_n3=kmeans_n3,
            cluster_mapping=cluster_mapping,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Cluster mapping
    # ------------------------------------------------------------------

    def build_cluster_mapping(self, model_n1, model_n2, model_n3) -> DataFrame:
        """
        Build a DataFrame mapping each N1 cluster to its N2 and N3 parents.

        Uses N1 centroids → transform N2 → get N2 assignment per centroid,
        then N2 centroids → transform N3 → get N3 assignment per centroid.

        Returns:
            DataFrame with columns: cluster_n1 (int), cluster_n2 (int), cluster_n3 (int).
        """
        self._logger.info("Construindo mapeamento N1→N2→N3")

        centroids_n1_df = self._centroids_as_df(model_n1, level=1)
        # Assign N2 label to each N1 centroid
        mapped_n1_n2 = (
            model_n2.transform(centroids_n1_df)
            .select(
                F.col("cluster_id").alias("cluster_n1"),
                F.col("prediction").alias("cluster_n2"),
            )
        )

        # Assign N3 label to each N2 centroid
        centroids_n2_df = self._centroids_as_df(model_n2, level=2)
        mapped_n2_n3 = (
            model_n3.transform(centroids_n2_df)
            .select(
                F.col("cluster_id").alias("cluster_n2"),
                F.col("prediction").alias("cluster_n3"),
            )
        )

        mapping = mapped_n1_n2.join(mapped_n2_n3, on="cluster_n2", how="inner")
        self._logger.info(
            "Mapeamento construído",
            registros=mapping.count(),
        )
        return mapping

    # ------------------------------------------------------------------
    # Assignment (Job 2)
    # ------------------------------------------------------------------

    def transform_new_embeddings(
        self,
        df: DataFrame,
        normalizer_model,
        pca_model,
        kmeans_n1_model,
        cluster_mapping: DataFrame,
        features_col: str = "embedding",
    ) -> DataFrame:
        """
        Assign cluster labels to new embeddings using pre-trained models.

        Flow:
            1. array → vector
            2. L2 normalise
            3. PCA transform
            4. KMeans N1 predict → cluster_n1
            5. Join cluster_mapping → cluster_n2, cluster_n3

        Args:
            df:               DataFrame with features_col as array<float>.
            normalizer_model: Fitted Normalizer.
            pca_model:        Fitted PCA model.
            kmeans_n1_model:  Fitted KMeans N1 model.
            cluster_mapping:  DataFrame(cluster_n1, cluster_n2, cluster_n3).
            features_col:     Column name of raw embeddings.

        Returns:
            Input DataFrame enriched with cluster_n1, cluster_n2, cluster_n3.
        """
        record_count = df.count()
        self._logger.info("Iniciando transform de novos embeddings", registros=record_count)

        t0 = time.perf_counter()
        df = array_to_vector_column(df, features_col, "features_raw")
        df_norm = normalizer_model.transform(df)
        df_pca = pca_model.transform(df_norm)

        df_n1 = kmeans_n1_model.transform(df_pca).withColumnRenamed(
            "prediction", "cluster_n1"
        )
        elapsed_n1 = time.perf_counter() - t0
        self._logger.info("Transform N1 concluído", tempo=f"{elapsed_n1:.2f}s")

        t1 = time.perf_counter()
        result = df_n1.join(
            cluster_mapping.select("cluster_n1", "cluster_n2", "cluster_n3"),
            on="cluster_n1",
            how="left",
        )
        elapsed_join = time.perf_counter() - t1
        self._logger.info("Join mapeamento concluído", tempo=f"{elapsed_join:.2f}s")

        # Drop intermediate vector columns
        result = result.drop("features_raw", "features_norm", "features_pca")
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_level(
        self,
        df: DataFrame,
        features_col: str,
        k: int,
        level: int,
        record_count: int,
    ):
        """Train KMeans for one hierarchy level and compute metrics."""
        self._logger.info(
            f"KMeans N{level} iniciado",
            k=k,
            features=features_col,
            registros=record_count,
        )

        start = time.perf_counter()
        model = self._strategy.fit(df, features_col, k)
        inertia = float(model.summary.trainingCost)
        elapsed = time.perf_counter() - start

        # Silhouette only makes sense with ≥ 2 clusters and ≥ 2 records
        silhouette = None
        if k >= 2 and record_count >= 2:
            try:
                evaluator = ClusteringEvaluator(
                    featuresCol=features_col,
                    predictionCol="prediction",
                    metricName="silhouette",
                    distanceMeasure="squaredEuclidean",
                )
                predictions = model.transform(df)
                silhouette = float(evaluator.evaluate(predictions))
                if silhouette < self._config.silhouette_threshold:
                    self._logger.warning(
                        f"Silhouette N{level} abaixo do limiar",
                        silhouette=f"{silhouette:.4f}",
                        limiar=self._config.silhouette_threshold,
                    )
            except Exception:
                self._logger.warning(f"Não foi possível calcular silhouette N{level}")

        self._logger.info(
            f"KMeans N{level} concluído",
            inércia=f"{inertia:.2f}",
            silhouette=f"{silhouette:.4f}" if silhouette is not None else "N/A",
            tempo=f"{elapsed:.2f}s",
        )

        predictions_df = model.transform(df)
        metrics = {"inertia": inertia, "silhouette": silhouette}
        return model, metrics, predictions_df

    def _centroids_as_df(self, model, level: int) -> DataFrame:
        """
        Convert KMeans cluster centres to a DataFrame for the next level.

        Returns DataFrame with columns: cluster_id (int), centroid (VectorUDT).
        """
        centres = model.clusterCenters()
        data = [(i, Vectors.dense(c)) for i, c in enumerate(centres)]
        schema = "cluster_id INT, centroid VECTOR"
        from pyspark.sql.types import StructType, StructField, IntegerType

        schema_struct = StructType(
            [
                StructField("cluster_id", IntegerType(), nullable=False),
                StructField("centroid", VectorUDT(), nullable=False),
            ]
        )
        return self._spark.createDataFrame(data, schema=schema_struct)
