"""
Unit tests for ClusteringService using local Spark and synthetic data.

Runs without any AWS / Glue / S3 / API dependencies.
SparkSession is created with master("local[*]").
"""

import pytest

from clustering_pipeline.config.settings import ClusteringConfig
from clustering_pipeline.services.clustering_service import (
    ClusteringService,
    SparkKMeansStrategy,
)
from clustering_pipeline.tests.fixtures.sample_embeddings import (
    generate_small_synthetic_embeddings,
)
from clustering_pipeline.utils.logger import PipelineLogger
from clustering_pipeline.utils.spark_utils import (
    array_to_vector_column,
    create_local_spark_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spark():
    """Shared local SparkSession for the entire test module."""
    session = create_local_spark_session("test-clustering-service")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def clustering_config():
    """Small k values so tests run fast on local Spark."""
    return ClusteringConfig(
        k_level_1=5,
        k_level_2=3,
        k_level_3=2,
        init_mode="k-means||",
        init_steps=2,
        max_iter=10,
        seed=42,
        tolerance=1e-4,
        pca_output_dims=8,
        evaluation_k_candidates=[2, 3, 5],
        evaluation_sample_fraction=0.5,
        silhouette_threshold=0.0,  # no warning noise in tests
        run_evaluation=False,
    )


@pytest.fixture(scope="module")
def embedding_df(spark):
    """
    DataFrame with synthetic 32-dimensional embeddings and 5 topics.
    Shape: 200 rows × {id, frase, embedding (array<float>)}.
    """
    from pyspark.sql.types import (
        ArrayType,
        FloatType,
        StringType,
        StructField,
        StructType,
    )

    records = generate_small_synthetic_embeddings(
        n_samples=200, n_dims=32, n_topics=5, seed=42
    )
    schema = StructType(
        [
            StructField("id", StringType()),
            StructField("frase", StringType()),
            StructField("embedding", ArrayType(FloatType())),
        ]
    )
    data = [(r["id"], r["frase"], r["embedding"]) for r in records]
    return spark.createDataFrame(data, schema=schema)


@pytest.fixture(scope="module")
def pipeline_result(spark, clustering_config, embedding_df):
    """Run train_pipeline once and reuse across tests."""
    logger = PipelineLogger("test", "ClusteringService")
    strategy = SparkKMeansStrategy(clustering_config)
    svc = ClusteringService(
        spark=spark,
        strategy=strategy,
        config=clustering_config,
        logger=logger,
    )
    return svc.train_pipeline(embedding_df)


# ---------------------------------------------------------------------------
# Tests — train_pipeline
# ---------------------------------------------------------------------------

class TestTrainPipeline:
    def test_models_not_none(self, pipeline_result):
        """All trained model objects must be non-None."""
        assert pipeline_result.normalizer_model is not None
        assert pipeline_result.pca_model is not None
        assert pipeline_result.kmeans_n1 is not None
        assert pipeline_result.kmeans_n2 is not None
        assert pipeline_result.kmeans_n3 is not None

    def test_cluster_mapping_has_records(self, pipeline_result, clustering_config):
        """
        cluster_mapping must have one row per N1 centroid.
        In our case: k_level_1 = 5, so we expect 5 rows.
        """
        mapping = pipeline_result.cluster_mapping
        count = mapping.count()
        assert count == clustering_config.k_level_1, (
            f"Expected {clustering_config.k_level_1} rows in cluster_mapping, got {count}"
        )

    def test_cluster_mapping_columns(self, pipeline_result):
        """cluster_mapping must contain cluster_n1, cluster_n2, cluster_n3."""
        cols = pipeline_result.cluster_mapping.columns
        assert "cluster_n1" in cols
        assert "cluster_n2" in cols
        assert "cluster_n3" in cols

    def test_metrics_structure(self, pipeline_result):
        """Metrics dict must have keys n1, n2, n3 each with inertia and silhouette."""
        metrics = pipeline_result.metrics
        for level in ["n1", "n2", "n3"]:
            assert level in metrics, f"Missing key '{level}' in metrics"
            assert "inertia" in metrics[level]
            assert "silhouette" in metrics[level]

    def test_inertia_positive(self, pipeline_result):
        """Training cost (inertia) must be a positive finite number."""
        for level in ["n1", "n2", "n3"]:
            inertia = pipeline_result.metrics[level]["inertia"]
            assert inertia is not None
            assert inertia > 0, f"Inertia at level {level} should be positive, got {inertia}"


# ---------------------------------------------------------------------------
# Tests — transform_new_embeddings
# ---------------------------------------------------------------------------

class TestTransformNewEmbeddings:
    def test_all_rows_assigned(self, spark, clustering_config, embedding_df, pipeline_result):
        """Every row must receive a cluster_n1 assignment (no nulls)."""
        logger = PipelineLogger("test", "ClusteringService")
        svc = ClusteringService(
            spark=spark,
            strategy=SparkKMeansStrategy(clustering_config),
            config=clustering_config,
            logger=logger,
        )
        result = svc.transform_new_embeddings(
            df=embedding_df,
            normalizer_model=pipeline_result.normalizer_model,
            pca_model=pipeline_result.pca_model,
            kmeans_n1_model=pipeline_result.kmeans_n1,
            cluster_mapping=pipeline_result.cluster_mapping,
            features_col="embedding",
        )

        total = result.count()
        assert total == 200, f"Expected 200 rows, got {total}"

        from pyspark.sql import functions as F

        null_n1 = result.filter(F.col("cluster_n1").isNull()).count()
        assert null_n1 == 0, f"{null_n1} rows have null cluster_n1"

    def test_output_cluster_columns(self, spark, clustering_config, embedding_df, pipeline_result):
        """Result must contain cluster_n1, cluster_n2, cluster_n3 columns."""
        logger = PipelineLogger("test", "ClusteringService")
        svc = ClusteringService(
            spark=spark,
            strategy=SparkKMeansStrategy(clustering_config),
            config=clustering_config,
            logger=logger,
        )
        result = svc.transform_new_embeddings(
            df=embedding_df,
            normalizer_model=pipeline_result.normalizer_model,
            pca_model=pipeline_result.pca_model,
            kmeans_n1_model=pipeline_result.kmeans_n1,
            cluster_mapping=pipeline_result.cluster_mapping,
            features_col="embedding",
        )
        for col in ["cluster_n1", "cluster_n2", "cluster_n3"]:
            assert col in result.columns, f"Column '{col}' missing from result"

    def test_cluster_ids_within_range(
        self, spark, clustering_config, embedding_df, pipeline_result
    ):
        """Cluster IDs must be within [0, k-1] for each level."""
        logger = PipelineLogger("test", "ClusteringService")
        svc = ClusteringService(
            spark=spark,
            strategy=SparkKMeansStrategy(clustering_config),
            config=clustering_config,
            logger=logger,
        )
        result = svc.transform_new_embeddings(
            df=embedding_df,
            normalizer_model=pipeline_result.normalizer_model,
            pca_model=pipeline_result.pca_model,
            kmeans_n1_model=pipeline_result.kmeans_n1,
            cluster_mapping=pipeline_result.cluster_mapping,
            features_col="embedding",
        )
        from pyspark.sql import functions as F

        max_n1 = result.agg(F.max("cluster_n1")).collect()[0][0]
        min_n1 = result.agg(F.min("cluster_n1")).collect()[0][0]

        assert min_n1 >= 0
        assert max_n1 < clustering_config.k_level_1
