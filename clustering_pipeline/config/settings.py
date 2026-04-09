"""
Central configuration for the Semantic Hierarchical Clustering Pipeline.
All parameters are loaded from environment variables with documented defaults.
No hardcoded values should exist outside this file.
"""

import os
from dataclasses import dataclass, field


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_list_int(key: str, default: list) -> list:
    raw = os.environ.get(key)
    if raw:
        return [int(v.strip()) for v in raw.split(",")]
    return default


@dataclass
class SparkConfig:
    """Spark session configuration."""

    app_name: str
    executor_memory: str
    driver_memory: str
    executor_cores: int
    shuffle_partitions: int
    iceberg_extensions: str
    iceberg_catalog_impl: str

    @classmethod
    def from_env(cls) -> "SparkConfig":
        return cls(
            app_name=_env_str("SPARK_APP_NAME", "semantic-clustering-pipeline"),
            executor_memory=_env_str("SPARK_EXECUTOR_MEMORY", "4g"),
            driver_memory=_env_str("SPARK_DRIVER_MEMORY", "4g"),
            executor_cores=_env_int("SPARK_EXECUTOR_CORES", 2),
            shuffle_partitions=_env_int("SPARK_SHUFFLE_PARTITIONS", 8),
            iceberg_extensions=_env_str(
                "SPARK_ICEBERG_EXTENSIONS",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            ),
            iceberg_catalog_impl=_env_str(
                "SPARK_ICEBERG_CATALOG_IMPL",
                "org.apache.iceberg.aws.glue.GlueCatalog",
            ),
        )


@dataclass
class ClusteringConfig:
    """KMeans hierarchical clustering configuration."""

    # Hierarchical levels
    k_level_1: int  # fine-grained: 1000
    k_level_2: int  # mid-level:    200
    k_level_3: int  # macro:         50

    # KMeans parameters
    init_mode: str       # "k-means||"
    init_steps: int      # 5
    max_iter: int        # 50
    seed: int            # 42
    tolerance: float     # 1e-4

    # PCA dimensionality reduction
    pca_output_dims: int  # 128

    # Evaluation (Elbow + Silhouette)
    evaluation_k_candidates: list  # [50, 100, 200, 300, 500, 750, 1000]
    evaluation_sample_fraction: float  # 0.2
    silhouette_threshold: float         # 0.3 — below this logs a warning

    # Whether to run evaluation phase during training
    run_evaluation: bool  # True

    @classmethod
    def from_env(cls) -> "ClusteringConfig":
        return cls(
            k_level_1=_env_int("CLUSTERING_K_LEVEL_1", 1000),
            k_level_2=_env_int("CLUSTERING_K_LEVEL_2", 200),
            k_level_3=_env_int("CLUSTERING_K_LEVEL_3", 50),
            init_mode=_env_str("CLUSTERING_INIT_MODE", "k-means||"),
            init_steps=_env_int("CLUSTERING_INIT_STEPS", 5),
            max_iter=_env_int("CLUSTERING_MAX_ITER", 50),
            seed=_env_int("CLUSTERING_SEED", 42),
            tolerance=_env_float("CLUSTERING_TOLERANCE", 1e-4),
            pca_output_dims=_env_int("CLUSTERING_PCA_OUTPUT_DIMS", 128),
            evaluation_k_candidates=_env_list_int(
                "CLUSTERING_EVAL_K_CANDIDATES",
                [50, 100, 200, 300, 500, 750, 1000],
            ),
            evaluation_sample_fraction=_env_float(
                "CLUSTERING_EVAL_SAMPLE_FRACTION", 0.2
            ),
            silhouette_threshold=_env_float("CLUSTERING_SILHOUETTE_THRESHOLD", 0.3),
            run_evaluation=os.environ.get("CLUSTERING_RUN_EVALUATION", "true").lower()
            == "true",
        )


@dataclass
class EmbeddingConfig:
    """
    Embedding generation configuration for the Edi Bins API.
    Does NOT include URL, key or model name — those belong in the concrete client.

    Edi Bins API limits:
        - max 20 parallel requests  (max_concurrent)
        - max 500 items per input array (batch_size)
    """

    batch_size: int            # 500 — Edi Bins limit per request
    max_concurrent: int        # 20  — Edi Bins parallel request limit
    retry_attempts: int        # 3
    retry_delay_seconds: float  # 2.0
    vector_dimensions: int     # defined by the API used

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            batch_size=_env_int("EMBEDDING_BATCH_SIZE", 500),
            max_concurrent=_env_int("EMBEDDING_MAX_CONCURRENT", 20),
            retry_attempts=_env_int("EMBEDDING_RETRY_ATTEMPTS", 3),
            retry_delay_seconds=_env_float("EMBEDDING_RETRY_DELAY_SECONDS", 2.0),
            vector_dimensions=_env_int("EMBEDDING_VECTOR_DIMENSIONS", 384),
        )


@dataclass
class StorageConfig:
    """Iceberg / Glue Catalog storage configuration."""

    # Glue Catalog
    glue_catalog: str   # e.g. "glue_catalog"
    database: str       # e.g. "semantic_clustering"

    # Full table names: glue_catalog.database.table
    table_embeddings: str
    table_clusters_output: str
    table_cluster_mapping: str
    table_cluster_names: str
    table_cluster_metrics: str
    table_frases_raw: str

    # S3 paths for saved models
    models_base_path: str
    model_normalizer_path: str
    model_pca_path: str
    model_kmeans_n1_path: str
    model_kmeans_n2_path: str
    model_kmeans_n3_path: str

    @classmethod
    def from_env(cls) -> "StorageConfig":
        catalog = _env_str("STORAGE_GLUE_CATALOG", "glue_catalog")
        database = _env_str("STORAGE_DATABASE", "semantic_clustering")
        prefix = f"{catalog}.{database}"

        models_base = _env_str("STORAGE_MODELS_BASE_PATH", "s3://my-bucket/models/")

        return cls(
            glue_catalog=catalog,
            database=database,
            table_embeddings=_env_str(
                "STORAGE_TABLE_EMBEDDINGS", f"{prefix}.embeddings"
            ),
            table_clusters_output=_env_str(
                "STORAGE_TABLE_CLUSTERS_OUTPUT", f"{prefix}.clusters_output"
            ),
            table_cluster_mapping=_env_str(
                "STORAGE_TABLE_CLUSTER_MAPPING", f"{prefix}.cluster_mapping"
            ),
            table_cluster_names=_env_str(
                "STORAGE_TABLE_CLUSTER_NAMES", f"{prefix}.cluster_names"
            ),
            table_cluster_metrics=_env_str(
                "STORAGE_TABLE_CLUSTER_METRICS", f"{prefix}.cluster_metrics"
            ),
            table_frases_raw=_env_str(
                "STORAGE_TABLE_FRASES_RAW", f"{prefix}.frases_raw"
            ),
            models_base_path=models_base,
            model_normalizer_path=_env_str(
                "STORAGE_MODEL_NORMALIZER_PATH", f"{models_base}normalizer"
            ),
            model_pca_path=_env_str(
                "STORAGE_MODEL_PCA_PATH", f"{models_base}pca"
            ),
            model_kmeans_n1_path=_env_str(
                "STORAGE_MODEL_KMEANS_N1_PATH", f"{models_base}kmeans_n1"
            ),
            model_kmeans_n2_path=_env_str(
                "STORAGE_MODEL_KMEANS_N2_PATH", f"{models_base}kmeans_n2"
            ),
            model_kmeans_n3_path=_env_str(
                "STORAGE_MODEL_KMEANS_N3_PATH", f"{models_base}kmeans_n3"
            ),
        )


@dataclass
class NamingConfig:
    """
    Cluster naming via LLM configuration.
    Does NOT include URL, key or model name — those belong in the concrete client.
    """

    samples_per_cluster: int  # 10 — representative phrases sent to the LLM
    max_concurrent: int       # 20
    retry_attempts: int       # 3

    @classmethod
    def from_env(cls) -> "NamingConfig":
        return cls(
            samples_per_cluster=_env_int("NAMING_SAMPLES_PER_CLUSTER", 10),
            max_concurrent=_env_int("NAMING_MAX_CONCURRENT", 20),
            retry_attempts=_env_int("NAMING_RETRY_ATTEMPTS", 3),
        )


@dataclass
class PipelineSettings:
    """
    Top-level settings object that aggregates all sub-configurations.
    Single source of truth for the entire pipeline.
    """

    spark: SparkConfig
    clustering: ClusteringConfig
    embedding: EmbeddingConfig
    storage: StorageConfig
    naming: NamingConfig

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        """Load all settings from environment variables."""
        return cls(
            spark=SparkConfig.from_env(),
            clustering=ClusteringConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            storage=StorageConfig.from_env(),
            naming=NamingConfig.from_env(),
        )
