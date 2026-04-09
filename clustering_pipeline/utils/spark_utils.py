"""
Spark utilities: SparkSession creation, schema definitions, and helper functions.
All Iceberg table schemas are defined here as constants.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from clustering_pipeline.config.settings import SparkConfig


# ---------------------------------------------------------------------------
# Iceberg Table Schemas
# ---------------------------------------------------------------------------

SCHEMA_EMBEDDINGS = StructType(
    [
        StructField("id", StringType(), nullable=False),
        StructField("frase", StringType(), nullable=False),
        StructField("embedding", ArrayType(FloatType()), nullable=False),
        StructField("created_at", TimestampType(), nullable=False),
    ]
)

SCHEMA_CLUSTERS_OUTPUT = StructType(
    [
        StructField("id", StringType(), nullable=False),
        StructField("frase", StringType(), nullable=False),
        StructField("embedding", ArrayType(FloatType()), nullable=False),
        StructField("cluster_n1", IntegerType(), nullable=True),
        StructField("nome_n1", StringType(), nullable=True),
        StructField("dor_n1", StringType(), nullable=True),
        StructField("cluster_n2", IntegerType(), nullable=True),
        StructField("nome_n2", StringType(), nullable=True),
        StructField("dor_n2", StringType(), nullable=True),
        StructField("cluster_n3", IntegerType(), nullable=True),
        StructField("nome_n3", StringType(), nullable=True),
        StructField("dor_n3", StringType(), nullable=True),
        StructField("processed_at", TimestampType(), nullable=False),
    ]
)

SCHEMA_CLUSTER_MAPPING = StructType(
    [
        StructField("cluster_n1", IntegerType(), nullable=False),
        StructField("cluster_n2", IntegerType(), nullable=False),
        StructField("cluster_n3", IntegerType(), nullable=False),
    ]
)

SCHEMA_CLUSTER_NAMES = StructType(
    [
        StructField("cluster_id", IntegerType(), nullable=False),
        StructField("nivel", IntegerType(), nullable=False),
        StructField("nome", StringType(), nullable=True),
        StructField("dor", StringType(), nullable=True),
        StructField("trained_at", TimestampType(), nullable=False),
    ]
)

SCHEMA_CLUSTER_METRICS = StructType(
    [
        StructField("run_id", StringType(), nullable=False),
        StructField("trained_at", TimestampType(), nullable=False),
        StructField("k_n1", IntegerType(), nullable=True),
        StructField("k_n2", IntegerType(), nullable=True),
        StructField("k_n3", IntegerType(), nullable=True),
        StructField("silhouette_n1", FloatType(), nullable=True),
        StructField("silhouette_n2", FloatType(), nullable=True),
        StructField("silhouette_n3", FloatType(), nullable=True),
        StructField("inertia_n1", FloatType(), nullable=True),
        StructField("inertia_n2", FloatType(), nullable=True),
        StructField("inertia_n3", FloatType(), nullable=True),
        StructField("total_records", LongType(), nullable=True),
        StructField("pca_dims", IntegerType(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# SparkSession factory
# ---------------------------------------------------------------------------

def create_spark_session(config: SparkConfig) -> SparkSession:
    """
    Create and configure a SparkSession for the pipeline.

    Configures:
    - Executor and driver memory
    - Shuffle partitions (low value for small DPU count)
    - Iceberg extensions and catalog implementation
    """
    builder = (
        SparkSession.builder.appName(config.app_name)
        .config("spark.executor.memory", config.executor_memory)
        .config("spark.driver.memory", config.driver_memory)
        .config("spark.executor.cores", config.executor_cores)
        .config("spark.sql.shuffle.partitions", config.shuffle_partitions)
        .config("spark.sql.extensions", config.iceberg_extensions)
        .config(
            "spark.sql.catalog.glue_catalog",
            config.iceberg_catalog_impl,
        )
        .config(
            "spark.sql.catalog.glue_catalog.warehouse",
            "s3://",  # overridden by Glue Catalog
        )
    )
    return builder.getOrCreate()


def create_local_spark_session(app_name: str = "test-local") -> SparkSession:
    """
    Create a local SparkSession for unit tests.
    Uses minimal resources and no Iceberg extensions.
    """
    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vectors_to_array_column(df, vector_col: str, output_col: str):
    """
    Convert a Spark ML DenseVector column to an ArrayType(FloatType()) column.
    Useful when persisting embeddings or cluster centres to Iceberg.
    """
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    return df.withColumn(
        output_col,
        vector_to_array(F.col(vector_col)).cast(ArrayType(FloatType())),
    )


def array_to_vector_column(df, array_col: str, output_col: str):
    """
    Convert an ArrayType(FloatType()) column to a Spark ML DenseVector column.
    Needed before feeding persisted embeddings into ML transformers.
    """
    from pyspark.ml.functions import array_to_vector

    return df.withColumn(output_col, array_to_vector(array_col))
