"""
Input and output validators for the clustering pipeline.
Validators fail fast with clear, descriptive error messages.
"""

from typing import Any

from pyspark.sql import DataFrame

from clustering_pipeline.config.settings import ClusteringConfig, StorageConfig
from clustering_pipeline.utils.logger import PipelineLogger


class ValidationError(Exception):
    """Raised when a validation check fails."""


class InputValidator:
    """
    Validates pre-conditions before jobs start processing data.
    All checks should be exhaustive — validate everything upfront, fail fast.
    """

    def __init__(self, logger: PipelineLogger):
        self._logger = logger

    # ------------------------------------------------------------------
    # DataFrame validators
    # ------------------------------------------------------------------

    def assert_not_empty(self, df: DataFrame, label: str) -> None:
        """Raise ValidationError if the DataFrame has no rows."""
        count = df.count()
        if count == 0:
            raise ValidationError(
                f"DataFrame '{label}' está vazio — esperava pelo menos 1 registro."
            )
        self._logger.info(f"Validação OK: '{label}' não está vazio", registros=count)

    def assert_columns_exist(self, df: DataFrame, columns: list, label: str) -> None:
        """Raise ValidationError if any of the required columns are missing."""
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValidationError(
                f"DataFrame '{label}' faltam colunas obrigatórias: {missing}. "
                f"Colunas presentes: {df.columns}"
            )
        self._logger.debug(f"Validação OK: '{label}' possui todas as colunas", colunas=columns)

    # ------------------------------------------------------------------
    # Clustering config validators
    # ------------------------------------------------------------------

    def assert_clustering_hierarchy(self, config: ClusteringConfig) -> None:
        """
        Ensure k_level_1 > k_level_2 > k_level_3.
        Hierarchical clustering requires each level to have fewer clusters.
        """
        if not (config.k_level_1 > config.k_level_2 > config.k_level_3):
            raise ValidationError(
                f"Configuração de clustering inválida: k1={config.k_level_1} deve ser "
                f"> k2={config.k_level_2} deve ser > k3={config.k_level_3}."
            )
        self._logger.info(
            "Validação OK: hierarquia de k",
            k1=config.k_level_1,
            k2=config.k_level_2,
            k3=config.k_level_3,
        )

    # ------------------------------------------------------------------
    # S3 / Iceberg validators
    # ------------------------------------------------------------------

    def assert_s3_path_accessible(self, spark: Any, path: str) -> None:
        """
        Verify an S3 path (model checkpoint) is reachable via Hadoop FS.
        Raises ValidationError if the path does not exist.
        """
        try:
            jvm = spark._jvm
            hadoop_conf = spark._jsc.hadoopConfiguration()
            fs = jvm.org.apache.hadoop.fs.FileSystem.get(
                jvm.java.net.URI(path), hadoop_conf
            )
            exists = fs.exists(jvm.org.apache.hadoop.fs.Path(path))
            if not exists:
                raise ValidationError(
                    f"Caminho S3 não encontrado: {path}. "
                    "Verifique se o modelo foi salvo corretamente."
                )
            self._logger.info("Validação OK: caminho S3 acessível", path=path)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(
                f"Erro ao verificar caminho S3 '{path}': {exc}"
            ) from exc

    def assert_table_exists(self, spark: Any, table_name: str) -> None:
        """
        Check that an Iceberg table exists in the Glue Catalog.
        Raises ValidationError if absent.
        """
        try:
            parts = table_name.split(".")
            if len(parts) == 3:
                catalog, database, table = parts
                exists = spark.catalog.tableExists(f"{database}.{table}", catalog)
            else:
                exists = spark.catalog.tableExists(table_name)
            if not exists:
                raise ValidationError(
                    f"Tabela Iceberg não encontrada: '{table_name}'. "
                    "Verifique o Glue Catalog e o database."
                )
            self._logger.info("Validação OK: tabela existe", tabela=table_name)
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(
                f"Erro ao verificar tabela '{table_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Embedding validators
    # ------------------------------------------------------------------

    def assert_embeddings_valid(
        self, embeddings: list, expected_dims: int, label: str
    ) -> None:
        """
        Validate that a list of embedding vectors has the expected dimensionality.
        """
        if not embeddings:
            raise ValidationError(f"Lista de embeddings '{label}' está vazia.")
        for i, emb in enumerate(embeddings):
            if len(emb) != expected_dims:
                raise ValidationError(
                    f"Embedding {i} em '{label}' tem dimensão {len(emb)}, "
                    f"esperada {expected_dims}."
                )
        self._logger.debug(
            f"Validação OK: embeddings '{label}'",
            total=len(embeddings),
            dims=expected_dims,
        )
