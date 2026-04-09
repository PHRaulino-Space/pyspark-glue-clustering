"""
IcebergStorageService — reads and writes Iceberg tables via the Glue Catalog.
"""

import time

from pyspark.sql import DataFrame, SparkSession

from clustering_pipeline.config.settings import StorageConfig
from clustering_pipeline.core.base_storage import BaseStorage
from clustering_pipeline.utils.logger import PipelineLogger


class IcebergStorageService(BaseStorage):
    """
    Concrete Iceberg storage implementation backed by the AWS Glue Catalog.

    Table names must be fully qualified: ``glue_catalog.database.table``.
    All read and write operations are logged with record counts and timings.
    """

    def __init__(
        self,
        spark: SparkSession,
        config: StorageConfig,
        logger: PipelineLogger,
    ):
        self._spark = spark
        self._config = config
        self._logger = logger

    # ------------------------------------------------------------------
    # BaseStorage implementation
    # ------------------------------------------------------------------

    def read_table(self, table_name: str) -> DataFrame:
        """
        Read a full Iceberg table.

        Args:
            table_name: Fully-qualified name, e.g. ``glue_catalog.db.embeddings``.

        Returns:
            Spark DataFrame with all columns and rows.
        """
        start = time.perf_counter()
        self._logger.info("Lendo tabela", tabela=table_name)

        df = self._spark.table(table_name)
        count = df.count()
        elapsed = time.perf_counter() - start

        self._logger.info(
            "Tabela lida",
            tabela=table_name,
            registros=count,
            tempo=f"{elapsed:.2f}s",
        )
        return df

    def read_table_with_filter(self, table_name: str, condition: str) -> DataFrame:
        """
        Read an Iceberg table with a predicate push-down filter.

        Args:
            table_name: Fully-qualified table name.
            condition:  SQL WHERE clause, e.g. ``"date = '2024-01-15'"``.

        Returns:
            Spark DataFrame with only matching rows.
        """
        start = time.perf_counter()
        self._logger.info("Lendo tabela com filtro", tabela=table_name, filtro=condition)

        df = self._spark.table(table_name).where(condition)
        count = df.count()
        elapsed = time.perf_counter() - start

        self._logger.info(
            "Tabela filtrada lida",
            tabela=table_name,
            filtro=condition,
            registros=count,
            tempo=f"{elapsed:.2f}s",
        )
        return df

    def write_table(
        self, df: DataFrame, table_name: str, mode: str = "append"
    ) -> None:
        """
        Write a DataFrame to an Iceberg table.

        Args:
            df:         DataFrame to persist.
            table_name: Fully-qualified target table name.
            mode:       ``"append"`` (default) or ``"overwrite"``.
        """
        count = df.count()
        self._logger.info(
            "Escrita iniciada",
            tabela=table_name,
            registros=count,
            modo=mode,
        )

        start = time.perf_counter()
        df.writeTo(table_name).using("iceberg").mode(mode).saveAsTable(table_name)
        elapsed = time.perf_counter() - start

        self._logger.info(
            "Escrita concluída",
            tabela=table_name,
            registros=count,
            tempo=f"{elapsed:.2f}s",
        )

    def table_exists(self, table_name: str) -> bool:
        """
        Check whether an Iceberg table exists in the Glue Catalog.

        Args:
            table_name: Fully-qualified table name.

        Returns:
            True if the table exists, False otherwise.
        """
        try:
            parts = table_name.split(".")
            if len(parts) == 3:
                _, database, table = parts
                return self._spark.catalog.tableExists(f"{database}.{table}")
            return self._spark.catalog.tableExists(table_name)
        except Exception:
            return False

    def get_existing_ids(self, table_name: str, id_column: str) -> DataFrame:
        """
        Return a DataFrame with only the distinct ID column — optimised for anti-join.

        Args:
            table_name: Fully-qualified table name.
            id_column:  Column name to select.

        Returns:
            Single-column DataFrame with distinct ID values.
        """
        self._logger.info(
            "Carregando IDs existentes",
            tabela=table_name,
            coluna=id_column,
        )
        df = self._spark.table(table_name).select(id_column).distinct()
        count = df.count()
        self._logger.info(
            "IDs carregados",
            tabela=table_name,
            total_ids=count,
        )
        return df
