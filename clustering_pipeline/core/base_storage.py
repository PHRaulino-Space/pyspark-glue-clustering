"""
Abstract base class for Iceberg storage operations via Glue Catalog.
"""

from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class BaseStorage(ABC):
    """
    Abstraction for reading and writing Iceberg tables through the Glue Catalog.

    Table names must be fully qualified: ``glue_catalog.database.table_name``.
    """

    @abstractmethod
    def read_table(self, table_name: str) -> DataFrame:
        """
        Read an entire Iceberg table.

        Args:
            table_name: Fully-qualified name, e.g. ``glue_catalog.db.embeddings``.

        Returns:
            Spark DataFrame with all columns and rows.
        """

    @abstractmethod
    def read_table_with_filter(self, table_name: str, condition: str) -> DataFrame:
        """
        Read an Iceberg table with a SQL WHERE-clause filter (predicate push-down).

        Args:
            table_name: Fully-qualified table name.
            condition:  SQL condition string, e.g. ``"date = '2024-01-15'"``.

        Returns:
            Spark DataFrame containing only matching rows.
        """

    @abstractmethod
    def write_table(
        self, df: DataFrame, table_name: str, mode: str = "append"
    ) -> None:
        """
        Write a DataFrame to an Iceberg table.

        Args:
            df:         DataFrame to persist.
            table_name: Fully-qualified target table name.
            mode:       Write mode — ``"append"`` (default) or ``"overwrite"``.
        """

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """
        Check whether an Iceberg table exists in the Glue Catalog.

        Args:
            table_name: Fully-qualified table name.

        Returns:
            True if the table exists, False otherwise.
        """

    @abstractmethod
    def get_existing_ids(self, table_name: str, id_column: str) -> DataFrame:
        """
        Return a DataFrame containing only the ID column of a table.

        Optimised for anti-join deduplication: loads only one column,
        avoiding full table scans of wide tables.

        Args:
            table_name: Fully-qualified table name.
            id_column:  Name of the ID column to select.

        Returns:
            DataFrame with a single column ``id_column`` containing distinct values.
        """
