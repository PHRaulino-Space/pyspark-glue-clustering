"""
Abstract base class for clustering strategies.
Decouples KMeans implementation from the orchestration logic,
allowing alternative algorithms to be swapped in without changing the jobs.
"""

from abc import ABC, abstractmethod

from pyspark.sql import DataFrame


class BaseClusteringStrategy(ABC):
    """
    Strategy interface for clustering algorithms.

    The concrete implementation wraps Spark ML KMeans (or any other algorithm)
    and exposes a uniform API consumed by ClusteringService.
    """

    @abstractmethod
    def fit(self, df: DataFrame, features_col: str, k: int):
        """
        Train a clustering model on the provided DataFrame.

        Args:
            df:           DataFrame containing the feature vectors.
            features_col: Name of the column holding the feature vectors
                          (must be a Spark ML VectorUDT column).
            k:            Number of clusters.

        Returns:
            A trained model object (e.g. KMeansModel).
        """

    @abstractmethod
    def transform(self, model, df: DataFrame, features_col: str) -> DataFrame:
        """
        Apply a trained model to assign cluster labels to each row.

        Args:
            model:        Trained model returned by fit().
            df:           DataFrame to label (must contain features_col).
            features_col: Name of the feature vector column.

        Returns:
            Input DataFrame with an additional ``prediction`` integer column
            containing the assigned cluster index.
        """

    @abstractmethod
    def save_model(self, model, path: str) -> None:
        """
        Persist a trained model to the given path (S3 or local FS).

        Args:
            model: Trained model to serialise.
            path:  Destination path, e.g. ``s3://bucket/models/kmeans_n1``.
        """

    @abstractmethod
    def load_model(self, path: str):
        """
        Load a previously saved model from the given path.

        Args:
            path: Source path, e.g. ``s3://bucket/models/kmeans_n1``.

        Returns:
            A trained model object ready for transform().
        """
