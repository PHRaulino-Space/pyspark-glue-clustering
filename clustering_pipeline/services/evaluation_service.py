"""
EvaluationService — Elbow Method + Silhouette to find the optimal k.

Runs over a sample of the data to keep evaluation time manageable,
then recommends a k based on the elbow point and silhouette score.
"""

import time

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame, SparkSession

from clustering_pipeline.config.settings import ClusteringConfig
from clustering_pipeline.utils.logger import PipelineLogger


class EvaluationService:
    """
    Determines the optimal number of clusters via Elbow Method + Silhouette score.

    Uses a random sample (config.evaluation_sample_fraction) for efficiency,
    which avoids full-scan KMeans training for every k candidate.
    """

    def __init__(
        self,
        spark: SparkSession,
        config: ClusteringConfig,
        logger: PipelineLogger,
    ):
        self._spark = spark
        self._config = config
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_optimal_k(
        self,
        df: DataFrame,
        features_col: str,
        k_candidates: list,
    ) -> dict:
        """
        Evaluate each k candidate and return a recommendation.

        Args:
            df:            DataFrame with feature vectors (post-normalisation/PCA).
            features_col:  Column name of the vector features.
            k_candidates:  Ordered list of k values to evaluate.

        Returns:
            Dict with keys:
                - "results":             list of {"k", "inertia", "silhouette"}
                - "recommended_k":       int
                - "recommendation_reason": str
        """
        sample_fraction = self._config.evaluation_sample_fraction
        sample_df = df.sample(fraction=sample_fraction, seed=self._config.seed).cache()
        sample_count = sample_df.count()

        self._logger.info(
            "Iniciando avaliação de k",
            k_candidates=k_candidates,
            amostra_fracao=sample_fraction,
            amostra_registros=sample_count,
        )

        evaluator = ClusteringEvaluator(
            featuresCol=features_col, metricName="silhouette", distanceMeasure="squaredEuclidean"
        )

        results = []
        for k in k_candidates:
            iter_start = time.perf_counter()
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
            model = kmeans.fit(sample_df)
            predictions = model.transform(sample_df)

            inertia = float(model.summary.trainingCost)
            silhouette = float(evaluator.evaluate(predictions))
            elapsed = time.perf_counter() - iter_start

            self._logger.info(
                "k avaliado",
                k=k,
                inércia=f"{inertia:.2f}",
                silhouette=f"{silhouette:.4f}",
                tempo=f"{elapsed:.2f}s",
            )

            if silhouette < self._config.silhouette_threshold:
                self._logger.warning(
                    "Silhouette abaixo do limiar",
                    k=k,
                    silhouette=f"{silhouette:.4f}",
                    limiar=self._config.silhouette_threshold,
                )

            results.append({"k": k, "inertia": inertia, "silhouette": silhouette})

        sample_df.unpersist()

        ks = [r["k"] for r in results]
        inertias = [r["inertia"] for r in results]
        silhouettes = [r["silhouette"] for r in results]

        elbow_k = self._find_elbow(ks, inertias)

        # Tiebreak: prefer k at elbow with highest silhouette
        elbow_idx = ks.index(elbow_k)
        elbow_silhouette = silhouettes[elbow_idx]

        reason = (
            f"cotovelo em k={elbow_k}, silhouette={elbow_silhouette:.4f}"
        )

        self._logger.info(
            "Avaliação concluída",
            k_recomendado=elbow_k,
            motivo=reason,
        )

        return {
            "results": results,
            "recommended_k": elbow_k,
            "recommendation_reason": reason,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_elbow(self, ks: list, inertias: list) -> int:
        """
        Detect the elbow point via the largest second-derivative (curvature).

        For each interior k, compute the finite-difference second derivative
        of the inertia curve.  The k with the largest positive curvature is
        the elbow — the point where adding more clusters yields diminishing returns.

        Args:
            ks:       Ordered list of k values.
            inertias: Inertia (WCSS) for each k in the same order.

        Returns:
            The k value at the elbow point.
        """
        if len(ks) <= 2:
            # Not enough points to compute a second derivative — return last k
            return ks[-1]

        # Normalise inertias so magnitudes don't dominate
        max_inertia = max(inertias)
        norm = [i / max_inertia for i in inertias]

        # Second finite difference: Δ²f[i] = f[i+1] - 2*f[i] + f[i-1]
        second_derivatives = [
            norm[i + 1] - 2 * norm[i] + norm[i - 1]
            for i in range(1, len(norm) - 1)
        ]

        # Elbow is at the maximum of |Δ²f| — largest curvature
        elbow_interior_idx = second_derivatives.index(max(second_derivatives))
        # Interior points start at index 1 in the original list
        return ks[elbow_interior_idx + 1]
