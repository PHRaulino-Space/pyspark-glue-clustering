"""
Abstract base class for all Glue jobs in the clustering pipeline.
Manages the full lifecycle: init → validate → run → teardown.
"""

import time
from abc import ABC, abstractmethod

from clustering_pipeline.config.settings import PipelineSettings
from clustering_pipeline.utils.logger import PipelineLogger
from clustering_pipeline.utils.spark_utils import create_spark_session


class BaseJob(ABC):
    """
    Base class for all Glue jobs.

    Concrete jobs must implement:
        - job_name (property)
        - validate_inputs()
        - run()

    Call job.execute() as the public entrypoint — it orchestrates the full lifecycle.
    """

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.spark = None
        self.logger = PipelineLogger(job_name=self.job_name, service_name="BaseJob")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def job_name(self) -> str:
        """Unique name for this job, used in logs and traceability."""

    @abstractmethod
    def validate_inputs(self) -> bool:
        """
        Validate all pre-conditions before processing any data.

        Must check that required tables, models, and configuration values
        are present and sane.  Fail fast with a descriptive error if any
        condition is not met — never proceed with invalid state.

        Returns True if all checks pass, raises otherwise.
        """

    @abstractmethod
    def run(self) -> None:
        """Main job logic. Called only after validate_inputs() succeeds."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_spark(self) -> None:
        """Initialize SparkSession from pipeline settings."""
        self.spark = create_spark_session(self.settings.spark)
        self.logger.info(
            "SparkSession inicializado",
            app=self.settings.spark.app_name,
            executor_memory=self.settings.spark.executor_memory,
            shuffle_partitions=self.settings.spark.shuffle_partitions,
        )

    def _teardown(self, start_time: float) -> None:
        """Stop Spark and log total duration."""
        elapsed = time.perf_counter() - start_time
        self.logger.info("Job finalizado", tempo_total=f"{elapsed:.2f}s")
        if self.spark:
            self.spark.stop()

    def execute(self) -> None:
        """
        Orchestrate the full job lifecycle.

        Steps:
            1. Initialize SparkSession
            2. validate_inputs()  — fails fast on bad state
            3. run()              — main logic, wrapped in try/except
            4. teardown           — stop Spark, log total duration

        Do NOT override this method in subclasses.
        """
        start_time = time.perf_counter()
        self.logger.info("Job iniciando")

        try:
            self._init_spark()

            self.logger.info("Validando entradas")
            self.validate_inputs()
            self.logger.info("Entradas validadas com sucesso")

            self.run()

        except Exception as exc:
            self.logger.error(
                "Job falhou com exceção",
                tipo=type(exc).__name__,
                erro=str(exc),
            )
            raise

        finally:
            self._teardown(start_time)
