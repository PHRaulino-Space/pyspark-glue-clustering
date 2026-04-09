"""
Centralized structured logger for the clustering pipeline.
Every log line includes: timestamp, job name, service name, level, message, and context.
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone


class PipelineLogger:
    """
    Structured logger with per-line context for full traceability.

    Log format:
        2024-01-15T10:23:45.123Z [job_training] [ClusteringService] INFO  KMeans N1 iniciado | k=1000 seed=42
    """

    def __init__(self, job_name: str, service_name: str):
        self.job_name = job_name
        self.service_name = service_name

        # Build an underlying stdlib logger (one per (job, service) pair)
        logger_name = f"{job_name}.{service_name}"
        self._logger = logging.getLogger(logger_name)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)
            self._logger.propagate = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
               f"{datetime.now(tz=timezone.utc).microsecond // 1000:03d}Z"

    def _format(self, level: str, message: str, context: dict) -> str:
        ts = self._now()
        ctx_str = ""
        if context:
            ctx_str = " | " + " ".join(f"{k}={v}" for k, v in context.items())
        level_padded = level.ljust(7)
        return (
            f"{ts} [{self.job_name}] [{self.service_name}] "
            f"{level_padded} {message}{ctx_str}"
        )

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def info(self, message: str, **context) -> None:
        """Log an INFO-level message with optional structured context."""
        self._logger.info(self._format("INFO", message, context))

    def warning(self, message: str, **context) -> None:
        """Log a WARNING-level message with optional structured context."""
        self._logger.warning(self._format("WARNING", message, context))

    def error(self, message: str, **context) -> None:
        """Log an ERROR-level message with optional structured context."""
        self._logger.error(self._format("ERROR", message, context))

    def debug(self, message: str, **context) -> None:
        """Log a DEBUG-level message with optional structured context."""
        self._logger.debug(self._format("DEBUG", message, context))

    # ------------------------------------------------------------------
    # Timer context manager
    # ------------------------------------------------------------------

    @contextmanager
    def timer(self, label: str, **context):
        """
        Context manager that logs execution time of a block.

        Usage::

            with logger.timer("KMeans N1", k=1000):
                model = kmeans.fit(df)
            # Logs: KMeans N1 concluído | tempo=12.34s k=1000
        """
        self.info(f"{label} iniciado", **context)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.info(f"{label} concluído", tempo=f"{elapsed:.2f}s", **context)
