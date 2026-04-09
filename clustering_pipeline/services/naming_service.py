"""
NamingService — generates human-readable names and pain descriptions for clusters.

Uses an LLM (via BaseCompletionClient) with async concurrency control.
Each cluster receives a name and a "dor" (pain point description) based on
the most representative sample phrases closest to its centroid.
"""

import asyncio
import time
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, TimestampType

from clustering_pipeline.config.settings import NamingConfig
from clustering_pipeline.core.base_completion_client import BaseCompletionClient
from clustering_pipeline.utils.logger import PipelineLogger


_SYSTEM_PROMPT_N1 = """Você é um especialista em análise de experiência do cliente.
Dado um conjunto de frases reais de clientes pertencentes ao mesmo cluster semântico,
responda SOMENTE com um JSON válido no seguinte formato (sem explicações extras):
{"nome": "<nome curto e descritivo do cluster>", "dor": "<descrição da principal dor/necessidade dos clientes>"}"""

_SYSTEM_PROMPT_N2 = """Você é um especialista em análise de experiência do cliente.
Dado um conjunto de nomes de sub-clusters semânticos, gere um nome de grupo maior e
uma descrição da dor comum que engloba todos esses sub-grupos.
Responda SOMENTE com JSON válido:
{"nome": "<nome macro do grupo>", "dor": "<dor/necessidade macro compartilhada>"}"""

_SYSTEM_PROMPT_N3 = """Você é um especialista em estratégia de produto e experiência do cliente.
Dado um conjunto de nomes de grupos de clusters, gere um nome de categoria macro e
a dor estratégica que representa todos esses grupos.
Responda SOMENTE com JSON válido:
{"nome": "<categoria estratégica macro>", "dor": "<dor estratégica central>"}"""

_SYSTEM_PROMPTS = {1: _SYSTEM_PROMPT_N1, 2: _SYSTEM_PROMPT_N2, 3: _SYSTEM_PROMPT_N3}


class NamingService:
    """
    Names clusters at all three hierarchy levels using an LLM.

    Level 1: receives sample phrases → generates granular name + pain.
    Level 2: receives N1 names within the group → generates mid-level name + pain.
    Level 3: receives N2 names within the group → generates macro name + pain.
    """

    def __init__(
        self,
        client: BaseCompletionClient,
        config: NamingConfig,
        logger: PipelineLogger,
    ):
        self._client = client
        self._config = config
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def name_all_clusters(
        self,
        df_clustered: DataFrame,
        text_col: str,
        cluster_mapping: DataFrame,
    ) -> DataFrame:
        """
        Generate names for every cluster at all three hierarchy levels.

        Flow:
            1. For each N1 cluster, sample config.samples_per_cluster phrases
               and call the LLM to get name + dor.
            2. Join N1 names with cluster_mapping to propagate to N2/N3 structure.
            3. For each N2 cluster, aggregate N1 names and call LLM.
            4. For each N3 cluster, aggregate N2 names and call LLM.
            5. Return a wide DataFrame with all names joined.

        Args:
            df_clustered:   DataFrame with cluster_n1, cluster_n2, cluster_n3, text_col.
            text_col:       Column containing the original phrase text.
            cluster_mapping: DataFrame(cluster_n1, cluster_n2, cluster_n3).

        Returns:
            DataFrame: cluster_n1, cluster_n2, cluster_n3,
                       nome_n1, dor_n1, nome_n2, dor_n2, nome_n3, dor_n3.
        """
        spark = df_clustered.sparkSession

        # ── N1 naming ──────────────────────────────────────────────────
        self._logger.info("Iniciando nomeação de clusters N1")
        n1_ids = [
            r["cluster_n1"]
            for r in df_clustered.select("cluster_n1").distinct().collect()
        ]
        n1_names = await self._name_clusters_n1(df_clustered, text_col, n1_ids)

        # Build N1 names DataFrame
        n1_schema = StructType(
            [
                StructField("cluster_n1", IntegerType(), nullable=False),
                StructField("nome_n1", StringType(), nullable=True),
                StructField("dor_n1", StringType(), nullable=True),
            ]
        )
        n1_df = spark.createDataFrame(
            [(cid, v["nome"], v["dor"]) for cid, v in n1_names.items()],
            schema=n1_schema,
        )

        # ── N2 naming ──────────────────────────────────────────────────
        self._logger.info("Iniciando nomeação de clusters N2")
        n2_names = await self._name_parent_clusters(
            child_names_df=n1_df,
            mapping_df=cluster_mapping,
            child_col="cluster_n1",
            parent_col="cluster_n2",
            child_name_col="nome_n1",
            level=2,
            spark=spark,
        )

        n2_schema = StructType(
            [
                StructField("cluster_n2", IntegerType(), nullable=False),
                StructField("nome_n2", StringType(), nullable=True),
                StructField("dor_n2", StringType(), nullable=True),
            ]
        )
        n2_df = spark.createDataFrame(
            [(cid, v["nome"], v["dor"]) for cid, v in n2_names.items()],
            schema=n2_schema,
        )

        # ── N3 naming ──────────────────────────────────────────────────
        self._logger.info("Iniciando nomeação de clusters N3")
        n3_names = await self._name_parent_clusters(
            child_names_df=n2_df,
            mapping_df=cluster_mapping.select("cluster_n2", "cluster_n3").distinct(),
            child_col="cluster_n2",
            parent_col="cluster_n3",
            child_name_col="nome_n2",
            level=3,
            spark=spark,
        )

        n3_schema = StructType(
            [
                StructField("cluster_n3", IntegerType(), nullable=False),
                StructField("nome_n3", StringType(), nullable=True),
                StructField("dor_n3", StringType(), nullable=True),
            ]
        )
        n3_df = spark.createDataFrame(
            [(cid, v["nome"], v["dor"]) for cid, v in n3_names.items()],
            schema=n3_schema,
        )

        # ── Assemble final DataFrame ────────────────────────────────────
        result = (
            cluster_mapping
            .join(n1_df, on="cluster_n1", how="left")
            .join(n2_df, on="cluster_n2", how="left")
            .join(n3_df, on="cluster_n3", how="left")
        )

        self._logger.info(
            "Nomeação concluída",
            clusters_n1=len(n1_names),
            clusters_n2=len(n2_names),
            clusters_n3=len(n3_names),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _name_clusters_n1(
        self,
        df: DataFrame,
        text_col: str,
        cluster_ids: list,
    ) -> dict:
        """
        Name every N1 cluster based on sample phrases.

        Returns:
            Dict {cluster_id: {"nome": ..., "dor": ...}}.
        """
        samples_per = self._config.samples_per_cluster
        # Collect samples per cluster to the driver
        cluster_samples = {}
        for cid in cluster_ids:
            rows = (
                df.filter(F.col("cluster_n1") == cid)
                .select(text_col)
                .limit(samples_per)
                .collect()
            )
            cluster_samples[cid] = [r[text_col] for r in rows]

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        total = len(cluster_ids)
        named = {}
        done = 0

        async def _name_one(cid: int):
            nonlocal done
            phrases = cluster_samples[cid]
            system_p, user_p = self._build_naming_prompt(phrases, level=1)
            result = await self._call_with_retry(system_p, user_p, semaphore)
            named[cid] = result
            done += 1
            self._logger.info(
                "Cluster nomeado",
                cluster_id=cid,
                nivel=1,
                amostras=len(phrases),
                nome_gerado=result.get("nome", "?"),
                progresso=f"{done}/{total}",
            )

        await asyncio.gather(*[_name_one(cid) for cid in cluster_ids])
        return named

    async def _name_parent_clusters(
        self,
        child_names_df: DataFrame,
        mapping_df: DataFrame,
        child_col: str,
        parent_col: str,
        child_name_col: str,
        level: int,
        spark: SparkSession,
    ) -> dict:
        """
        Name parent-level clusters based on the names of their children.

        Returns:
            Dict {parent_cluster_id: {"nome": ..., "dor": ...}}.
        """
        # Join mapping to get parent id per child name
        joined = mapping_df.join(child_names_df, on=child_col, how="left")
        parent_ids = [r[parent_col] for r in joined.select(parent_col).distinct().collect()]

        # Aggregate child names per parent
        parent_children = {}
        for pid in parent_ids:
            rows = (
                joined.filter(F.col(parent_col) == pid)
                .select(child_name_col)
                .collect()
            )
            parent_children[pid] = [r[child_name_col] for r in rows if r[child_name_col]]

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        total = len(parent_ids)
        named = {}
        done = 0

        async def _name_one(pid: int):
            nonlocal done
            phrases = parent_children[pid]
            system_p, user_p = self._build_naming_prompt(phrases, level=level)
            result = await self._call_with_retry(system_p, user_p, semaphore)
            named[pid] = result
            done += 1
            self._logger.info(
                "Cluster nomeado",
                cluster_id=pid,
                nivel=level,
                amostras=len(phrases),
                nome_gerado=result.get("nome", "?"),
                progresso=f"{done}/{total}",
            )

        await asyncio.gather(*[_name_one(pid) for pid in parent_ids])
        return named

    async def _call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        """Call the LLM with exponential back-off retry."""
        delay = 2.0
        async with semaphore:
            for attempt in range(1, self._config.retry_attempts + 1):
                try:
                    return await self._client.complete_json(system_prompt, user_prompt)
                except Exception as exc:
                    self._logger.error(
                        "Falha na chamada LLM",
                        tentativa=f"{attempt}/{self._config.retry_attempts}",
                        erro=str(exc),
                    )
                    if attempt < self._config.retry_attempts:
                        await asyncio.sleep(delay)
                        delay *= 2
                    else:
                        return {"nome": "ERRO", "dor": "ERRO"}

    def _build_naming_prompt(
        self, phrases: list, level: int
    ) -> tuple:
        """
        Build (system_prompt, user_prompt) for the given hierarchy level.

        Higher levels produce more abstract/macro names.

        Args:
            phrases: Sample phrases (N1) or child cluster names (N2/N3).
            level:   1 = granular, 2 = mid, 3 = macro.

        Returns:
            Tuple (system_prompt, user_prompt).
        """
        system_prompt = _SYSTEM_PROMPTS.get(level, _SYSTEM_PROMPT_N1)
        bullet_list = "\n".join(f"- {p}" for p in phrases)

        if level == 1:
            user_prompt = (
                f"Analise estas {len(phrases)} frases de clientes do mesmo cluster "
                f"e gere o nome e a dor do cluster:\n\n{bullet_list}"
            )
        elif level == 2:
            user_prompt = (
                f"Estes são os nomes dos {len(phrases)} sub-clusters que compõem este grupo. "
                f"Gere um nome de grupo e uma dor macro:\n\n{bullet_list}"
            )
        else:
            user_prompt = (
                f"Estes são os nomes dos {len(phrases)} grupos que compõem esta categoria. "
                f"Gere um nome de categoria estratégica e uma dor central:\n\n{bullet_list}"
            )

        return system_prompt, user_prompt
