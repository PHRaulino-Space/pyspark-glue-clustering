# Prompt para Claude Code — Pipeline de Clustering Semântico Hierárquico

## Contexto

Você deve implementar um pipeline completo de clustering semântico hierárquico usando AWS Glue 5.0 com PySpark. O sistema processa frases, gera embeddings via API externa, e agrupa semanticamente em 3 níveis hierárquicos usando KMeans.

O pipeline é dividido em dois jobs Glue independentes:
- **Job 1 — Treinamento**: estuda parâmetros ideais, treina os modelos KMeans e nomeia os grupos via LLM
- **Job 2 — Atribuição diária**: processa frases novas, gera embeddings e atribui aos clusters existentes

---

## Estrutura de Diretórios

```
clustering_pipeline/
├── config/
│   └── settings.py               # ÚNICA fonte de configuração
├── core/
│   ├── __init__.py
│   ├── base_job.py               # Classe abstrata base para todos os jobs
│   ├── base_embedding_client.py  # Classe abstrata para cliente de embedding
│   ├── base_completion_client.py # Classe abstrata para cliente de chat completion
│   ├── base_storage.py           # Classe abstrata para leitura/escrita Iceberg
│   └── base_clustering.py        # Classe abstrata para estratégia de clustering
├── services/
│   ├── __init__.py
│   ├── embedding_service.py      # Orquestra geração de embeddings em batch async
│   ├── clustering_service.py     # Orquestra KMeans hierárquico
│   ├── evaluation_service.py     # Elbow method + Silhouette
│   ├── naming_service.py         # Nomeia clusters via LLM
│   └── storage_service.py        # Leitura/escrita Iceberg via Glue Catalog
├── jobs/
│   ├── __init__.py
│   ├── job_training.py           # Job 1 — Treinamento
│   └── job_assignment.py         # Job 2 — Atribuição diária
├── utils/
│   ├── __init__.py
│   ├── logger.py                 # Logger centralizado com rastreabilidade
│   ├── spark_utils.py            # Helpers Spark (SparkSession, schema, etc)
│   └── validators.py             # Validações de entrada e saída
└── tests/
    ├── __init__.py
    ├── test_clustering_service.py
    ├── test_embedding_service.py
    └── fixtures/
        └── sample_embeddings.py  # Dados sintéticos para testes locais
```

---

## 1. Configuração Central — `config/settings.py`

Implemente uma dataclass ou classe de configuração que centralize **todos** os parâmetros do pipeline. Nenhum valor hardcoded deve existir fora deste arquivo.

```python
# Todos os parâmetros devem ser carregados de variáveis de ambiente
# com fallback para valores default documentados

@dataclass
class SparkConfig:
    app_name: str
    executor_memory: str
    driver_memory: str
    executor_cores: int
    shuffle_partitions: int
    iceberg_extensions: str
    iceberg_catalog_impl: str

@dataclass
class ClusteringConfig:
    # Níveis hierárquicos
    k_level_1: int          # ex: 1000
    k_level_2: int          # ex: 200
    k_level_3: int          # ex: 50

    # KMeans
    init_mode: str          # "k-means||"
    init_steps: int         # 5
    max_iter: int           # 50
    seed: int               # 42
    tolerance: float        # 1e-4

    # PCA
    pca_output_dims: int    # 128

    # Avaliação
    evaluation_k_candidates: list[int]   # [50, 100, 200, 300, 500, 750, 1000]
    evaluation_sample_fraction: float    # 0.2
    silhouette_threshold: float          # 0.3 — abaixo disso loga warning

@dataclass
class EmbeddingConfig:
    # Não inclui URL, chave ou modelo — implementado na classe concreta
    batch_size: int          # 1000
    max_concurrent: int      # 50
    retry_attempts: int      # 3
    retry_delay_seconds: float  # 2.0
    vector_dimensions: int   # definido pela API usada

@dataclass
class StorageConfig:
    # Glue Catalog
    glue_catalog: str        # ex: "glue_catalog"
    database: str            # ex: "semantic_clustering"

    # Nomes das tabelas — formato completo: glue_catalog.database.table
    table_embeddings: str           # "glue_catalog.database.embeddings"
    table_clusters_output: str      # "glue_catalog.database.clusters_output"
    table_cluster_mapping: str      # "glue_catalog.database.cluster_mapping"
    table_cluster_names: str        # "glue_catalog.database.cluster_names"
    table_cluster_metrics: str      # "glue_catalog.database.cluster_metrics"
    table_frases_raw: str           # "glue_catalog.database.frases_raw"

    # S3 — modelos salvos
    models_base_path: str    # "s3://bucket/models/"
    model_normalizer_path: str
    model_pca_path: str
    model_kmeans_n1_path: str
    model_kmeans_n2_path: str
    model_kmeans_n3_path: str

@dataclass
class NamingConfig:
    # Não inclui URL, chave ou modelo — implementado na classe concreta
    samples_per_cluster: int    # 10 — frases representativas enviadas ao LLM
    max_concurrent: int         # 20
    retry_attempts: int         # 3

@dataclass
class PipelineSettings:
    spark: SparkConfig
    clustering: ClusteringConfig
    embedding: EmbeddingConfig
    storage: StorageConfig
    naming: NamingConfig

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        # Carrega tudo de variáveis de ambiente
        ...
```

---

## 2. Logger Centralizado — `utils/logger.py`

Implemente um logger que inclua em cada linha:

- Timestamp ISO 8601
- Nome do job
- Nome do serviço/classe
- Nível (INFO, WARNING, ERROR, DEBUG)
- Mensagem
- Contexto estruturado (ex: `k=1000`, `batch=42/130`, `cluster_id=742`)

```python
# Formato esperado dos logs:
# 2024-01-15T10:23:45.123Z [job_training] [ClusteringService] INFO  KMeans N1 iniciado | k=1000 seed=42 features=features_pca
# 2024-01-15T10:24:10.456Z [job_training] [EvaluationService] INFO  Silhouette calculado | k=200 score=0.4821
# 2024-01-15T10:25:00.789Z [job_assignment] [EmbeddingService] INFO  Batch processado | batch=42/130 frases=1000 tempo=1.2s
# 2024-01-15T10:25:01.001Z [job_assignment] [EmbeddingService] ERROR Falha no batch | batch=43/130 tentativa=1/3 erro=TimeoutError

class PipelineLogger:
    def __init__(self, job_name: str, service_name: str): ...
    def info(self, message: str, **context): ...
    def warning(self, message: str, **context): ...
    def error(self, message: str, **context): ...
    def debug(self, message: str, **context): ...
    def timer(self, label: str):
        # context manager que loga o tempo de execução de um bloco
        ...
```

---

## 3. Classes Abstratas — `core/`

### `base_job.py`

```python
from abc import ABC, abstractmethod

class BaseJob(ABC):
    """
    Classe base para todos os jobs Glue.
    Gerencia ciclo de vida: init → validate → run → teardown
    """

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.spark = None
        self.logger = PipelineLogger(job_name=self.job_name)

    @property
    @abstractmethod
    def job_name(self) -> str:
        """Nome do job para logs e rastreabilidade"""
        ...

    @abstractmethod
    def validate_inputs(self) -> bool:
        """
        Valida pré-condições antes de executar.
        Ex: modelos existem no S3, tabelas existem no Glue Catalog.
        Deve falhar rápido com mensagem clara antes de processar dados.
        """
        ...

    @abstractmethod
    def run(self) -> None:
        """Lógica principal do job"""
        ...

    def execute(self) -> None:
        """
        Método público que orquestra o ciclo completo.
        Não deve ser sobrescrito.
        """
        # 1. init spark
        # 2. validate_inputs — falha rápido
        # 3. run com try/except
        # 4. teardown (fecha spark, loga duração total)
        ...
```

### `base_embedding_client.py`

```python
from abc import ABC, abstractmethod

class BaseEmbeddingClient(ABC):
    """
    Abstração para qualquer API de embedding.
    A implementação concreta define URL, modelo, autenticação e formato.
    """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Recebe lista de textos, retorna lista de vetores.
        Deve tratar erros, retries e rate limit internamente.
        """
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        ...

    @property
    @abstractmethod
    def vector_dimensions(self) -> int:
        """Dimensões do vetor gerado por esta API"""
        ...
```

### `base_completion_client.py`

```python
from abc import ABC, abstractmethod

class BaseCompletionClient(ABC):
    """
    Abstração para qualquer API de chat completion.
    Usada para nomear clusters.
    """

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Envia prompt e retorna resposta como string.
        Deve tratar retries internamente.
        """
        ...

    @abstractmethod
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Versão que garante resposta em JSON válido.
        Deve fazer parse e validar antes de retornar.
        """
        ...
```

### `base_storage.py`

```python
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame

class BaseStorage(ABC):
    """
    Abstração para leitura/escrita no Iceberg via Glue Catalog.
    """

    @abstractmethod
    def read_table(self, table_name: str) -> DataFrame:
        """Lê tabela Iceberg completa"""
        ...

    @abstractmethod
    def read_table_with_filter(self, table_name: str, condition: str) -> DataFrame:
        """Lê com filtro pushdown"""
        ...

    @abstractmethod
    def write_table(self, df: DataFrame, table_name: str, mode: str = "append") -> None:
        """Escreve DataFrame na tabela Iceberg"""
        ...

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        ...

    @abstractmethod
    def get_existing_ids(self, table_name: str, id_column: str) -> DataFrame:
        """Retorna DataFrame só com a coluna de IDs — otimizado para anti-join"""
        ...
```

### `base_clustering.py`

```python
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeansModel

class BaseClusteringStrategy(ABC):
    """
    Abstração para estratégia de clustering.
    Permite trocar KMeans por outro algoritmo sem mudar os jobs.
    """

    @abstractmethod
    def fit(self, df: DataFrame, features_col: str, k: int) -> KMeansModel:
        ...

    @abstractmethod
    def transform(self, model, df: DataFrame, features_col: str) -> DataFrame:
        ...

    @abstractmethod
    def save_model(self, model, path: str) -> None:
        ...

    @abstractmethod
    def load_model(self, path: str):
        ...
```

---

## 4. Serviços — `services/`

### `embedding_service.py`

Responsável por orquestrar a geração de embeddings em lote com controle de concorrência async.

```python
class EmbeddingService:
    def __init__(
        self,
        client: BaseEmbeddingClient,
        config: EmbeddingConfig,
        logger: PipelineLogger
    ): ...

    async def process_dataframe(
        self,
        df: DataFrame,         # DataFrame com coluna de texto
        text_col: str,
        id_col: str
    ) -> DataFrame:
        """
        Processa todos os textos do DataFrame em batches async paralelos.
        
        Fluxo:
        1. Coleta textos do DataFrame (collect apenas ids e textos)
        2. Divide em chunks de config.batch_size
        3. Processa chunks com asyncio + Semaphore(config.max_concurrent)
        4. Retorna DataFrame com colunas: id, texto, embedding (array<float>)
        
        Logs esperados:
        - Início: total de frases, total de batches
        - Por batch: batch N/total, tempo, sucesso/erro
        - Fim: total processado, tempo total, taxa (frases/s)
        """
        ...

    async def _process_chunk_with_retry(
        self,
        chunk_id: int,
        total_chunks: int,
        ids: list,
        texts: list,
        semaphore: asyncio.Semaphore
    ) -> list[dict]:
        """
        Processa um chunk com retry exponencial.
        Loga cada tentativa com contexto claro.
        """
        ...
```

### `evaluation_service.py`

Responsável pelo estudo do k ideal.

```python
class EvaluationService:
    def __init__(
        self,
        spark,
        config: ClusteringConfig,
        logger: PipelineLogger
    ): ...

    def find_optimal_k(
        self,
        df: DataFrame,
        features_col: str,
        k_candidates: list[int]
    ) -> dict:
        """
        Roda Elbow Method + Silhouette para cada k candidato.
        Usa amostra (config.evaluation_sample_fraction) para eficiência.

        Retorna dict com:
        {
            "results": [
                {"k": 100, "inertia": 9800.0, "silhouette": 0.41},
                {"k": 200, "inertia": 7200.0, "silhouette": 0.48},
                ...
            ],
            "recommended_k": 500,
            "recommendation_reason": "cotovelo em k=500, silhouette=0.54"
        }

        Logs esperados:
        - Por iteração: k=X | inércia=Y | silhouette=Z | tempo=Ws
        - Ao final: k recomendado com justificativa
        """
        ...

    def _find_elbow(self, ks: list[int], inertias: list[float]) -> int:
        """
        Detecta o cotovelo matematicamente (maior variação de segunda derivada).
        """
        ...
```

### `clustering_service.py`

Orquestra o KMeans hierárquico nos 3 níveis.

```python
class ClusteringService:
    def __init__(
        self,
        spark,
        strategy: BaseClusteringStrategy,
        config: ClusteringConfig,
        logger: PipelineLogger
    ): ...

    def train_pipeline(self, df: DataFrame) -> ClusteringPipelineResult:
        """
        Treina os 3 níveis hierárquicos completos.

        Fluxo:
        1. Normaliza vetores (Normalizer L2)
        2. Reduz dimensões (PCA: vector_dims → pca_output_dims)
        3. KMeans N1 (k=k_level_1) nos embeddings normalizados/reduzidos
        4. Extrai centróides do N1
        5. KMeans N2 (k=k_level_2) nos centróides do N1
        6. Extrai centróides do N2
        7. KMeans N3 (k=k_level_3) nos centróides do N2
        8. Gera tabela de mapeamento n1→n2→n3

        Retorna ClusteringPipelineResult com todos os modelos e o mapeamento.

        Logs esperados:
        - Por nível: iniciando KMeans | k=X | features=Y | registros=Z
        - Por nível: KMeans concluído | inércia=X | tempo=Ys
        - Ao final: pipeline completo | tempo_total=Xs
        """
        ...

    def build_cluster_mapping(
        self,
        model_n1,
        model_n2,
        model_n3
    ) -> DataFrame:
        """
        Gera DataFrame com mapeamento completo:
        cluster_n1 | cluster_n2 | cluster_n3

        Usa os centróides do N1 como input do transform do N2,
        e centróides do N2 como input do transform do N3.
        """
        ...

    def transform_new_embeddings(
        self,
        df: DataFrame,
        normalizer_model,
        pca_model,
        kmeans_n1_model,
        cluster_mapping: DataFrame,
        features_col: str = "embedding"
    ) -> DataFrame:
        """
        Atribui clusters a novos embeddings usando modelos já treinados.

        Fluxo:
        1. Normaliza
        2. Reduz com PCA
        3. model_n1.transform() → cluster_n1
        4. join com cluster_mapping → cluster_n2, cluster_n3

        Logs esperados:
        - Iniciando transform | registros=X
        - Transform N1 concluído | tempo=Ys
        - Join mapeamento concluído | tempo=Ys
        """
        ...

@dataclass
class ClusteringPipelineResult:
    normalizer_model: any
    pca_model: any
    kmeans_n1: any
    kmeans_n2: any
    kmeans_n3: any
    cluster_mapping: DataFrame    # n1→n2→n3
    metrics: dict                 # inércia e silhouette por nível
```

### `naming_service.py`

Nomeia cada cluster usando LLM com as frases mais representativas.

```python
class NamingService:
    def __init__(
        self,
        client: BaseCompletionClient,
        config: NamingConfig,
        logger: PipelineLogger
    ): ...

    async def name_all_clusters(
        self,
        df_clustered: DataFrame,
        text_col: str,
        cluster_mapping: DataFrame
    ) -> DataFrame:
        """
        Para cada cluster_n1, pega as N frases mais próximas do centróide
        e chama o LLM para gerar nome e descrição da dor.

        Retorna DataFrame:
        cluster_n1 | cluster_n2 | cluster_n3 | nome_n1 | dor_n1 | nome_n2 | dor_n2 | nome_n3 | dor_n3

        Fluxo:
        1. Para cada cluster único em N1: seleciona config.samples_per_cluster frases
        2. Chama LLM async com Semaphore(config.max_concurrent)
        3. LLM retorna JSON: {"nome": "...", "dor": "..."}
        4. Join com cluster_mapping para propagar nomes aos níveis superiores
        5. Para N2 e N3: agrega nomes dos N1 filhos e chama LLM para nome do grupo pai

        Logs esperados:
        - Por cluster: cluster_id=X | nivel=N1 | amostras=10 | nome_gerado="..."
        - Progresso: nomeados X/1000 clusters
        """
        ...

    def _build_naming_prompt(self, frases: list[str], nivel: int) -> tuple[str, str]:
        """
        Retorna (system_prompt, user_prompt) para o nível de hierarquia.
        Níveis mais altos pedem nomes mais abstratos/macro.
        O prompt deve pedir resposta SOMENTE em JSON: {"nome": "...", "dor": "..."}
        """
        ...
```

### `storage_service.py`

```python
class IcebergStorageService(BaseStorage):
    def __init__(self, spark, config: StorageConfig, logger: PipelineLogger): ...

    def read_table(self, table_name: str) -> DataFrame:
        """
        Lê tabela Iceberg via Spark.
        table_name deve estar no formato: glue_catalog.database.table
        Loga: tabela lida | registros=X | tempo=Ys
        """
        ...

    def write_table(self, df: DataFrame, table_name: str, mode: str = "append") -> None:
        """
        Escreve no Iceberg.
        Loga: escrita iniciada | tabela=X | registros=Y
             escrita concluída | tabela=X | tempo=Zs
        """
        ...

    def get_existing_ids(self, table_name: str, id_column: str) -> DataFrame:
        """
        SELECT DISTINCT id_column FROM table_name
        Otimizado — só carrega a coluna de ID para o anti-join.
        """
        ...
```

---

## 5. Jobs — `jobs/`

### `job_training.py`

```python
class JobTraining(BaseJob):
    """
    Job 1 — Treinamento dos modelos KMeans hierárquicos.
    Executado sob demanda (não diariamente).
    """

    @property
    def job_name(self) -> str:
        return "job_training"

    def validate_inputs(self) -> bool:
        """
        Valida:
        - Tabela de embeddings históricos existe e não está vazia
        - Caminho S3 para modelos está acessível
        - Parâmetros de k fazem sentido (k1 > k2 > k3)
        """
        ...

    def run(self) -> None:
        """
        Fluxo completo:

        FASE 1 — LEITURA
        - Lê tabela de embeddings históricos (StorageService)
        - Loga: total de registros carregados, dimensões dos vetores

        FASE 2 — ESTUDO DO K (opcional via flag na config)
        - EvaluationService.find_optimal_k()
        - Loga resultados da avaliação
        - Salva métricas na tabela cluster_metrics

        FASE 3 — TREINAMENTO
        - ClusteringService.train_pipeline()
        - Loga progresso de cada nível

        FASE 4 — PERSISTÊNCIA DOS MODELOS
        - Salva normalizer, pca, kmeans_n1, kmeans_n2, kmeans_n3 no S3
        - Salva cluster_mapping no Iceberg

        FASE 5 — NOMEAÇÃO
        - NamingService.name_all_clusters()
        - Salva cluster_names no Iceberg

        FASE 6 — MÉTRICAS FINAIS
        - Salva métricas do treino na tabela cluster_metrics
        - Loga resumo completo: tempo total, k usados, silhouettes
        """
        ...
```

### `job_assignment.py`

```python
class JobAssignment(BaseJob):
    """
    Job 2 — Atribuição diária de clusters a novos embeddings.
    Executado diariamente.
    """

    @property
    def job_name(self) -> str:
        return "job_assignment"

    def validate_inputs(self) -> bool:
        """
        Valida:
        - Modelos existem no S3 (normalizer, pca, kmeans_n1)
        - Tabela cluster_mapping existe
        - Tabela cluster_names existe
        - Tabela frases_raw do dia tem registros
        Falha rápido com mensagem clara se qualquer condição não for atendida.
        """
        ...

    def run(self) -> None:
        """
        Fluxo completo:

        FASE 1 — LEITURA PARALELA
        - Lê tabela embeddings (IDs já processados) — só coluna de ID
        - Lê tabela frases_raw (frases do dia)
        - Lê cluster_mapping
        - Lê cluster_names
        - Carrega modelos do S3: normalizer, pca, kmeans_n1
        - Loga: X frases no dia, Y embeddings já existentes

        FASE 2 — FILTRAGEM
        - Anti-join: frases_raw - embeddings existentes = frases_novas
        - Loga: X frases novas para processar, Y já tinham embedding

        FASE 3 — GERAÇÃO DE EMBEDDINGS
        - Se frases_novas vazio: pula fase, loga aviso
        - EmbeddingService.process_dataframe() com as frases novas
        - Salva embeddings novos na tabela embeddings (append)
        - Loga: X embeddings gerados, tempo total, taxa frases/s

        FASE 4 — DF COMPLETO
        - Union: embeddings novos + embeddings existentes do dia
        - Loga: total de embeddings para clusterizar

        FASE 5 — ATRIBUIÇÃO DE CLUSTERS
        - ClusteringService.transform_new_embeddings()
        - Join com cluster_mapping → cluster_n2, cluster_n3
        - Join com cluster_names → nome_n1, dor_n1, nome_n2, dor_n2, nome_n3, dor_n3
        - Loga: transform concluído, joins concluídos

        FASE 6 — OUTPUT FINAL
        - Salva resultado na tabela clusters_output
        - Loga: X registros salvos, colunas do output, tempo total do job
        """
        ...
```

---

## 6. Schemas das Tabelas Iceberg

Implemente os schemas como constantes ou dataclasses em `utils/spark_utils.py`:

```python
# embeddings
# id STRING, frase STRING, embedding ARRAY<FLOAT>, created_at TIMESTAMP

# clusters_output
# id STRING, frase STRING, embedding ARRAY<FLOAT>,
# cluster_n1 INT, nome_n1 STRING, dor_n1 STRING,
# cluster_n2 INT, nome_n2 STRING, dor_n2 STRING,
# cluster_n3 INT, nome_n3 STRING, dor_n3 STRING,
# processed_at TIMESTAMP

# cluster_mapping
# cluster_n1 INT, cluster_n2 INT, cluster_n3 INT

# cluster_names
# cluster_id INT, nivel INT,
# nome STRING, dor STRING,
# trained_at TIMESTAMP

# cluster_metrics
# run_id STRING, trained_at TIMESTAMP,
# k_n1 INT, k_n2 INT, k_n3 INT,
# silhouette_n1 FLOAT, silhouette_n2 FLOAT, silhouette_n3 FLOAT,
# inertia_n1 FLOAT, inertia_n2 FLOAT, inertia_n3 FLOAT,
# total_records BIGINT, pca_dims INT
```

---

## 7. Testes — `tests/`

### `fixtures/sample_embeddings.py`

Gere dados sintéticos para testes locais sem depender de API:

```python
def generate_synthetic_embeddings(
    n_samples: int = 4000,
    n_dims: int = 384,
    n_topics: int = 20,
    seed: int = 42
) -> list[dict]:
    """
    Gera embeddings sintéticos agrupados em n_topics temas.
    Cada tema é um centróide com ruído gaussiano ao redor.
    Retorna lista de dicts: {"id": str, "frase": str, "embedding": list[float]}
    """
    ...
```

### `test_clustering_service.py`

```python
# Testa com dados sintéticos localmente (sem Glue, sem S3)
# Usa SparkSession local: master("local[*]")
# Valida:
# - train_pipeline retorna modelos não nulos
# - cluster_mapping tem registros para cada centróide do N1
# - transform_new_embeddings atribui cluster a todos os registros
# - métricas de silhouette estão no range esperado
```

---

## 8. Requisitos e Boas Práticas

### Obrigatório em toda a implementação:

1. **Nenhum valor hardcoded** fora de `config/settings.py`
2. **Nenhuma credencial, URL ou nome de modelo** no código — apenas nas classes concretas (não implementadas)
3. **Logs estruturados** em toda operação relevante com contexto (tempo, contagens, identificadores)
4. **Fail fast**: `validate_inputs()` deve checar todas as pré-condições antes de processar qualquer dado
5. **Idempotência no Job 2**: rodar duas vezes no mesmo dia não deve duplicar registros (anti-join garante isso)
6. **PCA antes do KMeans**: sempre reduzir dimensões antes de clusterizar
7. **Normalização L2 antes do PCA**: vetores de embedding devem ser normalizados
8. **`shuffle.partitions` baixo**: configurar para 8-16 com 1-2 DPUs, não usar o padrão 200
9. **Async para I/O**: toda chamada de API (embedding e completion) deve ser async com Semaphore
10. **Retry com backoff exponencial**: toda chamada de API deve ter retry configurável

### Padrões de código:

- Type hints em todos os métodos
- Docstrings em todas as classes e métodos públicos
- Dataclasses para objetos de resultado (não dicts soltos)
- Context managers para operações com tempo mensurável

---

## 9. Entrypoints dos Jobs Glue

```python
# jobs/job_training.py — ao final do arquivo
if __name__ == "__main__":
    settings = PipelineSettings.from_env()
    job = JobTraining(settings)
    job.execute()

# jobs/job_assignment.py — ao final do arquivo
if __name__ == "__main__":
    settings = PipelineSettings.from_env()
    job = JobAssignment(settings)
    job.execute()
```

---

## Resumo do que NÃO implementar

Deixe como abstrato ou com `raise NotImplementedError`:

- `BaseEmbeddingClient.embed_batch()` — implementação concreta depende da API
- `BaseCompletionClient.complete()` — implementação concreta depende da API
- Credenciais, URLs e nomes de modelos de IA
- Nome do bucket S3, database e catalog Glue — vêm de variáveis de ambiente

O código deve compilar e os testes devem passar usando apenas os dados sintéticos dos fixtures, sem nenhuma dependência externa de API ou infraestrutura.
