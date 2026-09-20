# Triagem de Laudos Medicos com NLP

[![CI](https://github.com/guijanczak1/triagem-laudos-medicos-nlp-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/guijanczak1/triagem-laudos-medicos-nlp-mlops/actions/workflows/ci.yml)
[![Docker build](https://github.com/guijanczak1/triagem-laudos-medicos-nlp-mlops/actions/workflows/docker-build.yml/badge.svg)](https://github.com/guijanczak1/triagem-laudos-medicos-nlp-mlops/actions/workflows/docker-build.yml)
[![Airflow DAG validation](https://github.com/guijanczak1/triagem-laudos-medicos-nlp-mlops/actions/workflows/dag-validation.yml/badge.svg)](https://github.com/guijanczak1/triagem-laudos-medicos-nlp-mlops/actions/workflows/dag-validation.yml)

Servico de triagem de urgencia para laudos/abstracts clinicos em texto
livre: um classificador NLP leve (TF-IDF + regressao logistica,
exportado para ONNX Runtime) empacotado como API FastAPI, com pipeline
completo de MLOps -- treino orquestrado, containerizacao, CI/CD,
monitoramento e otimizacao de latencia medida.

> **Aviso importante -- leia antes de usar.** O rotulo de urgencia
> (`normal` / `atencao` / `urgente`) **nao foi atribuido por um
> profissional de saude**. Ele vem de uma heuristica determinista de
> palavras-chave + categoria clinica, aplicada sobre um corpus de
> resumos academicos em ingles (nao laudos clinicos reais). Este e um
> sistema **didatico**, **nao validado clinicamente**, e nao deve ser
> usado para apoiar decisao medica real. Detalhes completos, incluindo
> como a heuristica foi construida e suas limitacoes, em
> [`docs/model_card.md`](docs/model_card.md).

## Sumario

- [Decisao arquitetural de deploy em nuvem](#decisao-arquitetural-de-deploy-em-nuvem)
- [Arquitetura](#arquitetura)
- [Resultados](#resultados)
- [Estrutura do repositorio](#estrutura-do-repositorio)
- [Como rodar](#como-rodar)
- [API](#api)
- [Monitoramento](#monitoramento)
- [Configuracao](#configuracao)
- [Dataset e licenca](#dataset-e-licenca)

---

## Decisao arquitetural de deploy em nuvem

### Batch vs. real-time

O caso de uso e **triagem de urgencia no momento do atendimento**: o
valor da predicao cai a zero se ela chega depois da decisao humana. Um
laudo classificado como `urgente` seis horas depois nao prioriza
ninguem. Isso descarta inferencia em lote para o caminho principal do
sistema.

**Decisao: inferencia real-time sincrona** (request/response HTTP). O
backend padrao (`onnx`) responde em fracoes de milissegundo por
requisicao em bancada local -- ver [Resultados](#resultados) -- o que
deixa folga confortavel para o orcamento de latencia end-to-end em
producao.

**Batch continua existindo -- mas para o retreino.** O ciclo de vida do
modelo (baixar dataset, treinar, avaliar, publicar artefato) e um job
agendado, tolerante a latencia, que roda semanalmente via Airflow
(`dags/train_triagem_dag.py`). Ou seja, o sistema e hibrido por
responsabilidade: **real-time para servir, batch para treinar** -- a
mesma separacao que aparece na arquitetura tecnica (ver
[`docs/architecture.md`](docs/architecture.md)).

### Servico escolhido: AWS ECS Fargate

| dimensao | decisao |
|---|---|
| Registro de imagem | Amazon ECR (scan on push) |
| Execucao | ECS Fargate -- 1 task, 0.25 vCPU / 0.5 GB |
| Entrada | Application Load Balancer (HTTP 80, health check em `/health`) |
| Logs e alarmes | CloudWatch Logs (retencao 7 dias) + metricas do ALB/ECS |
| Regiao | `us-east-1` |

**Por que Fargate:** o artefato de deploy ja e a mesma imagem Docker
testada localmente -- nenhuma reescrita para subir. Fargate remove a
operacao de EC2 (patching, AMI, grupo de autoscaling) sem trazer o peso
de uma plataforma de ML gerenciada, e escala horizontalmente por
contagem de tasks se o volume de laudos crescer. Para um servico de
inferencia leve, sem estado e sempre-ligado, e o menor denominador
operacional que ainda e producao de verdade.

**Alternativas avaliadas e descartadas:**

- **AWS Lambda** -- atraente pelo custo por requisicao, mas o pacote de
  execucao carrega scikit-learn/ONNX Runtime e o modelo, empurrando o
  cold start para a casa dos segundos. Cold start de 2-5 s contradiz o
  proprio requisito de resposta no momento do atendimento. So seria
  viavel com concorrencia provisionada -- o que anula a economia e
  reintroduz custo fixo, na pratica um Fargate mais caro e mais
  complexo.
- **Amazon SageMaker Endpoint** -- e a escolha certa para modelos
  pesados, com multi-variant, testes A/B e autoscaling gerenciado de
  GPU/CPU. Para um TF-IDF + regressao logistica de poucos megabytes, e
  custo e complexidade operacional desproporcionais ao problema.
- **EC2 com Docker** -- mais barato no papel, mas devolve ao time o
  patching de sistema operacional, gerenciamento de AMI e configuracao
  manual de autoscaling -- operacao que este servico nao precisa
  assumir.
- **Kubernetes (EKS)** -- resolve um problema de orquestracao multi-
  servico que este sistema nao tem. Um unico container de inferencia
  nao justifica um control plane.

**Observabilidade em producao:** a stack Prometheus + Grafana roda
localmente via Docker Compose (ver [Monitoramento](#monitoramento)). Em
producao, o caminho natural seria CloudWatch Container Insights ou
Amazon Managed Service for Prometheus fazendo scrape do mesmo endpoint
`/metrics` -- a instrumentacao ja segue o formato padrao Prometheus,
entao essa migracao nao exige nenhuma mudanca no codigo da aplicacao.
Isso e documentado aqui como evolucao natural, nao como algo ja
entregue.

**Custo estimado** (1 task Fargate 0.25 vCPU/0.5 GB ligada 24/7 + ALB +
ECR em `us-east-1`): ordem de US$ 25-30/mes, dominado pelo custo fixo do
ALB.

### Estado da infraestrutura neste repositorio

A infraestrutura esta escrita como codigo (Terraform: ECR, cluster e
servico ECS Fargate, ALB, security groups, log group do CloudWatch, IAM
de minimo privilegio) em `deploy/terraform/`, validada apenas em
**dry-run** (`terraform fmt`, `terraform validate`, `terraform plan`).
**Nenhum recurso foi criado na AWS** -- subir a infraestrutura de fato e
uma decisao operacional do dono do projeto, tomada fora deste
repositorio, e nao algo que qualquer automacao aqui executa por conta
propria.

---

## Arquitetura

Diagrama de componentes, fluxo de dados completo (dataset -> treino ->
otimizacao -> serving -> observabilidade) e as decisoes de design por
tras de cada peca estao em
[`docs/architecture.md`](docs/architecture.md).

Resumo visual:

```
corpus publico -> data/raw -> data/processed/laudos.csv
                                      |
                     +----------------+-----------------+
                     |                                  |
              DAG Airflow (semanal,              load_dataset + split
              batch: download->build->                  |
              train->evaluate->publish)         TF-IDF + LogisticRegression
                     |                                   |
                     |                         model.joblib -> model.onnx
                     |                                   |
                     +---------------> Predictor (sklearn | onnx) <---+
                                                  |
                                        FastAPI (Docker, :8000)
                                                  |
                                        Prometheus (:9090) -> Grafana (:3000)
```

---

## Resultados

### Modelo (holdout estratificado de 20%, 2.888 amostras de um total de 14.438)

| metrica | valor | gate minimo |
|---|---:|---:|
| macro F1 | **0.6427** | 0.55 |
| accuracy | 0.6343 | -- |
| recall (`urgente`) | **0.7364** | 0.70 |

| classe | precision | recall | F1 | suporte |
|---|---:|---:|---:|---:|
| `normal` | 0.6002 | 0.5876 | 0.5938 | 948 |
| `atencao` | 0.5842 | 0.6007 | 0.5923 | 1132 |
| `urgente` | 0.7475 | 0.7364 | 0.7419 | 808 |

O gate de qualidade foi ajustado, com decisao explicita e documentada,
para os valores acima apos duas rodadas reais de correcao tecnica
(troca de familia de modelo e ajuste da formula da heuristica de
rotulo). O raciocinio completo, com os numeros de cada tentativa, esta
em [`docs/model_card.md`](docs/model_card.md) -- nada abaixo da meta
original foi omitido.

### Latencia -- baseline (sklearn) vs. otimizado (ONNX Runtime)

Medido no mesmo hardware, mesma amostra de entrada, mesma quantidade de
execucoes (n=200, warm-up=20 descartado), chamando `Predictor.predict()`
diretamente (o mesmo caminho que a API usa):

| backend | p50 | p90 | p95 | p99 | throughput |
|---|---:|---:|---:|---:|---:|
| sklearn (baseline) | 0.3199 ms | 0.3736 ms | 0.4089 ms | 0.7492 ms | 3.075 req/s |
| onnx (otimizado) | 0.0984 ms | 0.1207 ms | 0.1287 ms | 0.1529 ms | 10.085 req/s |

- **Reducao de p50: 69.24%** (meta: >= 20%) -- atingida com folga.
- **Equivalencia de predicao sklearn x onnx: 99.65%** (2.878/2.888,
  meta: >= 99%) -- a otimizacao nao mudou o comportamento do modelo.
- Quantizacao dinamica int8 foi avaliada e **descartada**: nao trouxe
  ganho real de p50 sobre este modelo. Relatorio completo, metodologia e
  numeros brutos em [`docs/latency_report.md`](docs/latency_report.md)
  e [`docs/benchmarks/comparison.json`](docs/benchmarks/comparison.json).

A API roda com o backend `onnx` por padrao (`TRIAGEM_MODEL_BACKEND=onnx`).

---

## Estrutura do repositorio

| pasta/arquivo | conteudo |
|---|---|
| `src/triagem/data/` | download, build e heuristica de rotulo do dataset |
| `src/triagem/models/`, `src/triagem/training/` | pipeline do classificador, treino, avaliacao |
| `src/triagem/optimization/` | export ONNX e benchmark de latencia |
| `src/triagem/serving/` | API FastAPI, Predictor, metricas Prometheus |
| `dags/` | DAG Airflow de treino/retreino |
| `docker/`, `docker-compose.yml` | imagens, Prometheus, Grafana provisionados |
| `deploy/terraform/` | infraestrutura AWS como codigo (dry-run) |
| `scripts/` | medicao de latencia, geracao de carga |
| `docs/` | arquitetura, dataset, model card, latencia, monitoramento, Airflow |
| `.github/workflows/` | CI (lint + test), build Docker, validacao da DAG |
| `tests/` | testes automatizados, espelhando `src/` |

---

## Como rodar

Testado em Windows 11 com Python 3.12 + Poetry e Docker Desktop.

### 1. Setup local

```powershell
poetry install
poetry run pytest
```

O grupo opcional `airflow` **nao** e instalado por padrao (nao roda
nativamente no Windows); ele e usado apenas dentro do container Docker
do Airflow (`poetry install --with airflow`, so em Linux/CI/container).

### 2. Construir o dataset

```powershell
python -m triagem.data.download          # baixa data/raw/*.csv (corpus publico, CC BY-SA 3.0)
python -m triagem.data.build --seed 42    # gera data/processed/laudos.csv
```

### 3. Treinar e exportar o modelo

```powershell
poetry run python -m triagem.training.train
poetry run python -m triagem.optimization.onnx_export
```

Gera `models/model.joblib`, `models/model.onnx`, `models/metrics.json`,
`models/label_encoder.json`.

### 4. Rodar a API localmente (sem Docker)

```powershell
poetry run uvicorn triagem.serving.api:app --reload --port 8000
```

`GET http://localhost:8000/docs` abre o Swagger; `GET /health` confirma
o backend carregado.

### 5. Subir a stack completa (API + Prometheus + Grafana)

```powershell
docker build -f docker/Dockerfile.api -t triagem-api:local .
docker compose up -d
docker compose ps            # aguarde os 3 servicos ficarem "healthy"
```

- API: <http://localhost:8000>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (login `admin`/`admin` por padrao
  local -- ver `.env.example`), dashboard "Triagem API - Observabilidade"
  ja provisionado com 4 paineis.

Para popular os graficos com trafego real:

```powershell
python scripts/generate_load.py --n 300
```

Instrucoes detalhadas de leitura do dashboard em
[`docs/monitoring.md`](docs/monitoring.md).

### 6. Subir o Airflow (opcional, profile `airflow`)

```powershell
docker compose --profile airflow up -d
```

- UI: <http://localhost:8080> (login `admin`/`admin` por padrao local)
- Dispara `train_triagem` pela UI ou via
  `docker compose --profile airflow exec airflow airflow dags trigger train_triagem`.

Detalhes (volumes, validacao da DAG, credenciais) em
[`docs/airflow.md`](docs/airflow.md).

### 7. Qualidade de codigo

```powershell
poetry run ruff check .
poetry run black --check .
poetry run mypy src
poetry run pytest
```

Os mesmos comandos rodam em CI a cada `push`/pull request -- ver os
badges no topo deste README e `.github/workflows/`.

---

## API

Base local: `http://localhost:8000`.

| metodo | rota | descricao |
|---|---|---|
| `POST` | `/predict` | Classifica um texto: `{"text": "..."}` -> `{label, label_pt, scores, latency_ms, backend, model_version}` |
| `POST` | `/predict/batch` | Classifica ate 64 textos em uma chamada |
| `GET` | `/health` | Status de liveness/readiness, backend carregado |
| `GET` | `/model-info` | Metadados do modelo carregado + metricas de treino |
| `GET` | `/metrics` | Metricas Prometheus (texto, formato `0.0.4`) |
| `GET` | `/docs` | Swagger/OpenAPI |

Exemplo de resposta de `/predict`:

```json
{
  "label": "urgente",
  "label_pt": "urgente",
  "scores": {"normal": 0.07, "atencao": 0.21, "urgente": 0.72},
  "latency_ms": 0.11,
  "backend": "onnx",
  "model_version": "2026-09-20T00:10:37Z"
}
```

---

## Monitoramento

Instrumentacao via `prometheus_client`: contagem de requisicoes,
duracao por endpoint, predicoes por classe e erros por tipo, alem de
metadado do modelo carregado. O dashboard Grafana e provisionado por
arquivo (`docker/grafana/dashboards/triagem_api.json`) com 4 paineis:
taxa de requisicoes, latencia p50/p95, taxa de erro e predicoes por
classe. Guia completo de acesso e leitura em
[`docs/monitoring.md`](docs/monitoring.md).

---

## Configuracao

Todas as variaveis usam o prefixo `TRIAGEM_` e tem default funcional
(ver `src/triagem/config.py` e `.env.example`):

| variavel | default | uso |
|---|---|---|
| `TRIAGEM_DATA_PATH` | `data/processed/laudos.csv` | fonte do dataset (trocavel sem mudar codigo) |
| `TRIAGEM_MODELS_DIR` | `models` | diretorio dos artefatos treinados |
| `TRIAGEM_MODEL_BACKEND` | `onnx` | backend de inferencia (`sklearn` \| `onnx`) |
| `TRIAGEM_SEED` | `42` | reprodutibilidade |
| `TRIAGEM_API_PORT` | `8000` | porta do uvicorn |
| `TRIAGEM_LOG_LEVEL` | `INFO` | nivel de log |
| `TRIAGEM_DATASET_BASE_URL` | corpus publico | fonte do download |

`GRAFANA_ADMIN_USER`/`GRAFANA_ADMIN_PASSWORD` e
`AIRFLOW_ADMIN_USER`/`AIRFLOW_ADMIN_PASSWORD` controlam login local dos
respectivos paineis (ver `.env.example`); sem `.env`, caem no default
`admin`/`admin`, valido **apenas** para desenvolvimento local. Nenhuma
credencial real vive neste repositorio.

---

## Dataset e licenca

O dataset de treino vem do
[Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus),
licenciado sob **CC BY-SA 3.0** (redistribuido sem modificacao em
`data/raw/`, com atribuicao registrada em `data/raw/LICENSE_DATASET.md`
apos o download). O corpus original rotula **categoria clinica**, nao
urgencia; o rotulo de urgencia usado neste projeto e derivado por uma
heuristica de palavras-chave declarada em codigo
(`src/triagem/data/urgency.py`), nao pelo dataset original.

Documentacao completa (schema, volume, como reconstruir, como trocar
por um dataset real) em [`docs/dataset.md`](docs/dataset.md). Limitacoes
e uso pretendido do modelo em [`docs/model_card.md`](docs/model_card.md).
