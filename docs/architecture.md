# Arquitetura

Este documento descreve a arquitetura tecnica do sistema: componentes,
fluxo de dados e as decisoes de design que sustentam o comportamento em
producao. A decisao de **implantacao em nuvem** (batch vs. real-time,
servico escolhido) esta no `README.md`, secao "Decisao arquitetural de
deploy em nuvem" -- aqui o foco e como o sistema funciona, nao onde ele
roda.

## Visao geral

O sistema tem duas responsabilidades desacopladas:

1. **Treino** (offline, em lote): baixa o corpus publico, deriva o rotulo
   de urgencia por heuristica, treina o classificador e publica o
   artefato. Roda sob demanda ou semanalmente via Airflow.
2. **Inferencia** (online, tempo real): recebe texto de um laudo/abstract
   via HTTP e devolve a classe de urgencia com score de confianca. Roda
   continuamente atras da API FastAPI.

As duas trocam apenas artefatos em disco (`models/model.joblib`,
`models/model.onnx`, `models/metrics.json`) -- nenhuma chamada direta
entre elas.

## Diagrama de componentes e fluxo de dados

```
              Medical Abstracts TC Corpus (CC BY-SA 3.0, GitHub raw)
                               |
                               | download_raw()
                               v
                     data/raw/*.csv (copia bruta)
                               |
                               | build_dataset() + map_urgency()
                               v
              data/processed/laudos.csv  (colunas: text, label, ...)
                               |
                 +-------------+--------------+
                 |                            |
                 v                            v
     DAG Airflow "train_triagem"     load_dataset() / split_dataset()
     (semanal, batch):                        |
     download -> build -> train ->            v
     evaluate -> publish            build_pipeline(): TF-IDF + LogisticRegression
                 |                             |
                 |                             | train() + evaluate()
                 |                             v
                 |               models/model.joblib + metrics.json
                 |                             |
                 |                             | export_to_onnx()
                 |                             v
                 |                     models/model.onnx
                 |                             |
                 +---------------> Predictor.load(backend="onnx"|"sklearn")
                                               |
                                               v
                                 FastAPI (Docker, porta 8000)
                                 /predict  /predict/batch
                                 /health  /model-info  /metrics
                                               |
                                               | scrape a cada 5s
                                               v
                                       Prometheus (porta 9090)
                                               |
                                               v
                                    Grafana (porta 3000)
                                 dashboard provisionado, 4 paineis
```

## Componentes

### Pipeline de dados (`src/triagem/data/`)

- `download.py` -- baixa os CSVs do corpus publico (idempotente, valida
  schema do cabecalho antes de gravar).
- `urgency.py` -- heuristica **pura e deterministica** que deriva o
  rotulo `normal | atencao | urgente` a partir de lexico clinico +
  categoria da condicao (prior de baixo peso). Cortes de quantil
  ajustados apenas no split de treino, reaplicados sem vazamento no
  split de teste. Documentada em detalhe, com justificativa, em
  `docs/model_card.md`.
- `build.py` -- funcao pura que combina as duas etapas acima em
  `data/processed/laudos.csv`.
- `loaders.py` -- unico ponto de leitura do dataset processado
  (`load_dataset`); trocar `TRIAGEM_DATA_PATH` por outro CSV com as
  mesmas colunas troca a fonte de dados sem alterar codigo.

### Modelo (`src/triagem/models/`, `src/triagem/training/`)

`TfidfVectorizer` (bigrams, `max_features=20000`) alimentando uma
`LogisticRegression` (`C=0.1`, `class_weight="balanced"`). A escolha de
um classificador linear em vez de uma arvore/floresta veio de um
diagnostico concreto: o rotulo e uma funcao aproximadamente linear das
features de entrada (contagem de termos + prior de categoria), e um
`RandomForestClassifier` testado como baseline nao reconstroi bem esse
tipo de fronteira sobre ~20 mil dimensoes esparsas de TF-IDF -- a troca
de familia de modelo, sozinha, ja levantou as metricas reais de forma
mensuravel. Numeros completos, incluindo as duas rodadas de ajuste que
levaram a esse pipeline, estao em `docs/model_card.md`.

`training/evaluate.py` calcula macro-F1, recall por classe e matriz de
confusao; o gate de qualidade (`QualityGate`) falha o treino (e a task
`evaluate_model` da DAG) se as metricas ficarem abaixo do minimo
configurado.

### Otimizacao de latencia (`src/triagem/optimization/`)

- `onnx_export.py` -- exporta o pipeline sklearn treinado para ONNX via
  `skl2onnx`, preservando as mesmas etapas (vetorizacao + classificacao)
  num unico grafo.
- `benchmark.py` -- mede p50/p90/p95/p99 e throughput de cada backend
  sobre a mesma amostra determinista, com warm-up declarado, e calcula
  equivalencia de predicao entre os dois backends no split de teste.
  Resultados reais em `docs/latency_report.md`.

### Predictor (`src/triagem/serving/predictor.py`)

Abstracao unica com **a mesma assinatura** para os dois backends
(`sklearn` e `onnx`) -- e isso que permite trocar o backend em producao
por variavel de ambiente, comparar latencia entre eles e testar
equivalencia de saida sem duplicar nenhuma logica de API. O backend
`onnx` nunca cai silenciosamente para `sklearn`: se o artefato pedido
nao existe, a chamada falha alto (`ModelArtifactNotFound`). O unico
lugar com fallback e o startup da API (proximo item), e ele e sempre
logado.

### API (`src/triagem/serving/api.py`)

FastAPI com o modelo carregado uma unica vez no `lifespan` (nunca por
requisicao). Backend padrao: `onnx` (mais rapido, ver resultados no
README); se `models/model.onnx` estiver ausente na subida, cai para
`sklearn` com um `WARNING` explicito no log e reflete isso em
`/health`/`/model-info`/na metrica `triagem_model_info` -- nunca de
forma silenciosa. Rotas: `POST /predict`, `POST /predict/batch`,
`GET /health`, `GET /model-info`, `GET /metrics`, `GET /docs`.

### Observabilidade (`src/triagem/serving/metrics.py`, `docker/prometheus/`, `docker/grafana/`)

Instrumentacao via `prometheus_client` (contadores de requisicao/erro,
histograma de duracao, contador de predicoes por classe, gauge de
metadado do modelo). Prometheus faz scrape de `/metrics` a cada 5s;
Grafana sobe com datasource e dashboard (4 paineis: requisicoes/s,
latencia p50/p95, taxa de erro, predicoes por classe) provisionados por
arquivo -- nenhum clique manual necessario. Detalhes operacionais em
`docs/monitoring.md`.

### Orquestracao (`dags/train_triagem_dag.py`)

DAG linear (`download_dataset >> build_dataset >> train_model >>
evaluate_model >> publish_artifacts`), cada task chamando uma funcao de
`src/triagem` -- nenhuma logica de treino vive na DAG. Roda apenas via
Docker (`docker compose --profile airflow up`), porque `apache-airflow`
nao instala nativamente no Windows. Detalhes em `docs/airflow.md`.

## Decisoes de design que sustentam a arquitetura

- **Predictor com assinatura unica para os dois backends** -- unico jeito
  de comparar latencia e equivalencia de saida sem duas implementacoes de
  API divergentes.
- **Fallback de backend apenas no startup da API, sempre logado** -- o
  `Predictor` em si nunca mascara um artefato ausente; a API decide, de
  forma visivel, se prefere degradar para `sklearn` a nao subir.
- **DAG sem logica de treino embutida** -- a mesma funcao de treino usada
  pela DAG e usada por quem roda `poetry run python -m
  triagem.training.train` manualmente; nao existem dois caminhos de
  treino divergentes.
- **Rotulo de urgencia como heuristica versionada em codigo, nao em
  dados** -- torna a derivacao auditavel e reproduzivel, mas nao a
  transforma em verdade clinica; ver o aviso em `docs/model_card.md`.
- **Serving em tempo real, treino em lote** -- a mesma separacao de
  responsabilidade que justifica a escolha de nuvem no README: a
  inferencia precisa responder no momento do atendimento, o retreino
  tolera uma janela semanal.
