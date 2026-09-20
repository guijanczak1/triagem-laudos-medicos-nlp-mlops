# Airflow local (Docker, profile `airflow`)

Orquestracao do pipeline de treino/retreino (`dags/train_triagem_dag.py`, T14:
`download_dataset >> build_dataset >> train_model >> evaluate_model >>
publish_artifacts`) via um container Airflow, opt-in dentro do mesmo
`docker-compose.yml` da stack de observabilidade (T18).

## Por que nao instalar o Airflow nativo (Windows)

`apache-airflow` fica **fora** do grupo principal de dependencias do Poetry
(`pyproject.toml`, grupo opcional `airflow`) porque nao instala nativamente
no Windows -- o pacote depende de `pwd`/`fcntl`, modulos que so existem em
sistemas POSIX. Ele so roda de verdade dentro da imagem Docker
(`docker/Dockerfile.airflow`, baseada em `apache/airflow:2.10.3-python3.12`)
ou em CI, num runner Ubuntu (T16). Em Windows, `poetry install` (sem
`--with airflow`) resolve normalmente; `poetry install --with airflow`
so deve ser usado dentro do container/CI.

## Subir o Airflow (Windows, PowerShell)

O profile `airflow` e **opt-in**: nunca sobe com um `docker compose up -d`
simples (que so traz `api` + `prometheus` + `grafana`, T18). Para subir o
Airflow **alem** disso, sem derrubar/reiniciar o que ja esta rodando:

```powershell
docker compose --profile airflow config      # valida o YAML resolvido
docker compose --profile airflow up -d       # builda (1a vez) e sobe tambem o airflow
```

Isso constroi `docker/Dockerfile.airflow` (tag `triagem-airflow:local`) e
inicia o container em modo `standalone` (webserver + scheduler + triggerer
no mesmo processo, metadata db sqlite) -- suficiente para desenvolvimento e
para validar a DAG localmente, sem precisar de um Postgres/Celery separado.

- UI: <http://localhost:8080>
- Login: `AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD` (ver `.env.example`);
  sem um `.env` local, cai no default `admin` / `admin` (**apenas local** --
  nunca use esse default fora da maquina de desenvolvimento).

Para derrubar so o Airflow, mantendo `api`/`prometheus`/`grafana` no ar:

```powershell
docker compose --profile airflow stop airflow
```

## Volumes montados

| Host | Container | Modo |
|---|---|---|
| `./dags` | `/opt/airflow/dags` | leitura/escrita (Airflow grava `.airflowignore`/cache aqui) |
| `./src` | `/opt/airflow/src` | somente leitura -- pacote `triagem` importado pelas tasks |
| `./data` | `/opt/airflow/data` | leitura/escrita -- `download_dataset`/`build_dataset` gravam aqui |
| `./models` | `/opt/airflow/models` | leitura/escrita -- `train_model` grava os artefatos |
| volume nomeado `airflow-home` | `/opt/airflow` (base) | metadata db (sqlite) + logs, persiste entre restarts |

Nenhum artefato e copiado para dentro da imagem: um `poetry run python -m
triagem.training.train` novo no host, ou uma edicao na DAG, e refletido no
container so reiniciando-o (sem rebuild da imagem) -- ver o comentario no
topo de `docker/Dockerfile.airflow`.

## Disparar a DAG

Pela UI (<http://localhost:8080> -> DAG `train_triagem` -> trigger manual),
ou pela CLI dentro do container:

```powershell
docker compose --profile airflow exec airflow airflow dags list
docker compose --profile airflow exec airflow airflow dags trigger train_triagem
```

`train_triagem` tem `schedule="@weekly"` e `catchup=False`, entao nao dispara
automaticamente varias execucoes retroativas ao ligar o container.

## Validar sem subir a UI inteira

Gate objetivo de import (T14 aplicado a este container):

```powershell
docker compose --profile airflow exec airflow airflow dags list-import-errors
```

Saida vazia (sem linhas) = a DAG importou sem erro. `airflow dags list` deve
listar `train_triagem` na coluna `dag_id`.

## Credenciais

`AIRFLOW_ADMIN_USER`/`AIRFLOW_ADMIN_PASSWORD` (login web) e
`GRAFANA_ADMIN_USER`/`GRAFANA_ADMIN_PASSWORD` (T18) vivem **apenas** num
`.env` local, nunca versionado (`.env.example` traz as chaves vazias, com o
default local documentado em comentario). A DAG em si nunca recebe
credencial como argumento literal (regra dura do papel Airflow) -- qualquer
credencial de infraestrutura real usaria Airflow Connections, nao Variables
nem env vars soltas.
