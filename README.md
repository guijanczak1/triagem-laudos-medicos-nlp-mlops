# Triagem de Laudos Medicos com NLP

Assistente de triagem de urgencia para laudos/abstracts clinicos, baseado em
NLP, com pipeline de MLOps (treino, servico de inferencia e monitoramento).

> Projeto em construcao. Este README e um placeholder minimo da fase de
> scaffold inicial e sera substituido por uma versao completa (instrucoes de
> execucao, decisao de arquitetura de nuvem, metricas e badges) assim que as
> demais etapas do projeto forem concluidas.

## Estrutura

Codigo em `src/triagem/`, testes em `tests/`.

## Setup local (Windows, Python 3.12, Poetry)

```powershell
poetry install
poetry run pytest
```

O grupo opcional `airflow` nao e instalado por padrao (nao roda nativamente no
Windows); ele e usado apenas dentro do container Docker do Airflow.
