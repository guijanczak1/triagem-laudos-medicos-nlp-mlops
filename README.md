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

## CI (GitHub Actions)

Workflow em `.github/workflows/ci.yml`, disparado em todo `push` e em
`pull_request` para `main`. Dois jobs, com os mesmos comandos rodados
localmente (Poetry, Python 3.12, cache de dependencias):

- **`lint`** — `poetry run ruff check .`, `poetry run black --check .` e
  `poetry run mypy src`.
- **`test`** — `poetry run pytest` (cobertura minima de 80% e a marca
  `network` deselecionada ja vem de `[tool.pytest.ini_options]` no
  `pyproject.toml`); o relatorio de cobertura (`coverage.xml`) fica
  disponivel como artifact do job na aba **Actions** do repositorio.

Resultado de cada execucao: aba **Actions** do GitHub, no run correspondente
ao commit/PR. Um job vermelho bloqueia o merge visualmente (nenhum gate usa
`continue-on-error`).
