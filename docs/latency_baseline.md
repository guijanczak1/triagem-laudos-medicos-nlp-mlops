# Baseline de latencia local em container

Baseline de latencia (PDF Etapa 1, tarefa T12): mede a inferencia in-process do
backend `sklearn` (T9) e a chamada HTTP ponta a ponta contra o container
`triagem-api:local` (T11), no mesmo hardware, com a mesma amostra de entrada e a
mesma quantidade de execucoes, com warm-up declarado.

## Metodologia

- Maquina: Windows-11-10.0.26200-SP0 (AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, 8 CPUs)
- Python 3.12.13 | scikit-learn 1.9.1 | fastapi 0.115.14 | numpy 2.5.3
- Gerado em: 2026-09-20T15:28:14.108969+00:00
- n = 200 execucoes medidas por alvo, warm-up = 20 chamadas descartadas antes da medicao
- Amostra de entrada com seed fixa (42), textos do dataset processado (`data/processed/laudos.csv`)

## Resultados (percentis em milissegundos)

| Alvo | n | warm-up | media | p50 | p90 | p95 | p99 | throughput (req/s) |
|---|---|---|---|---|---|---|---|---|
| in_process_sklearn | 200 | 20 | 0.34 | 0.32 | 0.42 | 0.47 | 0.73 | 2929.5 |
| http_container | 200 | 20 | 1.76 | 1.75 | 1.86 | 1.89 | 1.97 | 567.9 |

## Leitura honesta

- O alvo p95 end-to-end < 100 ms (PDF) e **informativo, nao bloqueante** -- o numero acima e reportado como medido, mesmo se estiver acima do alvo.
- Reportamos percentis (p50/p90/p95/p99), nao so a media: a media esconde a cauda, que e o que doi em producao.
- Numeros brutos desta execucao (JSON): `docs/benchmarks/baseline.json`.
- Este e o baseline (sklearn) da Etapa 1; a comparacao contra o backend ONNX otimizado (T20/T21) fica em `docs/latency_report.md`.

