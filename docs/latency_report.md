# Relatorio de latencia: baseline (sklearn) x ONNX

Comparacao formal de latencia entre o backend `sklearn` (baseline, T7) e o backend `onnx` (T20) do `Predictor` (T9), no mesmo hardware, com a mesma amostra de entrada e a mesma quantidade de execucoes, mais o teste de equivalencia de predicoes entre os dois backends no split de teste do projeto (T21).

## Metodologia

- Maquina: Windows-11-10.0.26200-SP0 (AMD64 Family 26 Model 68 Stepping 0, AuthenticAMD, 8 CPUs)
- Python 3.12.13 | scikit-learn 1.9.1 | onnxruntime 1.30.0 | skl2onnx 1.20.0 | numpy 2.5.3
- Gerado em: 2026-09-20T15:37:25.139591+00:00
- Seed: 42 (amostra de latencia e split de holdout deterministicos)
- Split de holdout: `split_dataset(test_size=0.2, seed=42)` sobre `data/processed/laudos.csv` -- o mesmo split que `triagem.training.train` usa para reportar `metrics.json` (T7) e que T20 usou para medir equivalencia (99.5152% em 2.888 linhas -- ver `models/model_onnx_meta.json`)
- Warm-up declarado por backend (descartado da medicao) + chamadas medidas individualmente via `Predictor.predict()` (mesmo caminho publico usado pela API)

## Latencia por backend (percentis em milissegundos)

| Backend | n | warm-up | media | p50 | p90 | p95 | p99 | throughput (req/s) |
|---|---|---|---|---|---|---|---|---|
| sklearn | 200 | 20 | 0.3250 | 0.3199 | 0.3736 | 0.4089 | 0.7492 | 3074.6 |
| onnx | 200 | 20 | 0.0990 | 0.0984 | 0.1207 | 0.1287 | 0.1529 | 10085.2 |

## Reducao de p50 (onnx sobre sklearn)

- Reducao medida: **69.24%** (meta: >= 20%) -- **ATINGIDA**.

## Equivalencia de predicoes sklearn x onnx

- Comparadas 2888 linhas do split de teste (holdout do seed acima): 2878 predicoes identicas.
- Equivalencia: **99.6537%** (meta: >= 99%) -- **ATINGIDA**.
- Esta e a mesma verificacao que T20 ja havia rodado sobre o split de teste completo (99.5152% em 2.888 linhas); o numero acima e medido de novo aqui, contra o artefato oficial deste projeto, como parte do entregavel formal desta tarefa.

## Leitura honesta

- Reportamos percentis (p50/p90/p95/p99), nao so a media: a media esconde a cauda, que e o que doi em producao.
- Quantizacao dinamica int8 foi avaliada em T20 e **descartada** -- nao trouxe ganho real de p50 neste modelo (ver `models/model_onnx_meta.json`: `quantization_check`). Este relatorio nao reaplica essa tentativa.
- Nenhum numero acima foi ajustado para bater a meta; se a reducao de p50 nao atingiu o alvo, isso esta reportado explicitamente acima, nao omitido.
- Numeros brutos desta execucao (JSON): `docs/benchmarks/comparison.json`.

