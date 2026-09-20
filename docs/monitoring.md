# Monitoramento: Prometheus + Grafana

Stack de observabilidade local (T17/T18/T19, PDF Etapa 3): a API expoe
`GET /metrics` (`triagem.serving.metrics`), o Prometheus faz scrape a cada
5s (`docker/prometheus/prometheus.yml`) e o Grafana ja sobe com o
datasource **e** o dashboard provisionados -- nenhum clique manual.

## Subir a stack

```powershell
docker build -f docker/Dockerfile.api -t triagem-api:local .
docker compose config      # valida o YAML resolvido
docker compose up -d       # sobe api (8000) + prometheus (9090) + grafana (3000)
```

Aguarde os 3 servicos ficarem `healthy`:

```powershell
docker compose ps
```

## Acessar o Grafana

- URL: <http://localhost:3000>
- Login: `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` (ver `.env.example`);
  sem um `.env` local, cai no default `admin` / `admin` (**apenas local** --
  nunca use esse default fora da maquina de desenvolvimento).
- Datasource **Prometheus** ja aparece em Connections -> Data sources, uid
  fixo `prometheus` (`docker/grafana/provisioning/datasources/prometheus.yml`).
- Dashboard **"Triagem API - Observabilidade"** ja aparece na home/lista de
  dashboards (pasta "General"), provisionado a partir de
  `docker/grafana/dashboards/triagem_api.json` via
  `docker/grafana/provisioning/dashboards/dashboard.yml`.

## Os 4 paineis do dashboard

| Painel | Query PromQL |
|---|---|
| Taxa de requisicoes (req/s) | `rate(triagem_requests_total[1m])` |
| Latencia (p50 / p95) | `histogram_quantile(0.50, sum(rate(triagem_request_duration_seconds_bucket[5m])) by (le))` e a mesma expressao com `0.95` |
| Taxa de erro | `sum(rate(triagem_errors_total[5m])) / sum(rate(triagem_requests_total[5m]))` |
| Predicoes por classe | `triagem_predictions_total` |

Os paineis 1, 2 e 4 usam exatamente as queries do backlog (T19). O range
padrao do dashboard e "ultimos 15 minutos", com auto-refresh de 5s.

**Nota sobre o painel de taxa de erro:** a query literal do backlog
(`rate(triagem_errors_total[5m]) / rate(triagem_requests_total[5m])`, sem
`sum()`) foi testada ao vivo contra a stack e **nunca produz series**:
`triagem_errors_total` tem labels `{endpoint,type}` e
`triagem_requests_total` tem labels `{endpoint,method,status}` -- o
Prometheus so casa vetores com o mesmo conjunto de labels na divisao, e como
os dois lados tem labels diferentes a divisao literal fica sempre vazia
(confirmado gerando um erro 422 real e consultando a API do Prometheus: o
contador subiu, a query literal continuou devolvendo 0 resultados). O
painel usa `sum(...)` dos dois lados antes de dividir -- mesmas duas
metricas, mesma janela de 5m, mas agora com labels compativeis -- para que
a taxa de erro realmente apareca no grafico quando houver erro.

## Gerar carga para ver os graficos

Um dashboard recem-provisionado comeca vazio (sem trafego, sem series). Para
popular os 4 paineis, dispare requisicoes reais contra `/predict` com
`scripts/generate_load.py` -- ele usa uma amostra deterministica com mix das
3 classes de urgencia (`normal`/`atencao`/`urgente`), nao so uma classe:

```powershell
python scripts/generate_load.py --n 300
```

O script espera `GET /health` responder 200 antes de comecar (timeout
configuravel via `--wait-timeout`), envia as requisicoes e imprime um resumo
(`ok`/`failed`/latencia media/throughput). Rodar contra outra instancia:

```powershell
python scripts/generate_load.py --n 100 --url http://127.0.0.1:8000
```

Depois de rodar, os 4 paineis devem mostrar series com dados dentro de
alguns segundos (scrape do Prometheus a cada 5s + refresh do dashboard a
cada 5s).

## Confirmar visualmente / via API

- **Visual**: abra <http://localhost:3000>, entre no dashboard "Triagem API
  - Observabilidade" e confira os 4 paineis com linhas nao-vazias.
- **Via API do Grafana** (sem abrir o navegador), com as credenciais do
  `.env`/default local:

  ```powershell
  curl -u admin:admin http://localhost:3000/api/dashboards/uid/triagem-api-observability
  ```

  Confirma que o dashboard foi provisionado (200 + JSON com os 4 `panels`).

- **Via API do Prometheus**, para confirmar que as series tem dados
  (sem depender do Grafana estar de pe):

  ```powershell
  curl "http://localhost:9090/api/v1/query?query=triagem_predictions_total"
  ```

## Onde salvar o print do dashboard

Se for anexar uma evidencia visual (ex.: para a entrega da Etapa 3), salve o
screenshot como `docs/screenshots/dashboard.png` (pasta a criar na hora --
nao versionada por padrao neste repo) e referencie o caminho no README ou no
relatorio de entrega. Nenhum print e obrigatorio neste documento: a
confirmacao "com dados" acima (visual ou via API) e o que comprova que o
dashboard funciona.

## Metricas expostas (contrato, T17)

Ver `src/triagem/serving/metrics.py` para a definicao exata. Resumo:

- `triagem_requests_total{endpoint,method,status}` (Counter)
- `triagem_request_duration_seconds{endpoint}` (Histogram)
- `triagem_predictions_total{label}` (Counter)
- `triagem_errors_total{endpoint,type}` (Counter)
- `triagem_inference_duration_seconds{backend}` (Histogram)
- `triagem_model_info{backend,model_version,model_type}` (Gauge, sempre 1)

Renomear qualquer uma delas exige atualizar
`docker/grafana/dashboards/triagem_api.json` no mesmo commit (regra dura de
`harness/observability.md`: metrica e contrato).
