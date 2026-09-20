# Deploy AWS -- runbook

Este documento cobre o deploy real da stack definida em
[`deploy/terraform/`](../deploy/terraform/) (ECR + ECS Fargate + ALB +
CloudWatch Logs + IAM), decidida em
["Decisao arquitetural de deploy em nuvem"](../README.md#decisao-arquitetural-de-deploy-em-nuvem).

> **Nada neste repositorio cria recursos na AWS por conta propria.**
> `deploy/terraform/*.tf` foi validado apenas com `terraform fmt`,
> `terraform validate` e `terraform plan` (nenhum `apply`). O unico script
> que pode escrever na AWS e [`scripts/deploy_aws.ps1`](../scripts/deploy_aws.ps1),
> e ele se recusa a aplicar nada a menos que quem o executa peca duas vezes
> (veja "Passo a passo" abaixo). Se voce so quer avaliar o projeto, nao
> execute a etapa de deploy -- `docker compose up` (raiz do repo) sobe a
> API + Prometheus + Grafana localmente sem tocar em AWS.

## Pre-requisitos

Antes de tentar um deploy real, tenha em maos:

1. **Conta AWS** com permissao para criar: repositorio ECR, cluster/servico/
   task definition ECS Fargate, Application Load Balancer + target group +
   security groups, log group CloudWatch, roles e policies IAM. Um usuario/
   role com a policy gerenciada `AdministratorAccess` cobre tudo isso para
   fins de curso; em producao restrinja para as acoes especificas destes
   recursos.
2. **Credenciais AWS configuradas localmente** -- `aws configure` (access
   key + secret) ou um perfil SSO (`aws configure sso`). O script confere
   a identidade resolvida (`aws sts get-caller-identity`, chamada
   somente-leitura) antes de qualquer outra coisa.
3. **Ferramentas instaladas e no PATH**: `docker` (Desktop ou Engine),
   `terraform` >= 1.5, `aws` CLI v2, PowerShell 7+ (`pwsh`).
4. **`deploy/terraform/terraform.tfvars`** criado a partir de
   [`terraform.tfvars.example`](../deploy/terraform/terraform.tfvars.example)
   (copie e ajuste `project_name`/`aws_region` se necessario -- os defaults
   ja batem com o que este runbook assume).
5. **Imagem buildavel localmente**: `models/model.joblib` e
   `models/model.onnx` precisam existir antes do build (rode
   `poetry run python -m triagem.training.train` e
   `poetry run python -m triagem.optimization.onnx_export` primeiro -- ver
   README "Como rodar"). O Dockerfile copia esses artefatos para dentro da
   imagem; sem eles o build falha ou sobe uma imagem sem modelo.
6. **Orcamento aprovado** para o custo mensal estimado abaixo -- este e o
   item que falta ser confirmado pelo usuario (ver
   `state/mlet-tech-challenge-fase-3/cloud_request.md`, fora deste
   repositorio).

## Custo estimado mensal

Estimativa honesta com as tarifas publicas da AWS para `us-east-1`
(consultadas em setembro/2026), assumindo **730 horas/mes** (media usada
pela propria AWS em suas calculadoras) e a stack exata que
`deploy/terraform/` cria: 1 task Fargate 0.25 vCPU / 0.5 GB rodando 24/7,
1 ALB, 1 repositorio ECR, 1 log group CloudWatch com retencao de 7 dias.
**Nao ha free tier assumido** (o free tier de 12 meses so vale para contas
novas, e mesmo assim nao cobre Fargate nem ALB) -- exceto o "always free"
de 5 GB/mes do CloudWatch Logs, citado explicitamente abaixo.

| Item | Tarifa (us-east-1) | Calculo | Custo/mes |
|---|---|---|---|
| Fargate -- vCPU | US$ 0,04048 / vCPU-hora | 0,25 vCPU x 730h x 0,04048 | **US$ 7,39** |
| Fargate -- memoria | US$ 0,004445 / GB-hora | 0,5 GB x 730h x 0,004445 | **US$ 1,62** |
| ALB -- taxa fixa | US$ 0,0225 / hora | 730h x 0,0225 | **US$ 16,43** |
| ALB -- LCU (uso) | US$ 0,008 / LCU-hora | trafego de demo/curso, estimado < 1 LCU medio | **US$ 0,50 -- 5,84** |
| ECR -- storage | US$ 0,10 / GB-mes | imagem ~0,88 GB (testada em T11), 1-2 tags retidas | **US$ 0,09 -- 0,18** |
| CloudWatch Logs -- ingestao | US$ 0,50 / GB (ate 10 TB) | trafego de demo, estimado < 5 GB/mes | **US$ 0,00** (dentro do free tier de 5 GB) |
| CloudWatch Logs -- storage | US$ 0,03 / GB-mes | retencao curta (7 dias), volume residual | **~US$ 0,00 -- 0,10** |
| Data transfer -- saida internet | US$ 0,09 / GB (apos 1a GB) | trafego de demo, estimado < 1 GB/mes | **US$ 0,00 -- 0,09** |
| **Total** | | | **~US$ 26 -- 31/mes** |

**Premissas explicitas (para revisar/contestar):**

- **Uptime 24/7 do zero ao mes inteiro.** Nao ha auto-stop noturno/fim de
  semana configurado. Se o uso real for so "ligar para demonstrar e
  desligar", o custo cai proporcionalmente (ex.: 8h/dia x 5 dias ~= 173h/mes
  em vez de 730h reduz Fargate + ALB fixo para ~1/4 do valor acima, mas o
  ALB tem uma taxa fixa por hora *ligada*, entao desligar o Fargate sozinho
  sem destruir o ALB nao economiza a maior parte do custo).
- **1 unica task, sem auto-scaling** -- `desired_count = 1` no Terraform;
  nao ha custo de escala adicional projetado.
- **LCU dominado por trafego baixo** (curso/demo, nao producao real). O
  calculo de LCU usa a MAIOR das 4 dimensoes (novas conexoes, conexoes
  ativas, bytes processados, avaliacoes de regra) por hora -- com pouco
  trafego, tende a ficar bem abaixo de 1 LCU-hora a maior parte do tempo;
  o teto de ~US$ 5,84/mes assume 1 LCU sustentado o mes inteiro, o que e
  o pior caso realista, nao o esperado.
- **Nenhum dominio/certificado**: listener e HTTP puro na porta 80 (sem
  HTTPS/ACM, sem Route53) -- ver `cloud_request.md` para confirmar se isso
  e aceitavel ou se precisa de um dominio + certificado (adiciona custo de
  Route53 hosted zone ~US$ 0,50/mes; ACM publico e gratuito).
- **NAT Gateway: nenhum.** O stack usa as subnets publicas da VPC default
  com IP publico direto na task (`assign_public_ip = true`), evitando o
  custo de NAT Gateway (~US$ 32/mes so de taxa fixa, o que dobraria o
  orcamento). Trade-off documentado em `deploy/terraform/main.tf`.
- **Sem backend remoto de state** (S3 + DynamoDB) nesta fase dry-run --
  custo desprezivel (centavos) se ativado depois, nao incluido acima.
- **Nao inclui** o esforco humano de configurar/monitorar, nem custos de
  contas de suporte AWS, nem alarmes CloudWatch adicionais (nenhum foi
  criado no Terraform atual).

Se o orcamento aprovado for menor que ~US$ 26/mes, a stack como desenhada
(ECS Fargate + ALB) nao cabe -- o ALB sozinho ja consome ~US$ 16,43/mes so
de taxa fixa, independente de uso. Alternativas mais baratas (e os
trade-offs ja descartados na decisao arquitetural do README) seriam Lambda
atras de uma Function URL (sem ALB, mas cold start de segundos) ou nenhum
deploy publico (so `docker compose up` local). Ver
`state/mlet-tech-challenge-fase-3/cloud_request.md`.

## Passo a passo

Todos os comandos abaixo assumem PowerShell 7+ na raiz do repositorio
clonado (Windows ou qualquer OS com `pwsh`).

### 1. Dry run (seguro, pode ser feito a qualquer momento, nao precisa de autorizacao)

```powershell
copy deploy\terraform\terraform.tfvars.example deploy\terraform\terraform.tfvars
# ajuste deploy\terraform\terraform.tfvars se necessario (region, project_name)

poetry run python -m triagem.training.train --seed 42
poetry run python -m triagem.optimization.onnx_export

.\scripts\deploy_aws.ps1
```

Isso builda a imagem localmente, roda `terraform init` + `terraform plan`
e **para**. Nenhuma chamada de escrita na AWS acontece. Revise o plano
impresso (deve mostrar `13 to add, 0 to change, 0 to destroy`, conforme
validado em T24) e o banner de custo antes de prosseguir.

### 2. Deploy real (exige autorizacao explicita do usuario -- HARNESS SS6)

So execute esta etapa depois de:

- Ler `state/mlet-tech-challenge-fase-3/cloud_request.md` e fornecer tudo
  que ele pede (conta, regiao, orcamento aprovado, etc.);
- Decidir, de fato, seguir com o deploy real (nao apenas "pode continuar"
  dito em qualquer etapa anterior do projeto -- HARNESS SS6 e explicito
  que isso nao vale como autorizacao de deploy).

```powershell
.\scripts\deploy_aws.ps1 -WhatIf:$false
```

O script builda a imagem, roda `terraform plan` de novo, mostra o banner
de custo estimado e **pede para digitar a palavra `apply`** em um prompt
interativo. Qualquer outra resposta (Enter vazio, "y", "yes", Ctrl+C)
cancela sem nenhuma chamada de escrita na AWS. Só com a palavra exata
`apply` o script segue para `docker push` (login no ECR primeiro) e
`terraform apply` do plano ja revisado.

Ao final, o script imprime `terraform output`, incluindo `alb_dns_name`.
Teste com:

```powershell
curl http://<alb_dns_name>/health
curl -X POST http://<alb_dns_name>/predict -H "Content-Type: application/json" -d '{"text": "patient presents with acute severe chest pain"}'
```

A primeira task pode levar 1-2 minutos para passar no health check do ALB
(`healthy_threshold = 3`, `interval = 30s`) depois do primeiro deploy.

### 3. Atualizando uma imagem ja implantada

Rodar `.\scripts\deploy_aws.ps1 -WhatIf:$false` de novo builda, empurra a
nova imagem com a mesma tag (`latest` por padrao) e reaplica o Terraform
(sem mudanca de infraestrutura, só recria a task definition se algo no
`.tf` mudou). Se a tag da imagem nao mudou, force o ECS a puxar a imagem
nova:

```powershell
aws ecs update-service --cluster triagem-cluster --service triagem-api --force-new-deployment --region us-east-1
```

## Teardown

Desligar a stack e obrigatorio ao final de qualquer avaliacao/demo real --
o custo roda enquanto os recursos existirem, mesmo sem trafego (ver
"Custo estimado mensal" acima: ALB e Fargate cobram por hora ligados, nao
por requisicao).

```powershell
cd deploy\terraform
terraform plan -destroy    # revise o que sera destruido antes de confirmar
terraform destroy
```

`terraform destroy` pede confirmacao interativa (`yes`) por padrao -- nao
ha flag `-auto-approve` em nenhum comando deste runbook. Confira depois:

```powershell
aws ecr describe-repositories --repository-names triagem-api --region us-east-1
aws ecs describe-clusters --clusters triagem-cluster --region us-east-1
```

Ambos devem retornar "nao encontrado" (ou cluster com `status: INACTIVE` e
0 servicos) apos o destroy. Se a imagem ficou no ECR e o repositorio nao
foi esvaziado antes do destroy, `terraform destroy` pode falhar em apagar
o repositorio ECR (ECR nao deixa apagar repo com imagens via Terraform por
padrao) -- nesse caso:

```powershell
aws ecr batch-delete-image --repository-name triagem-api --region us-east-1 --image-ids imageTag=latest
terraform destroy
```

## Troubleshooting rapido

- **`terraform plan` mostra mudanca inesperada em recurso que ja existia**:
  alguem rodou `apply` fora deste script (ex.: console AWS manualmente).
  Rode `terraform plan` e leia o diff antes de aplicar qualquer coisa.
- **Task fica `UNHEALTHY` no target group**: confira
  `aws logs tail /ecs/triagem-api --follow --region us-east-1` -- causa
  mais comum e falta de `models/model.onnx`/`models/model.joblib` na
  imagem buildada (ver pre-requisito 5).
- **`docker push` falha com 403**: o login do ECR expira apos 12h; rode o
  script de novo (ele refaz o login a cada execucao) em vez de reusar uma
  sessao antiga de `docker login`.
