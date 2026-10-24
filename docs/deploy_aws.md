# Deploy AWS -- runbook

Este documento cobre o deploy real da stack definida em
[`deploy/terraform/`](../deploy/terraform/) (ECR + AWS Lambda em imagem de
container + API Gateway HTTP API + CloudWatch Logs + IAM), decidida na
subsecao ["Instancia real implantada"](../README.md#decisao-arquitetural-de-deploy-em-nuvem)
de "Decisao arquitetural de deploy em nuvem". **Esta stack ja foi aplicada
de verdade** contra a conta AWS Academy Learner Lab do autor do projeto, sob
autorizacao explicita (HARNESS SS6) -- os detalhes e achados reais (nao
hipoteticos) estao documentados abaixo.

> `deploy/terraform/*.tf` e `scripts/deploy_aws.ps1` nao criam nada na AWS
> por conta propria -- toda chamada de escrita exige que quem executa o
> script digite a palavra `apply` num prompt interativo (veja "Passo a
> passo"). Se voce so quer avaliar o projeto sem tocar em AWS, `docker
> compose up` (raiz do repo) sobe a API + Prometheus + Grafana localmente.

## Por que Lambda aqui, e nao ECS Fargate + ALB

O README documenta a decisao arquitetural *geral* (producao, conta sem
restricao de orcamento) como **ECS Fargate + ALB**, e lista Lambda entre as
alternativas descartadas -- principalmente pelo cold start de scikit-learn/
ONNX Runtime. Essa analise continua valida para esse cenario.

A conta real usada neste projeto, porem, e um **AWS Academy Learner Lab**:
orcamento fixo de ~US$50 para o curso inteiro (nao mensal), com sessoes de
4 horas apos as quais recursos como EC2 sao encerrados automaticamente --
mas o ALB, especificamente, continua cobrando entre sessoes enquanto nao for
destruido manualmente (a propria documentacao da AWS Academy recomenda
apagar e recriar recursos assim por sessao). Diante disso:

- **Lambda (imagem de container) + API Gateway HTTP API** nao tem taxa fixa
  por hora -- cobra por requisicao, dentro do free tier perpetuo da Lambda
  (API Gateway HTTP API cobra US$ 1,00/milhao de requisicoes, irrelevante
  no volume de uma demo). Custo real medido nesta conta: ver "Custo estimado
  mensal" abaixo.
- A objecao de tamanho de pacote (limite de 250 MB para deploy via zip) que
  o README cita **nao se aplica** a este deploy: `deploy/terraform/main.tf`
  usa `package_type = "Image"`, que aceita imagens de ate 10 GB vindas do
  ECR -- a mesma imagem Docker testada localmente, sem reescrita.
- A objecao que **continua valida** e o cold start (import de scikit-learn/
  ONNX Runtime + leitura do artefato do modelo). Ele e aceito aqui
  deliberadamente como o trade-off certo para uma conta de estudante.

Se este projeto fosse implantado numa conta de producao sem a restricao de
orcamento do Academy, a recomendacao do README (ECS Fargate + ALB)
continuaria sendo a escolha certa -- a mudanca para Lambda e especifica
deste deploy, nao uma correcao da analise geral.

### Por que API Gateway, e nao uma Lambda Function URL direta

O desenho original desta subsecao era uma Function URL simples
(`authorization_type = "NONE"`) -- um recurso a menos que API Gateway, sem
custo proprio. Aplicado de verdade contra a conta AWS Academy Learner Lab,
**a invocacao anonima da Function URL retornou `403 Forbidden`**, mesmo com
`AuthType = NONE` e a resource policy correta (`lambda:InvokeFunctionUrl`
para `Principal "*"`, confirmada via `aws lambda get-policy`). Isolando a
causa: uma chamada **assinada (SigV4)** para a mesma URL (com
`authorization_type` temporariamente mudado para `AWS_IAM`) retornou `200`
-- ou seja, a funcao e a policy estavam corretas; o bloqueio e especifico a
acesso **anonimo** via Function URL. E quase certamente uma guarda de
Organizations do proprio AWS Academy contra Function URLs publicas
(vetor de abuso comum em contas de sandbox compartilhadas), nao um erro de
configuracao deste projeto.

API Gateway invoca a Lambda via `lambda:InvokeFunction` a partir do
principal de servico `apigateway.amazonaws.com` -- uma acao e um principal
diferentes do `lambda:InvokeFunctionUrl` + `Principal "*"` que foi
bloqueado -- e por isso nao cai na mesma restricao. Confirmado: `curl` sem
nenhuma credencial contra o endpoint do API Gateway retornou `200` em
`/health`, `/predict`, `/metrics` e `/docs`.

## Pre-requisitos

Antes de tentar um deploy real, tenha em maos:

1. **Conta AWS** com permissao para criar: repositorio ECR, funcao Lambda
   (imagem de container), API Gateway HTTP API, log group CloudWatch e,
   dependendo da conta, role/policy IAM (ver item 6 abaixo sobre AWS
   Academy).
2. **Credenciais AWS configuradas localmente** -- `aws configure` (access
   key + secret + **session token**, obrigatorio em contas AWS Academy
   Learner Lab) ou um perfil SSO. O script confere a identidade resolvida
   (`aws sts get-caller-identity`, chamada somente-leitura) antes de
   qualquer outra coisa.
3. **Ferramentas instaladas e no PATH**: `docker` (Desktop ou Engine, com
   `docker buildx`/BuildKit -- padrao em instalacoes recentes),
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
6. **Se a conta e um AWS Academy Learner Lab**: criacao de IAM role/policy
   costuma ser bloqueada no sandbox -- confirmado nesta conta real
   (`iam:GetRole`/`iam:ListPolicyVersions` retornam `AccessDenied` com
   "explicit deny" nas policies `Pvoclabs1`/`Pvoclabs2` anexadas ao papel do
   aluno). Preencha `execution_role_arn` em `terraform.tfvars` com o ARN da
   `LabRole` pre-existente da conta (ex.:
   `arn:aws:iam::<account_id>:role/LabRole` -- confirme com
   `aws iam get-role --role-name LabRole`, chamada somente-leitura). Sem
   isso, `terraform apply` falhara ao tentar criar
   `aws_iam_role.lambda_execution`.
7. **Orcamento aprovado** -- ver "Custo estimado mensal" abaixo; este e o
   item que falta ser confirmado pelo usuario (ver
   `state/mlet-tech-challenge-fase-3/cloud_request.md`, fora deste
   repositorio).

## Custo estimado mensal

Estimativa honesta com as tarifas publicas da AWS (consultadas em
outubro/2026) para `us-east-1`, assumindo a stack exata que
`deploy/terraform/` cria: 1 funcao Lambda (imagem de container, 2048 MB),
1 API Gateway HTTP API, 1 repositorio ECR, 1 log group CloudWatch com
retencao de 7 dias.

| Item | Tarifa (us-east-1) | Calculo (demo/curso) | Custo/mes |
|---|---|---|---|
| Lambda -- requisicoes | US$ 0,20 / 1M requisicoes, **1M gratis/mes (always free, nao expira)** | demo/curso, muito abaixo de 1M/mes | **US$ 0,00** |
| Lambda -- duracao | US$ 0,0000166667 / GB-s, **400.000 GB-s gratis/mes (always free)** | poucas centenas de invocacoes/mes a 2 GB x ~1-3 s | **US$ 0,00** |
| API Gateway HTTP API | US$ 1,00 / 1M requisicoes (sem tier gratis permanente garantido nesta conta) | poucas centenas de requisicoes/mes para demo/avaliacao | **~US$ 0,00** (fracao de centavo) |
| ECR -- storage | US$ 0,10 / GB-mes | imagem ~0,9 GB, 1 tag retida | **US$ 0,09** |
| CloudWatch Logs -- ingestao | US$ 0,50 / GB (ate 10 TB), **5 GB gratis/mes (always free)** | trafego de demo, estimado < 1 GB/mes | **US$ 0,00** |
| CloudWatch Logs -- storage | US$ 0,03 / GB-mes | retencao curta (7 dias), volume residual | **~US$ 0,00 -- 0,05** |
| Data transfer -- saida internet | US$ 0,09 / GB (apos 1a GB gratis) | trafego de demo, estimado < 1 GB/mes | **US$ 0,00** |
| **Total** | | | **~US$ 0,09 -- 0,15/mes** |

**Premissas explicitas (para revisar/contestar):**

- **Trafego de demo/curso, nao producao real.** O always-free tier da
  Lambda (1M requisicoes + 400.000 GB-s por mes) e o custo por requisicao do
  API Gateway cobrem com folga qualquer volume razoavel de
  avaliacao/demonstracao. Se este projeto virasse producao com trafego real
  sustentado, o calculo mudaria -- ver a comparacao com Fargate+ALB no
  README.
- **Sem concorrencia provisionada.** `deploy/terraform/main.tf` nao reserva
  nem provisiona concorrencia -- cada cold start apos periodo ocioso custa
  alguns segundos de latencia extra, nao dinheiro extra.
- **ECR e o maior item de custo residual**, e e proporcional ao numero de
  tags/imagens retidas, nao ao tempo. Apagar as tags antigas
  (`aws ecr batch-delete-image`) ou o repositorio inteiro no teardown zera
  esse valor tambem.
- **Nenhum dominio/certificado**: o endpoint padrao do API Gateway ja e
  HTTPS (certificado gerenciado pela AWS), sem custo de ACM/Route53
  adicional.
- **Sem backend remoto de state** (S3 + DynamoDB) -- custo desprezivel
  (centavos) se ativado depois, nao incluido acima.
- **Nao inclui** o esforco humano de configurar/monitorar, nem custos de
  contas de suporte AWS.

Esta e uma mudanca deliberada em relacao a decisao arquitetural *geral* do
README (ECS Fargate + ALB, ~US$26-31/mes) -- ver "Por que Lambda aqui, e nao
ECS Fargate + ALB" acima para o raciocinio completo.

## Passo a passo

Todos os comandos abaixo assumem PowerShell 7+ na raiz do repositorio
clonado (Windows ou qualquer OS com `pwsh`).

### 1. Dry run (seguro, pode ser feito a qualquer momento, nao precisa de autorizacao)

```powershell
copy deploy\terraform\terraform.tfvars.example deploy\terraform\terraform.tfvars
# ajuste deploy\terraform\terraform.tfvars se necessario (region, project_name,
# execution_role_arn se a conta for AWS Academy -- ver pre-requisito 6)

poetry run python -m triagem.training.train --seed 42
poetry run python -m triagem.optimization.onnx_export

.\scripts\deploy_aws.ps1
```

Isso builda a imagem localmente (sem provenance/SBOM attestations -- ver
nota no proprio script, necessario para a Lambda aceitar a imagem), roda
`terraform init` + `terraform plan` e **para**. Nenhuma chamada de escrita
na AWS acontece. Revise o plano impresso e o banner de custo antes de
prosseguir.

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
`apply` o script segue em **duas fases** (necessario porque uma funcao
Lambda com `package_type = "Image"` so pode ser criada depois que a imagem
ja existe na tag que ela referencia):

1. `terraform apply -target=...` cria so o repositorio ECR + o log group do
   CloudWatch.
2. `docker push` manda a imagem para esse repositorio.
3. `terraform apply` (sem target) cria a funcao Lambda, o API Gateway e a
   permissao de invocacao.

Ao final, o script imprime `terraform output`, incluindo `api_url`. Teste
com:

```powershell
curl <api_url>health
curl -X POST <api_url>predict -H "Content-Type: application/json" -d '{"text": "patient presents with acute severe chest pain"}'
```

A primeira requisicao apos a criacao (ou apos um periodo ocioso) e um cold
start -- pode levar alguns segundos a mais que as seguintes.

### 3. Atualizando uma imagem ja implantada

Rodar `.\scripts\deploy_aws.ps1 -WhatIf:$false` de novo builda e empurra a
nova imagem com a mesma tag (`latest` por padrao). Diferente do ECS, o
Lambda **nao detecta automaticamente** que o digest atras da mesma tag
mudou -- force a atualizacao:

```powershell
aws lambda update-function-code --function-name triagem-api --image-uri <ecr_repository_url>:latest --region us-east-1
```

(`<ecr_repository_url>` vem de `terraform output ecr_repository_url`.)

## Teardown

Desligar a stack elimina o unico custo residual real desta stack (imagens
retidas no ECR); a funcao Lambda e o API Gateway nao cobram nada enquanto
ociosos.

```powershell
cd deploy\terraform
terraform plan -destroy    # revise o que sera destruido antes de confirmar
terraform destroy
```

`terraform destroy` pede confirmacao interativa (`yes`) por padrao -- nao
ha flag `-auto-approve` em nenhum comando deste runbook. Confira depois:

```powershell
aws ecr describe-repositories --repository-names triagem-api --region us-east-1
aws lambda get-function --function-name triagem-api --region us-east-1
```

Ambos devem retornar "nao encontrado" apos o destroy. Se a imagem ficou no
ECR e o repositorio nao foi esvaziado antes do destroy, `terraform destroy`
pode falhar em apagar o repositorio ECR (ECR nao deixa apagar repo com
imagens via Terraform por padrao) -- nesse caso:

```powershell
aws ecr batch-delete-image --repository-name triagem-api --region us-east-1 --image-ids imageTag=latest
terraform destroy
```

## Troubleshooting rapido

- **`terraform apply` falha criando `aws_iam_role.lambda_execution`**:
  conta AWS Academy sem permissao para criar IAM roles. Preencha
  `execution_role_arn` em `terraform.tfvars` com o ARN de uma role
  existente (ex.: `LabRole`) e rode `terraform plan`/`apply` de novo (ver
  pre-requisito 6).
- **`terraform apply` falha criando `aws_lambda_function.api` com
  "the image manifest, config or layer media type ... is not supported"**:
  a imagem foi buildada com `docker buildx` incluindo attestations de
  provenance/SBOM (padrao em versoes recentes do Docker Desktop), que o
  Lambda rejeita. Rebuilde com `--provenance=false --sbom=false` (ja
  presente em `scripts/deploy_aws.ps1`) e faca o push de novo antes de
  tentar o apply.
- **`terraform apply` falha criando `aws_lambda_function.api` com
  "image ... not found" / `ImageNotFoundException`**: a imagem ainda nao
  foi empurrada para a tag que `container_image_tag` referencia, ou o ECR
  repo ainda nao existe. Siga a ordem de duas fases descrita em "Passo a
  passo" (ECR+logs -> push -> Lambda+API Gateway).
- **`curl <api_url>health` retorna `403 Forbidden`**: isso aconteceria se
  voce tivesse revertido para uma Lambda Function URL com
  `authorization_type = "NONE"` -- bloqueada para acesso anonimo nesta
  conta AWS Academy (ver "Por que API Gateway" acima). O stack atual usa
  API Gateway, que nao tem esse problema; se mesmo assim houver 403, confira
  `aws apigatewayv2_api`/rota/integracao com `terraform state show`.
- **`aws lambda get-function` mostra `LastUpdateStatus: Failed`**: confira
  `aws logs tail /aws/lambda/triagem-api --follow --region us-east-1` --
  causa mais comum e falta de `models/model.onnx`/`models/model.joblib` na
  imagem buildada (ver pre-requisito 5), que faz o container falhar o
  readiness check (`AWS_LWA_READINESS_CHECK_PATH=/health`) na inicializacao.
- **Primeira requisicao apos deploy/idle demora varios segundos, ou a
  invocacao direta falha com `Sandbox.Timedout`**: cold start de
  scikit-learn/ONNX Runtime pode passar de 30s dependendo da memoria
  alocada -- `lambda_memory_mb`/`lambda_timeout_seconds` em
  `terraform.tfvars` (defaults: 2048 MB / 60s) existem exatamente para
  isso; suba ainda mais se necessario.
- **`docker push` falha com 403**: o login do ECR expira apos 12h (ou
  antes, em sessoes AWS Academy Learner Lab); rode o script de novo (ele
  refaz o login a cada execucao) em vez de reusar uma sessao antiga de
  `docker login`.
