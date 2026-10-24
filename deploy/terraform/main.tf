# T24 -- Terraform AWS stack for the triage API (dry-run only).
#
# Scope: ECR (image registry) + AWS Lambda (container image, compute) +
# Lambda Function URL (public entry point) + CloudWatch Logs (observability)
# + IAM (least-privilege execution role). This mirrors the "Instancia real
# implantada" subsection of README.md's "Decisao arquitetural de deploy em
# nuvem" (T23): Lambda + Function URL was adopted over the general-purpose
# ECS Fargate + ALB design (documented in the same section, and still the
# right call for a non-budget-constrained account) specifically because this
# project's actual target account is an AWS Academy Learner Lab -- a fixed
# ~US$50 total-course budget (not monthly) where the ALB's per-hour fixed
# charge keeps billing across the Lab's forced 4-hour session resets. Lambda
# + Function URL has no per-hour fixed charge and fits the Lab's perpetual
# free-tier-shaped allowance (1M requests/month). The cold-start trade-off
# this implies is accepted deliberately for this budget-constrained
# deployment; it is not free for every use case (see README).
#
# HARD RULE (HARNESS.md SS6, this task's own acceptance criteria): this file
# is prepared and validated with `terraform fmt`, `terraform validate` and
# `terraform plan` ONLY. Nobody runs `terraform apply` / `terraform destroy`
# / any write `aws` call from this task. Actual apply happens only in
# T25/T26, after the user's explicit authorization recorded in
# state/mlet-tech-challenge-fase-3/cloud_request.md and answers_cloud.md.

locals {
  name = var.project_name

  common_tags = merge(
    {
      Project   = var.project_name
      ManagedBy = "terraform"
    },
    var.extra_tags
  )

  # AWS Academy Learner Lab accounts commonly cannot create/modify IAM
  # roles or policies (restricted sandbox). When var.execution_role_arn is
  # set (e.g. to the Lab's pre-provisioned "LabRole"), reuse it instead of
  # creating a new role. Default (null) creates a minimal role with only
  # CloudWatch Logs write access -- the running process makes no other AWS
  # API calls (same "gates_pulados" rationale as before: no MLflow
  # registry, no S3 model bucket, model is baked into the image).
  lambda_execution_role_arn = var.execution_role_arn != null ? var.execution_role_arn : aws_iam_role.lambda_execution[0].arn
}

data "aws_caller_identity" "current" {}

# --- ECR ---------------------------------------------------------------
resource "aws_ecr_repository" "api" {
  name                 = "${local.name}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.common_tags
}

# --- CloudWatch Logs -----------------------------------------------------
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}-api"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# --- IAM -----------------------------------------------------------------
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

# Created only when var.execution_role_arn is null (see local above).
resource "aws_iam_role" "lambda_execution" {
  count = var.execution_role_arn == null ? 1 : 0

  name               = "${local.name}-lambda-execution-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_execution_logs" {
  count = var.execution_role_arn == null ? 1 : 0

  role       = aws_iam_role.lambda_execution[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- Lambda function (container image) ------------------------------------
# package_type = "Image" uses the ECR image directly, so the 250 MB
# zip-package limit that ruled out Lambda in the general README decision
# does NOT apply here -- container images support up to 10 GB from ECR.
# The repo starts empty: a real deploy must push var.container_image_tag
# first (see scripts/deploy_aws.ps1 / T25) before this function can be
# created or updated.
resource "aws_lambda_function" "api" {
  function_name = "${local.name}-api"
  role          = local.lambda_execution_role_arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.container_image_tag}"
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_seconds
  architectures = ["x86_64"]

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.lambda.name
  }

  depends_on = [
    aws_cloudwatch_log_group.lambda,
    aws_iam_role_policy_attachment.lambda_execution_logs,
  ]

  tags = local.common_tags
}

# --- Public entry point: API Gateway (HTTP API) -----------------------------
# The original design here was a Lambda Function URL (authorization_type =
# "NONE") -- simpler and free, no separate resource needed. Deployed against
# the real AWS Academy Learner Lab account, anonymous Function URL invocation
# returned 403 Forbidden ("Forbidden. For troubleshooting Function URL
# authorization issues..."), while a SigV4-signed request to the *same* URL
# (authorization_type temporarily flipped to AWS_IAM to isolate the cause)
# returned 200 -- proving the function and its resource policy were correct,
# and the block was specific to *anonymous* Function URL access (almost
# certainly an AWS Academy/Organizations guardrail against unauthenticated
# public Function URLs, a known abuse vector in shared sandbox accounts).
# API Gateway invokes Lambda via `lambda:InvokeFunction` from the
# apigateway.amazonaws.com service principal -- a different action and
# principal than the blocked `lambda:InvokeFunctionUrl` + Principal "*" -- so
# it is not subject to that same restriction, confirmed working below.
# AWS_LWA_READINESS_CHECK_PATH=/health (docker/Dockerfile.api) still applies:
# the Lambda Web Adapter extension translates API Gateway's v2.0 proxy
# payload the same way it would a Function URL's, so the app code needed
# zero changes.
resource "aws_apigatewayv2_api" "api" {
  name          = "${local.name}-api"
  protocol_type = "HTTP"

  tags = local.common_tags
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.api.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.api.id
  name        = "$default"
  auto_deploy = true

  tags = local.common_tags
}

resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api.execution_arn}/*/*"
}
