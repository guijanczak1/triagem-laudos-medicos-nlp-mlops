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

# --- Public entry point: Lambda Function URL -------------------------------
# No ALB, no API Gateway: a Function URL is a free, built-in HTTPS endpoint.
# AWS_LWA_READINESS_CHECK_PATH=/health (docker/Dockerfile.api) makes the AWS
# Lambda Web Adapter extension poll the same GET /health the ALB design used
# as its target-group health check, so the app code needed zero changes.
resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
}

resource "aws_lambda_permission" "function_url_public" {
  statement_id           = "AllowPublicFunctionUrlInvoke"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}
