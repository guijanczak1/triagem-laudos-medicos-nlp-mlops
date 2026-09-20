# T24 -- Terraform AWS stack for the triage API (dry-run only).
#
# Scope: ECR (image registry) + ECS Fargate (compute) + ALB (public entry
# point) + CloudWatch Logs (observability) + IAM (least-privilege execution
# and task roles). This mirrors the "Decisao arquitetural de deploy em
# nuvem" in README.md / docs/architecture.md (T23): real-time synchronous
# inference behind ECS Fargate + ALB, chosen over Lambda (cold start +
# 250MB/50MB package limits are a poor fit for a TF-IDF + RandomForest/ONNX
# model) and SageMaker Endpoint (managed-inference cost/complexity not
# justified for a single stateless container) and plain EC2 (no managed
# scaling/patching).
#
# HARD RULE (HARNESS.md §6, this task's own acceptance criteria): this file
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
}

data "aws_caller_identity" "current" {}

# --- Networking: reuse the account's default VPC/subnets -------------------
# No VPC is created by this stack. A course-deliverable single-container
# service does not warrant provisioning (and tearing down) its own VPC; the
# default VPC's public subnets are sufficient for an ALB + Fargate task with
# a public IP (see var.assign_public_ip). A real production stack would
# likely bring its own VPC module instead -- documented here, not built,
# since the PDF only asks for the ECS/ALB/ECR/CloudWatch/IAM shape.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }

  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

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
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}-api"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# --- ECS cluster -----------------------------------------------------------
resource "aws_ecs_cluster" "main" {
  name = "${local.name}-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled" # opt-in only; avoids extra CloudWatch cost for a demo workload
  }

  tags = local.common_tags
}

# --- IAM ---------------------------------------------------------------
# Execution role: what ECS/Fargate itself needs (pull from ECR, write to the
# CloudWatch log group above) -- standard AWS-managed policy, nothing custom.
data "aws_iam_policy_document" "ecs_tasks_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-ecs-execution-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "execution_managed" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Task role: what the RUNNING CONTAINER (the triage API process) is allowed
# to do via the AWS SDK. Per docker/Dockerfile.api's own documented
# decision, the model artifacts are baked into the image at build time (no
# MLflow registry, no S3 model bucket, no DVC remote -- see
# state/.../backlog.json "gates_pulados"), so the running process makes no
# AWS API calls at all. The role therefore exists (ECS requires a task role
# to be assumable even if it grants nothing) but carries NO policy
# attachments -- true least privilege, not a placeholder "*" grant. Extend
# this role, deliberately, the day the app needs to call an AWS service.
resource "aws_iam_role" "task" {
  name               = "${local.name}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json

  tags = local.common_tags
}

# --- Security groups ---------------------------------------------------
resource "aws_security_group" "alb" {
  name_prefix = "${local.name}-alb-"
  description = "Allow inbound HTTP from allowed_ingress_cidr_blocks to the ALB; all egress."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP from allowed CIDR blocks"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidr_blocks
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${local.name}-alb-sg" })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "service" {
  name_prefix = "${local.name}-svc-"
  description = "Allow inbound traffic from the ALB security group on container_port only; all egress."
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "From ALB only"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "All outbound (ECR pull, CloudWatch Logs, etc.)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, { Name = "${local.name}-service-sg" })

  lifecycle {
    create_before_destroy = true
  }
}

# --- Application Load Balancer ------------------------------------------
resource "aws_lb" "main" {
  name               = "${local.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  tags = local.common_tags
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip" # required for awsvpc-mode Fargate tasks

  health_check {
    path                = var.health_check_path
    protocol            = "HTTP"
    matcher             = "200"
    healthy_threshold   = 3
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 5
  }

  tags = local.common_tags
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# --- ECS task definition + service ----------------------------------------
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.fargate_cpu
  memory                   = var.fargate_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "${local.name}-api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.container_image_tag}"
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      # Mirrors docker/Dockerfile.api's own container HEALTHCHECK, so ECS
      # can detect a stuck/unresponsive task independently of the ALB's
      # target-group health check.
      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request as u; u.urlopen('http://127.0.0.1:${var.container_port}/health', timeout=3)\" || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 15
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      environment = [
        { name = "TRIAGEM_API_PORT", value = tostring(var.container_port) }
      ]
    }
  ])

  tags = local.common_tags
}

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = var.assign_public_ip
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "${local.name}-api"
    container_port   = var.container_port
  }

  # The listener must exist before ECS tries to register targets through it.
  depends_on = [aws_lb_listener.http]

  tags = local.common_tags
}
