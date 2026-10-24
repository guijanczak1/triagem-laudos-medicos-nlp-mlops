# Input variables for the T24 dry-run stack (ECR + Lambda container image +
# Function URL + CloudWatch + IAM). No default here carries a real account
# id, ARN, or credential -- see terraform.tfvars.example for the values a
# real deploy (T25/T26, only after explicit user authorization) would need
# to supply.

variable "project_name" {
  description = "Short, lowercase, hyphen-safe prefix applied to every resource name (e.g. \"triagem\" -> \"triagem-api\")."
  type        = string
  default     = "triagem"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.project_name))
    error_message = "project_name must be lowercase alphanumeric/hyphen, start with a letter, 2-21 chars (keeps derived resource names within AWS length limits)."
  }
}

variable "aws_region" {
  description = "AWS region to deploy into. Defaults to us-east-1 per the acceptance criteria; override for another region."
  type        = string
  default     = "us-east-1"
}

variable "container_image_tag" {
  description = "Image tag (within the ECR repo this stack creates) that the Lambda function points at. The repo starts empty -- a real deploy must push this tag first (see scripts/deploy_aws.ps1 / T25) before the function can be created or updated."
  type        = string
  default     = "latest"
}

variable "lambda_memory_mb" {
  description = "Lambda memory allocation in MB (also proportionally scales CPU, which speeds up cold-start imports). 1024 MB was tried first against the real account and the cold start (scikit-learn + ONNX Runtime imports + model load) exceeded a 30s timeout (Sandbox.Timedout); 2048 MB resolves it comfortably."
  type        = number
  default     = 2048
}

variable "lambda_timeout_seconds" {
  description = "Lambda invocation timeout. Must be generous enough to absorb a cold start (scikit-learn/ONNX Runtime import + model load), not just the measured p50/p95 warm inference latency (see docs/latency_report.md) -- 30s was measured too short against the real account at 1024 MB memory; 60s at 2048 MB works."
  type        = number
  default     = 60
}

variable "execution_role_arn" {
  description = "ARN of an existing IAM role for Lambda to assume (e.g. an AWS Academy Learner Lab's pre-provisioned \"LabRole\", arn:aws:iam::<account_id>:role/LabRole). Set this when the account cannot create new IAM roles/policies -- a common AWS Academy restriction. When null (default), this stack creates its own minimal execution role with only CloudWatch Logs write access."
  type        = string
  default     = null
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention for the Lambda function's log group."
  type        = number
  default     = 7
}

variable "extra_tags" {
  description = "Additional resource tags merged on top of the stack's own (Project, ManagedBy)."
  type        = map(string)
  default     = {}
}
