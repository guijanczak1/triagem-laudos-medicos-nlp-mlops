# Input variables for the T24 dry-run stack (ECR + ECS Fargate + ALB +
# CloudWatch + IAM). No default here carries a real account id, ARN, or
# credential -- see terraform.tfvars.example for the values a real deploy
# (T25/T26, only after explicit user authorization) would need to supply.

variable "project_name" {
  description = "Short, lowercase, hyphen-safe prefix applied to every resource name (e.g. \"triagem\" -> \"triagem-api\", \"triagem-cluster\")."
  type        = string
  default     = "triagem"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.project_name))
    error_message = "project_name must be lowercase alphanumeric/hyphen, start with a letter, 2-21 chars (keeps derived resource names, e.g. ALB names, within AWS length limits)."
  }
}

variable "aws_region" {
  description = "AWS region to deploy into. Defaults to us-east-1 per the acceptance criteria; override for another region."
  type        = string
  default     = "us-east-1"
}

variable "container_port" {
  description = "TCP port the triage API listens on inside the container (matches EXPOSE 8000 in docker/Dockerfile.api and Settings.api_port default)."
  type        = number
  default     = 8000
}

variable "health_check_path" {
  description = "HTTP path the ALB target group polls for health (matches GET /health in src/triagem/serving/api.py)."
  type        = string
  default     = "/health"
}

variable "container_image_tag" {
  description = "Image tag (within the ECR repo this stack creates) that the ECS task definition points at. The repo starts empty -- a real deploy must push this tag first (see scripts/deploy_aws.ps1 / T25) before the ECS service can start a healthy task."
  type        = string
  default     = "latest"
}

variable "fargate_cpu" {
  description = "Fargate task CPU units. 256 = 0.25 vCPU, per the acceptance criteria."
  type        = number
  default     = 256
}

variable "fargate_memory" {
  description = "Fargate task memory in MiB. 512 = 0.5 GB, per the acceptance criteria."
  type        = number
  default     = 512
}

variable "desired_count" {
  description = "Number of ECS tasks the service keeps running."
  type        = number
  default     = 1
}

variable "log_retention_days" {
  description = "CloudWatch Logs retention for the ECS task's log group."
  type        = number
  default     = 7
}

variable "allowed_ingress_cidr_blocks" {
  description = "CIDR blocks allowed to reach the ALB on port 80. Defaults to open internet (0.0.0.0/0) since this is a public inference API demo; narrow this for anything beyond a course deliverable."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "assign_public_ip" {
  description = "Whether ECS tasks get a public IP. Required to be true unless private subnets + a NAT gateway are provisioned; this stack uses the account's default (public) subnets, so true is the correct default here."
  type        = bool
  default     = true
}

variable "extra_tags" {
  description = "Additional resource tags merged on top of the stack's own (Project, ManagedBy)."
  type        = map(string)
  default     = {}
}
