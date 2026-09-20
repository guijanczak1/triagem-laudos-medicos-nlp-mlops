# Terraform + provider version pins (T24, dry-run only -- see HARNESS.md §6
# and docs/deploy_aws.md for the human gate that must approve an actual
# `terraform apply`).
#
# --- State backend -----------------------------------------------------
# No `backend` block is declared here on purpose: state stays LOCAL
# (deploy/terraform/terraform.tfstate, gitignored -- see .gitignore's
# "Terraform" section) for this dry-run phase, since `terraform plan` never
# needs to persist state anywhere durable. Before a real `apply` is ever
# run (T25/T26, only after explicit user authorization), switch to a remote
# backend, e.g.:
#
#   terraform {
#     backend "s3" {
#       bucket         = "<account-specific-tfstate-bucket>"
#       key            = "triagem/terraform.tfstate"
#       region         = "us-east-1"
#       dynamodb_table = "<account-specific-lock-table>"
#       encrypt        = true
#     }
#   }
#
# Local state is fine for solo dry-run planning; it is NOT fine for a real,
# shared deployment (no locking, no durability, no team access).
terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}
