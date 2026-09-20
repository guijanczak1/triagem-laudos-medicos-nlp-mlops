# Outputs a real deploy (T25/T26) needs to push an image and reach the API.
# None of these leak a real account id/ARN by themselves -- they only exist
# once `apply` has actually run, which this task (T24) never does.

output "ecr_repository_url" {
  description = "Push target for `docker push` (see scripts/deploy_aws.ps1, T25). Format: <account_id>.dkr.ecr.<region>.amazonaws.com/<project_name>-api"
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster running the service."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service (useful for `aws ecs update-service --force-new-deployment` after a new image push)."
  value       = aws_ecs_service.api.name
}

output "alb_dns_name" {
  description = "Public DNS name of the ALB. http://<this>/health should return 200 once a healthy task is running."
  value       = aws_lb.main.dns_name
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch Logs group receiving the container's stdout/stderr."
  value       = aws_cloudwatch_log_group.api.name
}

output "aws_account_id_in_use" {
  description = "Account the current credentials resolve to -- sanity check before ever running apply against it."
  value       = data.aws_caller_identity.current.account_id
}
