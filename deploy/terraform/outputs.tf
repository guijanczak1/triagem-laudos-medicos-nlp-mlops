# Outputs a real deploy (T25/T26) needs to push an image and reach the API.
# None of these leak a real account id/ARN by themselves -- they only exist
# once `apply` has actually run, which this task (T24) never does.

output "ecr_repository_url" {
  description = "Push target for `docker push` (see scripts/deploy_aws.ps1, T25). Format: <account_id>.dkr.ecr.<region>.amazonaws.com/<project_name>-api"
  value       = aws_ecr_repository.api.repository_url
}

output "lambda_function_name" {
  description = "Name of the Lambda function (useful for `aws lambda update-function-code` after a new image push)."
  value       = aws_lambda_function.api.function_name
}

output "lambda_function_url" {
  description = "Public HTTPS endpoint (Lambda Function URL, no custom domain). <this>health should return 200 once an image has been pushed and the function invoked (the first request after a cold start may take a few seconds)."
  value       = aws_lambda_function_url.api.function_url
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch Logs group receiving the function's stdout/stderr."
  value       = aws_cloudwatch_log_group.lambda.name
}

output "aws_account_id_in_use" {
  description = "Account the current credentials resolve to -- sanity check before ever running apply against it."
  value       = data.aws_caller_identity.current.account_id
}
