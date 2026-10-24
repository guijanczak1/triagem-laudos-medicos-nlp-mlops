<#
.SYNOPSIS
    Builds, pushes and deploys the triage API to the AWS stack defined in
    deploy/terraform/ (ECR + Lambda container image + API Gateway HTTP API +
    CloudWatch + IAM).

.DESCRIPTION
    T25 (HARNESS.md SS6 -- "parada obrigatoria"). This script is the ONLY
    place in the repo that is allowed to perform AWS write calls, and even
    here it refuses to do so unless the caller asks twice:

      1. Run WITHOUT -WhatIf:$false (i.e. leave -WhatIf at its default,
         which is $true) -- the script only builds the image locally and
         runs `terraform plan`. Nothing is pushed to ECR, nothing is
         applied. This is the default and safe to run at any time.
      2. Run WITH -WhatIf:$false -- the script shows the plan, an estimated
         monthly cost banner, and then STOPS at an interactive prompt that
         requires typing the literal word `apply`. Anything else (including
         Ctrl+C, empty input, "y", "yes") aborts with no AWS write call
         made.

    Only when BOTH conditions hold (explicit -WhatIf:$false AND the typed
    confirmation) does the script call `docker push` and `terraform apply`.

    Do not remove either gate. Do not add -AutoApprove / -Force flags that
    skip the prompt. See docs/deploy_aws.md and
    state/mlet-tech-challenge-fase-3/cloud_request.md (outside this repo)
    for the human authorization this script depends on.

.PARAMETER ProjectName
    Must match terraform.tfvars' project_name (default "triagem"). Used to
    derive the ECR repository name (<ProjectName>-api) and the Lambda
    function name.

.PARAMETER AwsRegion
    AWS region. Must match terraform.tfvars' aws_region (default us-east-1).

.PARAMETER ImageTag
    Tag applied to the built image and pushed to ECR. Must match
    terraform.tfvars' container_image_tag (default "latest") or the Lambda
    function will point at a tag that was never pushed.

.PARAMETER TerraformDir
    Path to the Terraform root module. Default: deploy/terraform relative
    to the repo root (this script lives in scripts/, one level down).

.PARAMETER WhatIf
    Defaults to $true (dry run: local build + `terraform plan` only, no AWS
    write calls). Pass -WhatIf:$false to unlock the apply path -- this
    alone does not apply anything, it only reveals the confirmation prompt
    described above.

.EXAMPLE
    # Safe at any time: builds the image locally, shows the Terraform plan,
    # pushes nothing, applies nothing.
    .\scripts\deploy_aws.ps1

.EXAMPLE
    # Real deploy: only proceeds past the plan if the user then types
    # "apply" at the interactive prompt.
    .\scripts\deploy_aws.ps1 -WhatIf:$false
#>

[CmdletBinding()]
param(
    [string]$ProjectName = "triagem",
    [string]$AwsRegion = "us-east-1",
    [string]$ImageTag = "latest",
    [string]$TerraformDir = (Join-Path $PSScriptRoot "..\deploy\terraform"),
    [bool]$WhatIf = $true
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TerraformDir = Resolve-Path $TerraformDir
$DockerfilePath = Join-Path $RepoRoot "docker\Dockerfile.api"
$LocalImage = "$ProjectName-api:local"
$EcrRepoName = "$ProjectName-api"

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "==== $Title ====" -ForegroundColor Cyan
}

function Invoke-Checked {
    <# Runs an external command and stops the script if it exits non-zero.
       PowerShell does NOT stop on a failing native command by default even
       with $ErrorActionPreference = "Stop", so every docker/terraform/aws
       call in this script goes through this wrapper. #>
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host "-> $Description" -ForegroundColor DarkGray
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $Description (exit code $LASTEXITCODE)"
    }
}

# --- 0. Pre-flight: required tools + read-only AWS identity check ----------
Write-Section "Pre-flight checks"

foreach ($tool in @("docker", "terraform", "aws")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required tool '$tool' was not found on PATH. Install it before running this script."
    }
}

if (-not (Test-Path $DockerfilePath)) {
    throw "Dockerfile not found at $DockerfilePath"
}

$tfvarsPath = Join-Path $TerraformDir "terraform.tfvars"
if (-not (Test-Path $tfvarsPath)) {
    Write-Host "WARNING: $tfvarsPath does not exist yet." -ForegroundColor Yellow
    Write-Host "Copy terraform.tfvars.example to terraform.tfvars first (see docs/deploy_aws.md)." -ForegroundColor Yellow
    throw "Missing terraform.tfvars -- aborting before touching AWS."
}

# `aws sts get-caller-identity` is READ-ONLY: it only confirms which account
# the current credentials resolve to. It is intentionally run even in
# -WhatIf mode so the operator can catch "wrong account" mistakes before
# ever reaching the apply gate. No write call happens here.
Write-Host "Checking AWS credentials (read-only: sts get-caller-identity)..." -ForegroundColor DarkGray
try {
    $callerIdentityJson = aws sts get-caller-identity --output json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "aws sts get-caller-identity failed: $callerIdentityJson"
    }
    $callerIdentity = $callerIdentityJson | ConvertFrom-Json
    Write-Host "  Account: $($callerIdentity.Account)"
    Write-Host "  Identity ARN: $($callerIdentity.Arn)"
}
catch {
    throw "Could not resolve AWS credentials. Configure them (aws configure / env vars / SSO login) before running this script.`n$_"
}

Write-Host ""
Write-Host "Region:        $AwsRegion"
Write-Host "Project name:  $ProjectName"
Write-Host "Image tag:     $ImageTag"
Write-Host "Terraform dir: $TerraformDir"
Write-Host "Mode:          $(if ($WhatIf) { 'DRY RUN (-WhatIf, default)' } else { 'APPLY PATH UNLOCKED (-WhatIf:$false)' })" -ForegroundColor $(if ($WhatIf) { "Green" } else { "Yellow" })

# --- 1. Build the image locally (no AWS write call) -------------------------
# --provenance=false --sbom=false: modern `docker buildx` (BuildKit) attaches
# a build-provenance attestation manifest by default, turning the pushed
# image into a multi-manifest index. AWS Lambda's CreateFunction validation
# rejects that shape ("the image manifest, config or layer media type ... is
# not supported") -- found by actually hitting this error against the real
# account (T25). --platform linux/amd64 matches
# deploy/terraform/variables.tf's `architectures = ["x86_64"]" and the
# vendored docker/lambda-adapter/lambda-adapter binary (also x86_64).
Write-Section "Docker build (local only, no push)"

Invoke-Checked -Description "docker build --platform linux/amd64 --provenance=false --sbom=false -f docker/Dockerfile.api -t $LocalImage ." -Command {
    docker build --platform linux/amd64 --provenance=false --sbom=false -f $DockerfilePath -t $LocalImage $RepoRoot
}

# --- 2. Terraform init + plan (read-only against AWS) -----------------------
# No -out here: this stack's `aws_lambda_function.api` has package_type =
# "Image", which AWS validates against the ECR tag at CreateFunction time --
# a plan computed before the ECR repo exists (first-ever deploy) can't later
# be applied as-is once the repo and image exist. The plan below is for
# human review only; the real apply happens in two separate steps further
# down (ECR+logs first, then the Lambda/API Gateway resources once an image
# has been pushed).
Write-Section "Terraform plan (read-only, for review only -- not saved/reused)"

Push-Location $TerraformDir
try {
    Invoke-Checked -Description "terraform init" -Command {
        terraform init -input=false
    }

    Invoke-Checked -Description "terraform plan" -Command {
        terraform plan -input=false
    }
}
finally {
    Pop-Location
}

# --- 3. Dry-run stop (default path) -----------------------------------------
if ($WhatIf) {
    Write-Section "DRY RUN complete -- nothing was pushed or applied"
    Write-Host "Image built locally as $LocalImage."
    Write-Host "Terraform plan saved at $TerraformDir\tfplan.out -- review it above."
    Write-Host ""
    Write-Host "Nothing was pushed to ECR and no AWS resource was created or changed." -ForegroundColor Green
    Write-Host "To actually deploy, re-run:  .\scripts\deploy_aws.ps1 -WhatIf:`$false"
    Write-Host "You will be asked to type 'apply' before anything real happens." -ForegroundColor Yellow
    exit 0
}

# --- 4. Explicit human confirmation gate (only reached with -WhatIf:$false) -
Write-Section "REAL DEPLOY -- explicit confirmation required"

Write-Host "This will create billable AWS resources in account $($callerIdentity.Account), region $AwsRegion." -ForegroundColor Yellow
Write-Host "See docs/deploy_aws.md 'Custo estimado mensal' -- this stack (Lambda container image + API" -ForegroundColor Yellow
Write-Host "Gateway HTTP API + ECR + CloudWatch Logs) has NO fixed hourly charge; cost is pay-per-request" -ForegroundColor Yellow
Write-Host "and falls within Lambda's perpetual free tier for a course-deliverable demo (API Gateway HTTP" -ForegroundColor Yellow
Write-Host "API is USD 1.00/million requests after any free tier -- negligible at demo volume)." -ForegroundColor Yellow
Write-Host ""
Write-Host "Review the 'terraform plan' output printed above before answering." -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "Type the word 'apply' (without quotes) to proceed, anything else cancels"

if ($confirmation -cne "apply") {
    Write-Host ""
    Write-Host "Confirmation not given ('$confirmation' != 'apply'). Aborting -- no AWS write call made." -ForegroundColor Green
    exit 1
}

# --- 5. Terraform apply, phase 1: ECR repo + log group only -----------------
# aws_lambda_function.api (package_type = "Image") validates its image_uri
# against ECR at creation time -- the repo must exist, and an image must
# already be pushed to the tag it references, before it can be created. This
# phase only creates the resources that have no such dependency.
Write-Section "Terraform apply (phase 1/2: ECR + CloudWatch log group)"

Push-Location $TerraformDir
try {
    Invoke-Checked -Description "terraform apply -target=aws_ecr_repository.api -target=aws_cloudwatch_log_group.lambda" -Command {
        terraform apply -auto-approve -input=false -target=aws_ecr_repository.api -target=aws_cloudwatch_log_group.lambda
    }
}
finally {
    Pop-Location
}

# --- 6. ECR login + tag/push (first image write call in this script) -------
Write-Section "ECR login + push"

Invoke-Checked -Description "aws ecr get-login-password | docker login" -Command {
    $password = aws ecr get-login-password --region $AwsRegion
    if ($LASTEXITCODE -ne 0) { throw "aws ecr get-login-password failed" }
    $password | docker login --username AWS --password-stdin "$($callerIdentity.Account).dkr.ecr.$AwsRegion.amazonaws.com"
}

$EcrImageUri = "$($callerIdentity.Account).dkr.ecr.$AwsRegion.amazonaws.com/${EcrRepoName}:$ImageTag"

Invoke-Checked -Description "docker tag $LocalImage $EcrImageUri" -Command {
    docker tag $LocalImage $EcrImageUri
}

Invoke-Checked -Description "docker push $EcrImageUri" -Command {
    docker push $EcrImageUri
}

# --- 7. Terraform apply, phase 2: Lambda + API Gateway ----------------------
# The image now exists at the tag the Lambda function references, so
# CreateFunction's image validation succeeds.
Write-Section "Terraform apply (phase 2/2: Lambda + API Gateway)"

Push-Location $TerraformDir
try {
    Invoke-Checked -Description "terraform apply" -Command {
        terraform apply -auto-approve -input=false
    }

    Write-Host ""
    Invoke-Checked -Description "terraform output" -Command {
        terraform output
    }
}
finally {
    Pop-Location
}

Write-Section "Done"
Write-Host "If the Lambda function already existed and the image tag did not change, Lambda does NOT" -ForegroundColor DarkGray
Write-Host "automatically notice the new image digest behind the same tag. Force it to pick up the push:" -ForegroundColor DarkGray
Write-Host "  aws lambda update-function-code --function-name $ProjectName-api --image-uri $EcrImageUri --region $AwsRegion" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Check <api_url>health (see terraform output api_url above)." -ForegroundColor Green
Write-Host "The first request after creation/update is a cold start and may take a few seconds." -ForegroundColor DarkGray
Write-Host "Teardown procedure (ECR storage keeps billing a few cents until deleted): docs/deploy_aws.md 'Teardown'." -ForegroundColor Yellow
