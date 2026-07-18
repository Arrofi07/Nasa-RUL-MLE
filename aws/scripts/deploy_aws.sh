#!/usr/bin/env bash
# aws/scripts/deploy_aws.sh
# ==========================
# Full automated deployment of the NASA RUL API to AWS Lambda via ECR.
#
# What this script does (in order)
# ---------------------------------
# 1. Validate prerequisites (AWS CLI, Docker, jq)
# 2. Export the LSTM to ONNX if best_lstm.onnx doesn't exist yet
# 3. Create an ECR repository if it doesn't exist
# 4. Log Docker into ECR
# 5. Build the Lambda Docker image for linux/amd64 (Lambda runs on x86)
# 6. Tag and push the image to ECR
# 7. Create the Lambda function if it doesn't exist yet, OR update it
# 8. Create a Lambda Function URL (free public HTTPS endpoint — no API Gateway cost)
# 9. Wait for the update to finish and print the live URL
#
# AWS Free Tier coverage
# ----------------------
# Lambda:  1,000,000 requests/month + 400,000 GB-seconds compute — always free
# ECR:     500 MB/month free for 12 months (our image is ~300 MB)
# No API Gateway needed — Lambda Function URLs are free and give HTTPS directly
#
# Prerequisites
# -------------
#   brew install awscli jq          # macOS
#   aws configure                   # enter your AWS Access Key, Secret, region
#   python scripts/export_onnx.py   # creates models/best_lstm.onnx
#
# Usage
# -----
#   chmod +x aws/scripts/deploy_aws.sh
#   ./aws/scripts/deploy_aws.sh
#
#   # Override defaults:
#   AWS_REGION=eu-central-1 LAMBDA_NAME=nasa-rul-prod ./aws/scripts/deploy_aws.sh

set -euo pipefail  # exit on error, unset variable, or pipe failure

# ---------------------------------------------------------------------------
# Configuration — override any of these with environment variables
# ---------------------------------------------------------------------------

AWS_REGION="${AWS_REGION:-eu-central-1}"          # change to your preferred region
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_REPO_NAME="${ECR_REPO_NAME:-nasa-rul-api}"
LAMBDA_NAME="${LAMBDA_NAME:-nasa-rul-api}"
LAMBDA_MEMORY="${LAMBDA_MEMORY:-1024}"            # MB — 1 GB fits all models with room
LAMBDA_TIMEOUT="${LAMBDA_TIMEOUT:-30}"            # seconds — generous for cold start
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full ECR image URI — used for both push and Lambda update
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

# ---------------------------------------------------------------------------
# Colours for readable output
# ---------------------------------------------------------------------------
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"  # no colour

log()  { echo -e "${GREEN}[deploy]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]  ${NC} $*"; }
fail() { echo -e "${RED}[error] ${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Step 0 — Validate prerequisites
# ---------------------------------------------------------------------------
log "Validating prerequisites …"

command -v aws    >/dev/null 2>&1 || fail "AWS CLI not found. Install: brew install awscli"
command -v docker >/dev/null 2>&1 || fail "Docker not found. Install Docker Desktop."
command -v jq     >/dev/null 2>&1 || fail "jq not found. Install: brew install jq"
command -v python3 >/dev/null 2>&1 || fail "python3 not found."

# Verify AWS credentials are configured
aws sts get-caller-identity >/dev/null 2>&1 \
  || fail "AWS credentials not configured. Run: aws configure"

log "AWS account : ${AWS_ACCOUNT_ID}"
log "Region      : ${AWS_REGION}"
log "Lambda name : ${LAMBDA_NAME}"
log "Memory      : ${LAMBDA_MEMORY} MB"
log "Timeout     : ${LAMBDA_TIMEOUT} s"

# ---------------------------------------------------------------------------
# Step 1 — Export LSTM to ONNX (skip if already done)
# ---------------------------------------------------------------------------
log "Checking ONNX model …"

if [ ! -f "models/best_lstm.onnx" ]; then
    warn "models/best_lstm.onnx not found — running ONNX export now …"
    # onnxruntime must be installed locally to run the sanity check
    pip install onnxruntime --quiet
    python3 scripts/export_onnx.py \
      || fail "ONNX export failed. Check scripts/export_onnx.py for errors."
    log "ONNX export complete."
else
    log "models/best_lstm.onnx already exists — skipping export."
fi

# ---------------------------------------------------------------------------
# Step 2 — Create ECR repository (idempotent — safe to run multiple times)
# ---------------------------------------------------------------------------
log "Creating ECR repository '${ECR_REPO_NAME}' (if it doesn't exist) …"

aws ecr describe-repositories \
    --repository-names "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" \
    >/dev/null 2>&1 \
|| aws ecr create-repository \
    --repository-name "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    >/dev/null

log "ECR repository ready: ${ECR_URI}"

# ---------------------------------------------------------------------------
# Step 3 — Log Docker into ECR
# ---------------------------------------------------------------------------
log "Authenticating Docker with ECR …"

aws ecr get-login-password \
    --region "${AWS_REGION}" \
  | docker login \
    --username AWS \
    --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ---------------------------------------------------------------------------
# Step 4 — Build Docker image for linux/amd64
# ---------------------------------------------------------------------------
# Lambda runs on x86_64 regardless of your local machine architecture.
# --platform linux/amd64 ensures the image is built for Lambda's CPU,
# not your Mac's ARM (M1/M2) which would cause "exec format error" on Lambda.
log "Building Docker image for linux/amd64 …"
log "(This takes ~3-5 minutes on first build, much faster on rebuilds)"

docker build \
    --platform linux/amd64 \
    --file aws/lambda/Dockerfile \
    --tag "${ECR_REPO_NAME}:${IMAGE_TAG}" \
    .

log "Docker build complete."

# ---------------------------------------------------------------------------
# Step 5 — Tag and push to ECR
# ---------------------------------------------------------------------------
log "Pushing image to ECR …"

docker tag  "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}"
docker push "${ECR_URI}"

log "Image pushed: ${ECR_URI}"

# ---------------------------------------------------------------------------
# Step 6 — Create or update Lambda function
# ---------------------------------------------------------------------------
log "Creating/updating Lambda function '${LAMBDA_NAME}' …"

# Check whether the Lambda function already exists
LAMBDA_EXISTS=$(
  aws lambda get-function \
      --function-name "${LAMBDA_NAME}" \
      --region "${AWS_REGION}" \
      2>/dev/null \
  && echo "yes" \
  || echo "no"
)

if [ "${LAMBDA_EXISTS}" = "no" ]; then
    # ── First deploy: create the function ──────────────────────────────────
    log "Function not found — creating …"

    # We need an IAM role for Lambda to assume. Create a basic one if needed.
    ROLE_NAME="${LAMBDA_NAME}-role"
    ROLE_ARN=$(
      aws iam get-role \
          --role-name "${ROLE_NAME}" \
          --query "Role.Arn" \
          --output text \
          2>/dev/null \
      || echo "NOTFOUND"
    )

    if [ "${ROLE_ARN}" = "NOTFOUND" ]; then
        log "Creating IAM execution role '${ROLE_NAME}' …"

        # Trust policy: allows Lambda service to assume this role
        TRUST_POLICY='{
          "Version": "2012-10-17",
          "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action":    "sts:AssumeRole"
          }]
        }'

        ROLE_ARN=$(
          aws iam create-role \
              --role-name "${ROLE_NAME}" \
              --assume-role-policy-document "${TRUST_POLICY}" \
              --query "Role.Arn" \
              --output text
        )

        # Attach the managed policy that grants CloudWatch Logs access
        # (so you can see your print() statements in the AWS console)
        aws iam attach-role-policy \
            --role-name "${ROLE_NAME}" \
            --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

        # IAM role propagation takes a few seconds
        log "Waiting 15 s for IAM role to propagate …"
        sleep 15
    fi

    log "Using IAM role: ${ROLE_ARN}"

    # Create the Lambda function from the ECR image
    aws lambda create-function \
        --function-name    "${LAMBDA_NAME}" \
        --package-type     Image \
        --code             ImageUri="${ECR_URI}" \
        --role             "${ROLE_ARN}" \
        --memory-size      "${LAMBDA_MEMORY}" \
        --timeout          "${LAMBDA_TIMEOUT}" \
        --region           "${AWS_REGION}" \
        --environment      "Variables={
            KMP_DUPLICATE_LIB_OK=TRUE,
            OMP_NUM_THREADS=1,
            MKL_NUM_THREADS=1,
            LGBM_NUM_THREADS=1,
            TOKENIZERS_PARALLELISM=false
        }" \
        >/dev/null

    log "Lambda function created."

else
    # ── Subsequent deploys: update the image ───────────────────────────────
    log "Function exists — updating image …"

    aws lambda update-function-code \
        --function-name "${LAMBDA_NAME}" \
        --image-uri     "${ECR_URI}" \
        --region        "${AWS_REGION}" \
        >/dev/null

    log "Lambda function code updated."
fi

# ---------------------------------------------------------------------------
# Step 7 — Wait for update to complete
# ---------------------------------------------------------------------------
log "Waiting for Lambda to finish updating …"

aws lambda wait function-updated \
    --function-name "${LAMBDA_NAME}" \
    --region        "${AWS_REGION}"

log "Lambda is ready."

# ---------------------------------------------------------------------------
# Step 8 — Create or retrieve Function URL (free public HTTPS endpoint)
# ---------------------------------------------------------------------------
# Lambda Function URLs give you a stable HTTPS endpoint without API Gateway.
# API Gateway costs $3.50/million requests. Function URLs are free.
# auth-type NONE means public — no AWS Signature required to call the API.
log "Setting up Lambda Function URL …"

FUNCTION_URL=$(
  aws lambda get-function-url-config \
      --function-name "${LAMBDA_NAME}" \
      --region        "${AWS_REGION}" \
      --query         "FunctionUrl" \
      --output        text \
      2>/dev/null \
  || echo "NOTFOUND"
)

if [ "${FUNCTION_URL}" = "NOTFOUND" ]; then
    FUNCTION_URL=$(
      aws lambda create-function-url-config \
          --function-name "${LAMBDA_NAME}" \
          --auth-type     NONE \
          --cors          '{
              "AllowOrigins": ["*"],
              "AllowMethods": ["GET","POST"],
              "AllowHeaders": ["Content-Type"]
          }' \
          --region        "${AWS_REGION}" \
          --query         "FunctionUrl" \
          --output        text
    )

    # Allow public invocations (no IAM auth required)
    aws lambda add-permission \
        --function-name  "${LAMBDA_NAME}" \
        --statement-id   "FunctionURLAllowPublicAccess" \
        --action         "lambda:InvokeFunctionUrl" \
        --principal      "*" \
        --function-url-auth-type NONE \
        --region         "${AWS_REGION}" \
        >/dev/null 2>&1 \
    || true  # ignore if permission already exists
fi

# ---------------------------------------------------------------------------
# Step 9 — Health check
# ---------------------------------------------------------------------------
log "Waiting 5 s before health check …"
sleep 5

# Strip trailing slash from Function URL if present
BASE_URL="${FUNCTION_URL%/}"
HEALTH_URL="${BASE_URL}/health"

log "Testing: GET ${HEALTH_URL}"

HTTP_STATUS=$(curl --silent --output /dev/null --write-out "%{http_code}" "${HEALTH_URL}")

if [ "${HTTP_STATUS}" = "200" ]; then
    HEALTH_BODY=$(curl --silent "${HEALTH_URL}")
    log "Health check passed (HTTP ${HTTP_STATUS})"
    echo "${HEALTH_BODY}" | jq . 2>/dev/null || echo "${HEALTH_BODY}"
else
    warn "Health check returned HTTP ${HTTP_STATUS} — the Lambda may still be cold-starting."
    warn "Try: curl ${HEALTH_URL}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✅ Deployment complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Lambda function : ${LAMBDA_NAME}"
echo "  Region          : ${AWS_REGION}"
echo "  ECR image       : ${ECR_URI}"
echo ""
echo "  🌐 Base URL    : ${BASE_URL}"
echo "  ❤️  Health      : ${BASE_URL}/health"
echo "  📖 Swagger     : ${BASE_URL}/docs"
echo ""
echo "  Test with:"
echo "  curl ${BASE_URL}/health"
echo ""
echo "  Predict (XGBoost):"
echo "  curl -X POST ${BASE_URL}/predict/xgb \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"engine_id\":1,\"cycle\":50,\"setting_1\":-0.0007,\"setting_2\":-0.0004,\"setting_3\":100.0,\"sensor_1\":518.67,\"sensor_2\":641.82,\"sensor_3\":1589.70,\"sensor_4\":1400.60,\"sensor_5\":14.62,\"sensor_6\":21.61,\"sensor_7\":554.36,\"sensor_8\":2388.02,\"sensor_9\":9046.19,\"sensor_10\":1.30,\"sensor_11\":47.47,\"sensor_12\":521.66,\"sensor_13\":2388.02,\"sensor_14\":8138.62,\"sensor_15\":8.4195,\"sensor_16\":0.03,\"sensor_17\":392.0,\"sensor_18\":2388.0,\"sensor_19\":100.0,\"sensor_20\":39.06,\"sensor_21\":23.419}'"
echo ""
echo "  CloudWatch logs:"
echo "  aws logs tail /aws/lambda/${LAMBDA_NAME} --follow --region ${AWS_REGION}"
echo ""