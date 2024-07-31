#!/bin/bash

# Load environment variables from .env file
set -a
source .env
set +a

# Define variables
IMAGE_NAME="groq-moa-public"
GCR_IMAGE="gcr.io/groqlabs-demo-1/${IMAGE_NAME}"
PROJECT_NAME="groq-moa-public"
REGION="us-central1"

# Convert environment variables to --set-env-vars format
ENV_VARS=$(grep -v '^#' .env | xargs | sed 's/ /,/g')

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

# Tag the Docker image for Google Container Registry
echo "Tagging Docker image..."
docker tag ${IMAGE_NAME} ${GCR_IMAGE}

# Push the Docker image to Google Container Registry
echo "Pushing Docker image to Google Container Registry..."
docker push ${GCR_IMAGE}

# Deploy the Docker image to Google Cloud Run
echo "Deploying to Google Cloud Run..."
gcloud run deploy ${PROJECT_NAME} \
  --image ${GCR_IMAGE} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars ${ENV_VARS}

echo "Deployment completed."
