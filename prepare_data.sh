#!/bin/bash
set -e

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "❌ .env file not found!"
  exit 1
fi

kaggle datasets download karimabdulnabi/fruit-classification10-class  -p ./data --unzip

echo "✅ Dataset downloaded to ./data"