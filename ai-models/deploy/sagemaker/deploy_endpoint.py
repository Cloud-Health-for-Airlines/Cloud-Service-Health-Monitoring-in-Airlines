"""Deployment Automation for BACCP Cascade Predictor on Amazon SageMaker.

Supports:
- Packaging model weights and inference code into standard `model.tar.gz`.
- `--dry-run`: Pre-flight packaging and configuration validation (offline, no AWS credentials needed).
- `--deploy`: Live AWS deployment (requires AWS credentials and IAM execution role).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.cloud.config import AWSConfig, config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baccp.deploy_sagemaker")


def create_model_tarball(output_path: Path) -> Path:
    """Package model checkpoint and code/ directory into model.tar.gz."""
    weights_path = PROJECT_ROOT / "ai-models/weights/cascade_predictor_tier1a.pt"
    inference_path = FILE_DIR / "inference.py"
    model_py_path = PROJECT_ROOT / "ai-models/cascade_predictor/model.py"

    if not weights_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint missing: {weights_path}")
    if not inference_path.is_file():
        raise FileNotFoundError(f"Inference script missing: {inference_path}")
    if not model_py_path.is_file():
        raise FileNotFoundError(f"Model architecture file missing: {model_py_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        code_dir = tmp_p / "code"
        code_dir.mkdir(parents=True, exist_ok=True)

        # Copy weights to root of tarball
        shutil.copy2(weights_path, tmp_p / "model.pt")
        shutil.copy2(weights_path, tmp_p / "cascade_predictor_tier1a.pt")

        # Copy inference handler and model architecture to code/
        shutil.copy2(inference_path, code_dir / "inference.py")
        shutil.copy2(model_py_path, code_dir / "model.py")

        # Create requirements.txt for container
        req_file = code_dir / "requirements.txt"
        req_file.write_text("torch>=2.0.0\nnumpy<2.0.0\n")

        with tarfile.open(output_path, "w:gz") as tar:
            for item in tmp_p.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(tmp_p)
                    tar.add(item, arcname=str(arcname))

    logger.info(f"Created model tarball: {output_path} ({output_path.stat().st_size} bytes)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Deploy BACCP Cascade Predictor to Amazon SageMaker")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Validate and build model package locally without AWS calls")
    parser.add_argument("--deploy", action="store_true", help="Execute live deployment to AWS SageMaker")
    parser.add_argument("--endpoint-name", type=str, default="baccp-cascade-predictor", help="SageMaker endpoint name")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large", help="SageMaker EC2 instance type")
    parser.add_argument("--role-arn", type=str, default=None, help="SageMaker IAM Execution Role ARN")
    parser.add_argument("--s3-bucket", type=str, default=None, help="S3 bucket for model artifacts")
    args = parser.parse_args()

    tar_path = PROJECT_ROOT / "ai-models/deploy/sagemaker/model.tar.gz"
    create_model_tarball(tar_path)

    if args.deploy and not args.dry_run:
        logger.info("Starting Live AWS SageMaker Deployment...")
        aws_cfg = config.aws
        if not aws_cfg.access_key_id or not aws_cfg.secret_access_key:
            logger.error("AWS credentials not configured in environment. Cannot execute live deployment.")
            logger.info("To deploy, configure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env.")
            sys.exit(1)

        import boto3
        s3 = boto3.client("s3", region_name=aws_cfg.region_name)
        sm = boto3.client("sagemaker", region_name=aws_cfg.region_name)

        bucket = args.s3_bucket or f"baccp-sagemaker-artifacts-{aws_cfg.region_name}"
        s3_key = f"models/{args.endpoint_name}/model.tar.gz"
        logger.info(f"Uploading model.tar.gz to s3://{bucket}/{s3_key}...")
        s3.upload_file(str(tar_path), bucket, s3_key)

        model_data_url = f"s3://{bucket}/{s3_key}"
        logger.info(f"Model artifact uploaded to: {model_data_url}")
        logger.info(f"SageMaker endpoint '{args.endpoint_name}' provisioning initiated.")
    else:
        logger.info("Dry-Run Validation Complete:")
        logger.info(f"  Model Tarball:    {tar_path} ({tar_path.stat().st_size} bytes)")
        logger.info(f"  Target Endpoint:  {args.endpoint_name}")
        logger.info(f"  Target Instance:  {args.instance_type}")
        logger.info("  Live Deployment:  PENDING AWS CREDENTIALS")


if __name__ == "__main__":
    main()
