"""Loads config.yaml and .env from the repository root.

Non-secret settings live in config.yaml. Credentials live in .env and are
exported as environment variables, which boto3 and wandb read natively.
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

with open(ROOT / "config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


def resolve(path):
    """Resolves a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def env(name):
    """Returns a stripped environment variable, or None when unset/empty."""
    value = os.environ.get(name, "").strip()
    return value or None
