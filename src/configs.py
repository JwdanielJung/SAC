from dotenv import load_dotenv
from openai import OpenAI
from googleapiclient import discovery
import argparse

import os

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

perspective_client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=os.getenv("PERSPECTIVE_API_KEY"),
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Argument parser for model configuration."
    )

    # Choice for embed_model
    parser.add_argument(
        "--embed_model",
        type=str,
        choices=["openai", "uae", "mxbai"],
        default="uae",
        help="Embedding model to use (choices: 'openai', 'uae', 'mxbai', default: 'openai')",
    )

    # Choice for model
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama_guard", "moderation", "perspective", "gradsafe", "gpt4"],
        default="gpt4",
        help="Model to use (choices: 'llama_guard', 'moderation', 'perspective', 'gradsafe', 'gpt4', default: 'gpt4')",
    )

    # Boolean flag for is_prepared
    parser.add_argument(
        "--is_prepared",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Flag indicating if the system is prepared (default: True)",
    )

    # Lambda loss
    parser.add_argument(
        "--lambda_loss",
        type=float,
        default=0.05,
        help="Lambda loss value (default: 0.05)",
    )

    # Top-k parameter
    parser.add_argument(
        "--top_k", type=int, default=1, help="Top-k value for filtering (default: 1)"
    )

    return parser.parse_args()