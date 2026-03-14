"""
main.py — Entry point for the self-heal-agent.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-Heal Sidecar Agent")
    parser.add_argument(
        "--orchestrator-url",
        default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
        help="Base URL of the agent orchestrator (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    from orchestrator_client import OrchestratorClient

    client = OrchestratorClient(
        orchestrator_url=args.orchestrator_url,
        log_level=args.log_level,
    )

    try:
        asyncio.run(client.start())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Self-heal agent stopped.")


if __name__ == "__main__":
    main()
