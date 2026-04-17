#!/usr/bin/env python3
"""
CoT baseline entry point for MetaWorld.

Examples:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --task button-press --seed 42
    python scripts/run_baseline.py --task reach --no-verbose --output output/reach.json
"""

import os
import sys
import json
import argparse

from p2mw.env import make
from p2mw.baselines import CoTAgent


AZURE_ENDPOINT = (
    "https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model}"
    "/chat/completions?api-version=2023-03-15-preview"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CoT baseline robot controller for MetaWorld",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task",       type=str, default="door-open")
    p.add_argument("--model",      type=str, default="gpt-4-32k",
                   choices=["gpt-4", "gpt-4-32k", "gpt-35-turbo"])
    p.add_argument("--max-steps",  type=int, default=100)
    p.add_argument("--seed",       type=int, default=1)
    p.add_argument("--no-verbose", action="store_true")
    p.add_argument("--output",     type=str, default="output/baseline_run.json")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    endpoint = AZURE_ENDPOINT.format(model=args.model)
    env      = make(name=args.task, frame_stack=3, action_repeat=2,
                    seed=args.seed, train=True, device_id=-1)
    agent    = CoTAgent(
        api_endpoint=endpoint,
        api_key=api_key,
        task_name=args.task,
        max_steps=args.max_steps,
        verbose=not args.no_verbose,
    )
    result = agent.run_episode(env)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nLog saved → {args.output}")


if __name__ == "__main__":
    main()
