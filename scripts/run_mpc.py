#!/usr/bin/env python3
"""
LLM-MPC entry point for MetaWorld.

Examples:
    # Basic run (door-open, 5 candidates, 3-step horizon)
    python scripts/run_mpc.py

    # Door-close with more candidates and deeper lookahead
    python scripts/run_mpc.py --task door-close --candidates 7 --horizon 5

    # Quiet run, save log
    python scripts/run_mpc.py --task reach --no-verbose --output output/reach.json

    # Compare against CoT baseline on the same task
    python scripts/run_mpc.py --task door-open --baseline
"""

import os
import sys
import json
import argparse
import numpy as np

from p2mw.env import make
from p2mw.mpc import LLMMPCController
from p2mw.baselines import CoTAgent


AZURE_ENDPOINT = (
    "https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model}"
    "/chat/completions?api-version=2023-03-15-preview"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-MPC robot controller for MetaWorld",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task",       type=str,   default="door-open",
                   help="MetaWorld task name")
    p.add_argument("--model",      type=str,   default="gpt-4-32k",
                   choices=["gpt-4", "gpt-4-32k", "gpt-35-turbo"])
    p.add_argument("--candidates", type=int,   default=5,
                   help="Candidate actions generated per step")
    p.add_argument("--horizon",    type=int,   default=3,
                   help="Mental simulation lookahead steps")
    p.add_argument("--max-steps",  type=int,   default=100,
                   help="Maximum environment steps per episode")
    p.add_argument("--seed",       type=int,   default=1)
    p.add_argument("--no-verbose", action="store_true",
                   help="Suppress step-by-step output")
    p.add_argument("--output",     type=str,   default="output/mpc_run.json",
                   help="Path to save the episode log")
    p.add_argument("--baseline",   action="store_true",
                   help="Also run the CoT baseline for comparison")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_mpc(args: argparse.Namespace, api_key: str) -> dict:
    endpoint   = AZURE_ENDPOINT.format(model=args.model)
    env        = make(name=args.task, frame_stack=3, action_repeat=2,
                      seed=args.seed, train=True, device_id=-1)
    time_step  = env.reset()

    controller = LLMMPCController(
        api_endpoint=endpoint,
        api_key=api_key,
        task_name=args.task,
        n_candidates=args.candidates,
        sim_horizon=args.horizon,
        verbose=not args.no_verbose,
    )

    log = []
    total_reward = 0.0
    step = 0

    print(f"\n{'━'*62}")
    print(f"  LLM-MPC  |  task={args.task}  candidates={args.candidates}"
          f"  horizon={args.horizon}")
    print(f"{'━'*62}")

    while not time_step.last() and time_step["success"] != 1 and step < args.max_steps:
        obs      = np.array(time_step.observation)
        action   = controller.select_action(obs)
        prev_obs = obs.copy()

        time_step    = env.step(action)
        actual_obs   = np.array(time_step.observation)
        reward       = float(time_step.reward)
        total_reward += reward

        controller.update_after_step(prev_obs, action, actual_obs)
        log.append({
            "step":    step,
            "action":  action.tolist(),
            "reward":  reward,
            "success": float(time_step["success"]),
        })
        step += 1

    success = bool(time_step["success"] == 1)
    print(f"\n{'━'*62}")
    print(f"  LLM-MPC done | steps={step}  success={success}"
          f"  total_reward={total_reward:.3f}")
    print(f"{'━'*62}\n")

    return {
        "method":       "llm-mpc",
        "task":         args.task,
        "success":      success,
        "steps":        step,
        "total_reward": total_reward,
        "log":          log,
    }


def run_baseline(args: argparse.Namespace, api_key: str) -> dict:
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
    return agent.run_episode(env)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    results = [run_mpc(args, api_key)]

    if args.baseline:
        results.append(run_baseline(args, api_key))
        print("\n--- Comparison ---")
        for r in results:
            print(f"  {r['method']:15s}  success={r['success']}"
                  f"  steps={r['steps']}  reward={r['total_reward']:.3f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = results[0] if len(results) == 1 else results
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nLog saved → {args.output}")


if __name__ == "__main__":
    main()
