"""
CoT Baseline Agent — alternating action-prediction and self-reflection.

This is the original prompt-engineering approach:
  - Even steps: given current obs + history, predict action + next obs
  - Odd  steps: compare predicted vs. actual obs, reason about discrepancy

Usage:
    from p2mw.env import make
    from p2mw.baselines import CoTAgent

    env = make(name="door-open", frame_stack=3, action_repeat=2, seed=1, train=True)
    agent = CoTAgent(api_endpoint=..., api_key=..., task_name="door-open")
    result = agent.run_episode(env)
"""

from __future__ import annotations

import os
import json
import numpy as np
from typing import Dict, List, Optional

from p2mw.utils.api import post_with_retry
from p2mw.baselines.prompts.cot import (
    system_prompt,
    demo_prompt,
    new_task_prompt,
    interact_prompt,
    cot_prompt,
)


class CoTAgent:
    """
    Chain-of-thought baseline agent for MetaWorld.

    Alternates between two prompt types every other step:
      - Action prompt  (even steps): reason about the goal, output action + predicted obs
      - Reflect prompt (odd  steps): compare predicted vs. actual obs, update mental model
    """

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        task_name: str = "door-open",
        max_steps: int = 100,
        verbose: bool = True,
        max_wait: int = 40,
    ):
        self.api_endpoint = api_endpoint
        self.headers = {"Content-Type": "application/json", "api-key": api_key}
        self.task_name = task_name
        self.max_steps = max_steps
        self.verbose = verbose
        self.max_wait = max_wait

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(self, env) -> Dict:
        """
        Run one full episode.

        Args:
            env: A wrapped MetaWorld environment (output of ``p2mw.env.make``).

        Returns:
            Dict with keys: method, task, success, steps, total_reward, log.
        """
        time_step = env.reset()

        count = 0
        history_obs: List[np.ndarray] = []
        action = np.zeros(4)
        pred_obs: Optional[np.ndarray] = None
        log = []
        total_reward = 0.0

        if self.verbose:
            print(f"\n{'━'*60}")
            print(f"  CoT Baseline  |  task={self.task_name}")
            print(f"{'━'*60}")

        while not time_step.last() and time_step["success"] != 1 and count < self.max_steps:
            obs = np.array(time_step.observation)
            payload = self._build_payload(obs, action, history_obs, count, pred_obs)
            result = post_with_retry(self.api_endpoint, self.headers, payload,
                                     max_wait=self.max_wait)

            if count % 2 == 0:
                content = result["choices"][0]["message"]["content"]
                try:
                    action = self._parse_action(content)
                    pred_obs = self._parse_predicted_obs(content)
                except (IndexError, ValueError):
                    action = np.zeros(4)
                    pred_obs = None

                history_obs.append(obs)
                history_obs = history_obs[-10:]
                time_step = env.step(action)
                reward = float(time_step.reward)
                total_reward += reward
                log.append({
                    "step": count,
                    "action": action.tolist(),
                    "reward": reward,
                    "success": float(time_step["success"]),
                })
                if self.verbose:
                    print(f"  step {count:3d} | action={[round(a, 3) for a in action]}"
                          f"  reward={reward:.3f}")

            count += 1

        success = bool(time_step["success"] == 1)
        if self.verbose:
            print(f"\n  Done | steps={count}  success={success}"
                  f"  total_reward={total_reward:.3f}\n")

        return {
            "method": "baseline-cot",
            "task": self.task_name,
            "success": success,
            "steps": count,
            "total_reward": total_reward,
            "log": log,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        history_obs: List[np.ndarray],
        count: int,
        pred_obs: Optional[np.ndarray],
    ) -> dict:
        if count == 0:
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "assistant", "content": demo_prompt},
                {"role": "user",      "content": new_task_prompt.format(observation=obs)},
            ]
            temperature = 0.0
        else:
            turn = (
                interact_prompt.format(
                    previous_history=history_obs,
                    current_observation=obs,
                    previous_action=action,
                )
                if count % 2 == 0
                else cot_prompt.format(observation=obs)
            )
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "assistant", "content": demo_prompt},
                {"role": "user",      "content": new_task_prompt},
                {"role": "user",      "content": turn},
            ]
            temperature = 0.7

        return {"messages": messages, "max_tokens": 500, "temperature": temperature}

    @staticmethod
    def _parse_action(content: str) -> np.ndarray:
        raw = content.split("The predicted current action is [")[1].split("],")[0]
        return np.array([float(v.strip()) for v in raw.split(",") if v.strip()])

    @staticmethod
    def _parse_predicted_obs(content: str) -> np.ndarray:
        raw = content.split("The predicted next observation is [")[1].split("].")[0]
        return np.array([
            float(v.replace("]", "").strip())
            for v in raw.split(",") if v.strip()
        ])
