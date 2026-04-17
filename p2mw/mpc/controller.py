"""
LLM-MPC Controller — Language Model Model-Predictive Control for MetaWorld.

Novel contributions vs. existing LLM-for-robotics work:
  1. Semantic state abstraction   — raw 39-dim obs → structured text the LLM
                                    can actually reason about.
  2. Mental simulation            — for each candidate action, the LLM imagines
                                    a K-step future rollout *in language space*
                                    before committing to an action (imagined MPC).
  3. Surprise detection           — predicted vs. actual state comparison after
                                    every step; classifies surprise as low/medium/high.
  4. Adaptive strategy            — high-surprise events update an explicit
                                    strategy note injected into future prompts;
                                    3+ consecutive surprises trigger a deep
                                    strategy reconsideration call.
  5. Episodic memory              — compressed running log of surprising events
                                    prevents the LLM from repeating failed strategies.
"""

from __future__ import annotations

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

from .semantic import SemanticState, parse_obs
from .memory import EpisodicMemory, MemoryEntry
from . import prompts as P
from p2mw.utils.api import post_with_retry


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class LLMMPCController:
    """
    At every timestep:
      select_action(obs)        → choose the best action via mental simulation
      update_after_step(...)    → detect surprise and update episodic memory

    Typical usage:
        controller = LLMMPCController(api_endpoint, api_key, task_name="door-open")
        time_step = env.reset()
        while not done:
            obs = np.array(time_step.observation)
            action = controller.select_action(obs)
            prev_obs = obs.copy()
            time_step = env.step(action)
            controller.update_after_step(prev_obs, action, np.array(time_step.observation))
    """

    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        task_name: str = "door-open",
        n_candidates: int = 5,
        sim_horizon: int = 3,
        surprise_threshold_high: float = 0.08,
        surprise_threshold_medium: float = 0.03,
        consecutive_surprises_for_reset: int = 3,
        verbose: bool = True,
        temperature_generate: float = 0.8,
        temperature_simulate: float = 0.3,
        temperature_surprise: float = 0.2,
        max_wait_seconds: int = 40,
    ):
        self.api_endpoint = api_endpoint
        self.headers = {"Content-Type": "application/json", "api-key": api_key}
        self.task_name = task_name
        self.task_description = P.TASK_DESCRIPTIONS.get(task_name, task_name)
        self.n_candidates = n_candidates
        self.sim_horizon = sim_horizon
        self.surprise_threshold_high = surprise_threshold_high
        self.surprise_threshold_medium = surprise_threshold_medium
        self.consecutive_surprises_for_reset = consecutive_surprises_for_reset
        self.verbose = verbose
        self.temp_generate = temperature_generate
        self.temp_simulate = temperature_simulate
        self.temp_surprise = temperature_surprise
        self.max_wait = max_wait_seconds

        self.memory = EpisodicMemory()
        self.strategy_note: str = ""
        self.step: int = 0
        self.prev_obs: Optional[np.ndarray] = None
        self._last_predicted_text: Optional[str] = None
        self._recent_high_surprises: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Given the current observation, return the best 4-dim action.

        Pipeline:
          obs → semantic state → K candidates → mental simulation → best action
        """
        semantic = parse_obs(obs, self.prev_obs)
        semantic_text = semantic.to_text()

        if self.verbose:
            print(f"\n{'='*62}")
            print(f"Step {self.step:3d}  |  progress≈{semantic.progress_estimate():.1%}")
            print(semantic_text)

        candidates = self._generate_candidates(semantic_text)
        scored     = self._simulate_and_score(semantic_text, candidates)

        best_action, best_reasoning, best_sim_text, best_score = scored[0]

        if self.verbose:
            print(f"\n  ✓ Selected: {[round(a, 3) for a in best_action]}")
            print(f"    Reasoning:  {best_reasoning}")
            print(f"    Sim score:  {best_score}/10")
            if self.strategy_note:
                print(f"    Strategy:   {self.strategy_note}")

        self._last_predicted_text = best_sim_text
        return np.array(best_action, dtype=float)

    def update_after_step(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        actual_obs: np.ndarray,
    ) -> None:
        """
        Call after env.step() with the previous obs, executed action, and
        new obs.  Detects surprise, updates memory, and adjusts strategy.
        """
        actual_semantic = parse_obs(actual_obs, prev_obs)
        actual_text     = actual_semantic.to_text()
        surprise        = self._detect_surprise(prev_obs, actual_obs)

        entry = MemoryEntry(
            step=self.step,
            semantic_state_text=parse_obs(prev_obs).to_text(),
            action=action.tolist(),
            predicted_next_text=self._last_predicted_text,
            actual_next_text=actual_text,
            surprise_level=surprise["level"],
            surprise_cause=surprise.get("cause", ""),
            strategy_note=surprise.get("strategy", ""),
        )
        self.memory.add(entry)

        if surprise["level"] == "high":
            self._recent_high_surprises.append({"step": self.step, **surprise})
            if self.verbose:
                print(f"\n  ⚡ HIGH SURPRISE: {surprise.get('cause', '')}")
                print(f"     → {surprise.get('strategy', '')}")

            if len(self._recent_high_surprises) >= self.consecutive_surprises_for_reset:
                self._deep_strategy_update()
                self._recent_high_surprises = []

        elif surprise["level"] == "medium":
            if self.verbose:
                print(f"\n  ~ medium surprise: {surprise.get('cause', '')}")
            # Soft strategy update: adopt the note without a full LLM call
            if surprise.get("strategy"):
                self.strategy_note = surprise["strategy"]
            self._recent_high_surprises = []
        else:
            self._recent_high_surprises = []

        self.prev_obs = actual_obs.copy()
        self.step += 1

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self, semantic_text: str
    ) -> List[Tuple[List[float], str]]:
        """Ask the LLM to propose n_candidates actions with reasoning."""
        history = self.memory.get_context()
        if self.strategy_note:
            history += f"\nCurrent strategy: {self.strategy_note}"

        prompt = P.CANDIDATE_GENERATION_PROMPT.format(
            semantic_state=semantic_text,
            task_name=self.task_name,
            task_description=self.task_description,
            episode_history=history,
            k=self.n_candidates,
        )
        response = self._call_llm(
            [{"role": "system", "content": P.SYSTEM_PROMPT},
             {"role": "user",   "content": prompt}],
            temperature=self.temp_generate,
        )
        candidates = self._parse_candidates(response)

        if self.verbose:
            print(f"\n  Candidates ({len(candidates)}):")
            for i, (a, r) in enumerate(candidates):
                print(f"    {i+1}. {[round(x, 3) for x in a]}  — {r}")

        if not candidates:
            candidates = [([0.05, 0.0, 0.0, 0.0], "fallback: small forward nudge")]

        return candidates

    def _parse_candidates(
        self, text: str
    ) -> List[Tuple[List[float], str]]:
        candidates: List[Tuple[List[float], str]] = []
        for block in re.split(r"CANDIDATE\s+\d+:", text)[1:]:
            action_m    = re.search(r"Action:\s*\[([^\]]+)\]", block)
            reasoning_m = re.search(r"Reasoning:\s*(.+)",       block)
            if not action_m:
                continue
            try:
                action = [
                    max(-1.0, min(1.0, float(v.strip())))
                    for v in action_m.group(1).split(",")
                ]
                if len(action) != 4:
                    continue
                reasoning = reasoning_m.group(1).strip() if reasoning_m else ""
                candidates.append((action, reasoning))
            except ValueError:
                continue
        return candidates

    # ------------------------------------------------------------------
    # Mental simulation + scoring
    # ------------------------------------------------------------------

    def _simulate_and_score(
        self,
        semantic_text: str,
        candidates: List[Tuple[List[float], str]],
    ) -> List[Tuple[List[float], str, str, int]]:
        """
        For each candidate, ask the LLM to mentally simulate the next
        sim_horizon steps.  Returns candidates sorted by score descending.
        """
        if self.verbose:
            print(f"\n  Mental simulation (horizon={self.sim_horizon}):")

        scored: List[Tuple[List[float], str, str, int]] = []
        for action, reasoning in candidates:
            prompt = P.MENTAL_SIMULATION_PROMPT.format(
                semantic_state=semantic_text,
                action=[round(a, 3) for a in action],
                reasoning=reasoning,
                task_name=self.task_name,
                task_description=self.task_description,
                horizon=self.sim_horizon,
            )
            sim_text = self._call_llm(
                [{"role": "system", "content": P.SYSTEM_PROMPT},
                 {"role": "user",   "content": prompt}],
                temperature=self.temp_simulate,
            )
            score = self._parse_sim_score(sim_text)
            scored.append((action, reasoning, sim_text, score))

            if self.verbose:
                print(f"    {[round(a, 3) for a in action]}  →  score {score}/10")

        scored.sort(key=lambda x: x[3], reverse=True)
        return scored

    def _parse_sim_score(self, text: str) -> int:
        for pattern in [r"SIMULATION_SCORE:\s*(\d+)", r"[Ss]core[:\s]+(\d+)"]:
            m = re.search(pattern, text)
            if m:
                return min(10, max(0, int(m.group(1))))
        return 5

    # ------------------------------------------------------------------
    # Surprise detection
    # ------------------------------------------------------------------

    def _detect_surprise(
        self,
        prev_obs: np.ndarray,
        actual_obs: np.ndarray,
    ) -> Dict[str, str]:
        """
        Two-stage surprise detection:
          1. Fast heuristic: L2 delta on gripper + object positions.
          2. LLM analysis: called only when heuristic flags medium or high.
        """
        delta_gripper = np.linalg.norm(actual_obs[0:3] - prev_obs[0:3])
        delta_obj1    = np.linalg.norm(actual_obs[4:7] - prev_obs[4:7])
        motion_norm   = float(delta_gripper + delta_obj1)

        if motion_norm < self.surprise_threshold_medium:
            return {"level": "low", "cause": "", "strategy": ""}

        # Medium or high — run LLM surprise analysis
        actual_text  = parse_obs(actual_obs, prev_obs).to_text()
        differences  = (
            f"Gripper moved {delta_gripper:.3f} m  "
            f"(Δ=[{actual_obs[0]-prev_obs[0]:.3f},"
            f" {actual_obs[1]-prev_obs[1]:.3f},"
            f" {actual_obs[2]-prev_obs[2]:.3f}])\n"
            f"Object 1 moved {delta_obj1:.3f} m  "
            f"(Δ=[{actual_obs[4]-prev_obs[4]:.3f},"
            f" {actual_obs[5]-prev_obs[5]:.3f},"
            f" {actual_obs[6]-prev_obs[6]:.3f}])"
        )

        last_action = (
            self.memory.entries[-1].action
            if self.memory.entries else [0, 0, 0, 0]
        )
        prompt = P.SURPRISE_ANALYSIS_PROMPT.format(
            action=[round(a, 3) for a in last_action],
            predicted_state=self._last_predicted_text or "not available",
            actual_state=actual_text,
            differences=differences,
        )
        analysis = self._call_llm(
            [{"role": "system", "content": P.SYSTEM_PROMPT},
             {"role": "user",   "content": prompt}],
            temperature=self.temp_surprise,
        )
        return self._parse_surprise(analysis, motion_norm)

    def _parse_surprise(self, text: str, motion_norm: float) -> Dict[str, str]:
        level_m    = re.search(r"SURPRISE_LEVEL:\s*(low|medium|high)", text, re.I)
        cause_m    = re.search(r"CAUSE:\s*(.+)",    text)
        strategy_m = re.search(r"STRATEGY:\s*(.+)", text)

        # If LLM didn't classify, fall back to heuristic threshold
        if level_m:
            level = level_m.group(1).lower()
        else:
            level = "high" if motion_norm > self.surprise_threshold_high else "medium"

        return {
            "level":    level,
            "cause":    cause_m.group(1).strip()    if cause_m    else "",
            "strategy": strategy_m.group(1).strip() if strategy_m else "",
        }

    # ------------------------------------------------------------------
    # Deep strategy update
    # ------------------------------------------------------------------

    def _deep_strategy_update(self) -> None:
        """
        Called after consecutive_surprises_for_reset high-surprise events.
        Asks the LLM to fundamentally reconsider the current approach.
        """
        surprise_history = "\n".join(
            f"  Step {s['step']}: {s.get('cause','')}  → {s.get('strategy','')}"
            for s in self._recent_high_surprises
        )
        prompt = P.STRATEGY_UPDATE_PROMPT.format(
            n_surprises=len(self._recent_high_surprises),
            window=self.step,
            surprise_history=surprise_history,
            episode_history=self.memory.get_context(),
            task_name=self.task_name,
            task_description=self.task_description,
        )
        response = self._call_llm(
            [{"role": "system", "content": P.SYSTEM_PROMPT},
             {"role": "user",   "content": prompt}],
            temperature=0.5,
        )
        m = re.search(r"STRATEGY_UPDATE:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        if m:
            self.strategy_note = m.group(1).strip()
        if self.verbose:
            print(f"\n  🔄 DEEP STRATEGY UPDATE: {self.strategy_note}")

    # ------------------------------------------------------------------
    # LLM API call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 900,
    ) -> str:
        """POST to the configured endpoint; rate-limit retry via shared helper."""
        payload = {
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        result = post_with_retry(
            self.api_endpoint, self.headers, payload, max_wait=self.max_wait
        )
        return result["choices"][0]["message"]["content"]
