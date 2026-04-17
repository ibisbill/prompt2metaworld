# Prompt2MetaWorld

Zero-shot robot control in [MetaWorld](https://meta-world.github.io/) using large language models.

Two methods are included:

| Method | File | Idea |
|---|---|---|
| **CoT Baseline** | `scripts/run_baseline.py` | Alternating action-prediction and self-reflection prompts |
| **LLM-MPC** | `scripts/run_mpc.py` | LLM as imagined world model; K-step lookahead planning |

---

## LLM-MPC: Language Model Model-Predictive Control

Most prior work treats an LLM as a **policy** (obs → action). LLM-MPC treats it as an **imagined world model**: for each candidate action the LLM mentally simulates a K-step future in language space and scores it, then commits to the best-scoring action. No learned dynamics model is required.

```
obs (39-dim)
    │
    ▼
Semantic Abstraction          p2mw/mpc/semantic.py
  raw vector → structured text (positions, distances, velocities, rotation)
    │
    ▼
Candidate Generation          p2mw/mpc/controller.py  [temp=0.8]
  LLM proposes N actions with physical reasoning
    │
    ├─ for each candidate ──────────────────────────────────────┐
    │                                                            ▼
    │                          Mental Simulation   [temp=0.3]
    │                            LLM imagines K-step future in language space
    │                            scores trajectory 0–10
    │                                                            │
    └──────────── pick highest-scoring action ◄─────────────────┘
    │
    ▼
env.step(action)
    │
    ▼
Surprise Detection            p2mw/mpc/controller.py  [temp=0.2]
  L2 heuristic (Δgripper + Δobj) → LLM causal analysis if flagged
  classifies: low / medium / high
    │
    ├── medium → update strategy note (no extra LLM call)
    └── high × 3 → deep strategy reconsideration call
    │
    ▼
Episodic Memory               p2mw/mpc/memory.py
  sliding window of recent steps + compressed summary of surprising events
  injected into future candidate-generation prompts
    │
    └──────────────────────────────────────────── next step
```

### Designs

| # | Design | Where |
|---|---|---|
| 1 | **Semantic state abstraction** — 39-dim obs → structured text the LLM can reason about | `p2mw/mpc/semantic.py` |
| 2 | **Mental simulation (imagined MPC)** — LLM scores each candidate via K-step language rollout | `p2mw/mpc/controller.py` |
| 3 | **Two-stage surprise detection** — fast L2 heuristic + LLM causal analysis on demand | `p2mw/mpc/controller.py` |
| 4 | **Adaptive strategy** — surprise events update a strategy note; repeated failures trigger deep reconsideration | `p2mw/mpc/controller.py` |
| 5 | **Episodic memory** — compressed log of surprises prevents repeating failed strategies | `p2mw/mpc/memory.py` |

---

## Repository Structure

```
prompt2metaworld/
│
├── p2mw/                        # installable package  (pip install -e .)
│   ├── __init__.py
│   ├── env/
│   │   ├── __init__.py          # exports: make
│   │   └── wrapper.py           # MetaWorld → gym → dm_env wrapper chain
│   ├── baselines/
│   │   ├── __init__.py          # exports: CoTAgent
│   │   ├── cot_agent.py         # CoTAgent class: run_episode(), _build_payload()
│   │   └── prompts/
│   │       ├── __init__.py
│   │       ├── cot.py           # system, demo, interact, cot prompt strings
│   │       └── meta.py          # meta-learning style prompt strings
│   ├── mpc/
│   │   ├── __init__.py          # exports: LLMMPCController, parse_obs, SemanticState, EpisodicMemory
│   │   ├── controller.py        # LLMMPCController: select_action(), update_after_step()
│   │   ├── memory.py            # EpisodicMemory: sliding window + compression
│   │   ├── prompts.py           # prompt templates + TASK_DESCRIPTIONS registry
│   │   └── semantic.py          # SemanticState dataclass + parse_obs()
│   └── utils/
│       ├── __init__.py          # exports: post_with_retry
│       └── api.py               # shared Azure OpenAI POST with rate-limit retry
│
├── scripts/
│   ├── run_mpc.py               # LLM-MPC runner  (--baseline for side-by-side compare)
│   └── run_baseline.py          # CoT baseline runner
│
├── configs/
│   ├── mpc.yaml                 # all LLM-MPC hyperparameters with comments
│   └── tasks.yaml               # task descriptions + object labels
│
├── tests/
│   ├── test_semantic.py         # parse_obs + SemanticState unit tests
│   └── test_memory.py           # EpisodicMemory add/compress/retrieve unit tests
│
├── pyproject.toml               # package metadata + console_scripts entry points
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ibisbill/prompt2metaworld.git
cd prompt2metaworld
pip install -e ".[dev]"          # installs p2mw + pytest
```

Install MetaWorld and MuJoCo following the [official MetaWorld instructions](https://github.com/Farama-Foundation/Metaworld).

Set your Azure OpenAI key:

```bash
export OPENAI_API_KEY=<your_key>
```

After `pip install -e .`, two console commands are registered:

```bash
p2mw-mpc       # equivalent to python scripts/run_mpc.py
p2mw-baseline  # equivalent to python scripts/run_baseline.py
```

---

## Usage

### LLM-MPC

```bash
# Default: door-open, 5 candidates, 3-step lookahead
python scripts/run_mpc.py

# Custom task and planning depth
python scripts/run_mpc.py --task drawer-open --candidates 7 --horizon 5

# Quiet run, custom log path
python scripts/run_mpc.py --task reach --no-verbose --output output/reach.json

# Head-to-head comparison: LLM-MPC vs. CoT on the same episode
python scripts/run_mpc.py --task door-open --baseline
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--task` | `door-open` | MetaWorld task name (see list below) |
| `--model` | `gpt-4-32k` | Azure OpenAI deployment name |
| `--candidates` | `5` | Candidate actions generated per step |
| `--horizon` | `3` | Mental simulation lookahead steps |
| `--max-steps` | `100` | Maximum environment steps per episode |
| `--seed` | `1` | Random seed |
| `--no-verbose` | — | Suppress step-by-step console output |
| `--output` | `output/mpc_run.json` | Path to save the episode log |
| `--baseline` | — | Also run the CoT baseline and print a comparison |

**Supported tasks:**

`door-open` · `door-close` · `drawer-open` · `drawer-close` · `button-press` · `reach` · `push` · `pick-place` · `hammer` · `peg-insert-side`

Task descriptions and object labels are in [`configs/tasks.yaml`](configs/tasks.yaml).

### CoT Baseline

```bash
python scripts/run_baseline.py --task door-open
python scripts/run_baseline.py --task button-press --seed 42 --no-verbose
```

The CoT agent alternates between two prompt types every other step:
- **Even steps** — reason about the current state, output the next action and a predicted next observation.
- **Odd steps** — compare the predicted observation to the actual one, explain the discrepancy, adjust the mental model.

Prompt templates are in [`p2mw/baselines/prompts/cot.py`](p2mw/baselines/prompts/cot.py) (chain-of-thought) and [`p2mw/baselines/prompts/meta.py`](p2mw/baselines/prompts/meta.py) (meta-learning style, with full success-trajectory demonstrations).

---

## Configuration

All LLM-MPC hyperparameters are documented in [`configs/mpc.yaml`](configs/mpc.yaml):

```yaml
candidates: 5          # candidate actions per step
horizon: 3             # mental simulation lookahead
surprise_threshold_high: 0.08
surprise_threshold_medium: 0.03
consecutive_surprises_for_reset: 3
temperature_generate: 0.8
temperature_simulate: 0.3
temperature_surprise: 0.2
memory_window_size: 8
memory_compress_threshold: 12
```

These are the defaults baked into `LLMMPCController`; override them via `--candidates`, `--horizon`, or by passing constructor arguments directly.

---

## Testing

The unit tests cover the pure-Python core and require no MetaWorld install:

```bash
pytest tests/ -v
```

| Test file | What it covers |
|---|---|
| `tests/test_semantic.py` | `parse_obs()` field extraction, distances, velocities, rotation, `to_text()`, `progress_estimate()` |
| `tests/test_memory.py` | `EpisodicMemory` add, high-surprise filtering, strategy retrieval, compression, summary content |

---

## LLM-MPC vs. CoT Baseline

| Aspect | CoT Baseline | LLM-MPC |
|---|---|---|
| Action selection | Direct: obs → action | Lookahead: obs → N candidates → K-step simulation → best |
| Future reasoning | None | Mental simulation per candidate |
| Failure recovery | None | Surprise detection → strategy note → deep reconsideration |
| Memory | Last 10 raw observations | Compressed episodic log (surprising events only) |
| LLM calls / step | 1 | N + 1 simulations + 0–1 surprise analysis |

---

## Observation & Action Space

| | Dim | Layout |
|---|---|---|
| Observation | 39 | `[0:3]` gripper pos · `[3]` gripper state · `[4:7]` obj1 pos · `[7:11]` obj1 quat · `[11:14]` obj2 pos · `[14:18]` obj2 quat · `[18:36]` previous timestep (same layout) · `[36:39]` goal pos |
| Action | 4 | `[dx, dy, dz, gripper_force]`, all in `[−1, 1]` |

---

## License

[MIT License](LICENSE)
