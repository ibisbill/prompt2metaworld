# Prompt2MetaWorld

Zero-shot robot control in [MetaWorld](https://meta-world.github.io/) using large language models — no training, no fine-tuning.

Two approaches are included:

1. **CoT baseline** — alternating action-prediction and self-reflection prompts.
2. **LLM-MPC** — novel system treating the LLM as an imagined world model for lookahead planning.

---

## LLM-MPC: Language Model Model-Predictive Control

Most prior work using LLMs for robot control treats the model as a **policy**: observation → action. LLM-MPC treats it as an **imagined world model**: for each candidate action, the LLM mentally simulates a K-step future rollout in language space, then commits to the highest-scoring action. This brings a classical MPC planning loop into the language domain — without a learned dynamics model.

### Algorithm

```
┌────────────────────────────────────────────────────────────────┐
│                       LLM-MPC Control Loop                     │
│                                                                │
│  obs (39-dim)                                                  │
│    │                                                           │
│    ▼                                                           │
│  Semantic Abstraction                                          │
│    raw vector → structured text (pos, dist, vel, rotation)    │
│    │                                                           │
│    ▼                                                           │
│  Candidate Generation  (temp=0.8)                             │
│    LLM proposes N actions with physical reasoning             │
│    │                                                           │
│    ▼                                                           │
│  Mental Simulation  (temp=0.3)  ← for each candidate          │
│    LLM imagines K-step future rollout in language space       │
│    scores each trajectory 0–10                                │
│    │                                                           │
│    ▼                                                           │
│  Best action selected → env.step()                            │
│    │                                                           │
│    ▼                                                           │
│  Surprise Detection                                            │
│    heuristic L2(Δgripper + Δobj) → LLM analysis if flagged   │
│    classifies: low / medium / high                            │
│    │                                                           │
│    ▼                                                           │
│  Adaptive Strategy                                             │
│    medium surprise → soft strategy note update                │
│    3+ high surprises → deep strategy reconsideration call     │
│    │                                                           │
│    ▼                                                           │
│  Episodic Memory                                               │
│    surprising events + strategy changes compressed into       │
│    running log, injected into future prompts                  │
│    │                                                           │
│    └──────────────────────────────────────────────────────┐   │
│                                                 next step  │   │
└────────────────────────────────────────────────────────────┘
```

### Novel Contributions

| Contribution | Description |
|---|---|
| **Semantic state abstraction** | 39-dim obs → structured text with positions, distances, velocities, rotation — what the LLM can actually reason about |
| **Mental simulation (imagined MPC)** | For each of N candidates, LLM imagines a K-step future in language space and scores it; best-scoring action is selected |
| **Two-stage surprise detection** | Fast L2 heuristic filters obvious cases; LLM performs causal analysis only when motion is anomalous |
| **Adaptive strategy** | High-surprise events update a strategy note injected into future prompts; 3+ consecutive surprises trigger a full strategy reconsideration |
| **Episodic memory** | Compressed log of surprising events prevents repeating failed strategies across the episode |

---

## Repository Structure

```
prompt2metaworld/
├── p2mw/                         # installable Python package
│   ├── env/
│   │   ├── __init__.py
│   │   └── wrapper.py            # MetaWorld gym + dm_env wrappers
│   ├── baselines/
│   │   ├── cot_agent.py          # CoTAgent class
│   │   └── prompts/
│   │       ├── cot.py            # chain-of-thought prompt templates
│   │       └── meta.py           # meta-learning prompt templates
│   ├── mpc/
│   │   ├── controller.py         # LLMMPCController
│   │   ├── memory.py             # EpisodicMemory
│   │   ├── prompts.py            # MPC prompt templates + task registry
│   │   └── semantic.py           # SemanticState + parse_obs()
│   └── utils/
│       └── api.py                # shared Azure OpenAI retry helper
├── scripts/
│   ├── run_mpc.py                # LLM-MPC runner (main entry point)
│   └── run_baseline.py           # CoT baseline runner
├── configs/
│   ├── mpc.yaml                  # LLM-MPC hyperparameters
│   └── tasks.yaml                # task descriptions and object metadata
├── tests/
│   ├── test_semantic.py          # SemanticState + parse_obs unit tests
│   └── test_memory.py            # EpisodicMemory unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ibisbill/prompt2metaworld.git
cd prompt2metaworld
pip install -e .
```

Install MetaWorld following the [official instructions](https://github.com/Farama-Foundation/Metaworld).

```bash
export OPENAI_API_KEY=<your_key>
```

---

## Usage

### LLM-MPC

```bash
# Default: door-open, 5 candidates, 3-step horizon
python scripts/run_mpc.py

# More candidates, deeper lookahead
python scripts/run_mpc.py --task door-close --candidates 7 --horizon 5

# Quiet run, save log
python scripts/run_mpc.py --task reach --no-verbose --output output/reach.json

# Compare LLM-MPC against CoT baseline on the same task
python scripts/run_mpc.py --task door-open --baseline
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--task` | `door-open` | MetaWorld task name |
| `--model` | `gpt-4-32k` | Azure OpenAI deployment |
| `--candidates` | `5` | Candidate actions per step |
| `--horizon` | `3` | Mental simulation lookahead steps |
| `--max-steps` | `100` | Max environment steps |
| `--seed` | `1` | Random seed |
| `--no-verbose` | — | Suppress step-by-step output |
| `--output` | `output/mpc_run.json` | Log file path |
| `--baseline` | — | Also run CoT baseline for comparison |

**Supported tasks:** `door-open`, `door-close`, `drawer-open`, `drawer-close`, `button-press`, `reach`, `push`, `pick-place`, `hammer`, `peg-insert-side`

### CoT Baseline

```bash
python scripts/run_baseline.py --task door-open
```

---

## Testing

```bash
pip install pytest
pytest tests/
```

The tests cover `SemanticState` parsing and `EpisodicMemory` compression/retrieval without requiring a MetaWorld install.

---

## LLM-MPC vs. CoT Baseline

| Aspect | CoT Baseline | LLM-MPC |
|---|---|---|
| Action selection | Direct: obs → action | Lookahead: obs → candidates → simulation → best |
| Future reasoning | None | K-step mental simulation per candidate |
| Failure recovery | None | Surprise detection + strategy update |
| Memory | Last 10 observations | Compressed episodic log of surprising events |
| LLM calls per step | 1 | N+1 (candidates + simulations) + 1 (surprise) |

---

## Observation & Action Space

| Space | Dim | Description |
|---|---|---|
| Observation | 39 | Gripper pos (3) + gripper state (1) + object 1 pos/quat (7) + object 2 pos/quat (7) × 2 timesteps + goal pos (3) |
| Action | 4 | Δ gripper xyz (3) + gripper force (1), all in [−1, 1] |

---

## License

[MIT License](LICENSE)
