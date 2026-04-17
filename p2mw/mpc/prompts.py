"""
Prompt templates for the LLM-MPC controller.

Design philosophy:
  - Every prompt has a strict output format so responses can be parsed reliably.
  - Prompts are separated by concern: candidate generation, simulation,
    surprise analysis, and strategy updates are independent calls.
  - Temperature is tuned per prompt: low for simulation/parsing, higher for
    creative candidate generation.
"""

# ---------------------------------------------------------------------------
# System prompt — shared by all calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise controller for a Sawyer robot arm in the MetaWorld simulation.

Robot specifications:
  - Observation: 39-dim vector (parsed and given to you as structured text)
  - Action: [dx, dy, dz, gripper_force], each value in [-1.0, 1.0]
      dx/dy/dz : desired change in gripper position (up to ~5 cm per step at 100 Hz)
      gripper_force: +1 = fully closed, -1 = fully open
  - The gripper must physically contact an object before it can exert force on it.

Physics principles you must respect:
  1. Close the distance to the target object before trying to manipulate it.
  2. Large actions cause overshooting — use smaller values when near the target.
  3. The gripper must approach from the correct angle to exert useful force.
  4. Joint torques are limited — pushing perpendicular to a revolving joint is most efficient.
  5. Momentum persists — a moving object continues in the same direction if force is removed.

Always reason step-by-step before producing your final answer.
"""

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

CANDIDATE_GENERATION_PROMPT = """\
You are deciding the next action for a robot arm.

Current semantic state:
{semantic_state}

Task: {task_name}
Description: {task_description}

Episode history (what has been tried so far):
{episode_history}

Generate exactly {k} distinct candidate actions. Each candidate should represent
a meaningfully different strategy (not just minor numeric variations).

For each candidate, use this exact format:

CANDIDATE <number>:
Reasoning: <one sentence of physical reasoning>
Action: [<dx>, <dy>, <dz>, <gripper_force>]
Expected: <one sentence describing the immediate effect>
"""

# ---------------------------------------------------------------------------
# Mental simulation
# ---------------------------------------------------------------------------

MENTAL_SIMULATION_PROMPT = """\
You are simulating the future outcome of a candidate robot action.

Current semantic state:
{semantic_state}

Candidate action: {action}
Motivation: {reasoning}

Task: {task_name} — {task_description}

Simulate the robot executing this action and the next {horizon} control steps
(assuming the same action is repeated with natural follow-through):

STEP 1:
  Gripper position (estimated): (x, y, z)
  Object 1 position (estimated): (x, y, z)
  Key physical event: <what happens>

STEP 2:
  Gripper position (estimated): (x, y, z)
  Object 1 position (estimated): (x, y, z)
  Key physical event: <what happens>

STEP 3:
  Gripper position (estimated): (x, y, z)
  Object 1 position (estimated): (x, y, z)
  Key physical event: <what happens>

Overall assessment:
  Progress toward goal (0–10): <score>
  Main risk: <one sentence>
  Verdict: <good | mediocre | poor>

SIMULATION_SCORE: <integer 0-10>
"""

# ---------------------------------------------------------------------------
# Surprise analysis
# ---------------------------------------------------------------------------

SURPRISE_ANALYSIS_PROMPT = """\
After executing action {action}, the robot's actual new state differs from expectations.

Predicted next state:
{predicted_state}

Actual observed state:
{actual_state}

Measured differences:
{differences}

Analyse this discrepancy:

SURPRISE_LEVEL: <low | medium | high>
CAUSE: <one sentence explaining why the prediction was wrong>
STRATEGY: <one sentence describing what should change in the next steps>
"""

# ---------------------------------------------------------------------------
# Deep strategy update (triggered after repeated high-surprise events)
# ---------------------------------------------------------------------------

STRATEGY_UPDATE_PROMPT = """\
The robot has encountered {n_surprises} high-surprise events in the past {window} steps,
indicating that the current approach is not working.

Surprise log:
{surprise_history}

Full episode context:
{episode_history}

Task: {task_name} — {task_description}

Based on this pattern of failures, produce a fundamentally revised strategy:
  1. What approach has failed and why?
  2. What alternative approach should be tried?
  3. What specific action adjustments does this imply?

STRATEGY_UPDATE: <2–3 sentences with a concrete, actionable new approach>
"""

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS: dict[str, str] = {
    "door-open":     ("Push the door open by rotating it around its revolving joint. "
                      "The gripper must contact the door handle and push it sideways."),
    "door-close":    ("Push the door closed by rotating it back to the closed position "
                      "around its revolving joint."),
    "drawer-open":   ("Grasp the drawer handle and pull it outward away from the cabinet."),
    "drawer-close":  ("Push the drawer handle inward toward the cabinet to close it."),
    "button-press":  ("Move the gripper directly above the button and press it downward "
                      "until it is fully depressed."),
    "reach":         ("Move the gripper to the goal position as directly as possible."),
    "push":          ("Push the puck along the table surface to the goal position."),
    "pick-place":    ("Grasp the object, lift it, and place it at the goal position."),
    "hammer":        ("Strike the nail with the hammer head to drive it into the surface."),
    "peg-insert-side": ("Insert the peg into the hole from the side."),
}
