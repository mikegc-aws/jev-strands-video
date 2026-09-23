# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "strands-agents>=1.56,<2",
#     "typesafe-sdk>=0.7.1,<0.8",
# ]
# ///
"""
Jev + Strands — switch the model mid-conversation to dial cost up and down (single-file demo)
=============================================================================================

*** EXPERIMENTAL — not a production pattern. ***
This demo shows that mid-conversation model switching is POSSIBLE and explores
how it behaves. It works, but it is by no means perfect. Scaling UP to a bigger
model on a hard turn is reasonable; scaling DOWN to a cheaper model is risky —
the smaller model still has to read (and keep up with) all the prior context,
and a "simple-looking" follow-up can secretly depend on hard earlier reasoning,
so quality can quietly degrade. See the "Honest caveats" section in this demo's
README before borrowing any of this. Treat it as a starting point, not a recipe.

What this shows (the experiment)
--------------------------------
ONE Strands agent, ONE continuous multi-turn conversation — and we swap the
underlying Amazon Bedrock model *between turns* based on how hard the latest
question is. Easy turn -> cheap/fast "small" model. Hard turn -> capable/pricey
"big" model. The conversation history carries across the swap untouched, so a
turn answered by the small model still sees everything the big model said
earlier, and vice-versa.

Why this works
--------------
In Strands the conversation lives on the AGENT (`agent.messages`), not on the
model. The model is just the component that generates the next turn. So
reassigning `agent.model` before a turn changes *who* answers next without
disturbing the accumulated history. Because our three models share a large
context window, there's no context-size mismatch when we switch — we're purely
trading capability (and cost) per turn.

The flow you'll see printed to the console, once per turn:

    next user turn
        -> Jev PICKS the size (one narrow question: small / medium / big?)
        -> that winning label maps straight to a Bedrock model
        -> we set `agent.model` to it
        -> the SAME agent answers, continuing the SAME conversation

Design intent (read this before editing):
------------------------------------------
* Jev is ONLY a classifier. Because we have exactly three discrete sizes, we let
  it answer the question directly — "which size does this turn need?" — via a
  `Choice` over the labels small / medium / big. It returns the winning label
  (its highest-probability option) plus a confidence. No numeric score to
  re-bucket, no cutoffs to tune.
* We still own the mapping. The label -> model lookup lives in `MODELS_BY_SIZE`,
  and because the Choice labels ARE the sizes, the winning choice indexes
  straight into it. Jev classifies; our code decides what a label means and
  which model it runs.

The three sizes (Amazon Bedrock, Anthropic Claude family):
----------------------------------------------------------
    small  -> Claude Haiku 4.5   (global.anthropic.claude-haiku-4-5-20251001-v1:0)
    medium -> Claude Sonnet 4.6  (global.anthropic.claude-sonnet-4-6)
    big    -> Claude Opus 4.6    (global.anthropic.claude-opus-4-6-v1)

These are cross-Region inference-profile IDs (the "global." prefix). Swap them
for whatever you have model access to in your account/region — the switching
logic doesn't care what the concrete IDs are.

Run it with uv from the repo root (no manual install — deps are declared
inline above, and --env-file loads OPENROUTER_API_KEY from .env):

    uv run --env-file .env demos/model_switching/main.py

This calls Bedrock, so you need AWS credentials with access to the three models
above (plus OPENROUTER_API_KEY for Jev).

Optimized for readability, not production robustness.
"""

from __future__ import annotations

import os

# --- Strands: the agent + the Bedrock model provider -----------------------
from strands import Agent
from strands.models import BedrockModel

# --- Jev: the TypeSafe classifier client ------------------------------------
from typesafe_sdk import TypeSafeClient, Choice


# ---------------------------------------------------------------------------
# 0. Config / secrets
# ---------------------------------------------------------------------------
# OPENROUTER_API_KEY comes from the environment. Run via
# `uv run --env-file .env ...` and uv loads it from the .env file for you.
#
# Jev over OpenRouter. The leading "~" in the model id is intentional.
JEV_MODEL = "~typesafe/jev-latest"
jev = TypeSafeClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api",
)


# ---------------------------------------------------------------------------
# 1. The t-shirt-sized model catalog
# ---------------------------------------------------------------------------
# One Bedrock model ID per size. These are the ONLY Bedrock-specific details in
# the demo; everything else is generic switching logic. Change the IDs to models
# you have access to and the rest of the file keeps working.
MODELS_BY_SIZE: dict[str, str] = {
    # Fast + cheap. Good enough for greetings, lookups, simple rewrites.
    "small": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    # Balanced. The everyday workhorse for normal reasoning and coding.
    "medium": "global.anthropic.claude-sonnet-4-6",
    # Most capable (and priciest). Reserve for genuinely hard, multi-step work.
    "big": "global.anthropic.claude-opus-4-6-v1",
}


# ---------------------------------------------------------------------------
# 2. Ask Jev to pick the size (semantic step)
# ---------------------------------------------------------------------------
# We have exactly three discrete buckets and three models, so we ask Jev the
# question directly: WHICH size fits this turn? `Choice` returns the winning
# label (its highest-probability option) plus a confidence — no numeric score
# to re-bucket, no cutoffs to tune. The labels ARE the sizes, so the winning
# choice maps straight to a model. Jev still only classifies; we still own the
# label->model mapping (MODELS_BY_SIZE) and could still override on low
# confidence if we wanted to.
def classify_size(turn: str):
    """Ask Jev ONE narrow question: which t-shirt size does this turn need?

    Returns the answer object for the `size` question, which carries:
      * .choice       -> the winning label: "small" | "medium" | "big"
      * .confidence   -> how sure Jev is
      * .probabilities -> the full distribution over the three labels
    """
    response = jev.system_one(
        model=JEV_MODEL,
        state=f"A user said the following turn in a conversation with an AI assistant:\n\n{turn}",
        questions={
            "size": Choice(
                instructions=(
                    "Which size model is needed to answer this turn *well*? "
                    "Pick the smallest one that can do the job."
                ),
                criteria={
                    # The keys are the t-shirt sizes themselves — the winning
                    # label indexes straight into MODELS_BY_SIZE.
                    "small": "Trivial: a greeting, a simple fact, or a one-step request answerable in a sentence.",
                    "medium": "Moderate: ordinary reasoning, explanation, or a routine coding task.",
                    "big": "Hard: multi-step reasoning, novel problem-solving, tricky design, or deep analysis.",
                },
            ),
        },
    )
    return response.answers["size"]


# ---------------------------------------------------------------------------
# 4. One agent, one conversation — swap the model turn by turn
# ---------------------------------------------------------------------------
def run_turn(agent: Agent, turn: str) -> None:
    """Classify the turn, switch the agent's model to match, then answer.

    The KEY line is `agent.model = BedrockModel(...)`: it changes which model
    generates the next turn WITHOUT touching `agent.messages`, so the
    conversation continues seamlessly on the newly chosen model.
    """
    answer = classify_size(turn)   # Jev picks the size directly (semantic step)
    size = answer.choice           # the winning label IS the size

    print("\n" + "=" * 70)
    print(f"USER: {turn}")
    print(f"[jev]    chosen size      : {size}  (confidence {answer.confidence:.2f})")
    print(f"[policy] bedrock model    : {MODELS_BY_SIZE[size]}")
    print(f"[state]  history so far   : {len(agent.messages)} messages (persists across the swap)")

    # THE SWITCH: change the model on the LIVE agent. History is untouched.
    agent.model = BedrockModel(model_id=MODELS_BY_SIZE[size])

    # Same agent, same conversation — it now answers on the chosen model.
    result = agent(turn)

    print("\nASSISTANT REPLY:")
    print(result)


def main() -> None:
    # One agent, reused for every turn. It starts on the medium model; each turn
    # may switch it up or down. We give it no tools — this experiment is purely
    # about swapping the generating model across a continuous conversation.
    agent = Agent(model=BedrockModel(model_id=MODELS_BY_SIZE["medium"]))

    # A scripted conversation that deliberately spans the difficulty range so you
    # can watch the model tier move small -> medium -> big turn by turn. Turns 2+
    # refer back to earlier turns, which only works because the shared history
    # survives every model swap.
    conversation = [
        # Trivial -> expect "small".
        "Hey! I'm building a small weather app. What's the capital of France?",
        # Moderate, builds on turn 1 -> expect "medium".
        "For that app, explain the difference between a process and a thread, with an example.",
        # Hard, builds on the whole thread -> expect "big".
        (
            "Now design the concurrency model for that weather app's backend so it "
            "stays fair across regions during network partitions. Walk through the "
            "failure modes and the trade-offs of your approach."
        ),
        # Back down to trivial -> expect "small" again (proves it dials DOWN too).
        "Thanks! Can you give the app a fun one-line tagline?",
    ]

    for turn in conversation:
        run_turn(agent, turn)

    print("\n" + "=" * 70)
    print(f"Conversation complete: {len(agent.messages)} total messages in one continuous thread.")


if __name__ == "__main__":
    main()
