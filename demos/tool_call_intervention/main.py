# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "strands-agents>=1.56,<2",
#     "typesafe-sdk>=0.7.1,<0.8",
# ]
# ///
"""
Jev + Strands Interventions — a single-file teaching demo
=========================================================

What this shows
---------------
How to gate a proposed tool call *before it runs* using the real Strands
Interventions API, with Jev (TypeSafe's classifier model) acting purely as the
semantic judge.

The flow you'll see printed to the console:

    model proposes tool call
        -> Jev classifies it (narrow yes/no questions)
        -> plain Python `if` statements decide (the policy)
        -> intervention returns Guide(...) or Proceed()

Design intent (read this before editing):
------------------------------------------
* Jev is ONLY a classifier. It answers small, atomic questions and returns
  probabilities. It never decides what to do.
* The *policy* is ordinary, deterministic Python. Thresholds and the mapping
  from "classification" to "intervention decision" live in `before_tool_call`
  as readable `if` statements you can trace by eye.
* We register the handler through `Agent(interventions=[...])` — the supported
  entry point — rather than poking at hook events by hand.

To keep the demo legible we use one tool (`get_weather`) and one deliberately
incomplete request ("What's the weather?" with no location). A well-behaved
model *should* ask for the city, but models sometimes guess. This handler
catches the guess: Jev notices the arguments aren't grounded in the
conversation, and the policy converts that into a `Guide(...)` telling the
model to ask the user first.

Run it with uv from the repo root (no manual install — deps are declared
inline above, and --env-file loads OPENROUTER_API_KEY from .env):

    uv run --env-file .env demos/tool_call_intervention/main.py

The agent itself uses Strands' default Bedrock provider, so it also relies on
your ambient AWS credentials.

Optimized for readability, not production robustness.
"""

from __future__ import annotations

import json
import os

# --- Strands: the agent + the real Interventions API -----------------------
from strands import Agent, tool
from strands.interventions import Guide, InterventionHandler, Proceed

# --- Jev: the TypeSafe classifier client ------------------------------------
from typesafe_sdk import TypeSafeClient, Noul


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
# 1. One simple tool
# ---------------------------------------------------------------------------
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a specific city.

    Args:
        location: The city to look up, e.g. "Seattle" or "Paris".
    """
    # Fake data — the point of the demo is the intervention, not real weather.
    return f"It's 21°C and sunny in {location}."


# ---------------------------------------------------------------------------
# 2. Helpers: turn Strands state into something Jev can read
# ---------------------------------------------------------------------------
def render_conversation(messages: list[dict]) -> str:
    """Flatten Strands' message list into plain text for the classifier state.

    Strands stores each message as {"role": ..., "content": [blocks]}. We only
    need the human-readable text blocks here, so we skip tool/plumbing blocks.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        for block in message.get("content", []):
            if isinstance(block, dict) and "text" in block:
                lines.append(f"{role}: {block['text']}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def classify_tool_call(conversation: str, tool_name: str, tool_input: dict):
    """Ask Jev several NARROW questions about the proposed tool call.

    Each question is atomic (one judgment). Jev returns, per question, the
    probability that the answer is "yes" (`.noul`, in 0..1). We do NOT ask Jev
    to decide anything — just to observe.
    """
    state = (
        "A conversation between a user and an AI assistant is below, followed "
        "by a tool call the assistant now wants to make.\n\n"
        f"--- CONVERSATION ---\n{conversation}\n\n"
        f"--- PROPOSED TOOL CALL ---\n"
        f"tool: {tool_name}\n"
        f"arguments: {json.dumps(tool_input)}"
    )

    response = jev.system_one(
        model=JEV_MODEL,
        state=state,
        questions={
            # Does the chosen tool actually serve what the user asked for?
            "matches_intent": Noul(
                instructions="Does the proposed tool match what the user is actually asking for?"
            ),
            # Is the assistant missing information it needs to call correctly?
            "missing_info": Noul(
                instructions="Is required information missing that the tool needs to run correctly?"
            ),
            # Are the argument VALUES supported by the conversation, or invented?
            "args_grounded": Noul(
                instructions="Are the tool's argument values grounded in facts the user actually provided?"
            ),
            # Is the assistant jumping ahead before it should?
            "premature": Noul(
                instructions="Is it premature to call this tool now, before clarifying with the user?"
            ),
        },
    )
    return response.answers


# ---------------------------------------------------------------------------
# 3. The intervention handler
#    Jev = semantic classifier. Python `if` = the policy.
# ---------------------------------------------------------------------------
class JevToolCallReviewer(InterventionHandler):
    """Reviews each proposed tool call before it executes.

    `before_tool_call` runs *before* Strands executes the tool. We hand the
    conversation + proposed call to Jev, then use deterministic thresholds to
    return a typed decision: Guide(...) to send the model back with feedback,
    or Proceed() to let the call run.
    """

    name = "jev-tool-call-reviewer"

    # Probability thresholds — these are the policy knobs, owned by us, not Jev.
    # Tune them to taste; higher = stricter about blocking.
    YES = 0.65  # treat a Jev probability at/above this as a confident "yes"

    def before_tool_call(self, event):
        tool_name = event.tool_use["name"]
        tool_input = event.tool_use.get("input", {})
        conversation = render_conversation(event.agent.messages)

        print("\n" + "=" * 70)
        print(f"[intervention] model proposes: {tool_name}({json.dumps(tool_input)})")

        # --- Jev classifies (semantic step) --------------------------------
        answers = classify_tool_call(conversation, tool_name, tool_input)
        matches_intent = answers["matches_intent"].noul
        missing_info = answers["missing_info"].noul
        args_grounded = answers["args_grounded"].noul
        premature = answers["premature"].noul

        print("[intervention] Jev classifications (probability of 'yes'):")
        print(f"    matches_intent : {matches_intent:.2f}")
        print(f"    missing_info   : {missing_info:.2f}")
        print(f"    args_grounded  : {args_grounded:.2f}")
        print(f"    premature      : {premature:.2f}")

        # --- Python policy decides (deterministic step) --------------------
        # Each branch maps a classification pattern to a typed Strands action.
        if matches_intent < self.YES:
            decision = Guide(
                feedback=(
                    "That tool doesn't match what the user asked for. "
                    "Reconsider which tool (if any) fits their request."
                )
            )
            print("[policy] tool does not match intent -> Guide")
            return decision

        if missing_info >= self.YES:
            decision = Guide(
                feedback=(
                    "You're missing information this tool needs. Ask the user "
                    "for the required details before calling it."
                )
            )
            print("[policy] required info is missing -> Guide")
            return decision

        if args_grounded < self.YES:
            decision = Guide(
                feedback=(
                    "The arguments aren't grounded in anything the user said — "
                    "they look guessed. Ask the user to confirm the values "
                    "instead of inventing them."
                )
            )
            print("[policy] arguments not grounded in conversation -> Guide")
            return decision

        if premature >= self.YES:
            decision = Guide(
                feedback=(
                    "It's too early to call this tool. Clarify with the user "
                    "first, then try again."
                )
            )
            print("[policy] call is premature -> Guide")
            return decision

        # All checks passed: let the tool run unchanged.
        print("[policy] all checks passed -> Proceed")
        return Proceed()


# ---------------------------------------------------------------------------
# 4. Wire it up and run one deliberately incomplete request
# ---------------------------------------------------------------------------
def main() -> None:
    agent = Agent(
        tools=[get_weather],
        # Register through the supported interventions config — NOT by mutating
        # hook events by hand.
        interventions=[JevToolCallReviewer()],
        # This system prompt models a common failure mode: an "eager" agent that
        # would rather guess than ask. We instruct it to assume a default city
        # (Seattle) so it actually proposes a tool call with an ungrounded
        # argument — which is exactly what we want the intervention to catch.
        # In a real app the policy is your safety net for exactly this behavior.
        system_prompt=(
            "You are an eager weather assistant. Always answer weather questions "
            "by calling the get_weather tool immediately. Never ask the user for "
            "clarification — if no city is given, just assume Seattle and call "
            "the tool with location='Seattle'."
        ),
    )

    # Deliberately incomplete: no city is given. The eager model will guess
    # "Seattle" and call get_weather. Jev flags the argument as ungrounded, and
    # the policy converts that into a Guide telling the model to ask first.
    user_request = "What's the weather?"

    print(f"\nUSER: {user_request}")
    result = agent(user_request)

    print("\n" + "=" * 70)
    print("FINAL ASSISTANT REPLY:")
    print(result)


if __name__ == "__main__":
    main()
