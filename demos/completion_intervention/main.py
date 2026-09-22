# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "strands-agents>=1.56,<2",
#     "typesafe-sdk>=0.7.1,<0.8",
# ]
# ///
"""
Jev + Strands Interventions — "is the agent actually done?" (single-file demo)
==============================================================================

What this shows
---------------
How to catch an agent that tries to *stop too early*, using the real Strands
Interventions API at the `after_model_call` lifecycle point, with Jev
(TypeSafe's classifier model) acting purely as the semantic judge of whether
the user's task is finished.

`after_model_call` is the moment the model has produced a candidate final
response but *before* that response is handed back to the user. It's the right
hook for a completion check: we get to inspect what the model wants to say and,
if the task isn't really done, send it back into the loop.

The flow you'll see printed to the console:

    user goal + model's candidate response
        -> Jev classifies it (a few narrow questions)
        -> plain Python `if` statements decide (the policy)
        -> intervention returns Guide(...) or Proceed()

Why Guide() sends the model "back into the loop":
-------------------------------------------------
Per the Strands docs, returning `Guide(...)` from `after_model_call`
*discards* the candidate response and *retries* the model with the feedback
injected as a user message. So a Guide here literally makes the agent try
again, now knowing what it still owes the user. `Proceed()` accepts the
response as final.

Design intent (read this before editing):
------------------------------------------
* Jev is ONLY a classifier. It answers small, atomic questions and returns
  a probability (Noul) or a labeled choice (Choice). It never decides what to
  do.
* The *policy* is ordinary, deterministic Python. Thresholds and the mapping
  from "classification" to "intervention decision" live in `after_model_call`
  as readable `if` statements you can trace by eye.
* We register the handler through `Agent(interventions=[...])` — the supported
  entry point — rather than poking at hook events by hand.
* CONVERGENCE: `after_model_call` + Guide has no built-in retry cap (the docs
  warn about this), and the event carries no attempt counter, so the handler
  counts the guides it issues and stops after a couple of tries, `Proceed()`-ing
  anyway. The demo can never loop forever.

Why a SCRIPTED model (and not a real LLM)?
------------------------------------------
The teaching moment is: premature answer -> intervention -> recovery. Driving
that with a real LLM is unreliable — depending on the prompt the model either
refuses to finish (and never converges) or answers everything on the first try
(so the premature moment never happens). To make the flow identical on every
run we give the Strands `Agent` a tiny SCRIPTED model (`ScriptedModel` below)
that:

  1. first returns a deliberately PREMATURE answer (only Paris), then
  2. after it's been guided, returns the COMPLETE answer (Paris + Tokyo).

Only the agent's *model* is scripted. Jev is still the real classifier doing
real work on the real candidate text — which is the part the demo is about.

Run it with uv from the repo root (no manual install — deps are declared
inline above, and --env-file loads OPENROUTER_API_KEY from .env):

    uv run --env-file .env demos/completion_intervention/main.py

Because the agent's model is scripted, this demo needs NO AWS/Bedrock
credentials — only OPENROUTER_API_KEY for Jev.

Optimized for readability, not production robustness.
"""

from __future__ import annotations

import os
from typing import AsyncIterable

# --- Strands: the agent, its Model base class, the real Interventions API ---
from strands import Agent
from strands.interventions import Guide, InterventionHandler, Proceed
from strands.models import Model

# --- Jev: the TypeSafe classifier client ------------------------------------
from typesafe_sdk import TypeSafeClient, Choice, Noul


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
# 1. A scripted "model" so the teaching flow is identical every run
# ---------------------------------------------------------------------------
# The two responses the fake model will hand back, in order. The first is
# deliberately PREMATURE (Paris only); the second is COMPLETE (Paris + Tokyo).
PREMATURE_ANSWER = "It's 21°C and sunny in Paris."
COMPLETE_ANSWER = "It's 21°C and sunny in Paris, and 24°C and clear in Tokyo."


class ScriptedModel(Model):
    """A minimal Strands model that returns canned responses in sequence.

    Strands models are async generators of "stream events". A plain text
    assistant turn is just this sequence of chunks:

        messageStart -> contentBlockStart -> contentBlockDelta(text)
                     -> contentBlockStop  -> messageStop(end_turn)

    We yield exactly that, pulling the text from `self._responses` one call at
    a time. This is the ONLY scripted piece — Jev's classification downstream is
    real. Not a general-purpose model; just enough to make the demo deterministic.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    async def stream(self, *args, **kwargs) -> AsyncIterable[dict]:
        # Hand back the next scripted response; repeat the last one if the agent
        # somehow calls more times than we scripted (keeps things safe).
        text = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    # The rest of the Model interface is abstract but unused by this demo.
    def update_config(self, **model_config) -> None:
        return None

    def get_config(self) -> dict:
        return {}

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError("ScriptedModel does not support structured_output")


# ---------------------------------------------------------------------------
# 2. Helper: read the assistant's candidate response out of the event
# ---------------------------------------------------------------------------
def message_text(message: dict | None) -> str:
    """Join the human-readable text blocks of a single Strands message.

    Strands stores a message as {"role": ..., "content": [blocks]}. A block is
    text when it has a "text" key; tool-use / tool-result blocks are skipped.
    Returns "" if there's no message or no text.
    """
    if not message:
        return ""
    texts = [
        block["text"]
        for block in message.get("content", [])
        if isinstance(block, dict) and "text" in block
    ]
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# 3. Ask Jev whether the task is actually finished
# ---------------------------------------------------------------------------
def classify_completion(user_goal: str, candidate_response: str):
    """Ask Jev a few NARROW questions about the candidate final response.

    Each question is atomic (one judgment):
      * completeness   -> Choice: COMPLETE / PARTIAL / NOT_ANSWERED
      * work_remains   -> Noul:   is there meaningful work still to do?
      * stopping_early -> Noul:   is the model trying to stop prematurely?

    Jev only OBSERVES. It returns a labeled choice (+ per-label probabilities
    and a confidence) and yes-probabilities. The policy below does the deciding.
    """
    state = (
        "A user gave an AI assistant a goal. The assistant then produced the "
        "candidate final response below. Judge the response against the goal.\n\n"
        f"--- USER GOAL ---\n{user_goal}\n\n"
        f"--- ASSISTANT CANDIDATE RESPONSE ---\n{candidate_response}"
    )

    response = jev.system_one(
        model=JEV_MODEL,
        state=state,
        questions={
            # How fully does the response satisfy the goal? One label wins.
            "completeness": Choice(
                instructions=(
                    "How completely does the assistant's response satisfy the "
                    "user's goal?"
                ),
                criteria={
                    # Every part of the request is addressed.
                    "COMPLETE": "The response fully satisfies everything the user asked for.",
                    # Some of it is addressed; some is still missing.
                    "PARTIAL": "The response addresses part of the request but leaves some of it unfinished.",
                    # None of it is actually answered.
                    "NOT_ANSWERED": "The response does not actually answer the user's request.",
                },
            ),
            # Is there real, substantive work still owed to the user?
            "work_remains": Noul(
                instructions="Is there meaningful work still left to do to satisfy the user's goal?"
            ),
            # Is the assistant wrapping up / signing off before finishing?
            "stopping_early": Noul(
                instructions="Is the assistant trying to end the conversation before the goal is fully met?"
            ),
        },
    )
    return response.answers


# ---------------------------------------------------------------------------
# 4. The intervention handler
#    Jev = semantic classifier. Python `if` = the policy.
# ---------------------------------------------------------------------------
class JevCompletionReviewer(InterventionHandler):
    """Reviews each candidate final response before it reaches the user.

    `after_model_call` runs *after* the model produces a response but *before*
    it's returned. We hand the user's goal + the candidate response to Jev,
    then use deterministic thresholds to return a typed decision:
      * Guide(...)  -> discard this response and RETRY, telling the model what
                       still needs doing (this is the "back into the loop" step).
      * Proceed()   -> accept the response as the final answer.
    """

    name = "jev-completion-reviewer"

    # Probability threshold — the policy knob, owned by us, not Jev.
    YES = 0.65  # treat a Jev probability at/above this as a confident "yes"

    # Convergence guard: after_model_call + Guide has NO built-in retry cap
    # (the Strands docs warn about this), so we enforce our own. `AfterModelCallEvent`
    # carries no attempt counter, so we count the guides we've issued this turn
    # ourselves and stop after MAX_GUIDES, returning Proceed() to guarantee the
    # loop always terminates.
    MAX_GUIDES = 3

    def __init__(self) -> None:
        self._guides_issued = 0  # reset each time we accept a response (Proceed)

    def after_model_call(self, event):
        user_goal = self._user_goal(event.agent.messages)

        # The just-produced candidate response is on the EVENT, not in
        # agent.messages yet — Strands hasn't committed it to history when this
        # hook fires. Read it from stop_response.message. It's None only if the
        # model call failed.
        stop_response = getattr(event, "stop_response", None)
        candidate_message = stop_response.message if stop_response else None
        candidate = message_text(candidate_message)

        print("\n" + "=" * 70)
        print(f"[intervention] reviewing candidate response (guides so far: {self._guides_issued})")
        print(f"[intervention] user goal: {user_goal!r}")
        print(f"[intervention] candidate: {candidate!r}")

        # No text to judge (model error) -> nothing to review, let it continue.
        if not candidate:
            print("[policy] no assistant text to review -> Proceed")
            return Proceed()

        # --- Convergence guard (deterministic, checked first) --------------
        if self._guides_issued >= self.MAX_GUIDES:
            print(
                f"[policy] already guided {self._guides_issued}x (MAX_GUIDES="
                f"{self.MAX_GUIDES}) -> Proceed (stop guiding to guarantee convergence)"
            )
            self._guides_issued = 0
            return Proceed()

        # --- Jev classifies (semantic step) --------------------------------
        answers = classify_completion(user_goal, candidate)
        completeness = answers["completeness"].choice
        completeness_conf = answers["completeness"].confidence
        work_remains = answers["work_remains"].noul
        stopping_early = answers["stopping_early"].noul

        print("[intervention] Jev classifications:")
        print(f"    completeness   : {completeness} (confidence {completeness_conf:.2f})")
        print(f"    work_remains   : {work_remains:.2f}  (probability of 'yes')")
        print(f"    stopping_early : {stopping_early:.2f}  (probability of 'yes')")

        # --- Python policy decides (deterministic step) --------------------
        # Each branch maps a classification pattern to a typed Strands action.

        # 1) The model didn't actually answer at all -> send it back.
        if completeness == "NOT_ANSWERED":
            print("[policy] response does not answer the goal -> Guide (retry)")
            return self._guide(
                "You haven't actually answered the user's request yet. "
                "Do the work the user asked for before responding."
            )

        # 2) The model answered part of it but is trying to stop -> send it back.
        if completeness == "PARTIAL" and work_remains >= self.YES:
            print("[policy] partial answer with work remaining -> Guide (retry)")
            return self._guide(
                "You've only partially completed the request. Re-read the "
                "user's goal, identify every part you have NOT addressed "
                "yet, and finish those before giving a final answer."
            )

        # 3) It looks complete but Jev thinks the model is bailing early, and
        #    work still remains -> nudge it to finish rather than sign off.
        if stopping_early >= self.YES and work_remains >= self.YES:
            print("[policy] stopping prematurely with work remaining -> Guide (retry)")
            return self._guide(
                "It looks like you're wrapping up before the task is done. "
                "Make sure every part of the user's goal is fully handled, "
                "then respond."
            )

        # Otherwise: the task is complete. Accept the response as final and
        # reset the per-turn guide counter for the next request.
        print("[policy] task looks complete -> Proceed (accept response)")
        self._guides_issued = 0
        return Proceed()

    # -- tiny helpers ------------------------------------------------------
    def _guide(self, feedback: str) -> Guide:
        """Return a Guide (discard + retry) and count it for the convergence cap."""
        self._guides_issued += 1
        return Guide(feedback=feedback)

    @staticmethod
    def _user_goal(messages: list[dict]) -> str:
        """The original user goal is the first user message in the transcript."""
        for message in messages:
            if message.get("role") == "user":
                for block in message.get("content", []):
                    if isinstance(block, dict) and "text" in block:
                        return block["text"]
        return "(no user goal found)"


# ---------------------------------------------------------------------------
# 5. Wire it up and run one request the model will (deterministically) under-answer
# ---------------------------------------------------------------------------
def main() -> None:
    agent = Agent(
        # The scripted model: premature answer first, complete answer after a
        # guide. This is what makes the demo identical on every run.
        model=ScriptedModel([PREMATURE_ANSWER, COMPLETE_ANSWER]),
        # Register through the supported interventions config — NOT by mutating
        # hook events by hand.
        interventions=[JevCompletionReviewer()],
    )

    # Two-part goal on purpose. The scripted model answers only Paris first; Jev
    # classifies that as PARTIAL with work remaining, and the policy returns a
    # Guide that discards the response and retries. On the retry the scripted
    # model returns the Paris + Tokyo answer -> Jev says COMPLETE -> Proceed.
    user_request = "What's the weather in Paris and in Tokyo?"

    print(f"\nUSER: {user_request}")
    result = agent(user_request)

    print("\n" + "=" * 70)
    print("FINAL ASSISTANT REPLY:")
    print(result)


if __name__ == "__main__":
    main()
