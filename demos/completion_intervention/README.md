# Completion Intervention

Catch an agent that tries to **stop before the task is done**, using the real
[Strands Interventions API](https://strandsagents.com/docs/api/python/strands.interventions.actions/)
at the `after_model_call` lifecycle point, with [Jev](https://typesafe.ai)
acting purely as the semantic judge of "is this actually finished?"

## The flow

```
user goal + model's candidate response
    → Jev classifies it (a few narrow questions)
    → plain Python `if` statements decide (the policy)
    → intervention returns Guide(...) or Proceed()
```

- **Why `after_model_call`.** It fires the moment the model has produced a
  candidate final response but *before* that response goes back to the user —
  the natural place to ask "are we done?" Returning `Guide(...)` here
  **discards** the candidate response and **retries** the model with the
  feedback injected as a user message, so a Guide literally sends the agent
  back into the loop. `Proceed()` accepts the response as final.
- **Jev is only a classifier.** It answers small, atomic questions about the
  candidate response and returns typed results: a `Choice` for completeness
  (`COMPLETE` / `PARTIAL` / `NOT_ANSWERED`) plus two `Noul` yes/no probabilities
  (is meaningful work still remaining, is the model stopping prematurely). It
  never decides the action.
- **The policy is deterministic Python.** An `after_model_call` handler maps
  those classifications to a typed decision (`Guide(...)` or `Proceed()`)
  through readable `if` statements against a single threshold you can tune.
- **Convergence is enforced by us.** `after_model_call` + `Guide` has no
  built-in retry cap (the Strands docs warn about this), and the event carries
  no attempt counter, so the handler counts the guides it issues and stops after
  `MAX_GUIDES`, returning `Proceed()` to guarantee the loop always terminates.
- **Registered the supported way.** The handler is attached via
  `Agent(interventions=[...])`, not by mutating hook events by hand.

## The scenario

A deliberately two-part request — the weather in **Paris and Tokyo**. To make
the teaching flow identical on every run, the agent uses a tiny **scripted
model** (`ScriptedModel`) instead of a real LLM: it returns a *premature*
answer (Paris only) on the first call, and the *complete* answer (Paris +
Tokyo) after it's been guided. Jev classifies the first candidate as `PARTIAL`
with work remaining, and the policy returns a `Guide(...)` that discards it and
retries. The scripted model then returns the complete answer, Jev says
`COMPLETE`, and the policy returns `Proceed()`.

Only the agent's *model* is scripted — **Jev is still doing real classification
on the real candidate text**, which is the part the demo is teaching.

## Why a scripted model?

Driving the premature → intervention → recovery flow with a real LLM is
unreliable: depending on the prompt the model either refuses to finish (and the
loop never converges) or answers everything on the first try (so the premature
moment never happens). A scripted model makes the demonstration deterministic
so the intervention is always the visible star. Swapping in a real model
(e.g. Strands' Bedrock provider) is a one-line change if you want to see it
behave with a live LLM.

## Run it

From the repo root:

```bash
uv run --env-file .env demos/completion_intervention/main.py
```

Because the agent's model is scripted, this demo needs **only**
`OPENROUTER_API_KEY` (for Jev) — no AWS/Bedrock credentials required. See the
[top-level README](../../README.md) for setup.

## Expected output (abridged)

```
USER: What's the weather in Paris and in Tokyo?
It's 21°C and sunny in Paris.
======================================================================
[intervention] reviewing candidate response (guides so far: 0)
[intervention] user goal: "What's the weather in Paris and in Tokyo?"
[intervention] candidate: "It's 21°C and sunny in Paris."
[intervention] Jev classifications:
    completeness   : PARTIAL (confidence 0.88)
    work_remains   : 0.91  (probability of 'yes')
    stopping_early : 0.86  (probability of 'yes')
[policy] partial answer with work remaining -> Guide (retry)
It's 21°C and sunny in Paris, and 24°C and clear in Tokyo.
======================================================================
[intervention] reviewing candidate response (guides so far: 1)
    completeness   : COMPLETE (confidence 0.90)
[policy] task looks complete -> Proceed (accept response)
```

Exact labels and probabilities will vary run to run. (Strands streams the
discarded first response to the console before the retry — that's expected.)

> Implementation note: at `after_model_call` the candidate response is **not**
> in `agent.messages` yet — Strands hasn't committed it to history. The handler
> reads it from `event.stop_response.message`. The event also carries no attempt
> counter, so the handler counts the guides it issues itself to enforce a
> convergence cap.
