# Tool Call Intervention

Gate a proposed agent tool call **before it runs**, using the real
[Strands Interventions API](https://strandsagents.com/docs/user-guide/concepts/agents/interventions/),
with [Jev](https://typesafe.ai) acting purely as the semantic classifier.

## The flow

```
model proposes a tool call
    → Jev classifies it (narrow yes/no questions)
    → plain Python `if` statements decide (the policy)
    → intervention returns Guide(...) or Proceed()
```

- **Jev is only a classifier.** It answers small, atomic yes/no questions
  (`Noul`) about the proposed call — does the tool match intent, is required
  info missing, are the arguments grounded in the conversation, is the call
  premature — and returns a probability for each. It never decides the action.
- **The policy is deterministic Python.** A `before_tool_call` handler maps
  those probabilities to a typed decision (`Guide(...)` or `Proceed()`) through
  readable `if` statements against a single threshold you can tune.
- **Registered the supported way.** The handler is attached via
  `Agent(interventions=[...])`, not by mutating hook events by hand.

## The scenario

One tool (`get_weather`) and one deliberately incomplete request
("What's the weather?" with no city). The agent is given an intentionally
*eager* system prompt so it guesses a location (`Seattle`) and proposes the
tool call — exactly the failure mode the handler is meant to catch. Jev flags
the guessed argument as ungrounded / missing info, and the policy returns a
`Guide(...)` that sends the model back to ask the user for a city.

## Run it

From the repo root:

```bash
uv run --env-file .env demos/tool_call_intervention/main.py
```

See the [top-level README](../../README.md) for prerequisites and setup
(`OPENROUTER_API_KEY` plus AWS Bedrock access).

## Expected output (abridged)

```
USER: What's the weather?
Tool #1: get_weather
======================================================================
[intervention] model proposes: get_weather({"location": "Seattle"})
[intervention] Jev classifications (probability of 'yes'):
    matches_intent : 0.70
    missing_info   : 0.77
    args_grounded  : 0.04
    premature      : 0.84
[policy] required info is missing -> Guide
...assistant then asks the user which city they meant.
```

Exact probabilities will vary run to run.
