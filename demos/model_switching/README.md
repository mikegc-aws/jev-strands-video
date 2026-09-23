# Model Switching

> **⚠️ This is an experiment, not a recommended pattern.** It's here to show
> that model switching mid-conversation is *possible* and to explore how it
> behaves — not to say you should do it. It works, but it is by no means
> perfect. Read the [caveats](#honest-caveats) before you take any of this into
> a real system. In particular: scaling *up* to a bigger model on a hard turn is
> reasonable; scaling *down* to a smaller model has real downsides (see below).

Run **one** Strands agent across **one** continuous multi-turn conversation, and
swap the underlying Amazon Bedrock model *between turns* based on how hard the
latest question is. Easy turn → cheap/fast `small` model. Hard turn →
capable/pricey `big` model. The conversation history carries across every swap
untouched.

Uses [Jev](https://typesafe.ai) to pick a size for each turn and ordinary Python
to map that size to a model — the same "Jev classifies, Python decides" split as
the rest of the collection, applied here to *dial cost up and down within a
single conversation*.

## The flow (once per turn)

```
next user turn
    → Jev PICKS the size (one narrow question: small / medium / big?)
    → that winning label maps straight to a Bedrock model
    → agent.model is set to it
    → the SAME agent answers, continuing the SAME conversation
```

Because there are exactly three sizes and three models, we let Jev answer the
question directly with a `Choice` over the labels `small` / `medium` / `big`. It
returns the winning label (its highest-probability option) plus a confidence —
no numeric score to re-bucket and no cutoffs to tune. Our code still owns the
label→model mapping.

## Why it works

In Strands the conversation lives on the **agent** (`agent.messages`), not on
the model. The model is just the component that generates the next turn. So
reassigning `agent.model` before a turn changes *who answers next* without
disturbing the accumulated history:

```python
agent = Agent(model=BedrockModel(model_id=MODELS_BY_SIZE["medium"]))  # one agent, one history

for turn in conversation:
    size = classify_size(turn).choice                    # Jev picks small/medium/big
    agent.model = BedrockModel(model_id=MODELS_BY_SIZE[size])  # dial up/down — history persists
    result = agent(turn)                                 # continues the SAME conversation
```

Because the three models share a large context window, there's no context-size
mismatch when we switch — we're purely trading capability (and cost) per turn.

## Different shape from a per-request router

A router builds a fresh agent per request, so each request is isolated. This
demo does the opposite on purpose: a **single long-lived agent** whose model is
turned up and down turn by turn, inside one coherent thread. That's the whole
point of the experiment — dial cost *within* a conversation, not across
independent calls.

## The sizes

| Size | Bedrock model | When |
| --- | --- | --- |
| `small` | Claude Haiku 4.5 (`global.anthropic.claude-haiku-4-5-20251001-v1:0`) | greetings, quick facts, tiny rewrites |
| `medium` | Claude Sonnet 4.6 (`global.anthropic.claude-sonnet-4-6`) | everyday reasoning, explanation, routine coding |
| `big` | Claude Opus 4.6 (`global.anthropic.claude-opus-4-6-v1`) | multi-step reasoning, tricky design, deep analysis |

These are cross-Region inference-profile IDs (the `global.` prefix). Swap them
in `MODELS_BY_SIZE` for whatever you have model access to — the switching logic
doesn't care what the concrete IDs are.

## The scenario

A scripted four-turn conversation that spans the difficulty range, so you can
watch the model tier move `small → medium → big` and back down again while the
thread stays coherent (each later turn refers back to earlier ones):

1. _"…What's the capital of France?"_ → **small**
2. _"…explain the difference between a process and a thread…"_ → **medium**
3. _"Now design the concurrency model … across regions during network partitions…"_ → **big**
4. _"…give the app a fun one-line tagline?"_ → **small** again (proves it dials *down* too)

Each turn prints the running `agent.messages` count so you can see history
accumulating across the swaps.

## Honest caveats

This is an experiment. It works, but it is not perfect, and the two directions
of switching are **not** symmetric:

- **Scaling *up* is the sound half.** Escalating to a bigger model when a turn
  gets genuinely hard is reasonable — you're spending more only when the work
  warrants it, and the bigger model inherits the full context.
- **Scaling *down* is where it gets risky.** Handing a follow-up to a smaller,
  cheaper model can quietly degrade quality: it still has to *read* the whole
  prior conversation (including the big model's careful reasoning) and may not
  keep up with it. A "simple-looking" follow-up can secretly depend on hard
  context, so a cheap model can drop nuance, contradict earlier turns, or lose
  the thread — while the classifier happily rates the surface question as easy.
- **Voice can shift.** Different models answer different turns, so tone and style
  may drift across the conversation.
- **Not a free lunch.** A cheaper model saves on *generation* (output tokens) but
  still pays to *read* all prior context (input tokens). The savings are on the
  output side only.
- **Latency is real.** Rebuilding the model each turn and the extra Jev call add
  overhead; this isn't tuned for speed.
- **The classifier isn't infallible.** Jev returns a confidence, and this demo
  ignores it. A more careful version would set a floor — e.g. never scale *down*
  on low confidence, biasing toward the safer (bigger) model when unsure.

Treat it as a starting point for exploration, not a production recipe.

## Run it

From the repo root:

```bash
uv run --env-file .env demos/model_switching/main.py
```

This calls Bedrock, so you need AWS credentials with model access to all three
models above (plus `OPENROUTER_API_KEY` for Jev). See the
[top-level README](../../README.md) for prerequisites and setup.

## Expected output (abridged)

```
======================================================================
USER: Hey! I'm building a small weather app. What's the capital of France?
[jev]    chosen size      : small  (confidence 1.00)
[policy] bedrock model    : global.anthropic.claude-haiku-4-5-20251001-v1:0
[state]  history so far   : 0 messages (persists across the swap)
...
======================================================================
USER: Now design the concurrency model for that weather app's backend ...
[jev]    chosen size      : big  (confidence 0.98)
[policy] bedrock model    : global.anthropic.claude-opus-4-6-v1
[state]  history so far   : 4 messages (persists across the swap)
...
======================================================================
Conversation complete: 8 total messages in one continuous thread.
```

Exact sizes, confidences, and message counts will vary run to run.
