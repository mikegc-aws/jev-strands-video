# Jev on Strands — Demos

A small, growing collection of runnable demo scripts showing how to use
[Jev](https://typesafe.ai) (TypeSafe's classifier model) alongside the
[Strands Agents SDK](https://strandsagents.com).

The theme across every demo: **Jev is the semantic classifier, ordinary Python
is the policy.** Jev answers narrow, typed questions and returns
probabilities — it never decides what to do. Your code turns those
probabilities into decisions with plain, readable logic you can trace by eye.

Each demo is a single self-contained script with its dependencies declared
inline (PEP 723), so there's no environment to manage — [`uv`](https://docs.astral.sh/uv/)
installs what it needs on the fly.

## Demos

| Demo | What it shows |
| --- | --- |
| [`jev_basics`](demos/jev_basics/) | The "hello world" for Jev, as a **notebook** — **no agent, no Strands**. Walks the three question types (`Choice`, `Score`, `Noul`) one cell at a time, then combines them and lets plain Python make the decision. Start here. |
| [`tool_call_intervention`](demos/tool_call_intervention/) | Gate a proposed agent tool call *before it runs* using the real Strands Interventions API. Jev classifies the proposed call; a deterministic policy returns `Guide(...)` or `Proceed()`. |
| [`completion_intervention`](demos/completion_intervention/) | Catch an agent that tries to *stop before the task is done*, at the `after_model_call` point. Jev classifies whether the goal is `COMPLETE` / `PARTIAL` / `NOT_ANSWERED`; a deterministic policy returns `Guide(...)` (retry) or `Proceed()`. |
| [`model_switching`](demos/model_switching/) | An experiment (not a production pattern): *one* agent, *one* continuous conversation, and the Bedrock model swapped *between turns* (`small` / `medium` / `big`) to dial cost up and down. Jev picks the size with a `Choice`; ordinary Python maps it to a model and reassigns `agent.model` while `agent.messages` history persists across the swap. |

_More demos will be added over time — each one lives in its own folder under
`demos/`._

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — runs the
  scripts and handles dependencies.
- An **OpenRouter API key** for Jev. Get one at
  [openrouter.ai/keys](https://openrouter.ai/keys).
- **AWS credentials with Amazon Bedrock access** — the Strands-based demos use
  Bedrock as their default model provider. Make sure model access is enabled in
  the Bedrock console. (The [`jev_basics`](demos/jev_basics/) demo needs *only*
  the OpenRouter key — no AWS access required.)

## Setup

Copy the example env file and add your key:

```bash
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY
```

`.env` is gitignored, so your key stays out of version control.

## Running a demo

From the repo root, point `uv` at any demo's `main.py`. `--env-file` loads your
key from `.env`:

```bash
uv run --env-file .env demos/tool_call_intervention/main.py
```

`uv` reads the inline dependency block at the top of the script, installs the
packages into an ephemeral environment, and runs it. No `pip install`, no
virtualenv.

## Repo layout

```
.
├── README.md                       # you are here
├── LICENSE
├── .env.example                    # template — copy to .env
├── .gitignore
└── demos/                          # one folder per demo
    ├── jev_basics/
    │   ├── README.md
    │   └── jev_basics.ipynb
    ├── tool_call_intervention/
    │   ├── README.md
    │   └── main.py
    ├── completion_intervention/
    │   ├── README.md
    │   └── main.py
    └── model_switching/
        ├── README.md
        └── main.py
```

## Adding a new demo

1. Create a new folder under `demos/`, e.g. `demos/my_new_demo/`.
2. Add a `main.py` with an inline PEP 723 dependency block at the top:
   ```python
   # /// script
   # requires-python = ">=3.10"
   # dependencies = ["strands-agents>=1.56,<2", "typesafe-sdk>=0.7.1,<0.8"]
   # ///
   ```
3. Add a short `README.md` in the folder explaining what the demo shows and how
   to run it.
4. Add a row to the **Demos** table above.

## License

[MIT](LICENSE)
