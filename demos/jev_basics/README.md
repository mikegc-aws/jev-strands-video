# Jev Basics

The "hello world" for [Jev](https://typesafe.ai) as a notebook \u2014 **just the Jev
SDK over OpenRouter**, no Strands, no Bedrock, no AWS. It walks the three question
types one cell at a time, then combines them.

Jev classifies; plain Python decides.

| Type | What it does | Key field |
| --- | --- | --- |
| `Choice` | pick one option from a set | `.choice`, `.confidence` |
| `Score` | rate on an ordered rubric (low \u2192 high) | `.score`, `.legend` |
| `Noul` | one yes/no judgment | `.noul` (probability of "yes") |

## Run it

Needs `OPENROUTER_API_KEY` in the environment. From the repo root:

```bash
export $(grep -v '^#' .env | xargs)   # load OPENROUTER_API_KEY from .env
uv run --with 'typesafe-sdk>=0.7.1,<0.8' --with jupyter jupyter lab demos/jev_basics/jev_basics.ipynb
```

Or open `jev_basics.ipynb` in VS Code / Jupyter with a kernel that has
`typesafe-sdk` installed (`pip install typesafe-sdk`).
