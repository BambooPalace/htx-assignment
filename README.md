# LLM Evaluation Harness

A lightweight evaluation pipeline for testing a mock end-to-end chat system (`/chat`) and an OpenAI-compatible judge model (`/model`).

## Author's notes
When using LLM as a judge, it s key to ensure this step is trust worthy and accurate. In reality it will take some time to choose a suitable evaluating model and fix a prompt template. 

A real person need to act as the expert judge of the accuracy of this step, by manually label the prediction against the expected answers with ~ 5-10% of test dataset. You dont need to start with a best reasoning model for this step as it depends case by case on the data complexity. A working LLM model to achieve your desired accuracy (>95%) against the human labelor is sufficient. 
(In this demo, i m unable to demonstrate above process due to the assumptions the model and evalaution prompt is fixed. )

It s worth mentioning it s key to ask LLM judge to output binary labels instead on scores on scales, which cause instability due to model indeterministic nature. Use majority votes out of multiple runs also help to reduce the output randomness.


## What it does

1. **Mock API** (`app/server.py`)
   - `POST /chat` — accepts `{"input": "..."}` and returns dummy chat output plus intermediate fields (`query`, `contexts`) for debugging.
   - `POST /v1/chat/completions` (also aliased as `/model`) — OpenAI-compatible judge endpoint that returns random `true`, `false`, or `mock output` labels.

2. **Predict** (`predict.py`) — runs each row in `data/test.jsonl` through `/chat` and writes merged results to `data/outputs.jsonl`.

3. **Evaluate** (`evaluate.py`) — uses `/model` as a judge (temperature `0`, `n` runs per case), computes accuracy via majority vote, and calls `diagnose()` for failures. Logs progress and details to the terminal and `logs/log_<timestamp>.text`.

## Project layout

```
app/server.py       # Mock /chat and /model endpoints
predict.py          # Run test cases against /chat
evaluate.py         # Judge outputs and produce accuracy report
utils.py            # Logging and vote parsing helpers
data/test.jsonl     # 50 sample Q&A test cases
tests/              # Unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Start the mock API in one terminal:

```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Run predictions:

```bash
python predict.py
# optional: limit samples
python predict.py --n-samples 10
```

Run the full evaluation pipeline:

```bash
python evaluate.py --n-runs 3 --n-samples 50
```

Use `--skip-predict` to evaluate an existing `data/outputs.jsonl` without re-running `/chat`.

## Outputs

- `data/outputs.jsonl` — chat predictions with intermediate fields
- `data/evaluation_report.json` — accuracy summary and failure diagnoses
- `logs/log_<ddmmyyHHMMSS>.text` — terminal log mirror

## Scoring approach

The judge model is prompted to return only `true` or `false`. Each case is evaluated `n` times; malformed responses (e.g. `mock output`) are logged as errors and excluded from that run's vote. Final per-case label is the majority of valid votes; overall accuracy is the pass rate across scored cases.

## Tests

```bash
pytest
```

## Error handling

- Malformed JSONL lines raise clear parse errors.
- Endpoint failures during predict are captured per row with an `error` field.
- Malformed judge outputs are logged and skipped for voting.
- Chat errors during evaluate are recorded and included in the failure report.

## With more time

- Replace mock endpoints with real chat and judge models.
- Add semantic similarity scoring alongside LLM-as-judge.
- Persist structured experiment metadata (model version, prompt hash, timestamps).
- Add CI workflow and integration tests against a test server fixture.
