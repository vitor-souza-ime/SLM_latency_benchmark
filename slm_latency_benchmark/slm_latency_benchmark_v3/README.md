# SLM Latency Benchmark v3 — English-only, GPU-local vs OpenRouter

## Research objective

Compare **client-observed inference latency** for the same four ~8B generative language models, locally on **NVIDIA GeForce RTX 4070 12 GB (Ollama)** and remotely through **OpenRouter**. There are **no CPU benchmarks**. Answer accuracy is a small, secondary diagnostic.

> **Availability warning (26 October 2026):** The OpenRouter model pages may list a model even when it has no live serving endpoint. At the time this project was prepared, `qwen/qwen3-8b` returned an empty `endpoints` array. The older Nemotron Nano 8B `:free` page is listed as free, but its current live endpoint could not be independently verified. The program **checks model endpoints first**, **stops before any inference if a model is unavailable**, or (only with `--skip-unavailable`) drops the unavailable model from **both** local and remote runs. This prevents false comparisons. The Nemotron 3.5 *Embed 8B* and *Rerank 8B* are not replacements: they do not generate chat responses.

## Four model pairs

| Model | Ollama model | OpenRouter model |
| --- | --- | --- |
| Llama 3.1 8B Instruct | `llama3.1:8b` | `meta-llama/llama-3.1-8b-instruct` |
| Llama 3.1 Nemotron Nano 8B v1 | `Randomblock1/nemotron-nano:8b-instruct-q4_K_M` | `nvidia/llama-3.1-nemotron-nano-8b-v1:free` |
| Ministral 3 8B | `ministral-3:8b` | `mistralai/ministral-8b-2512` |
| Qwen3 8B | `qwen3:8b` | `qwen/qwen3-8b` |

The Nemotron Ollama artifact is a community-distributed Q4_K_M build: record its digest and verify it represents the intended NVIDIA checkpoint. Model weights and exact backend quantization may still differ from remote providers. Qwen's thinking is disabled in both settings where supported; Nemotron receives `detailed thinking off` as its system prompt. These model-specific prompt differences are fixed across the **two environments of each model** and are recorded in `models.json`.

## Economical protocol

- **12 English prompts:** 6 multiple-choice questions with verified answer keys, 6 length-controlled generation tasks (2 short, 2 medium, 2 long).
- **Default: 2 repetitions** per prompt and model, plus **1 warmup** per model per environment.
- If all four remote endpoints are available: **96 measured local + 96 measured remote requests**, plus **4 local + 4 remote warmups**; exactly **100 remote requests maximum**.
- The default remote request safety limit is 100; increasing repetitions requires explicitly changing `--max-remote-requests`.
- This is a low-cost **pilot**, not a sufficiently powered final Q1 experimental campaign. Replicate in separate sessions/days for stronger confidence intervals and P95 estimates.
- The full 60-question answer-key set and 12 generation prompts are also included (72 total), **all English**, but the full profile is **not enabled by default**.

OpenRouter free model variants may be subject to **50 requests/day on free-tier accounts**, higher limits depending on account/credit status. The model being free does not mean it is available or free of rate limits. Confirm your own account limits before a large test. Paid models are charged per token; check actual prices using the catalog first.

## Install

Python 3.10+, Ollama, GPU with NVIDIA drivers, and (for remote calls) an OpenRouter key are required.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull Randomblock1/nemotron-nano:8b-instruct-q4_K_M
ollama pull ministral-3:8b
ollama pull qwen3:8b
```

Optional English-language web application:

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

Store your key in the terminal environment; do not add it to the repository or output data:

```bash
# Linux / macOS
export OPENROUTER_API_KEY='YOUR_KEY'
# Windows PowerShell
$env:OPENROUTER_API_KEY = 'YOUR_KEY'
```

## Zero-cost checks before inference

```bash
# Offline: predicts counts, no remote query and no inference
python benchmark/run.py --dry-run

# Online read-only lookup: no inference or charged tokens
python benchmark/check_catalog.py
```

The catalog may show **zero active endpoints for Qwen3 8B**; if so, do **not** claim a four-model local-vs-remote result. The `--skip-unavailable` option can exclude it from both environments without compromising a matched comparison of the remaining models. `--local-only --model-ids 4` can separately test Qwen's local GPU performance, but that alone cannot contribute to the paired comparison.

## Run experiment

```bash
# Only run models with confirmed remote endpoints (paired design).
python benchmark/run.py --skip-unavailable

# Test only GPU-local models — no OpenRouter key needed.
python benchmark/run.py --local-only

# One-model minimal pilot, e.g. Llama only
python benchmark/run.py --model-ids 1 --repeats 1

# Preview a larger study without running it
python benchmark/run.py --repeats 3 --max-remote-requests 148 --dry-run
```

The benchmark verifies `ollama ps` reports **100% GPU** for local models after warmup. It stops rather than silently running on CPU; only bypass via `--allow-unverified-gpu` for debugging, never use that for published data. The OpenRouter preflight blocks remote runs with no active endpoints; only bypass it with `--skip-remote-preflight` when absolutely certain and document why.

For a final paper, set an explicit `provider` value for each model in `models.json` using active provider names, and check the actual provider after inference with `benchmark/enrich_providers.py`. Unpinned routing may change the serving hardware and the observed latency.

Results are saved under `results/<timestamp>/raw_results.csv` and `experiment_config.json`. The raw CSV includes all prompt texts, responses, errors, timing breakdowns, token counts, GPU memory observations, model IDs and requested providers.

## Analyze latency and generate English scientific plots

```bash
python benchmark/analyze.py --input results/EXPERIMENT_FOLDER/raw_results.csv
python benchmark/enrich_providers.py --input results/EXPERIMENT_FOLDER/raw_results.csv
```

The analysis produces CSV summary tables, per-prompt matched comparisons, 95% bootstrap CIs, boxplots, TTFC plots, estimated output throughput, latency CDF, accuracy and latency/accuracy views. Figures and report titles are entirely in English.

**Primary outcome:** end-to-end (E2E) latency measured with `time.perf_counter()` at the Python client. **Secondary latency outcome:** TTFC, time to first *visible text*, not necessarily first decoder token. The remote result includes network and queuing overhead; it does **not** represent cloud accelerator runtime alone. Only compare matched prompts using per-prompt median latency; do not treat repeated calls on the same prompt as independent samples.

Output text length is not guaranteed identical merely by requesting a target word count. Report actual completion tokens and consider matching token-count bins for a follow-up controlled-length campaign. GPU VRAM vs server quantization, network congestion, and server/provider changes are confounders to document.

## Validate project

```bash
python -m unittest discover -s tests -v
python benchmark/make_dataset.py
```

Tests use mocked streaming data exclusively to validate the parser and commands. **No synthetic numbers are presented as measured benchmark results.**

## References

- [OpenRouter models and providers](https://openrouter.ai/models)
- [OpenRouter API and pricing](https://openrouter.ai/pricing)
- [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [Ollama chat API](https://docs.ollama.com/api/chat)
- [Nemotron Nano model on OpenRouter](https://openrouter.ai/nvidia/llama-3.1-nemotron-nano-8b-v1:free)
- [Qwen3 8B on OpenRouter](https://openrouter.ai/qwen/qwen3-8b)
