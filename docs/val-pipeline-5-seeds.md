# Validation Pipeline: 5-Seed Evaluation (RSS & RS)

This document describes the validation pipeline for running multi-seed evaluations on HealthBench Hard using both Random Stratified Sampling (RSS) and Random Sampling (RS) strategies.

## Overview

Run HealthBench Hard evaluations on 5 different samples (seeds 42-46) for both baseline and BODHI v0.1.3, comparing results across:
- **Sampling strategies**: RSS (stratified) vs RS (random)
- **Evaluation modes**: Baseline vs BODHI v0.1.3

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | `gpt-4o-mini` |
| Evaluation | `healthbench_hard` |
| Seeds | 42, 43, 44, 45, 46 |
| Samples per seed | 200 |
| BODHI version | v0.1.3 |
| Threads | 10 |

## Prerequisites

1. **Generate sample files** using `notebooks/create_multi_seed_samples.ipynb`:
   - Run with `STRATEGY = 'rss'` to create `data/data-5-seeds-200RSS/`
   - Run with `STRATEGY = 'rs'` to create `data/data-5-seeds-200RS/`

2. **Set your API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Start

### Run RSS evaluations only:
```bash
./scripts/run_rss_eval.sh
```

### Run RS evaluations only:
```bash
./scripts/run_rs_eval.sh
```

### Run both (full pipeline):
```bash
./scripts/run_all_eval.sh
```

### Run specific seeds:
```bash
# Only seeds 42 and 43
./scripts/run_rss_eval.sh 42 43
./scripts/run_rs_eval.sh 42 43
```

## Output Structure

```
Results/
├── results-5-seeds-200rss/
│   ├── baseline-seed42/
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS.json
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_allresults.json
│   │   └── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS.html
│   ├── baseline-seed{43,44,45,46}/
│   ├── bodhiv0.1.3-seed42/
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi.json
│   │   ├── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi_allresults.json
│   │   └── healthbench_hard_gpt-4o-mini_YYYYMMDD_HHMMSS_bodhi.html
│   └── bodhiv0.1.3-seed{43,44,45,46}/
└── results-5-seeds-200rs/
    └── (same structure as RSS)
```

## Running with Different Models

The default model is `gpt-4o-mini`. To run with a different model, set the `MODEL` environment variable:

```bash
# Run with gpt-4.1-mini
export OPENAI_API_KEY="your-api-key-here"
export MODEL="gpt-4.1-mini"
export RESULTS_DIR="Results/results-5-seeds-200rss-gpt41mini"
./scripts/run_rss_eval.sh
```

### Available Models

Models must be configured in `simple-evals/simple_evals.py`. Currently supported:
- `gpt-4o-mini` (default)
- `gpt-4.1-mini`
- `gpt-4.1`
- `gpt-4.1-nano`
- `gpt-4o`
- And many others (see `simple_evals.py` for full list)

### Adding a New Model

To add a new model, edit `simple-evals/simple_evals.py` and add an entry in the models dictionary:

```python
"your-model-name": lambda: ChatCompletionSampler(
    model="your-model-id",
    system_message=OPENAI_SYSTEM_MESSAGE_API,
    max_tokens=2048,
),
```

## Scripts Reference

| Script | Description |
|--------|-------------|
| `scripts/run_rss_eval.sh` | Run RSS (stratified) evaluations |
| `scripts/run_rs_eval.sh` | Run RS (random) evaluations |
| `scripts/run_all_eval.sh` | Run both pipelines sequentially |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `MODEL` | `gpt-4o-mini` | Model to evaluate |
| `RESULTS_DIR` | `Results/results-5-seeds-200rss` | Output directory for results |
