# BODHI v0.1.3 Results - HealthBench Hard (200 samples)

## Overview

**Version:** BODHI v0.1.3
**Model:** gpt-4o-mini
**Dataset:** HealthBench Hard
**Samples:** 200
**Date:** 2026-01-14

---

## Results Summary

| Version | Strategy | Score | level:cluster | axis:completeness |
|---------|----------|-------|---------------|-------------------|
| v0.1.2 (baseline) | Original | 1.16% | 83.3% | 0.0% |
| **v0.1.3** | **Specificity + Active Inquiry** | **3.42%** | **82.5%** | **0.0%** |

**Improvement: +2.26% over baseline**

---

## Overall Score

| Metric | v0.1.2 Baseline | v0.1.3 | Change |
|--------|:---------------:|:------:|:------:|
| **Score** | 1.16% | **3.42%** | **+2.26%** |
| **level:cluster** | 83.3% | 82.5% | -0.8% |
| **level:example** | 0.0% | 0.0% | = |

---

## Theme-Level Results

| Theme | v0.1.2 | v0.1.3 | Change |
|-------|:------:|:------:|:------:|
| **Emergency Referrals** | 11.2% | **17.4%** | **+6.2%** |
| **Hedging** | 2.3% | **9.0%** | **+6.7%** |
| **Context Seeking** | 4.5% | **5.9%** | **+1.4%** |
| **Complex Responses** | 0.0% | **1.3%** | **+1.3%** |
| **Communication** | 3.5% | 0.9% | -2.6% |
| **Health Data Tasks** | 4.1% | 0.0% | -4.1% |
| **Global Health** | 0.0% | 0.0% | = |

---

## What Changed in v0.1.3

### Key Insight

HealthBench completeness is about **specific response patterns**, not just topic coverage.

### Three Core Changes

```
BE SPECIFIC:
- Include specific numbers: dosages, frequencies, timeframes
- Mention emergency numbers: "Call 911 (in the US)"

ACTIVELY ASK about concerning symptoms:
- Don't say "if you experience X" - ASK "Are you experiencing X right now?"

INCLUDE ALTERNATIVES when they exist:
- Don't just give one option - mention alternatives
```

---

## Files Reference

| File | Description |
|------|-------------|
| [healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.html](healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.html) | Interactive report |
| [healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.json](healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.json) | Metrics |

---

## Command to Reproduce

```bash
cd /Users/cajas.sebastian/Desktop/HumbleAILLMs
pip install -e /Users/cajas.sebastian/Desktop/bodhi-llm

OPENAI_API_KEY="your-key" python -m simple-evals.simple_evals \
  --eval healthbench_hard \
  --model gpt-4o-mini \
  --use-bodhi \
  --n-threads 10 \
  --sample-file data/hard_200_sample_ids.json \
  --output-dir Results/v0.1.3
```

---

## Conclusion

v0.1.3 achieves **3.42%**, a **+2.26%** improvement over v0.1.2 baseline (1.16%).

Key wins:
- Emergency referrals: +6.2%
- Hedging: +6.7%
- Complex responses: +1.3% (first non-zero)
