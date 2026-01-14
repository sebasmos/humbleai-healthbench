# BODHI v0.1.3 Results - HealthBench Hard (200 samples)

## Overview

**Version:** BODHI v0.1.3 (Attempt 6 - Specificity + Active Inquiry)
**Model:** gpt-4o-mini
**Dataset:** HealthBench Hard
**Samples:** 200
**Date:** 2026-01-14

---

## Results Summary

### v0.1.3 Attempt History

| Attempt | Strategy | Score | level:cluster | axis:completeness |
|---------|----------|-------|---------------|-------------------|
| v0.1.2 (baseline) | Original | 1.16% | 83.3% | 0.0% |
| v0.1.3 attempt 1 | Mental checklist | 1.47% | 83.1% | 0.0% |
| v0.1.3 attempt 2 | Integrated guidelines | 2.19% | 81.3% | 0.0% |
| v0.1.3 attempt 3 | Explicit enumeration | 0.06% | 79.4% | 0.0% |
| v0.1.3 attempt 4 | Length forcing | 0.36% | 82.1% | 0.0% |
| v0.1.3 attempt 5 | Answer-first | 0.0% | 76.5% | 0.0% |
| **v0.1.3 attempt 6** | **Specificity + Active Inquiry** | **3.42%** | **82.5%** | **0.0%** |

**Best result: v0.1.3 attempt 6 (3.42%) - EXCEEDS 3% TARGET**

---

## Key Improvements (Attempt 6)

### Overall Score

| Metric | v0.1.2 Baseline | v0.1.3 (Best) | Change |
|--------|:---------------:|:-------------:|:------:|
| **Score** | 1.16% | **3.42%** | **+2.26%** |
| **level:cluster** | 83.3% | 82.5% | -0.8% |
| **level:example** | 0.0% | 0.0% | = |

---

## Theme-Level Results (Best Attempt)

| Theme | v0.1.2 | v0.1.3 (Best) | Change |
|-------|:------:|:-------------:|:------:|
| **Emergency Referrals** | 11.2% | **17.4%** | **+6.2%** |
| **Hedging** | 2.3% | **9.0%** | **+6.7%** |
| **Context Seeking** | 4.5% | **5.9%** | **+1.4%** |
| **Complex Responses** | 0.0% | **1.3%** | **+1.3%** |
| **Communication** | 3.5% | 0.9% | -2.6% |
| **Health Data Tasks** | 4.1% | 0.0% | -4.1% |
| **Global Health** | 0.0% | 0.0% | = |

---

## What Made Attempt 6 Work

### The Key Insight

Previous attempts focused on **topic coverage** ("address all parts"). But HealthBench completeness is about **specific response patterns**:

1. **Missing specifics** - Model says "take ibuprofen" instead of "ibuprofen 200-400mg every 4-6 hours"
2. **Passive vs Active** - Model says "if you experience..." instead of "Are you experiencing...?"
3. **Missing alternatives** - Model says "see annually" instead of "annually OR every couple of years"

### The Three Changes

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

## axis:completeness Still 0%

Despite the overall improvement, axis:completeness remains at 0.0%. This suggests:

1. The improvements came from OTHER axes (hedging, context-seeking, emergency behavior)
2. Completeness failures require even more specific prompting
3. May need fine-tuning or post-processing to fully solve

---

## Files Reference

| File | Description |
|------|-------------|
| [healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.html](healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.html) | Best attempt (3.42%) interactive report |
| [healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.json](healthbench_hard_gpt-4o-mini_20260114_211821_bodhi.json) | Best attempt metrics |

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

v0.1.3 (attempt 6) achieves **3.42%**, exceeding the 3% target and representing a **+2.26%** improvement over v0.1.2 baseline (1.16%).

Key wins:
- Emergency referrals: 11.2% → 17.4% (+6.2%)
- Hedging: 2.3% → 9.0% (+6.7%)
- Complex responses: 0.0% → 1.3% (first non-zero!)

The "Specificity + Active Inquiry" strategy successfully changed HOW the model phrases responses, not just WHAT it covers.
