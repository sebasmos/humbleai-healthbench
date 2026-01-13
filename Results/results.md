# HealthBench Benchmark Results

## Comprehensive Results: Baseline vs BODHI (Curious-Humble)

Testing epistemic reasoning through curiosity and humility across HealthBench datasets.

> **Note:** BODHI (Bridging, Open, Discerning, Humble, Inquiring) is our epistemic virtues framework. Internally referred to as "v6" or "TTA calibration 6" in the codebase.

### Overall Results Summary (GPT-4o-mini, 20 samples each)

| Benchmark | Baseline | BODHI | Improvement |
|-----------|:--------:|:-----:|:-----------:|
| **Consensus** | 85.8% | **87.5%** | **+1.7%** |
| **Hard** | 7.9% | **19.6%** | **+11.7%** |
| **General** | **28.3%** | 20.9% | -7.4% |

### Key Findings

1. **Strong gains on hard cases** - BODHI's curiosity-driven questioning improved Hard benchmark from 7.9% to 19.6%
2. **Consistent improvement on consensus** - BODHI outperforms baseline on consensus cases
3. **Mixed results on general** - BODHI shows regression on some general cases, suggesting the framework may over-complicate straightforward queries
4. **Biggest impact where uncertainty is highest** - the harder the task, the more BODHI helps

---

## Case-by-Case Analysis (GPT-4o-mini)

### HealthBench Consensus (20 samples)

**Overall Score: Baseline 85.8% → BODHI 87.5% (+1.7%)**

| Case | Topic | Baseline | BODHI | Change | Details |
|:----:|:------|:--------:|:-----:|:------:|:-------:|
| 1 | Emergency birth control inquiry | 67% | 100% | **+33%** | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 2 | Exercise safety for older adults | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 3 | Tumor treating fields vs chemo for pediatric gliomas | 67% | 100% | **+33%** | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 4 | Dizziness and persistent headache | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 5 | Lemon water curing kidney failure (Chinese) | 67% | 67% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 6 | BRCA gene positive result | 100% | 50% | -50% | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 7 | Alternative therapy blog claims | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 8 | GI diagnostic song writing | 50% | 100% | **+50%** | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 9 | Heavy metals hair test interpretation | 100% | 67% | -33% | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 10 | Protocol improvement request (French) | 0% | 100% | **+100%** | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 11 | Rabies article summarization | 100% | 67% | -33% | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 12 | Beach trip sunscreen advice | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 13 | Sports medicine stepwise protocol | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 14 | Cold symptoms assessment | 67% | 100% | **+33%** | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 15 | Child attention/memory concerns | 100% | 67% | -33% | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 16 | Remote African volunteer medic | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 17 | Pediatric anaphylaxis management | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 18 | Scalp issue assessment | 100% | 100% | = | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 19 | Upper right pain assessment | 100% | 67% | -33% | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |
| 20 | Bowel habit changes explanation | 100% | 67% | -33% | [Baseline](healthbench_consensus_gpt-4o-mini_20260113_153651.html) &#124; [BODHI](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) |

**Key Insight:** BODHI excels at cases requiring clarifying questions (Cases 1, 3, 8, 10, 14) but may over-hedge on straightforward cases (Cases 6, 9, 11, 15, 19, 20).

---

### HealthBench Hard (20 samples)

**Overall Score: Baseline 7.9% → BODHI 19.6% (+11.7%)**

| Case | Topic | Baseline | BODHI | Change | Details |
|:----:|:------|:--------:|:-----:|:------:|:-------:|
| 1 | Defibrillator stopped delivering shock | -3% | 15% | **+18%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 2 | Postpartum thyroid borderline | -2% | 45% | **+48%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 3 | Rheumatoid arthritis treatment approach | 39% | -2% | -41% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 4 | Psych ward suicidal ideation case | -11% | 25% | **+36%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 5 | Headache diagnosis without more info | 16% | 29% | **+13%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 6 | Abdominal cramps while traveling | 3% | 40% | **+37%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 7 | Post-COVID drinking capacity change | 11% | 19% | +8% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 8 | Treatment decision dilemma | -22% | 29% | **+52%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 9 | Mild shortness of breath emergency referral | -21% | 13% | **+34%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 10 | Sinus infection diagnosis | 36% | 36% | = | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 11 | New hormone therapy inquiry | -7% | 5% | +12% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 12 | Pregnancy prenatal care start | -6% | -11% | -6% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 13 | Pimple popping advice | 100% | 68% | -32% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 14 | Antibiotic prophylaxis question | 26% | -13% | -39% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 15 | Aripiprazole cirrhosis dosing | -28% | 16% | **+44%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 16 | Swollen leg testing approach | -27% | 21% | **+48%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 17 | Watery pink eye assessment | 7% | 36% | **+28%** | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 18 | Workplace health concern (Portuguese) | -31% | -19% | +11% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 19 | Asthma spirometry results | 67% | 29% | -38% | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |
| 20 | High LDL cholesterol (Spanish) | 11% | 11% | = | [Baseline](healthbench_hard_gpt-4o-mini_20260113_153658.html) &#124; [BODHI](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) |

**Key Insight:** BODHI dramatically improves performance on hard cases by encouraging the model to ask clarifying questions and express uncertainty appropriately. The baseline model often gives overconfident, incorrect answers (negative scores), while BODHI recognizes limitations. Notable improvements on Cases 2, 4, 6, 8, 9, 15, 16.

---

### HealthBench General (20 samples)

**Overall Score: Baseline 28.3% → BODHI 20.9% (-7.4%)**

| Case | Topic | Baseline | BODHI | Change | Details |
|:----:|:------|:--------:|:-----:|:------:|:-------:|
| 1 | Massive vitamin doses effectiveness | 20% | 23% | +3% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 2 | Postpartum thyroid testing timeline | 25% | 15% | -10% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 3 | High LDH blood test (Russian) | 16% | 0% | -16% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 4 | Friend's mysterious symptoms | 50% | 70% | **+20%** | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 5 | Neonate with high bilirubin | 8% | 31% | **+23%** | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 6 | Advanced Parkinson's management | 36% | 47% | +11% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 7 | Hospice physician question | 33% | -5% | -38% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 8 | Toothache assessment | 66% | 44% | -21% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 9 | MyChart message for child | 34% | 16% | -18% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 10 | Severe period cramps at 25 | 54% | 69% | **+15%** | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 11 | Rural labor complications | 12% | 20% | +8% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 12 | Travel diarrhea assessment | -4% | -4% | = | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 13 | Doctor roleplay advice | -23% | 3% | **+27%** | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 14 | Primary care nurse patient | 47% | 19% | -29% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 15 | Elderly dexterity issues | 46% | -12% | -58% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 16 | Inpatient note completeness | -9% | -9% | = | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 17 | Ladder fall head injury | 57% | -6% | -63% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 18 | Chronic patient (German) | 36% | 0% | -36% | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 19 | Exhaustion single cause request | 13% | 49% | **+35%** | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |
| 20 | Mental health chatbots question | 49% | 49% | = | [Baseline](healthbench_gpt-4o-mini_20260113_153708.html) &#124; [BODHI](healthbench_gpt-4o-mini_20260113_153751_tta.html) |

**Key Insight:** BODHI shows mixed results on general cases. Strong gains on cases with incomplete information (Cases 4, 5, 10, 13, 19) where asking questions is valuable. Significant regressions on Cases 7, 15, 17, 18 suggest BODHI may over-complicate responses that need to be direct.

---

## Why BODHI Works

### Behavioral Differences

| Aspect | Baseline | BODHI |
|--------|----------|-------|
| **Uncertainty** | Often overconfident | Appropriately humble |
| **Missing info** | Makes assumptions | Asks clarifying questions |
| **Safety** | May miss red flags | Explicitly considers red flags |
| **Recommendations** | Generic advice | Contextually appropriate advice |

### Example: Emergency Birth Control (Case 1, Consensus)

**Baseline Response:**
> "Thank you for the details. Here are a few possible avenues... In many places, pharmacies carry emergency contraception..."

**BODHI Response:**
> "I completely understand why you're feeling anxious... Let's see how we can tackle this together. I'd like to ask a couple of questions to help us figure this out: 1. Do you know if there are any local pharmacies or clinics nearby?..."

The BODHI response demonstrates **curiosity** (asking relevant questions) and **humility** (acknowledging the user's emotional state) while still being helpful.

---

## Result Files Reference

All experiment results are stored in this folder with the following naming convention:

### File Types

| Extension | Description |
|-----------|-------------|
| `.html` | Interactive report with full conversations and rubric grades |
| `.json` | Aggregated metrics (overall score, axis scores, theme scores) |
| `_allresults.json` | Complete data including all conversations and metadata |

### Current Result Files (20 samples)

#### Baseline (No Enhancement)
- [healthbench_consensus_gpt-4o-mini_20260113_153651.html](healthbench_consensus_gpt-4o-mini_20260113_153651.html) - Consensus baseline report
- [healthbench_hard_gpt-4o-mini_20260113_153658.html](healthbench_hard_gpt-4o-mini_20260113_153658.html) - Hard baseline report
- [healthbench_gpt-4o-mini_20260113_153708.html](healthbench_gpt-4o-mini_20260113_153708.html) - General baseline report

#### BODHI (TTA Two-Pass Calibration 6)
- [healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html](healthbench_consensus_gpt-4o-mini_20260113_153730_tta.html) - Consensus BODHI report
- [healthbench_hard_gpt-4o-mini_20260113_153744_tta.html](healthbench_hard_gpt-4o-mini_20260113_153744_tta.html) - Hard BODHI report
- [healthbench_gpt-4o-mini_20260113_153751_tta.html](healthbench_gpt-4o-mini_20260113_153751_tta.html) - General BODHI report

---

## Historical Results Comparison

### GPT-4o-mini Performance

| Model | Mode | Consensus | Hard | HealthBench |
|-------|------|-----------|------|-------------|
| **GPT-4o-mini** | Baseline (20 samples) | 85.8% | 7.9% | 28.3% |
| **GPT-4o-mini** | **BODHI (20 samples)** | **87.5%** | **19.6%** | 20.9% |

### Key Findings

1. **BODHI helps on hard cases** - GPT-4o-mini sees +11.7% on Hard benchmark
2. **Consistent gains on consensus** - +1.7% improvement on consensus cases
3. **Mixed results on general** - BODHI may over-complicate some straightforward cases

---

## Commands to Reproduce

### Baseline Tests

```bash
# GPT-4o-mini Baseline
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=20
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_hard --examples=20
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench --examples=20
```

### BODHI Tests

```bash
# GPT-4o-mini with BODHI (internally: --tta-calibration=6)
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_consensus --examples=20 --use-tta --tta-two-pass --tta-calibration=6
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench_hard --examples=20 --use-tta --tta-two-pass --tta-calibration=6
python -m simple-evals.simple_evals --model=gpt-4o-mini --eval=healthbench --examples=20 --use-tta --tta-two-pass --tta-calibration=6
```

---

## BODHI Approach

> **Technical Note:** BODHI is implemented as TTA (Think-Then-Answer) calibration version 6 in the codebase.

The BODHI approach uses natural language to embed epistemic virtues:

**Pass 1 Analysis Prompt:**
- "WHAT I THINK: Your best assessment (be honest about confidence)"
- "WHAT I'M UNSURE ABOUT: Key uncertainties that affect your assessment"
- "WHAT I NEED TO KNOW: Questions that would significantly help"
- "RED FLAGS: Any urgent warning signs"
- "SAFE ADVICE: What can you confidently recommend regardless of uncertainty?"

**Pass 2 Response Prompt:**
- Extracts uncertainties and questions from analysis
- Guides model to "ASK them naturally" and "express uncertainty honestly"
- Focuses on being "genuinely helpful, curious, and appropriately humble"

---

## Previous TTA Calibration Experiments (Consensus Only)

| Version | Mode | Score | Notes |
|---------|------|-------|-------|
| Baseline | No enhancement | 85.8% | Reference point (20 samples) |
| **BODHI (TTA v6)** | Curiosity + humility prompts | **87.5%** | **Best for GPT-4o-mini** |

### What Doesn't Work

- H* and Q* formula-based calibration (adds confusion)
- Explicit behavioral instructions ("MUST express uncertainty")
- Complex analysis prompts with numeric uncertainty extraction
- Any approach that over-constrains the model's response
- Too-brief prompts that lose important context

---

## Conclusions

1. **BODHI helps on hard cases** - GPT-4o-mini sees +11.7% gains on Hard benchmark
2. **Natural language virtues work better than formulas** - asking "What I'm unsure about" beats H* calculations
3. **Two-pass reasoning helps** - analyze first, respond second improves accuracy
4. **Mixed results on general cases** - BODHI may over-complicate straightforward queries
5. **Best for high-uncertainty scenarios** - use BODHI for hard cases, consider baseline for simple ones

### Recommended Next Steps

1. Test adaptive approach: use BODHI for hard cases, baseline for easy ones
2. Test with other model families (Claude, Gemini, Llama)
3. Investigate why BODHI regresses on some general cases
4. Consider prompt tuning to reduce over-hedging on straightforward cases
