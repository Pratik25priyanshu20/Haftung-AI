# Research Questions and Hypotheses

## Overview

This study evaluates the impact of Retrieval-Augmented Generation (RAG) on automated
accident causation analysis using three system variants:

- **S1 (No-RAG):** LLM-only causation analysis without legal context retrieval
- **S2 (RAG):** LLM with adaptive retrieval of German traffic law (StVO) and case law
- **S3 (RAG+Constraints):** S2 with claim-level constraint enforcement and validation loop

All variants are evaluated on 30 text-based German accident scenarios across 6 categories.

---

## RQ1: Does RAG improve causation accuracy?

**Hypothesis H1:** RAG-augmented variants (S2, S3) achieve significantly higher causation
accuracy than the baseline (S1), as measured by taxonomy-based classification.

**Metrics:**
| Metric | Description | Source |
|---|---|---|
| `causation_accuracy_taxonomy` | Exact match of predicted vs GT cause taxonomy ID | `cause_taxonomy.py` |
| `factors_f1` | F1 score for contributing factors | `factors_f1.py` |
| `responsibility_mae` | Mean absolute error of responsibility percentages | `responsibility_mae.py` |

**Statistical Tests:**
- Paired t-test (per-scenario accuracy)
- Wilcoxon signed-rank test (non-parametric)
- Cohen's d effect size
- Bonferroni correction for multiple comparisons

**Expected Outcome:** S2 > S1 for causation accuracy, S3 >= S2.

---

## RQ2: Does RAG reduce hallucination?

**Hypothesis H2:** RAG reduces hallucination by grounding claims in retrieved legal text.
S3 further reduces hallucination through claim-level validation.

**Metrics:**
| Metric | Description | Source |
|---|---|---|
| `nli_hallucination_rate` | Fraction of claims classified as contradiction or neutral by NLI | `nli_hallucination.py` |
| `nli_contradiction_rate` | Fraction of claims classified as contradiction only | `nli_hallucination.py` |
| `nli_entailment_rate` | Fraction of claims entailed by scenario text | `nli_hallucination.py` |

**NLI Model:** `cross-encoder/nli-deberta-v3-base`

**Expected Outcome:** S1 > S2 > S3 for hallucination rate (lower is better).

---

## RQ3: Does RAG improve consistency across reruns?

**Hypothesis H3:** RAG provides deterministic legal grounding that reduces output variance
across multiple runs of the same scenario.

**Metrics:**
| Metric | Description | Source |
|---|---|---|
| `avg_entropy` | Mean Shannon entropy of cause distribution across N reruns | `hallucination_entropy.py` |
| `consistency_rate` | Fraction of scenarios with identical cause across all reruns | `hallucination_entropy.py` |
| `ece` | Expected Calibration Error | `calibration.py` |
| `brier_score` | Brier score of confidence vs accuracy | `calibration.py` |

**Protocol:** 5 reruns per scenario per variant.

**Expected Outcome:** S2, S3 have lower entropy and higher consistency than S1.

---

## RQ4: How does retrieval quality correlate with accuracy?

**Hypothesis H4:** Higher retrieval quality (measured by precision and relevance of
retrieved StVO paragraphs) correlates with higher causation accuracy.

**Metrics:**
| Metric | Description | Source |
|---|---|---|
| `precision@5` | Fraction of top-5 retrieved chunks matching relevant StVO | `retrieval_quality.py` |
| `MRR` | Mean Reciprocal Rank of first relevant chunk | `retrieval_quality.py` |
| `nDCG@5` | Normalized Discounted Cumulative Gain at k=5 | `retrieval_quality.py` |

**Analysis:** Scatter plot of retrieval score vs per-scenario accuracy; Pearson/Spearman correlation.

**Expected Outcome:** Positive correlation between retrieval quality and accuracy for S2/S3.

---

## Metric-to-RQ Mapping

| Metric | RQ1 | RQ2 | RQ3 | RQ4 |
|---|---|---|---|---|
| causation_accuracy_taxonomy | x | | | |
| factors_f1 | x | | | |
| responsibility_mae | x | | | |
| nli_hallucination_rate | | x | | |
| nli_contradiction_rate | | x | | |
| avg_entropy | | | x | |
| consistency_rate | | | x | |
| ece | | | x | |
| brier_score | | | x | |
| precision@5 | | | | x |
| MRR | | | | x |
| nDCG@5 | | | | x |

---

## Experimental Design

- **Scenarios:** 30 German accident narratives (200-400 words each)
- **Categories:** rear_end, side_collision, head_on, intersection, pedestrian, single_vehicle (5 each)
- **Variants:** S1, S2, S3 (total: 90 evaluations for single-run, 450 for stability)
- **Knowledge Base:** 7 StVO files + 5 BGH case law rulings
- **LLM:** Groq (Llama 3.3 70B)
- **Embedding:** BAAI/bge-large-en-v1.5
- **Vector DB:** Qdrant
- **Reranker:** Cross-encoder
