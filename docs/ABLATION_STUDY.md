# Ablation study design

## Question

When Pythia-160M is LoRA fine-tuned on Python code, do the features found by a sparse autoencoder change by more than they would from retraining the SAE with a different seed?

## Main measurement

For each feature in an SAE, take its best decoder cosine to other seeds of the same model, and subtract its best cosine to SAEs of the other model:

```
shift(f) = mean match to own model's other seeds − mean match to the other model's seeds
```

The reported shift averages this over all live features, in both directions. A value of 0 means the other model's SAE looks no different from another seed. Intervals are bootstrap 95% CIs over features.

Features within one SAE are not independent, so those intervals are somewhat too narrow. The seed range is reported as well and is part of the decision rule.

## Hypotheses and decision rules

These are fixed in the code before any full run, so results can't move the goalposts.

| | Claim | Supported only if |
|---|---|---|
| **H1** | Fine-tuning changes features beyond seed noise | CI lower bound > 0, every seed's shift > 0, and shift ≥ 0.02 |
| **H2** | The change grows with training | Spearman ≥ 0.8 across 0%, 10%, 30%, 100% of updates, and H1 is supported |
| **H3** | The change is specific to the domain | Python CI is above the placebo CI, the new-feature rate exceeds its seed-noise rate, and new features are more Python-selective than stable ones (one-sided Mann-Whitney p < 0.05) |

The 0.02 threshold is a judgment call: a practical minimum, not a derived value. H2 has only four points, so treat it as descriptive. No multiple-comparison correction is applied across the ablation table, so read those rows as exploratory.

**Calibration:** `control_above_layer` applies LoRA only above the hooked layer, so activations there can't change and the shift should be near zero. If it isn't, the report flags every other result as unreliable.

## Reference configuration

| Setting | Value |
|---|---|
| Model and layer | Pythia-160M, layer 6 output (0-indexed) |
| LoRA | r=16, alpha=32, dropout 0.05, on `query_key_value`, `dense_h_to_4h`, `dense_4h_to_h`, lr 3e-4 |
| Fine-tuning data | 100k Python functions, 2 epochs, early stopping on repos held out from training |
| SAE | BatchTopK, 6144 features (8×), k=64, 15k steps, batch 4096, auxiliary dead-feature loss |
| SAE data | 1.5M training and 200k test tokens, from repos fine-tuning never saw |
| Seeds | 3 per side for the reference, 2 per side elsewhere; the two sides never share seeds |

## Ablation groups

Each cell changes one thing from the reference and is run in this priority order. If time runs out, the core answers already exist. On a Kaggle T4, fine-tuning all models took about 5.5 hours and each SAE about 18 minutes, so the full study of 94 SAEs needs about 34 GPU hours across several sessions. The feature analysis behind H3 runs as soon as the reference SAEs exist.

| Group | Cells | What it tells you |
|---|---|---|
| `reference` | base vs LoRA on Python | H1 |
| `design` | same comparison on Wikipedia; base model on Python vs Wikipedia | Is the change limited to Python tokens? How big is a pure data effect? |
| `control` | LoRA above the hook; LoRA trained on Wikipedia | False-positive calibration; H3 placebo |
| `dose` | 10% and 30% of updates | H2 |
| `layer` | layers 2 and 10 | Where in the model the change shows up |
| `lora` | rank 4, rank 64; lr ÷3, lr ×10/3; full fine-tune | Whether the result is specific to LoRA settings |
| `sparsity` | k = 16, 32, 128 | Reconstruction vs sparsity trade-off |
| `architecture` | TopK; ReLU + L1 at 3 strengths | Whether BatchTopK is justified here |
| `width` | 2048, 12288 features | Whether matching depends on dictionary size |
| `pipeline` | old design with mismatched data; padding; unit-norm tokens; 200k tokens; no auxiliary loss; all old SAE choices together | How much each original choice distorted results |

Greedy vs Hungarian matching, decoder cosine vs activation correlation, and matched vs same-index frequency shift are computed for every cell, with no extra training.

These are measured on the first seed pair of each cell. Hungarian matching is skipped above 13,000 live features, which the configured widths don't reach.

## Metrics

| Metric | Meaning |
|---|---|
| FVU | Fraction of variance unexplained on held-out tokens |
| L0 | Mean active features per token |
| Dead fraction | Features that never fire on held-out tokens |
| CE recovered | 1 when the SAE splice matches the original model's loss, 0 when it's no better than replacing activations with their mean |
| Perplexity | Base vs tuned on held-out Python and on Wikipedia; the second shows any loss on general text |

## Feature classes

From the first seed's SAE, using fixed thresholds:

- **Reproducible:** matches another seed of its own model at cosine ≥ 0.7
- **New:** reproducible, with best match to the other model < 0.5
- **Stable:** reproducible, with best match to the other model ≥ 0.7

The same rule is applied across seeds of one model to measure how often seed noise alone produces "new" features.

These thresholds are conventions, not measured boundaries. Raw scores are saved in `features_*.csv`, so other cut-offs can be checked.

## Known limits

- LoRA restricts change to low-rank updates, so results describe LoRA fine-tuning. The `full_finetune` cell checks how far they carry over.
- Results apply to this model, these layers and this dataset.
- Top-activating examples are for reading, not proof of what a feature means.
- All SAEs are trained on one fine-tuning seed, so variation across fine-tuning runs isn't measured.
