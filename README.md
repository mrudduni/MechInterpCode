# MechInterpCode

Do sparse autoencoder (SAE) features change when a model is fine-tuned on a narrow domain, by more than they change from retraining the SAE with a new seed?

This repo LoRA fine-tunes [Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m) on Python code, trains SAEs on the base and tuned models using identical tokens, and compares them against seed noise, placebo training and a zero-change control.

## Run on Kaggle

1. Create a notebook and set **Accelerator: GPU T4** and **Internet: On**.
2. Run this in a cell:

```bash
!git clone --depth 1 https://github.com/mrudduni/MechInterpCode
!python MechInterpCode/run_experiment.py --preset main
```

3. Use **Save Version → Save & Run All** so the run continues after you close the tab.

**Check the wiring first (optional):**

```bash
!python MechInterpCode/run_experiment.py --preset smoke --out /kaggle/working/smoke
```

**Continuing after the time limit:** the script stops cleanly at 11 hours. Add that version's output to a new notebook as an input and run the same command. Finished work is found automatically and skipped.

**A smaller first run:** the first four groups answer the main questions.

```bash
!python MechInterpCode/run_experiment.py --preset main --groups reference,design,control,dose
```

## Outputs

Written to `/kaggle/working/outputs/`:

| File | Contents |
|---|---|
| `report.md` | Hypothesis verdicts, tables and figures |
| `ablation_results.csv` | One row per comparison |
| `finetune_results.csv` | Perplexity and parameter change per model |
| `features_tuned.csv`, `features_base.csv` | Per-feature scores and classes |
| `example_features.md` | Top-activating tokens, for reading |
| `resume_lines.md` | Resume bullets filled in with your measured numbers (`main` preset only) |
| `memory_log.csv` | Peak RAM and VRAM, and free disk, per stage |
| `models/`, `saes/`, `cells/` | Saved work used for resuming |

## Staying inside Kaggle's limits

| Limit | What the script does |
|---|---|
| **VRAM** | Skips logits and later layers while caching activations. Holds one model at a time. Halves the batch and retries if CUDA runs out of memory. Keeps activations on the GPU only if they fit in half the free VRAM |
| **RAM** | Stores tokens as flat arrays, memory-maps activations and streams feature correlations in batches |
| **`/kaggle/working` (20 GB)** | Saves LoRA adapters instead of full models. Activation caches go to scratch disk |
| **Scratch disk** | Stores activations as float16 and evicts the oldest caches when space runs low; they're rebuilt when needed |
| **Session time** | Stops cleanly at 11 hours and resumes from saved work |

If you still hit a limit, lower `n_train`, `d_sae` or `ft_batch` in the `Preset` class.

## Local tests

```bash
pip install -r requirements.txt pytest
```

```bash
pytest tests -q
```

## Repo layout

| Path | Purpose |
|---|---|
| `run_experiment.py` | The whole pipeline, with all config at the top |
| `docs/ABLATION_STUDY.md` | Study design and decision rules |
| `docs/AUDIT.md` | What was wrong with the first version |
| `tests/test_pipeline.py` | CPU tests using a tiny model |

## License

Apache-2.0
