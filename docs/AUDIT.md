# Audit of the original notebooks

What was wrong with the first version, and how `run_experiment.py` handles it.

## Results were invalid

| Problem | Evidence | Fix |
|---|---|---|
| Saved fine-tuned model is entirely NaN | All 162,322,944 parameters in the saved `model.safetensors` are NaN, and the config says `dtype: float16` | Weights are always cast to float32 and trained under autocast. Non-finite values stop the run |
| Saved activation cache is entirely NaN | All 200,000 rows of `act_cache_finetuned.npy` are NaN | Every activation batch is checked before it is written |
| NaN was hidden | Notebook 2 applied `nan_to_num(nan=0)` while collecting | Removed. A non-finite value raises an error |

## The experiment could not answer its question

| Problem | Why it matters | Fix |
|---|---|---|
| Base SAE saw Wikipedia and tuned SAE saw Python | Any difference could come from the data, not the model | Both SAEs see identical tokens. The old design stays as the `legacy_confound` ablation |
| No seed baseline | SAEs differ between seeds, so a low cosine alone means nothing | Several seeds per side. The headline number is the shift beyond seed noise |
| Frequency shift and Jaccard compared feature `i` to feature `i` | Indices of separately trained SAEs do not correspond | Compared only through one-to-one matched pairs |
| "Hungarian" matching was greedy `argmax` | Many features could match one feature, which inflates similarity | Real Hungarian matching, with the greedy many-to-one rate reported |
| Only decoder direction was compared | Misses features that behave alike with a rotated direction | Activation correlation on the same tokens is reported too |
| Any low-cosine feature was called "new" | Most of those may be seed noise | "New" must reproduce across seeds and be absent from the base model. The rule's false-positive rate is measured on seed noise |
| Report text read meaning into frequency changes | "Positive shift = new concept" was never tested | Removed. Top-activating examples are given for manual reading |

## Training and data bugs

| Problem | Effect | Fix |
|---|---|---|
| `padding="max_length"` with no attention mask while caching | Padding entered the activation data | Dynamic padding with a mask. Padding and separator positions are excluded and tested |
| 200,000 tokens at 512 per document | Only about 390 functions | About 1.7M tokens from documents the fine-tune never saw |
| Layer hook used `out[0]` | Current `transformers` returns a bare tensor, so this keeps only the first sequence | Handles both forms and checks the shape |
| `code_search_net` no longer resolves on the Hub | Data loading fails | `Nan-Do/code-search-net-python` (parquet) and `Salesforce/wikitext` |
| Dataset taken by row order, using its first 80k rows | Biased towards whichever repos come first | Splits by hash of repo or article, then shuffled |
| Perplexity check read a removed `labels` column | The cell crashes | Token-weighted loss, ignoring padding |
| Effective L1 about 7.5e-6 with mean reduction | Likely almost no sparsity pressure | BatchTopK sets sparsity directly |
| Unit-norm per token | Removes activation magnitude | One dataset-wide scale. Kept as an ablation |
| Decoder norm clamp only shrank rows | Rows could stay below unit norm | Decoder rows are renormalised every step, with tangent gradients |
| "Best" SAE selected on training loss | Not an evaluation | Fixed step budget, metrics on held-out documents |
| "Mean L0" counted live features | Mislabelled | True mean active features per token |

## Repo clutter

- One-off file names such as `notebook1-finetune-pythia (1).ipynb`, and hard-coded `feeder-dataset` paths.
- Three notebooks with copied SAE classes and copied config, which had drifted apart (`d_model` was 512 in one version and 768 in another).
- An unused `HF_TOKEN` lookup, and unused `wandb` and `bitsandbytes` installs.
- Results stored with pickle in `.npy` files.

All of this is now one script, with config in one place.
