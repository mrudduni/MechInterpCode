# =============================================================================
# NOTEBOOK 3: Feature Comparison Pipeline — Base vs Fine-tuned Pythia-160M
# AIMS DTU MechInterp 2026 — Phase 3 of 3
#
# Run AFTER notebook 2. Attach notebook-1 and notebook-2 outputs as datasets.
# CPU is sufficient for this notebook.
# =============================================================================

# ── 0. Install ────────────────────────────────────────────────────────────────
import subprocess, sys
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
pip("transformers>=4.40.0", "datasets>=2.19.0", "scikit-learn",
    "matplotlib", "seaborn", "umap-learn", "tqdm")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import umap
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 2. Check mounted datasets ─────────────────────────────────────────────────
print("Available input datasets:")
for d in os.listdir("/kaggle/input"):
    print(f"  /kaggle/input/{d}")

# ── 3. Paths & config ─────────────────────────────────────────────────────────
# UPDATE THESE to match what printed above
NB2_DIR  = Path("/kaggle/input/feeder-dataset-2")   # notebook-2 outputs (SAEs, features, stats)
NB1_DIR  = Path("/kaggle/input/feeder-dataset")     # notebook-1 outputs (model weights)
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(exist_ok=True)

BASE_MODEL_ID = "EleutherAI/pythia-160m"

CFG = dict(
    d_model       = 768,    # must match notebook 2
    d_sae         = 2048,   # must match notebook 2
    cache_layer   = 6,
    top_k_changed = 50,
    top_k_stable  = 50,
    cosine_thresh = 0.9,
    probe_prompts = [
        "def fibonacci(n):",
        "import numpy as np",
        "class LinearRegression:",
        "SELECT * FROM users WHERE",
        "Once upon a time in a land far away",
        "The patient was administered 10mg of",
        "In legal proceedings, the defendant",
        "The mitochondria is the powerhouse",
        "git commit -m 'fix bug'",
        "print('Hello, world!')",
    ],
)

# ── 4. Load artefacts ─────────────────────────────────────────────────────────
print("\n[1/7] Loading SAE artefacts ...")

feats_base = np.load(NB2_DIR / "features_base.npy")
feats_ft   = np.load(NB2_DIR / "features_ft.npy")
stats_base = np.load(NB2_DIR / "stats_base.npy", allow_pickle=True).item()
stats_ft   = np.load(NB2_DIR / "stats_ft.npy",   allow_pickle=True).item()

print(f"  Feature matrices : {feats_base.shape}  (base)  {feats_ft.shape}  (ft)")
assert feats_base.shape == (CFG["d_sae"], CFG["d_model"]), \
    f"Shape mismatch: expected ({CFG['d_sae']}, {CFG['d_model']}), got {feats_base.shape}. Update d_sae in CFG."
assert feats_ft.shape == (CFG["d_sae"], CFG["d_model"]), \
    f"Shape mismatch: expected ({CFG['d_sae']}, {CFG['d_model']}), got {feats_ft.shape}. Update d_sae in CFG."
print("  ✓ Shape check passed")

# ── 5. SAE class (must match notebook 2 — no b_dec) ──────────────────────────
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))

    def encode(self, x):
        return F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    def decode(self, z):
        return z @ self.W_dec + self.b_pre

    def forward(self, x):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

def load_sae(path, d_model, d_sae):
    sae        = SparseAutoencoder(d_model, d_sae)
    state_dict = torch.load(path, map_location="cpu")
    state_dict.pop("b_dec", None)   # remove if present from older saves
    sae.load_state_dict(state_dict, strict=False)
    sae.eval()
    return sae

sae_base = load_sae(NB2_DIR / "sae_base.pt", CFG["d_model"], CFG["d_sae"])
sae_ft   = load_sae(NB2_DIR / "sae_ft.pt",   CFG["d_model"], CFG["d_sae"])
print("  ✓ SAEs loaded")

# ── 6. Feature matching ───────────────────────────────────────────────────────
print("\n[2/7] Computing pairwise cosine similarities ...")

F_base = feats_base / (np.linalg.norm(feats_base, axis=1, keepdims=True) + 1e-8)
F_ft   = feats_ft   / (np.linalg.norm(feats_ft,   axis=1, keepdims=True) + 1e-8)

cos_sim         = cosine_similarity(F_base, F_ft)
best_match_idx  = cos_sim.argmax(axis=1)
best_match_cos  = cos_sim[np.arange(len(cos_sim)), best_match_idx]

print(f"  Median best-match cosine : {np.median(best_match_cos):.4f}")
print(f"  % features cos ≥ {CFG['cosine_thresh']} : "
      f"{(best_match_cos >= CFG['cosine_thresh']).mean()*100:.1f}%")

# ── 7. Quantitative metrics ───────────────────────────────────────────────────
print("\n[3/7] Quantitative metrics ...")

def jaccard_sets(freq_base, freq_ft, thresh=1e-3):
    set_b = set(np.where(freq_base > thresh)[0])
    set_f = set(np.where(freq_ft   > thresh)[0])
    inter = len(set_b & set_f)
    union = len(set_b | set_f)
    return inter / union if union > 0 else 0.0

metrics = {
    "mean_cosine_similarity"  : float(best_match_cos.mean()),
    "median_cosine_similarity": float(np.median(best_match_cos)),
    "fraction_matched_09"     : float((best_match_cos >= 0.9).mean()),
    "fraction_matched_07"     : float((best_match_cos >= 0.7).mean()),
    "fraction_lost_05"        : float((best_match_cos < 0.5).mean()),
    "jaccard_active_features" : jaccard_sets(
                                    stats_base["activation_freq"],
                                    stats_ft["activation_freq"]),
    "delta_dead_features"     : int(
                                    (stats_ft["activation_freq"]   < 1e-4).sum()
                                  - (stats_base["activation_freq"] < 1e-4).sum()),
    "mean_freq_shift"         : float(np.abs(
                                    stats_ft["activation_freq"] -
                                    stats_base["activation_freq"]).mean()),
}

print(json.dumps(metrics, indent=2))
with open(WORK_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── 8. Identify changed / stable features ────────────────────────────────────
print("\n[4/7] Identifying changed & stable features ...")

freq_shift   = np.abs(stats_ft["activation_freq"] - stats_base["activation_freq"])
change_score = (1 - best_match_cos) + 0.5 * (freq_shift / (freq_shift.max() + 1e-8))

idx_changed = np.argsort(change_score)[::-1][:CFG["top_k_changed"]]
idx_stable  = np.argsort(best_match_cos)[::-1][:CFG["top_k_stable"]]

# ── 9. Probe-based feature labelling ─────────────────────────────────────────
print("\n[5/7] Labelling features with probe prompts ...")

tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Use dtype= instead of deprecated torch_dtype=
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, dtype=torch.float32).eval()
ft_model   = AutoModelForCausalLM.from_pretrained(
    str(NB1_DIR / "pythia-160m-python-ft"), dtype=torch.float32).eval()

def get_layer_acts(prompt, model, layer_idx):
    ids  = tokenizer(prompt, return_tensors="pt")["input_ids"]
    acts = []
    def hook(module, inp, out):
        acts.append(out[0].detach().float())
    h = model.gpt_neox.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()
    return acts[0].squeeze(0)   # [T, D]

def top_features_for_prompt(prompt, model, sae, layer_idx, top_n=10):
    acts   = get_layer_acts(prompt, model, layer_idx)
    acts   = acts / (acts.norm(dim=-1, keepdim=True) + 1e-8)
    with torch.no_grad():
        z = sae.encode(acts)
    mean_z  = z.mean(0).numpy()
    top_idx = np.argsort(mean_z)[::-1][:top_n]
    return top_idx, mean_z[top_idx]

feature_labels = {}

for prompt in tqdm(CFG["probe_prompts"], desc="  Probing"):
    top_base, vals_base = top_features_for_prompt(
        prompt, base_model, sae_base, CFG["cache_layer"])
    top_ft, vals_ft = top_features_for_prompt(
        prompt, ft_model, sae_ft, CFG["cache_layer"])

    for idx, val in zip(top_base[:5], vals_base[:5]):
        feature_labels.setdefault(f"base_{idx}", []).append((prompt, float(val)))
    for idx, val in zip(top_ft[:5], vals_ft[:5]):
        feature_labels.setdefault(f"ft_{idx}", []).append((prompt, float(val)))

# ── 10. Visualisations ────────────────────────────────────────────────────────
print("\n[6/7] Generating plots ...")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# 10a. Cosine histogram
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(best_match_cos, bins=80, color="#3A86FF", edgecolor="white", linewidth=0.4)
ax.axvline(CFG["cosine_thresh"], color="#FF006E", linestyle="--",
           label=f"threshold = {CFG['cosine_thresh']}")
ax.set_xlabel("Best-match cosine similarity (base → ft)")
ax.set_ylabel("Number of features")
ax.set_title("Feature Preservation: Base vs Fine-tuned Pythia-160M")
ax.legend()
plt.tight_layout()
plt.savefig(WORK_DIR / "fig_cosine_histogram.png")
plt.close()

# 10b. Frequency scatter
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(stats_base["activation_freq"], stats_ft["activation_freq"],
           s=4, alpha=0.3, c=best_match_cos, cmap="RdYlGn", vmin=0, vmax=1)
lo = 0
hi = max(stats_base["activation_freq"].max(),
         stats_ft["activation_freq"].max()) * 1.05
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="no change")
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Best-match cosine")
ax.set_xlabel("Activation frequency (base)")
ax.set_ylabel("Activation frequency (fine-tuned)")
ax.set_title("Feature Frequency Shift per Feature")
ax.legend()
plt.tight_layout()
plt.savefig(WORK_DIR / "fig_frequency_scatter.png")
plt.close()

# 10c. UMAP
print("  UMAP (this may take ~2 min) ...")
n_sample = min(1000, CFG["d_sae"])
idx_s    = np.random.choice(CFG["d_sae"], n_sample, replace=False)
combined = np.vstack([F_base[idx_s], F_ft[idx_s]])
labels   = np.array(["base"] * n_sample + ["ft"] * n_sample)

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
emb     = reducer.fit_transform(combined)

fig, ax = plt.subplots(figsize=(9, 7))
for lab, col in [("base", "#3A86FF"), ("ft", "#FF006E")]:
    mask = labels == lab
    ax.scatter(emb[mask, 0], emb[mask, 1], s=5, alpha=0.5, c=col, label=lab)
ax.set_title("UMAP of SAE Feature Directions (Base vs Fine-tuned)")
ax.legend(markerscale=3)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig(WORK_DIR / "fig_umap_features.png")
plt.close()

# 10d. Top-50 changed features bar chart
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(CFG["top_k_changed"]),
       best_match_cos[idx_changed],
       color=plt.cm.RdYlGn(best_match_cos[idx_changed]))
ax.set_xlabel("Feature rank (most changed first)")
ax.set_ylabel("Best-match cosine similarity")
ax.set_title(f"Top-{CFG['top_k_changed']} Most Changed Features")
ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig(WORK_DIR / "fig_top_changed.png")
plt.close()

# 10e. Frequency delta for top changed features
delta_freq  = (stats_ft["activation_freq"] - stats_base["activation_freq"])[idx_changed]
colours_bar = ["#FF006E" if d > 0 else "#3A86FF" for d in delta_freq]
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(delta_freq)), delta_freq, color=colours_bar)
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("Feature rank (most changed first)")
ax.set_ylabel("Δ activation frequency (ft − base)")
ax.set_title("Activation Frequency Change for Most-Changed Features")
plt.tight_layout()
plt.savefig(WORK_DIR / "fig_freq_delta.png")
plt.close()

print("  ✓ All figures saved.")

# ── 11. Feature change report ─────────────────────────────────────────────────
print("\n[7/7] Writing feature change report ...")

report_lines = [
    "# Feature Change Report: Pythia-160M Base vs Python Fine-tuned",
    f"Layer analysed : {CFG['cache_layer']} / 11",
    f"SAE width      : {CFG['d_sae']}",
    "",
    "## Quantitative Metrics",
]
for k, v in metrics.items():
    report_lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float)
                        else f"- **{k}**: {v}")

report_lines += [
    "",
    f"## Top-{CFG['top_k_changed']} Most Changed Features",
    "| rank | feature_id | best_match_cos | Δfreq | top_probe_prompts |",
    "|------|-----------|---------------|-------|-------------------|",
]
for rank, feat_idx in enumerate(idx_changed):
    cos_val   = best_match_cos[feat_idx]
    dfreq_val = delta_freq[rank]
    prompts   = [p for p, _ in feature_labels.get(f"base_{feat_idx}", [])[:2]]
    report_lines.append(
        f"| {rank+1} | {feat_idx} | {cos_val:.3f} | {dfreq_val:+.4f} "
        f"| {'; '.join(prompts) or 'N/A'} |"
    )

report_lines += [
    "",
    f"## Top-{CFG['top_k_stable']} Most Stable Features",
    "| rank | feature_id | best_match_cos | freq_base | freq_ft |",
    "|------|-----------|---------------|-----------|---------| ",
]
for rank, feat_idx in enumerate(idx_stable):
    cos_val = best_match_cos[feat_idx]
    fb      = stats_base["activation_freq"][feat_idx]
    ff      = stats_ft["activation_freq"][feat_idx]
    report_lines.append(
        f"| {rank+1} | {feat_idx} | {cos_val:.4f} | {fb:.4f} | {ff:.4f} |"
    )

report_lines += [
    "",
    "## Interpretation Notes",
    "- Features with cosine < 0.5 represent new or reorganised directions.",
    "- Positive Δfreq = new concept introduced by fine-tuning.",
    "- Negative Δfreq = concept suppressed by fine-tuning.",
    "- Stable features (cos ≥ 0.9) = knowledge preserved across fine-tuning.",
    "",
    "## Figures",
    "- `fig_cosine_histogram.png`  — distribution of best-match cosines",
    "- `fig_frequency_scatter.png` — per-feature frequency drift",
    "- `fig_umap_features.png`     — UMAP of feature directions",
    "- `fig_top_changed.png`       — cosine for most-changed features",
    "- `fig_freq_delta.png`        — frequency delta for most-changed features",
]

with open(WORK_DIR / "feature_change_report.md", "w") as f:
    f.write("\n".join(report_lines))

print("✅ Notebook 3 complete. Outputs in /kaggle/working/:")
print("   ├── metrics.json")
print("   ├── feature_change_report.md")
print("   ├── fig_cosine_histogram.png")
print("   ├── fig_frequency_scatter.png")
print("   ├── fig_umap_features.png")
print("   ├── fig_top_changed.png")
print("   └── fig_freq_delta.png")
