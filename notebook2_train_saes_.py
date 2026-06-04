# =============================================================================
# NOTEBOOK 2: Train Sparse Autoencoders on Base & Fine-tuned Pythia-160M
# AIMS DTU MechInterp 2026 — Phase 1 & 2 (SAE training)
#
# Run AFTER notebook 1. Attach notebook 1 output as a dataset.
# Kaggle: GPU T4, Internet ON
# =============================================================================

# ── 0. Install ────────────────────────────────────────────────────────────────
import subprocess, sys
def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])
pip("transformers>=4.40.0", "datasets>=2.19.0", "tqdm", "einops")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 2. Check input directory ──────────────────────────────────────────────────
print("Available input datasets:")
for d in os.listdir("/kaggle/input"):
    print(f"  /kaggle/input/{d}")

# ── 3. Config ─────────────────────────────────────────────────────────────────
# UPDATE INPUT_DIR to match the dataset name printed above
INPUT_DIR = Path("/kaggle/input/feeder-dataset")   # <-- change to your dataset name
WORK_DIR  = Path("/kaggle/working")

BASE_MODEL_ID = "EleutherAI/pythia-160m"
FT_MODEL_DIR  = INPUT_DIR / "pythia-160m-python-ft"

SAE_CFG = dict(
    d_model        = 768,      # Pythia-160M hidden dim — must match notebook 1
    d_sae          = 2048,     # expansion ~2.7x — safe for 200k token corpus
    l1_coeff       = 2e-5,     # small — prevents dead feature collapse
    lr             = 3e-4,
    batch_size     = 1024,
    n_epochs       = 20,
    seed           = 42,
    cache_layer    = 6,
    cache_n_tokens = 200_000,
    device         = "cuda" if torch.cuda.is_available() else "cpu",
)

random.seed(SAE_CFG["seed"])
np.random.seed(SAE_CFG["seed"])
torch.manual_seed(SAE_CFG["seed"])
print(f"Device : {SAE_CFG['device']}")
print(f"d_model: {SAE_CFG['d_model']}  d_sae: {SAE_CFG['d_sae']}")

# ── 4. Sparse Autoencoder ─────────────────────────────────────────────────────
# NOTE: No b_dec parameter — matches the architecture saved by notebook 1
class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae   = d_sae
        self.b_pre  = nn.Parameter(torch.zeros(d_model))
        self.W_enc  = nn.Parameter(torch.nn.init.kaiming_uniform_(
                          torch.empty(d_model, d_sae)))
        self.b_enc  = nn.Parameter(torch.zeros(d_sae))
        self.W_dec  = nn.Parameter(self.W_enc.data.T.clone())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalise_decoder(self):
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1.0)
        self.W_dec.data /= norms

    def loss(self, x: torch.Tensor, l1_coeff: float):
        x_hat, z = self(x)
        l2 = (x_hat - x).pow(2).mean()
        l1 = l1_coeff * z.abs().mean()
        return l2 + l1, {"l2": l2.item(), "l1": l1.item(),
                         "mean_active": (z > 0).float().mean().item()}

# ── 5. Activation collection ──────────────────────────────────────────────────
def collect_activations(model_id_or_path, layer_idx, n_tokens,
                        cache_path, tokenizer, device):
    if cache_path.exists():
        acts = np.load(cache_path)
        print(f"  Loaded cache: {acts.shape}  NaN: {np.isnan(acts).sum()}")
        if np.isnan(acts).sum() == 0:
            return acts
        print("  Cache has NaN — recollecting...")
        cache_path.unlink()

    print(f"  Collecting {n_tokens:,} tokens from layer {layer_idx} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_id_or_path), torch_dtype=torch.float32).to(device).eval()

    act_list = []
    total    = 0

    def hook_fn(module, inp, out):
        hidden = out[0].detach().cpu().float()
        hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1.0, neginf=-1.0)
        act_list.append(hidden)

    handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook_fn)

    from datasets import load_dataset as lds
    wiki = lds("wikitext", "wikitext-103-raw-v1", split="train")

    with torch.no_grad():
        for sample in tqdm(wiki, desc="  Collecting"):
            if total >= n_tokens:
                break
            text = sample["text"].strip()
            if len(text) < 50:
                continue
            ids = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=512)["input_ids"].to(device)
            model(ids)
            total += ids.shape[1]

    handle.remove()
    del model
    torch.cuda.empty_cache()

    acts = torch.cat(act_list, dim=1).squeeze(0).float().numpy()[:n_tokens]
    print(f"  Shape: {acts.shape}  NaN: {np.isnan(acts).sum()}  Mean: {acts.mean():.4f}")
    np.save(cache_path, acts)
    return acts

# ── 6. SAE training ───────────────────────────────────────────────────────────
def train_sae(acts: np.ndarray, cfg: dict, tag: str) -> SparseAutoencoder:
    X = torch.tensor(acts, dtype=torch.float32)

    if torch.isnan(X).any() or torch.isinf(X).any():
        print("  WARNING: NaN/Inf in activations — cleaning...")
        X = torch.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    loader    = DataLoader(TensorDataset(X), batch_size=cfg["batch_size"],
                           shuffle=True, drop_last=True)
    sae       = SparseAutoencoder(cfg["d_model"], cfg["d_sae"]).to(cfg["device"])
    opt       = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], eps=1e-8)
    best_loss = float("inf")
    save_path = WORK_DIR / f"sae_{tag}.pt"
    last_z    = None
    info      = {"l2": 0.0, "l1": 0.0, "mean_active": 0.0}

    for epoch in trange(cfg["n_epochs"], desc=f"SAE [{tag}]"):
        epoch_losses = []

        for (batch,) in loader:
            batch = batch.to(cfg["device"])
            opt.zero_grad()
            loss, info = sae.loss(batch, cfg["l1_coeff"])

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            opt.step()
            sae.normalise_decoder()
            epoch_losses.append(loss.item())

            with torch.no_grad():
                last_z = sae.encode(batch).detach()

        if not epoch_losses:
            print(f"  ERROR: all NaN at epoch {epoch+1}")
            continue

        avg = np.mean(epoch_losses)
        if avg < best_loss and not math.isnan(avg):
            best_loss = avg
            torch.save(sae.state_dict(), save_path)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            dead = int((last_z.mean(0) < 1e-4).sum().item()) if last_z is not None else -1
            print(f"  epoch {epoch+1:3d}  loss={avg:.4f}  "
                  f"l2={info['l2']:.4f}  l1={info['l1']:.4f}  "
                  f"active={info['mean_active']:.3f}  dead={dead}/{cfg['d_sae']}")
            if dead > cfg["d_sae"] * 0.8:
                print("  WARNING: >80% dead features — reduce l1_coeff")

    if not save_path.exists():
        torch.save(sae.state_dict(), save_path)

    sae.load_state_dict(torch.load(save_path, map_location="cpu"))
    print(f"  ✓ SAE [{tag}] saved → {save_path}")
    return sae

# ── 7. Load tokenizer ─────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ── 8. Base model activations (Wikipedia — domain neutral) ───────────────────
print("\n[1/4] Base model activations ...")
acts_base = collect_activations(
    BASE_MODEL_ID,
    layer_idx  = SAE_CFG["cache_layer"],
    n_tokens   = SAE_CFG["cache_n_tokens"],
    cache_path = WORK_DIR / "act_cache_base.npy",
    tokenizer  = tokenizer,
    device     = SAE_CFG["device"],
)

# ── 9. Fine-tuned model activations ──────────────────────────────────────────
print("\n[2/4] Fine-tuned model activations ...")
ft_cache = INPUT_DIR / "act_cache_finetuned.npy"

if ft_cache.exists():
    acts_ft   = np.load(ft_cache)
    nan_count = np.isnan(acts_ft).sum()
    print(f"  Loaded: {acts_ft.shape}  NaN: {nan_count}  Mean: {np.nanmean(acts_ft):.4f}")
    if nan_count > 0:
        print("  NaN in ft cache — recollecting from saved model ...")
        acts_ft = collect_activations(
            FT_MODEL_DIR,
            layer_idx  = SAE_CFG["cache_layer"],
            n_tokens   = SAE_CFG["cache_n_tokens"],
            cache_path = WORK_DIR / "act_cache_finetuned_clean.npy",
            tokenizer  = tokenizer,
            device     = SAE_CFG["device"],
        )
else:
    print("  Cache not found — collecting from saved model ...")
    acts_ft = collect_activations(
        FT_MODEL_DIR,
        layer_idx  = SAE_CFG["cache_layer"],
        n_tokens   = SAE_CFG["cache_n_tokens"],
        cache_path = WORK_DIR / "act_cache_finetuned_clean.npy",
        tokenizer  = tokenizer,
        device     = SAE_CFG["device"],
    )

print(f"\nBase acts : {acts_base.shape}")
print(f"FT acts   : {acts_ft.shape}")
assert not np.isnan(acts_base).any(), "Base activations contain NaN — abort"
assert not np.isnan(acts_ft).any(),   "FT activations contain NaN — abort"
print("✓ Both activation caches are clean")

# ── 10. Train SAEs ────────────────────────────────────────────────────────────
print("\n[3/4] Training SAE on base model activations ...")
sae_base = train_sae(acts_base, SAE_CFG, tag="base")

print("\n[4/4] Training SAE on fine-tuned model activations ...")
sae_ft = train_sae(acts_ft, SAE_CFG, tag="ft")

# ── 11. Extract feature matrices ─────────────────────────────────────────────
feats_base = sae_base.W_dec.detach().cpu().float().numpy()
feats_ft   = sae_ft.W_dec.detach().cpu().float().numpy()
np.save(WORK_DIR / "features_base.npy", feats_base)
np.save(WORK_DIR / "features_ft.npy",   feats_ft)
print(f"\nFeature matrices: {feats_base.shape}")

# ── 12. Activation frequency stats ───────────────────────────────────────────
def feature_stats(sae, acts, cfg):
    X = torch.tensor(acts, dtype=torch.float32)
    X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    loader   = DataLoader(TensorDataset(X), batch_size=4096, shuffle=False)
    sum_fire = torch.zeros(cfg["d_sae"])
    sum_act  = torch.zeros(cfg["d_sae"])
    n = 0
    sae.eval().to("cpu")
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="  Stats"):
            z         = sae.encode(batch)
            sum_fire += (z > 0).float().sum(0)
            sum_act  += z.sum(0)
            n        += z.shape[0]
    return {
        "activation_freq": (sum_fire / n).numpy(),
        "mean_activation": (sum_act  / n).numpy(),
    }

print("\nComputing feature statistics ...")
stats_base = feature_stats(sae_base, acts_base, SAE_CFG)
stats_ft   = feature_stats(sae_ft,   acts_ft,   SAE_CFG)
np.save(WORK_DIR / "stats_base.npy", stats_base)
np.save(WORK_DIR / "stats_ft.npy",   stats_ft)

# ── 13. Summary ───────────────────────────────────────────────────────────────
dead_base = (stats_base["activation_freq"] < 1e-4).sum()
dead_ft   = (stats_ft["activation_freq"]   < 1e-4).sum()
print(f"\n{'='*50}")
print(f"SAE summary  (d_sae={SAE_CFG['d_sae']})")
print(f"  Dead features (base) : {dead_base}")
print(f"  Dead features (ft)   : {dead_ft}")
print(f"  Active L0  (base)    : {(stats_base['activation_freq'] > 1e-4).sum()}")
print(f"  Active L0  (ft)      : {(stats_ft['activation_freq']   > 1e-4).sum()}")
print(f"{'='*50}")
print("\n✅ Notebook 2 complete. Outputs:")
print("   ├── sae_base.pt / sae_ft.pt")
print("   ├── features_base.npy / features_ft.npy")
print("   └── stats_base.npy / stats_ft.npy")
