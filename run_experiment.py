#!/usr/bin/env python3
"""Do SAE features change when Pythia-160M is LoRA fine-tuned on Python code?

Single-file, resumable pipeline for one Kaggle T4 GPU:

    python run_experiment.py --preset main          # everything (no args also works)
    python run_experiment.py --preset smoke         # minutes on CPU, checks wiring only
    python run_experiment.py --stage report         # rebuild the report from saved results
    python run_experiment.py --groups reference,design,control,dose

Stages: fine-tune LoRA adapters -> cache activations -> train SAEs -> compare
SAE pairs against seed noise -> write report, figures and resume lines.

Every finished unit is saved, so re-running skips it. To continue on Kaggle,
attach the previous version's output as an input; it is adopted automatically.
See docs/ABLATION_STUDY.md for the design and docs/AUDIT.md for why it looks like this.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU: no DataParallel surprises
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PYTHON_DATASET = "Nan-Do/code-search-net-python"  # parquet; the old `code_search_net` id no longer resolves
WIKI_DATASET = "Salesforce/wikitext"
LORA_TARGETS = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
GB = 1024**3


# ── Configuration ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Preset:
    name: str = "main"
    base_model: str = "EleutherAI/pythia-160m"
    layer: int = 6
    layer_low: int = 2
    layer_high: int = 10
    seq_len: int = 256
    # Fine-tuning
    ft_docs: int = 100_000
    ft_eval_docs: int = 2_000
    ft_epochs: int = 2
    ft_batch: int = 16
    ft_accum: int = 2
    ft_eval_every: int = 500  # optimizer updates
    ft_patience: int = 4
    lora_r: int = 16
    lora_lr: float = 3e-4
    # Activations (stored on disk as float16)
    n_train: int = 1_500_000
    n_test: int = 200_000
    small_corpus: int = 200_000
    acts_batch: int = 32
    # SAE
    d_sae: int = 6144
    k: int = 64
    steps: int = 15_000
    sae_batch: int = 4096
    widths: tuple = (2048, 12288)
    sparsities: tuple = (16, 32, 128)
    l1s: tuple = (1.0, 3.0, 10.0)
    legacy_batch: int = 1024
    ref_seeds: int = 3
    other_seeds: int = 2
    # Evaluation
    ce_docs: int = 64
    corr_tokens: int = 100_000
    hungarian_max: int = 13_000  # about 1.2 GB of RAM at this size; skipped above it
    bootstraps: int = 1000
    n_examples: int = 30
    # Runtime
    data_rows: int = 0  # >0: stream only this many raw rows
    ft_reserve_hours: float = 1.5
    sae_reserve_hours: float = 0.25


PRESETS = {
    "main": Preset(),
    "smoke": Preset(
        name="smoke", seq_len=64, ft_docs=64, ft_eval_docs=16, ft_epochs=1, ft_batch=4, ft_accum=1,
        ft_eval_every=4, n_train=6000, n_test=2000, small_corpus=1500, acts_batch=8, d_sae=1536, k=16,
        steps=150, sae_batch=512, widths=(768, 2304), sparsities=(8, 32), legacy_batch=256, ce_docs=4,
        corr_tokens=1500, bootstraps=100, n_examples=5, data_rows=15_000, ft_reserve_hours=0, sae_reserve_hours=0,
    ),
}


@dataclass(frozen=True)
class FtSpec:
    corpus: str = "python"
    method: str = "lora"  # lora | full
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 3e-4
    layers_from: int | None = None  # None: every layer
    docs_frac: float = 1.0
    doses: tuple = ()
    seed: int = 0


@dataclass(frozen=True)
class SaeSpec:
    model: str = "base"
    corpus: str = "python"
    layer: int = 6
    include_pad: bool = False
    n_train: int = 1_500_000
    arch: str = "batchtopk"  # batchtopk | topk | relu_l1
    d_sae: int = 6144
    k: int = 64
    l1: float = 3.0
    norm: str = "dataset"  # dataset | token
    aux: bool = True
    steps: int = 15_000
    batch: int = 4096
    lr: float = 3e-4
    seed: int = 0

    def key(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class Cell:
    name: str
    group: str
    question: str
    a: SaeSpec
    b: SaeSpec
    n_seeds: int = 2


def stable_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def model_registry(p: Preset) -> dict[str, FtSpec]:
    r, lr = p.lora_r, p.lora_lr
    ref = FtSpec(r=r, alpha=2 * r, lr=lr)
    return {
        "ft": replace(ref, doses=(0.1, 0.3)),
        "placebo": replace(ref, corpus="wiki"),  # more training, off-domain
        "ft_above": replace(ref, layers_from=p.layer + 1, docs_frac=0.25),  # cannot move the hooked layer
        "ft_r4": replace(ref, r=4, alpha=8),
        "ft_r64": replace(ref, r=64, alpha=128),
        "ft_lr_low": replace(ref, lr=lr / 3),
        "ft_lr_high": replace(ref, lr=lr * 10 / 3),
        "ft_full": FtSpec(method="full", lr=5e-5),
    }


def build_study(p: Preset) -> list[Cell]:
    r = SaeSpec(layer=p.layer, n_train=p.n_train, d_sae=p.d_sae, k=p.k, steps=p.steps, batch=p.sae_batch)
    base, ft = r, replace(r, model="ft")

    def both(name, group, question, **kw):
        return Cell(name, group, question, replace(base, **kw), replace(ft, **kw))

    def vs(name, group, question, model):
        return Cell(name, group, question, base, replace(r, model=model))

    legacy_steps = max(1, round(20 * p.small_corpus / p.legacy_batch))
    cells = [
        Cell("reference", "reference", "Base vs LoRA on the same Python tokens", base, ft, p.ref_seeds),
        both("neutral_corpus", "design", "Same comparison on Wikipedia tokens", corpus="wiki"),
        Cell("data_effect", "design", "Same base model, Python vs Wikipedia tokens", base, replace(r, corpus="wiki")),
        vs("control_above_layer", "control", "LoRA only above the hook: true change is zero", "ft_above"),
        vs("control_placebo", "control", "LoRA on Wikipedia: training without the domain", "placebo"),
        vs("dose_10", "dose", "After 10% of LoRA updates", "ft@10"),
        vs("dose_30", "dose", "After 30% of LoRA updates", "ft@30"),
        Cell("legacy_confound", "pipeline", "Old design: base on Wikipedia, tuned on Python", replace(r, corpus="wiki"), ft),
        both("include_padding", "pipeline", "Old choice: keep padding tokens", include_pad=True),
        both("token_unit_norm", "pipeline", "Old choice: unit-norm every token", norm="token"),
        both("small_corpus", "pipeline", "Old choice: 200k tokens", n_train=p.small_corpus),
        both("no_aux_loss", "pipeline", "Remove the dead-feature auxiliary loss", aux=False),
        both("legacy_sae", "pipeline", "All old SAE choices together", arch="relu_l1", l1=7.5e-6, norm="token",
             include_pad=True, n_train=p.small_corpus, aux=False, d_sae=2048 if p.d_sae > 2048 else p.d_sae,
             steps=legacy_steps, batch=p.legacy_batch),
        both("arch_topk", "architecture", "TopK instead of BatchTopK", arch="topk"),
        *[both(f"arch_l1_{l:g}", "architecture", f"ReLU + L1 ({l:g})", arch="relu_l1", l1=l, aux=False) for l in p.l1s],
        *[both(f"k_{k}", "sparsity", f"k = {k}", k=k) for k in p.sparsities],
        *[both(f"width_{w}", "width", f"{w} features", d_sae=w) for w in p.widths],
        both("layer_low", "layer", f"Layer {p.layer_low}", layer=p.layer_low),
        both("layer_high", "layer", f"Layer {p.layer_high}", layer=p.layer_high),
        vs("lora_r4", "lora", "LoRA rank 4", "ft_r4"),
        vs("lora_r64", "lora", "LoRA rank 64", "ft_r64"),
        vs("lora_lr_low", "lora", "Lower LoRA learning rate", "ft_lr_low"),
        vs("lora_lr_high", "lora", "Higher LoRA learning rate", "ft_lr_high"),
        vs("full_finetune", "lora", "Full fine-tune instead of LoRA", "ft_full"),
    ]
    return cells


# ── Runtime helpers ──────────────────────────────────────────────────────────
class Ctx:
    """Paths and clocks for one run."""

    out = Path("outputs")
    cache = Path(".cache_mechinterp")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deadline = math.inf


class OutOfTime(Exception):
    pass


class StopForward(Exception):
    """Raised from a hook so a forward pass ends at the layer we need."""


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    Ctx.out.mkdir(parents=True, exist_ok=True)
    with open(Ctx.out / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_deps() -> None:
    missing = [m for m in ("datasets", "transformers", "peft", "scipy", "pandas", "matplotlib", "safetensors")
               if importlib.util.find_spec(m) is None]
    if missing:
        print("Installing", missing, flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    # Kaggle's image pairs peft with a torchao too old for it, and peft then refuses to wrap any
    # layer. Nothing here uses torchao, so drop it from the throwaway session.
    from peft.import_utils import is_torchao_available

    try:
        is_torchao_available()
    except ImportError as e:
        print(f"Removing incompatible torchao: {e}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])
        importlib.invalidate_caches()
        is_torchao_available()


def free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mem_snapshot(tag: str) -> None:
    """Append RAM, VRAM and disk use to memory_log.csv so limits are visible before they bite."""
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    except ImportError:
        rss = float("nan")
    vram = torch.cuda.max_memory_allocated() / GB if torch.cuda.is_available() else 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    Ctx.out.mkdir(parents=True, exist_ok=True)
    disk_out, disk_cache = shutil.disk_usage(Ctx.out).free / GB, shutil.disk_usage(Ctx.cache).free / GB
    path = Ctx.out / "memory_log.csv"
    if not path.exists():
        path.write_text("time,tag,peak_ram_gb,peak_vram_gb,free_out_gb,free_cache_gb\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')},{tag},{rss:.2f},{vram:.2f},{disk_out:.1f},{disk_cache:.1f}\n")


def make_room(needed_bytes: int) -> None:
    """Evict the least recently used activation caches; they are cheap to rebuild."""
    margin = 2 * GB
    caches = sorted(Ctx.cache.glob("acts_*.npy"), key=lambda f: f.stat().st_mtime)
    while shutil.disk_usage(Ctx.cache).free < needed_bytes + margin and caches:
        victim = caches.pop(0)
        log(f"  disk low: evicting {victim.name}")
        victim.unlink()


def check_time(reserve_hours: float, what: str) -> None:
    if time.time() + reserve_hours * 3600 > Ctx.deadline:
        raise OutOfTime(what)


def assert_finite_tensor(t: torch.Tensor, what: str) -> None:
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"Non-finite values in {what}. Refusing to continue.")


def assert_finite_model(model: nn.Module, what: str) -> None:
    for name, param in model.named_parameters():
        assert_finite_tensor(param, f"{what}:{name}")


def adopt_previous_outputs(root: Path = Path("/kaggle/input")) -> int:
    """Kaggle: copy finished work from an attached earlier run so it is not redone. Returns SAEs found."""
    if not root.exists():
        return 0
    found = 0
    for manifest in root.rglob("run_manifest.json"):
        src = manifest.parent
        log(f"Adopting earlier results from {src}")
        for sub in ("models", "saes", "cells"):
            if (src / sub).exists():
                shutil.copytree(src / sub, Ctx.out / sub, dirs_exist_ok=True)
        for name in ("feature_summary.json", "example_features.md", "features_tuned.csv", "features_base.csv"):
            if (src / name).exists():
                shutil.copy2(src / name, Ctx.out / name)
        found += len(list((src / "saes").glob("*.pt")))
    return found


# ── Text data: repo/article-disjoint splits, compact token storage ───────────
def raw_table(corpus: str, p: Preset):
    """Return (dataset, text column, group ids). Groups keep one repo/article in one split."""
    from datasets import Dataset, load_dataset

    if corpus == "python":
        name, kw, text_col = PYTHON_DATASET, {}, "code"
    else:
        name, kw, text_col = WIKI_DATASET, {"name": "wikitext-103-raw-v1"}, "text"
    if p.data_rows:
        stream = load_dataset(name, split="train", streaming=True, **kw)
        ds = Dataset.from_list(list(stream.take(p.data_rows)))
    else:
        ds = load_dataset(name, split="train", **kw)
    if corpus == "python":
        ds = ds.select_columns(["repo", "code"])
        return ds, text_col, np.array(list(ds["repo"]), dtype=object)
    groups = np.zeros(len(ds), dtype=np.int64)
    article = 0
    for start in range(0, len(ds), 20_000):  # chunks keep RAM flat on 1.8M rows
        for i, line in enumerate(ds[start:start + 20_000]["text"]):
            if line.startswith(" = ") and not line.startswith(" = ="):
                article += 1
            groups[start + i] = article
    return ds, text_col, groups


def split_of(group) -> str:
    bucket = int(hashlib.md5(str(group).encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "eval" if bucket < 85 else "interp"


def tokenizer_for(p: Preset):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(p.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class TokenSet:
    """Documents as one flat int32 array plus offsets; far lighter than Python lists."""

    def __init__(self, ids: np.ndarray, offsets: np.ndarray):
        self.ids, self.offsets = ids, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def doc(self, i: int) -> np.ndarray:
        return self.ids[self.offsets[i]:self.offsets[i + 1]]

    def save(self, path: Path) -> None:
        np.savez(path, ids=self.ids, offsets=self.offsets)

    @classmethod
    def load(cls, path: Path) -> "TokenSet":
        d = np.load(path)
        return cls(d["ids"], d["offsets"])


def get_tokens(corpus: str, split: str, p: Preset, max_docs: int = 0, max_tokens: int = 0, pad: bool = False) -> TokenSet:
    tag = stable_hash([corpus, split, max_docs, max_tokens, pad, p.seq_len, p.data_rows, p.base_model])
    path = Ctx.cache / f"tokens_{corpus}_{split}_{tag}.npz"
    if path.exists():
        return TokenSet.load(path)
    ds, text_col, groups = raw_table(corpus, p)
    verdict = {g: split_of(g) for g in set(groups.tolist())}
    idx = np.array([i for i, g in enumerate(groups.tolist()) if verdict[g] == split], dtype=np.int64)
    if len(idx) == 0:
        raise RuntimeError(f"Split {corpus}/{split} is empty; raise data_rows.")
    np.random.default_rng(0).shuffle(idx)
    tok = tokenizer_for(p)
    eos = tok.eos_token_id
    min_chars = 40 if corpus == "python" else 200
    chunks, lengths, total = [], [], 0
    for start in range(0, len(idx), 2000):
        texts = [t for t in ds[idx[start:start + 2000].tolist()][text_col] if len(t.strip()) >= min_chars]
        if not texts:
            continue
        for ids in tok(texts, truncation=True, max_length=p.seq_len - 1, add_special_tokens=False)["input_ids"]:
            if len(ids) < 8:
                continue
            doc = [eos] + ids
            if pad:
                doc = doc + [eos] * (p.seq_len - len(doc))
            chunks.append(np.asarray(doc, dtype=np.int32))
            lengths.append(len(doc))
            total += len(doc)
            if (max_docs and len(lengths) >= max_docs) or (max_tokens and total >= max_tokens):
                break
        if (max_docs and len(lengths) >= max_docs) or (max_tokens and total >= max_tokens):
            break
    del ds, groups, verdict
    free_memory()
    ts = TokenSet(np.concatenate(chunks), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    ts.save(path)
    log(f"  tokens {corpus}/{split}: {len(ts):,} docs, {len(ts.ids):,} tokens")
    return ts


def batches(ts: TokenSet, batch_size: int, eos: int, pad: bool = False, rng: np.random.Generator | None = None, doc_ids=None):
    """Yield (ids, attention_mask). Shuffled batches are length-bucketed to cut wasted padding."""
    order = np.arange(len(ts)) if doc_ids is None else np.asarray(doc_ids)
    if rng is not None:
        rng.shuffle(order)
        pool = batch_size * 50
        lens = np.diff(ts.offsets)
        order = np.concatenate([p_[np.argsort(lens[p_])] for p_ in np.array_split(order, max(1, len(order) // pool))])
        starts = np.arange(0, len(order), batch_size)
        rng.shuffle(starts)
    else:
        starts = np.arange(0, len(order), batch_size)
    for s in starts:
        docs = [ts.doc(i) for i in order[s:s + batch_size]]
        width = max(len(d) for d in docs)
        ids = np.full((len(docs), width), eos, dtype=np.int64)
        mask = np.zeros((len(docs), width), dtype=np.int64)
        for row, d in enumerate(docs):
            ids[row, :len(d)] = d
            mask[row, :len(d)] = 1
        yield torch.from_numpy(ids), torch.from_numpy(mask)


# ── Models and LoRA fine-tuning ──────────────────────────────────────────────
def load_base(p: Preset) -> nn.Module:
    from transformers import AutoModelForCausalLM

    # Pythia ships as float16 and recent transformers keeps that dtype. Training float16
    # weights is what turned the old run entirely NaN, so always cast explicitly.
    model = AutoModelForCausalLM.from_pretrained(p.base_model).float()
    model.config.use_cache = False
    return model


def model_dir(key: str) -> Path:
    name, _, dose = key.partition("@")
    return Ctx.out / "models" / name / (f"dose_{dose}" if dose else "final")


def load_model(key: str, p: Preset) -> nn.Module:
    model = load_base(p)
    if key != "base":
        name = key.partition("@")[0]
        spec = model_registry(p)[name]
        if spec.method == "full":
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_dir(key)).float()
            model.config.use_cache = False
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_dir(key)).merge_and_unload()
    assert_finite_model(model, key)
    return model.eval().to(Ctx.device)


class ModelSlot:
    """Hold at most one model in memory."""

    key, model = None, None

    @classmethod
    def get(cls, key: str, p: Preset) -> nn.Module:
        if cls.key != key:
            cls.clear()
            cls.model, cls.key = load_model(key, p), key
        return cls.model

    @classmethod
    def clear(cls) -> None:
        cls.model, cls.key = None, None
        free_memory()


def autocast():
    return torch.amp.autocast("cuda", dtype=torch.float16, enabled=Ctx.device == "cuda")


@torch.no_grad()
def eval_loss(model: nn.Module, ts: TokenSet, eos: int, batch_size: int) -> float:
    """Per-token cross-entropy in nats, ignoring padding."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for ids, mask in batches(ts, batch_size, eos):
        ids, mask = ids.to(Ctx.device), mask.to(Ctx.device)
        with autocast():
            logits = model(input_ids=ids, attention_mask=mask).logits
        targets = ids[:, 1:].masked_fill(mask[:, 1:] == 0, -100)
        total += F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                                 ignore_index=-100, reduction="sum").item()
        count += int(mask[:, 1:].sum())
    model.train(was_training)
    if not math.isfinite(total):
        raise FloatingPointError("Non-finite evaluation loss.")
    return total / max(count, 1)


def relative_param_change(model: nn.Module, base: nn.Module) -> float:
    base_params = dict(base.named_parameters())
    diff, norm = 0.0, 0.0
    for name, param in model.named_parameters():
        ref = base_params[name].to(param.device)
        diff += (param.detach() - ref).pow(2).sum().item()
        norm += ref.pow(2).sum().item()
    return math.sqrt(diff / max(norm, 1e-12))


def finetune(key: str, p: Preset) -> None:
    final = model_dir(key)
    if (final / "metrics.json").exists():
        return
    check_time(p.ft_reserve_hours, f"fine-tune {key}")
    batch, accum = p.ft_batch, p.ft_accum
    while True:
        try:
            _finetune(key, p, batch, accum)
            break
        except torch.cuda.OutOfMemoryError:
            if batch <= 2:
                raise
            batch, accum = batch // 2, accum * 2
            log(f"  CUDA OOM: retrying {key} with batch={batch}, accum={accum}")
            free_memory()
    ModelSlot.clear()
    mem_snapshot(f"finetune:{key}")


def _finetune(key: str, p: Preset, batch: int, accum: int) -> None:
    spec = model_registry(p)[key]
    torch.manual_seed(spec.seed)
    rng = np.random.default_rng(spec.seed)
    tok = tokenizer_for(p)
    eos = tok.eos_token_id
    train = get_tokens(spec.corpus, "train", p, max_docs=max(8, int(p.ft_docs * spec.docs_frac)))
    held_out = {c: get_tokens(c, "eval", p, max_docs=p.ft_eval_docs) for c in ("python", "wiki")}
    ModelSlot.clear()

    model = load_base(p).to(Ctx.device)
    base_losses = {c: eval_loss(model, ts, eos, batch) for c, ts in held_out.items()}
    if spec.method == "lora":
        from peft import LoraConfig, get_peft_model

        n_layers = model.config.num_hidden_layers
        layers = list(range(spec.layers_from, n_layers)) if spec.layers_from is not None else None
        model = get_peft_model(model, LoraConfig(
            r=spec.r, lora_alpha=spec.alpha, lora_dropout=spec.dropout, target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM", layers_to_transform=layers, layers_pattern="layers" if layers else None))
    params = [q for q in model.parameters() if q.requires_grad]
    n_trainable = sum(q.numel() for q in params)
    log(f"Fine-tuning {key}: {spec.method}, {n_trainable:,} trainable params, batch {batch}x{accum}, lr {spec.lr:g}")

    updates_per_epoch = max(1, len(train) // batch // accum)
    total = updates_per_epoch * p.ft_epochs
    warmup = max(1, int(0.05 * total))
    opt = torch.optim.AdamW(params, lr=spec.lr, weight_decay=0.0 if spec.method == "lora" else 0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warmup) * (
        0.5 * (1 + math.cos(math.pi * min(1.0, u / total)))))
    scaler = torch.amp.GradScaler("cuda", enabled=Ctx.device == "cuda")
    dose_at = {max(1, int(d * total)): d for d in spec.doses}

    def snapshot():
        return {n: q.detach().cpu().clone() for n, q in model.named_parameters() if q.requires_grad}

    best, best_state, bad, update, skipped = math.inf, None, 0, 0, 0
    model.train()
    done = False
    for _ in range(p.ft_epochs):
        micro = 0
        opt.zero_grad(set_to_none=True)
        for ids, mask in batches(train, batch, eos, rng=rng):
            ids, mask = ids.to(Ctx.device), mask.to(Ctx.device)
            with autocast():
                loss = model(input_ids=ids, attention_mask=mask, labels=ids.masked_fill(mask == 0, -100)).loss / accum
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > 20:
                    raise FloatingPointError(f"{key}: repeated non-finite training loss.")
                opt.zero_grad(set_to_none=True)
                micro = 0
                continue
            scaler.scale(loss).backward()
            micro += 1
            if micro < accum:
                continue
            micro = 0
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            update += 1
            if update in dose_at:
                dose_dir = model_dir(f"{key}@{round(dose_at[update] * 100)}")
                dose_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(dose_dir)
            if update % p.ft_eval_every == 0 or update == total:
                current = eval_loss(model, held_out[spec.corpus], eos, batch)
                log(f"  {key} update {update}/{total}  held-out loss {current:.4f}  train {loss.item() * accum:.4f}")
                if current < best - 1e-4:
                    best, best_state, bad = current, snapshot(), 0
                else:
                    bad += 1
                    if bad >= p.ft_patience:
                        done = True
                        break
        if done:
            break
    if best_state is not None:
        state = dict(model.named_parameters())
        for name, value in best_state.items():
            state[name].data.copy_(value.to(state[name].device))
    assert_finite_model(model, key)

    final = model_dir(key)
    final.mkdir(parents=True, exist_ok=True)
    for dose in spec.doses:  # early stopping can end training before a dose is reached
        dose_dir = model_dir(f"{key}@{round(dose * 100)}")
        if not dose_dir.exists():
            log(f"  {key}: stopped before the {dose:.0%} dose; using the final adapter for it")
            dose_dir.mkdir(parents=True)
            model.save_pretrained(dose_dir)
    if spec.method == "lora":
        model.save_pretrained(final)  # adapter only, a few MB per run
        merged = model.merge_and_unload()
    else:
        merged = model
        merged.half().save_pretrained(final)  # 325 MB instead of 650 MB
        merged.float()
    merged.eval()
    assert_finite_model(merged, key)
    losses = {c: eval_loss(merged, ts, eos, batch) for c, ts in held_out.items()}
    base_cpu = load_base(p)
    metrics = {
        "key": key, "method": spec.method, "corpus": spec.corpus, "lora_r": spec.r, "lr": spec.lr,
        "trainable_params": n_trainable, "updates": update, "skipped_batches": skipped,
        "base_loss_python": base_losses["python"], "base_loss_wiki": base_losses["wiki"],
        "tuned_loss_python": losses["python"], "tuned_loss_wiki": losses["wiki"],
        "base_ppl_python": math.exp(base_losses["python"]), "tuned_ppl_python": math.exp(losses["python"]),
        "base_ppl_wiki": math.exp(base_losses["wiki"]), "tuned_ppl_wiki": math.exp(losses["wiki"]),
        "relative_param_change": relative_param_change(merged, base_cpu),
    }
    del base_cpu, merged, model, opt, sched
    free_memory()
    (final / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"  {key}: python ppl {metrics['base_ppl_python']:.2f} -> {metrics['tuned_ppl_python']:.2f}, "
        f"wiki ppl {metrics['base_ppl_wiki']:.2f} -> {metrics['tuned_ppl_wiki']:.2f}")


# ── Activation cache: memory-mapped float16 on scratch disk ──────────────────
def layer_tensor(output) -> torch.Tensor:
    """transformers has returned both a tuple and a bare tensor from this layer."""
    h = output[0] if isinstance(output, (tuple, list)) else output
    if h.dim() != 3:
        raise RuntimeError(f"Expected [batch, seq, dim] from the hooked layer, got {tuple(h.shape)}")
    return h


def active_mask(mask: torch.Tensor, include_pad: bool) -> torch.Tensor:
    """Positions that count as data: real tokens, excluding the leading document separator."""
    if include_pad:
        return torch.ones_like(mask, dtype=torch.bool)
    keep = mask.bool().clone()
    keep[:, 0] = False
    return keep


@torch.no_grad()
def forward_to_layer(model: nn.Module, ids, mask, layer: int) -> torch.Tensor:
    """Run only up to the hooked layer: no logits (about 3 GB per batch) and no later blocks."""
    caught = []

    def hook(_module, _inputs, output):
        caught.append(layer_tensor(output))
        raise StopForward

    handle = model.gpt_neox.layers[layer].register_forward_hook(hook)
    try:
        model.gpt_neox(input_ids=ids, attention_mask=mask)
    except StopForward:
        pass
    finally:
        handle.remove()
    return caught[0]


class ActLayout:
    """Where train and test tokens sit in an activation file (documents never straddle the split)."""

    def __init__(self, ts: TokenSet, p: Preset, include_pad: bool):
        counts = np.diff(ts.offsets) - (0 if include_pad else 1)
        self.act_offsets = np.concatenate([[0], np.cumsum(counts)])
        self.total = int(self.act_offsets[-1])
        self.test_doc = int(np.searchsorted(self.act_offsets, min(p.n_train, int(self.total * 0.88))))
        self.test_start = int(self.act_offsets[self.test_doc])


def interp_tokens(corpus: str, include_pad: bool, p: Preset) -> tuple[TokenSet, ActLayout]:
    wanted = int((p.n_train + p.n_test) * 1.02)
    ts = get_tokens(corpus, "interp", p, max_tokens=wanted, pad=include_pad)
    return ts, ActLayout(ts, p, include_pad)


def acts_path(model: str, corpus: str, layer: int, include_pad: bool) -> Path:
    tag = "pad" if include_pad else "nopad"
    return Ctx.cache / f"acts_{model.replace('@', '-')}_{corpus}_L{layer}_{tag}.npy"


def ensure_acts(model_key: str, corpus: str, layer: int, include_pad: bool, p: Preset) -> Path:
    path = acts_path(model_key, corpus, layer, include_pad)
    if path.exists():
        os.utime(path)
        return path
    ts, layout = interp_tokens(corpus, include_pad, p)
    model = ModelSlot.get(model_key, p)
    d = model.config.hidden_size
    make_room(layout.total * d * 2)
    eos = tokenizer_for(p).eos_token_id
    tmp = path.with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(layout.total, d))
    bs, pos = p.acts_batch, 0
    log(f"  caching {path.name}: {layout.total:,} tokens, {layout.total * d * 2 / GB:.1f} GB")
    t0 = time.time()
    docs = np.arange(len(ts))
    cursor = 0
    while cursor < len(docs):
        chunk = docs[cursor:cursor + bs]
        try:
            ids, mask = next(batches(ts, len(chunk), eos, doc_ids=chunk))
            h = forward_to_layer(model, ids.to(Ctx.device), None if include_pad else mask.to(Ctx.device), layer)
        except torch.cuda.OutOfMemoryError:
            bs = max(1, bs // 2)
            log(f"  CUDA OOM while caching: batch -> {bs}")
            free_memory()
            continue
        rows = h[active_mask(mask, include_pad).to(h.device)].float()
        assert_finite_tensor(rows, f"activations of {model_key}")
        half = rows.half()
        assert_finite_tensor(half, f"float16 activations of {model_key}")
        mm[pos:pos + len(half)] = half.cpu().numpy()
        pos += len(half)
        cursor += len(chunk)
    if pos != layout.total:
        raise RuntimeError(f"Token bookkeeping mismatch: wrote {pos}, expected {layout.total}")
    mm.flush()
    del mm
    os.replace(tmp, path)
    log(f"  cached in {time.time() - t0:.0f}s")
    return path


def read_acts(path: Path, start: int, stop: int) -> torch.Tensor:
    """Copy a slice out of the memmap as a float16 CPU tensor."""
    arr = np.load(path, mmap_mode="r")
    return torch.from_numpy(np.array(arr[start:stop]))


def place_acts(x: torch.Tensor) -> torch.Tensor:
    """Keep float16 activations on the GPU only when they fit in half of the free VRAM."""
    if Ctx.device == "cuda":
        free, _ = torch.cuda.mem_get_info()
        if x.numel() * 2 < free * 0.5:
            return x.to(Ctx.device)
    return x


# ── Sparse autoencoder ───────────────────────────────────────────────────────
class SAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, arch: str, k: int, norm: str):
        super().__init__()
        self.arch, self.k, self.norm = arch, k, norm
        dec = F.normalize(torch.randn(d_sae, d_in), dim=1)
        self.W_dec = nn.Parameter(dec)
        self.W_enc = nn.Parameter(dec.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae), requires_grad=arch == "relu_l1")
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("theta", torch.tensor(0.0))
        self.register_buffer("mean_raw", torch.zeros(d_in))

    def prep(self, x: torch.Tensor):
        if self.norm == "token":
            n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return x / n, n
        return x * self.scale, self.scale

    @staticmethod
    def unprep(x_hat: torch.Tensor, s) -> torch.Tensor:
        return x_hat * s if torch.is_tensor(s) and s.dim() else x_hat / s

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def sparsify(self, pre: torch.Tensor, training: bool = False) -> torch.Tensor:
        act = F.relu(pre)
        if self.arch == "relu_l1":
            return act
        if self.arch == "topk":
            vals, idx = act.topk(min(self.k, act.size(-1)), dim=-1)
            return torch.zeros_like(act).scatter(-1, idx, vals)
        if training:
            n = min(self.k * act.size(0), act.numel())
            vals, idx = act.flatten().topk(n)
            z = torch.zeros_like(act).flatten().scatter(0, idx, vals).view_as(act)
            positive = vals[vals > 0]
            if positive.numel():
                floor = positive.min().detach()
                self.theta = floor if self.theta == 0 else 0.99 * self.theta + 0.01 * floor
            return z
        return act * (act > self.theta)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsify(self.pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def reconstruct_raw(self, x: torch.Tensor) -> torch.Tensor:
        x_n, s = self.prep(x)
        return self.unprep(self.decode(self.encode(x_n)), s)

    @torch.no_grad()
    def renorm(self) -> None:
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


def fit_sae(spec: SaeSpec, x_train: torch.Tensor, verbose_every: int = 2500) -> SAE:
    torch.manual_seed(spec.seed)
    gen = torch.Generator(device=x_train.device).manual_seed(spec.seed)
    d_in, device = x_train.shape[1], Ctx.device
    sae = SAE(d_in, spec.d_sae, spec.arch, spec.k, spec.norm).to(device)

    sample = x_train[:min(200_000, len(x_train))].to(device).float()
    sae.mean_raw.copy_(sample.mean(0))
    if spec.norm == "dataset":
        sae.scale.copy_(math.sqrt(d_in) / sample.norm(dim=1).mean())
    sae.b_dec.data.copy_(sae.prep(sample)[0].mean(0))
    del sample

    params = [q for q in sae.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=spec.lr)
    decay_from = int(0.8 * spec.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 1.0 if s < decay_from else max(0.0, (spec.steps - s) / max(1, spec.steps - decay_from)))
    since_fired = torch.zeros(spec.d_sae, device=device)
    dead_after = max(1, min(len(x_train), 500_000) / spec.batch)
    use_aux = spec.aux and spec.arch != "relu_l1"
    k_aux = min(d_in // 2, spec.d_sae)
    l1_warmup = max(1, int(0.05 * spec.steps))

    for step in range(spec.steps):
        idx = torch.randint(len(x_train), (spec.batch,), generator=gen, device=x_train.device)
        x, _ = sae.prep(x_train[idx].to(device).float())
        pre = sae.pre(x)
        z = sae.sparsify(pre, training=True)
        x_hat = sae.decode(z)
        residual = x - x_hat
        loss = residual.pow(2).sum(-1).mean()
        if spec.arch == "relu_l1":
            loss = loss + spec.l1 * min(1.0, (step + 1) / l1_warmup) * z.sum(-1).mean()
        since_fired += 1
        since_fired[(z > 0).any(0)] = 0
        if use_aux:
            dead = since_fired > dead_after
            if dead.any():
                n = min(k_aux, int(dead.sum()))
                vals, ids = pre.masked_fill(~dead, -1e9).topk(n, dim=-1)
                z_aux = torch.zeros_like(pre).scatter(-1, ids, F.relu(vals))
                aux = (z_aux @ sae.W_dec - residual.detach()).pow(2).sum(-1).mean()
                loss = loss + aux / 32
        if not torch.isfinite(loss):
            raise FloatingPointError(f"SAE {spec.key()}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():  # keep decoder updates tangent to the unit sphere
            g, w = sae.W_dec.grad, sae.W_dec
            g -= (g * w).sum(1, keepdim=True) * w
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        sae.renorm()
        if verbose_every and (step + 1) % verbose_every == 0:
            fvu = residual.pow(2).sum() / (x - x.mean(0)).pow(2).sum()
            log(f"    step {step + 1}/{spec.steps}  fvu {fvu.item():.4f}  l0 {(z > 0).float().sum(-1).mean().item():.1f}  "
                f"dead {(since_fired > dead_after).float().mean().item():.3f}")
    return sae


@torch.no_grad()
def eval_sae(sae: SAE, x_test: torch.Tensor, batch: int = 4096) -> dict:
    device = Ctx.device
    mean = x_test[:min(len(x_test), 100_000)].to(device).float().mean(0)
    sse, sst, nnz, n = 0.0, 0.0, 0.0, 0
    fires = torch.zeros(sae.W_dec.shape[0], device=device)
    for start in range(0, len(x_test), batch):
        x = x_test[start:start + batch].to(device).float()
        x_n, s = sae.prep(x)
        z = sae.encode(x_n)
        x_hat = sae.unprep(sae.decode(z), s)
        sse += (x_hat - x).pow(2).sum().item()
        sst += (x - mean).pow(2).sum().item()
        active = z > 0
        nnz += active.sum().item()
        fires += active.sum(0).float()
        n += len(x)
    rates = (fires / max(n, 1)).cpu()
    return {"fvu": sse / max(sst, 1e-12), "l0": nnz / max(n, 1), "dead_frac": float((rates == 0).float().mean()), "rates": rates}


@torch.no_grad()
def ce_with_replacement(model: nn.Module, ts: TokenSet, docs, eos: int, layer: int, include_pad: bool, replace_fn=None) -> float:
    """Per-token CE when `replace_fn` overwrites active positions of the hooked layer."""
    handle = None
    state = {}
    if replace_fn is not None:
        def hook(_module, _inputs, output):
            h = layer_tensor(output).clone()
            m = state["mask"]
            h[m] = replace_fn(h[m].float()).to(h.dtype)
            return (h, *output[1:]) if isinstance(output, (tuple, list)) else h

        handle = model.gpt_neox.layers[layer].register_forward_hook(hook)
    total, count = 0.0, 0
    try:
        for ids, mask in batches(ts, 16, eos, doc_ids=docs):
            ids, mask = ids.to(Ctx.device), mask.to(Ctx.device)
            state["mask"] = active_mask(mask, include_pad)
            logits = model(input_ids=ids, attention_mask=None if include_pad else mask).logits
            effective = torch.ones_like(mask) if include_pad else mask
            targets = ids[:, 1:].masked_fill(effective[:, 1:] == 0, -100)
            total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), targets.reshape(-1),
                                     ignore_index=-100, reduction="sum").item()
            count += int(effective[:, 1:].sum())
    finally:
        if handle is not None:
            handle.remove()
    return total / max(count, 1)


def ce_recovered(sae: SAE, spec: SaeSpec, p: Preset) -> float:
    """1 = splice is as good as the original; 0 = no better than replacing activations by their mean."""
    ts, layout = interp_tokens(spec.corpus, spec.include_pad, p)
    docs = np.arange(layout.test_doc, min(layout.test_doc + p.ce_docs, len(ts)))
    if len(docs) == 0:
        return float("nan")
    model = ModelSlot.get(spec.model, p)
    eos = tokenizer_for(p).eos_token_id
    run = lambda fn=None: ce_with_replacement(model, ts, docs, eos, spec.layer, spec.include_pad, fn)
    orig = run()
    mean = sae.mean_raw
    ablated = run(lambda h: mean.expand_as(h))
    spliced = run(sae.reconstruct_raw)
    return (ablated - spliced) / max(ablated - orig, 1e-9)


def sae_path(spec: SaeSpec) -> Path:
    return Ctx.out / "saes" / f"{spec.key()}.pt"


def ensure_sae(spec: SaeSpec, p: Preset) -> None:
    path = sae_path(spec)
    if path.exists():
        return
    check_time(p.sae_reserve_hours, f"SAE {spec.key()}")
    acts = ensure_acts(spec.model, spec.corpus, spec.layer, spec.include_pad, p)
    _, layout = interp_tokens(spec.corpus, spec.include_pad, p)
    ModelSlot.clear()  # the model is not needed while the SAE trains
    n = min(spec.n_train, layout.test_start)
    log(f"  SAE {spec.key()} model={spec.model} corpus={spec.corpus} L{spec.layer} {spec.arch} "
        f"d={spec.d_sae} k={spec.k} seed={spec.seed} on {n:,} tokens")
    x_train = place_acts(read_acts(acts, 0, n))
    sae = fit_sae(spec, x_train, verbose_every=max(1, spec.steps // 3))
    del x_train
    free_memory()
    x_test = read_acts(acts, layout.test_start, layout.total)
    metrics = eval_sae(sae, x_test)
    del x_test
    metrics["ce_recovered"] = ce_recovered(sae, spec, p)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"spec": asdict(spec), "state": {k: v.cpu() for k, v in sae.state_dict().items()}, "metrics": metrics}, tmp)
    os.replace(tmp, path)
    log(f"    fvu {metrics['fvu']:.4f}  l0 {metrics['l0']:.1f}  dead {metrics['dead_frac']:.3f}  ce_rec {metrics['ce_recovered']:.3f}")
    del sae
    free_memory()
    mem_snapshot(f"sae:{spec.key()}")


def load_saved(spec: SaeSpec) -> dict:
    return torch.load(sae_path(spec), map_location="cpu", weights_only=True)


def build_sae(saved: dict, d_in: int | None = None) -> SAE:
    spec = saved["spec"]
    state = saved["state"]
    sae = SAE(state["W_dec"].shape[1], state["W_dec"].shape[0], spec["arch"], spec["k"], spec["norm"])
    sae.load_state_dict(state)
    return sae.eval().to(Ctx.device)


# ── Comparing dictionaries ───────────────────────────────────────────────────
def unit_alive(saved: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit decoder rows of features that fired on held-out tokens, with their original indices."""
    rates = saved["metrics"]["rates"]
    alive = (rates > 0).nonzero().squeeze(-1)
    if len(alive) == 0:
        alive = torch.arange(len(rates))
    return F.normalize(saved["state"]["W_dec"][alive].float(), dim=1), alive


@torch.no_grad()
def best_cosine(wa: torch.Tensor, wb: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """For each row of `wa`, the highest cosine to any row of `wb`."""
    wb = wb.to(Ctx.device)
    out = [(wa[i:i + chunk].to(Ctx.device) @ wb.T).max(1).values.cpu() for i in range(0, len(wa), chunk)]
    return torch.cat(out) if out else torch.empty(0)


def hungarian(wa: torch.Tensor, wb: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one matching maximising total cosine. Returns (a index, b index, cosine)."""
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        cos = (wa.to(Ctx.device) @ wb.to(Ctx.device).T).cpu().numpy()
    rows, cols = linear_sum_assignment(cos, maximize=True)
    return rows, cols, cos[rows, cols]


@torch.no_grad()
def activation_correlation(sae_a: SAE, sae_b: SAE, xa: torch.Tensor, xb: torch.Tensor, batch: int = 2048) -> torch.Tensor:
    """Per-feature max Pearson correlation of SAE A's features with SAE B's, on aligned tokens.
    Streams through batches, holding only d_sae x d_sae sums rather than the feature matrices."""
    device = Ctx.device
    da, db = sae_a.W_dec.shape[0], sae_b.W_dec.shape[0]
    sa, sb = torch.zeros(da, device=device).double(), torch.zeros(db, device=device).double()
    saa, sbb = sa.clone(), sb.clone()
    sab = torch.zeros(da, db, device=device).double()
    n = 0
    for start in range(0, len(xa), batch):
        za = sae_a.encode(sae_a.prep(xa[start:start + batch].to(device).float())[0]).double()
        zb = sae_b.encode(sae_b.prep(xb[start:start + batch].to(device).float())[0]).double()
        sa += za.sum(0); sb += zb.sum(0)
        saa += za.pow(2).sum(0); sbb += zb.pow(2).sum(0)
        sab += za.T @ zb
        n += len(za)
    cov = sab / n - torch.outer(sa, sb) / n**2
    std_a = (saa / n - (sa / n).pow(2)).clamp(min=0).sqrt()
    std_b = (sbb / n - (sb / n).pow(2)).clamp(min=0).sqrt()
    corr = cov / (std_a[:, None] * std_b[None, :]).clamp(min=1e-12)
    keep = std_a > 0
    return corr.max(1).values[keep].float().cpu()


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seeded(spec: SaeSpec, n: int, offset: int) -> list[SaeSpec]:
    """Sides of a comparison never share seeds: a shared SAE init and batch order would fake similarity."""
    return [replace(spec, seed=offset + i) for i in range(n)]


def sides(cell: Cell) -> tuple[list[SaeSpec], list[SaeSpec]]:
    return seeded(cell.a, cell.n_seeds, 0), seeded(cell.b, cell.n_seeds, 100)


def cell_path(cell: Cell) -> Path:
    return Ctx.out / "cells" / f"{cell.name}_{stable_hash([asdict(cell.a), asdict(cell.b), cell.n_seeds])}.json"


def run_cell(cell: Cell, p: Preset) -> dict:
    path = cell_path(cell)
    if path.exists():
        return json.loads(path.read_text())
    log(f"Cell {cell.name} [{cell.group}]: {cell.question}")
    a_specs, b_specs = sides(cell)
    for s in a_specs + b_specs:
        ensure_sae(s, p)
    row = compare_cell(cell, a_specs, b_specs, p)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2))
    log(f"  shift {row['delta']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  within {row['within']:.3f}  cross {row['cross']:.3f}")
    ModelSlot.clear()
    return row


def compare_cell(cell: Cell, a_specs: list[SaeSpec], b_specs: list[SaeSpec], p: Preset) -> dict:
    A, B = [load_saved(s) for s in a_specs], [load_saved(s) for s in b_specs]
    ua, ub = [unit_alive(s) for s in A], [unit_alive(s) for s in B]

    def direction(left, right):
        """Per-feature: match to other seeds of its own side, minus match to the other side."""
        within, cross, delta, by_seed, cos_cross = [], [], [], [], []
        for i, (w, _) in enumerate(left):
            same = torch.stack([best_cosine(w, left[j][0]) for j in range(len(left)) if j != i]).mean(0)
            other = torch.stack([best_cosine(w, r[0]) for r in right])
            within.append(same.mean().item())
            cross.append(other.mean().item())
            cos_cross.append(other.flatten())
            d = (same - other.mean(0)).numpy()
            delta.append(d)
            by_seed.append(float(d.mean()))
        return within, cross, delta, by_seed, torch.cat(cos_cross).numpy()

    w_a, c_ab, d_a, seeds_a, cos_ab = direction(ua, ub)
    w_b, c_ba, d_b, seeds_b, cos_ba = direction(ub, ua)
    deltas = np.concatenate(d_a + d_b)
    ci_low, ci_high = bootstrap_ci(deltas, p.bootstraps)
    seed_deltas = seeds_a + seeds_b
    all_cross = np.concatenate([cos_ab, cos_ba])

    row = {
        "cell": cell.name, "group": cell.group, "question": cell.question,
        "model_a": cell.a.model, "model_b": cell.b.model, "corpus_a": cell.a.corpus, "corpus_b": cell.b.corpus,
        "layer": cell.a.layer, "arch": cell.b.arch, "d_sae": cell.b.d_sae, "k": cell.b.k, "n_seeds": cell.n_seeds,
        "within": float(np.mean(w_a + w_b)), "cross": float(np.mean(c_ab + c_ba)),
        "delta": float(deltas.mean()), "ci_low": ci_low, "ci_high": ci_high,
        "seed_delta_min": min(seed_deltas), "seed_delta_max": max(seed_deltas),
        "delta_base_to_tuned": float(np.concatenate(d_a).mean()), "delta_tuned_to_base": float(np.concatenate(d_b).mean()),
        "frac_cos_ge_09": float((all_cross >= 0.9).mean()), "frac_cos_ge_07": float((all_cross >= 0.7).mean()),
        "frac_cos_lt_05": float((all_cross < 0.5).mean()),
    }
    for side, saved in (("a", A), ("b", B)):
        for m in ("fvu", "l0", "dead_frac", "ce_recovered"):
            row[f"{m}_{side}"] = float(np.mean([s["metrics"][m] for s in saved]))

    # Matching-method ablation (first seed pair only: it is the slow part).
    (wa, ia), (wb, ib), (wa1, _) = ua[0], ub[0], ua[1]
    with torch.no_grad():
        nn_idx = (wa.to(Ctx.device) @ wb.to(Ctx.device).T).argmax(1).cpu().numpy()
    row["greedy_many_to_one"] = 1 - len(np.unique(nn_idx)) / max(len(nn_idx), 1)
    row["hung_cross"] = row["hung_within"] = row["freq_shift_matched"] = float("nan")
    rates_a, rates_b = A[0]["metrics"]["rates"].numpy(), B[0]["metrics"]["rates"].numpy()
    log_ratio = lambda x, y: np.abs(np.log10((x + 1e-6) / (y + 1e-6)))
    if max(len(wa), len(wb)) <= p.hungarian_max:
        rows, cols, cos = hungarian(wa, wb)
        row["hung_cross"] = float(cos.mean())
        row["hung_within"] = float(hungarian(wa, wa1)[2].mean())
        row["freq_shift_matched"] = float(np.median(log_ratio(rates_a[ia.numpy()[rows]], rates_b[ib.numpy()[cols]])))
    row["freq_shift_indexwise"] = float(np.median(log_ratio(rates_a, rates_b))) if len(rates_a) == len(rates_b) else float("nan")

    # Functional comparison needs the same tokens on both sides.
    row["corr_cross"] = row["corr_within"] = row["act_cosine"] = float("nan")
    if cell.a.corpus == cell.b.corpus and cell.a.include_pad == cell.b.include_pad and cell.a.layer == cell.b.layer:
        _, layout = interp_tokens(cell.a.corpus, cell.a.include_pad, p)
        stop = min(layout.total, layout.test_start + p.corr_tokens)
        pa = ensure_acts(cell.a.model, cell.a.corpus, cell.a.layer, cell.a.include_pad, p)
        pb = ensure_acts(cell.b.model, cell.b.corpus, cell.b.layer, cell.b.include_pad, p)
        ModelSlot.clear()
        xa, xb = read_acts(pa, layout.test_start, stop), read_acts(pb, layout.test_start, stop)
        sae_a, sae_a1, sae_b = build_sae(A[0]), build_sae(A[1]), build_sae(B[0])
        row["corr_cross"] = float(activation_correlation(sae_a, sae_b, xa, xb).mean())
        row["corr_within"] = float(activation_correlation(sae_a, sae_a1, xa, xa).mean())
        row["act_cosine"] = float(F.cosine_similarity(xa.float(), xb.float(), dim=1).mean())
        del sae_a, sae_a1, sae_b, xa, xb
    del A, B, ua, ub
    free_memory()
    return row


# ── Which features are new, lost, or stable? ─────────────────────────────────
@torch.no_grad()
def firing_rates(sae: SAE, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    fires = torch.zeros(sae.W_dec.shape[0], device=Ctx.device)
    for start in range(0, len(x), batch):
        fires += (sae.encode(sae.prep(x[start:start + batch].to(Ctx.device).float())[0]) > 0).sum(0).float()
    return (fires / max(len(x), 1)).cpu().numpy()


def classify(repro: np.ndarray, other: np.ndarray, repro_min=0.7, new_max=0.5, stable_min=0.7) -> np.ndarray:
    """A feature is only new/stable if a second seed of the same model finds it too."""
    label = np.full(len(repro), "unreproducible", dtype=object)
    good = repro >= repro_min
    label[good] = "intermediate"
    label[good & (other < new_max)] = "new"
    label[good & (other >= stable_min)] = "stable"
    return label


@torch.no_grad()
def top_contexts(sae: SAE, feats: np.ndarray, x: torch.Tensor, ts: TokenSet, layout: ActLayout, tok, n: int = 4, radius: int = 12) -> dict:
    device = Ctx.device
    feats_t = torch.as_tensor(feats, device=device)
    best_vals = torch.full((len(feats), n), -1.0, device=device)
    best_idx = torch.zeros((len(feats), n), dtype=torch.long, device=device)
    for start in range(0, len(x), 4096):
        z = sae.encode(sae.prep(x[start:start + 4096].to(device).float())[0])[:, feats_t]
        vals, idx = z.topk(min(n, z.size(0)), dim=0)
        vals, idx = vals.T, idx.T + start
        merged_v = torch.cat([best_vals, vals], 1)
        merged_i = torch.cat([best_idx, idx], 1)
        top = merged_v.topk(n, dim=1).indices
        best_vals, best_idx = merged_v.gather(1, top), merged_i.gather(1, top)
    out = {}
    for row, feat in enumerate(feats):
        lines = []
        for v, rel in zip(best_vals[row].tolist(), best_idx[row].tolist()):
            if v <= 0:
                continue
            act = layout.test_start + rel
            doc = int(np.searchsorted(layout.act_offsets, act, side="right") - 1)
            pos = act - int(layout.act_offsets[doc]) + 1  # +1: the separator was excluded
            ids = ts.doc(doc)
            left = tok.decode(ids[max(0, pos - radius):pos].tolist())
            mid = tok.decode(ids[pos:pos + 1].tolist())
            right = tok.decode(ids[pos + 1:pos + 1 + radius // 2].tolist())
            lines.append(f"{v:.2f}: {repr(left)[1:-1]}<<{repr(mid)[1:-1]}>>{repr(right)[1:-1]}")
        out[int(feat)] = lines
    return out


def analyze_features(ref: Cell, p: Preset) -> dict:
    """Count new/lost features, calibrate the rule against seed noise, and test domain selectivity."""
    import pandas as pd
    from scipy.stats import mannwhitneyu

    path = Ctx.out / "feature_summary.json"
    a_specs, b_specs = sides(ref)
    A, B = [load_saved(s) for s in a_specs], [load_saved(s) for s in b_specs]
    ua, ub = [unit_alive(s) for s in A], [unit_alive(s) for s in B]
    summary = {}
    tables = {}

    def analyse(side, own, other_side, saved, spec):
        w0, idx0 = own[0]
        repro = torch.stack([best_cosine(w0, own[j][0]) for j in range(1, len(own))]).mean(0).numpy()
        other_all = torch.stack([best_cosine(w0, r[0]) for r in other_side]).max(0).values.numpy()
        label = classify(repro, other_all)
        # Same rule with one seed on each side, so real and noise rates are like for like.
        one_other = best_cosine(w0, other_side[0][0]).numpy()
        rate_real = float((classify(repro, one_other) == "new").mean())
        rate_noise = float("nan")
        if len(own) >= 3:
            first = best_cosine(w0, own[1][0]).numpy()
            pseudo = best_cosine(w0, own[2][0]).numpy()
            rate_noise = float((classify(first, pseudo) == "new").mean())
        sae = build_sae(saved[0])
        ts_py, lay_py = interp_tokens("python", False, p)
        _, lay_wk = interp_tokens("wiki", False, p)
        x_py = read_acts(ensure_acts(spec.model, "python", spec.layer, False, p), lay_py.test_start, lay_py.total)
        x_wk = read_acts(ensure_acts(spec.model, "wiki", spec.layer, False, p), lay_wk.test_start, lay_wk.total)
        ModelSlot.clear()
        r_py, r_wk = firing_rates(sae, x_py)[idx0.numpy()], firing_rates(sae, x_wk)[idx0.numpy()]
        selectivity = np.log10((r_py + 1e-6) / (r_wk + 1e-6))
        table = pd.DataFrame({
            "feature": idx0.numpy(), "class": label, "seed_reproducibility": repro, "best_match_other_model": other_all,
            "fire_rate_python": r_py, "fire_rate_wiki": r_wk, "python_selectivity_log10": selectivity,
        })
        new_sel, stable_sel = selectivity[label == "new"], selectivity[label == "stable"]
        info = {
            "n_alive": len(label), **{f"n_{c}": int((label == c).sum()) for c in ("new", "stable", "intermediate", "unreproducible")},
            "new_rate_one_seed": rate_real, "new_rate_seed_noise": rate_noise,
            "median_selectivity_new": float(np.median(new_sel)) if len(new_sel) else float("nan"),
            "median_selectivity_stable": float(np.median(stable_sel)) if len(stable_sel) else float("nan"),
            "selectivity_auc": float("nan"), "selectivity_p": float("nan"),
        }
        if len(new_sel) and len(stable_sel):
            u, pval = mannwhitneyu(new_sel, stable_sel, alternative="greater")
            info["selectivity_auc"], info["selectivity_p"] = float(u / (len(new_sel) * len(stable_sel))), float(pval)
        picks = table[table["class"] == "new"].sort_values("seed_reproducibility", ascending=False).head(p.n_examples)["feature"].to_numpy()
        tok = tokenizer_for(p)
        contexts = top_contexts(sae, picks, x_py, ts_py, lay_py, tok) if len(picks) else {}
        table.to_csv(Ctx.out / f"features_{side}.csv", index=False)
        del sae, x_py, x_wk
        free_memory()
        tables[side] = table
        return info, contexts

    summary["tuned"], ctx_new = analyse("tuned", ub, ua, B, ref.b)   # new = appeared with tuning
    summary["base"], ctx_lost = analyse("base", ua, ub, A, ref.a)    # new here = lost with tuning
    lines = ["# Example features", "", "Top-activating held-out tokens, marked `<<like this>>`. These are prompts for manual reading,",
             "not labels: inspect them before claiming a feature means anything.", ""]
    for title, ctx, side in (("Appeared after fine-tuning", ctx_new, "tuned"), ("Lost after fine-tuning", ctx_lost, "base")):
        lines += [f"## {title}", ""]
        table = tables[side].set_index("feature")
        for feat, rows in ctx.items():
            lines.append(f"### Feature {feat} (reproducibility {table.loc[feat, 'seed_reproducibility']:.2f}, "
                         f"best match {table.loc[feat, 'best_match_other_model']:.2f}, "
                         f"python selectivity {table.loc[feat, 'python_selectivity_log10']:+.2f})")
            lines += [f"- `{r}`" for r in rows] + [""]
    (Ctx.out / "example_features.md").write_text("\n".join(lines), encoding="utf-8")
    path.write_text(json.dumps(summary, indent=2))
    return summary


# ── Reporting ────────────────────────────────────────────────────────────────
INK, MUTED, GRID, SERIES_1, SERIES_2 = "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
MIN_EFFECT = 0.02


def load_rows(p: Preset):
    import pandas as pd

    rows = []
    for cell in build_study(p):
        path = cell_path(cell)
        if path.exists():
            rows.append(json.loads(path.read_text()))
    return pd.DataFrame(rows)


def verdict_h1(r) -> tuple[str, str]:
    consistent = r.seed_delta_min > 0
    excludes = r.ci_low > 0
    big = r.delta >= MIN_EFFECT
    ok = consistent and excludes and big
    return ("SUPPORTED" if ok else "NOT SUPPORTED",
            f"shift {r.delta:+.4f}, 95% CI [{r.ci_low:+.4f}, {r.ci_high:+.4f}], seed range [{r.seed_delta_min:+.4f}, "
            f"{r.seed_delta_max:+.4f}], minimum effect {MIN_EFFECT}")


def write_report(p: Preset, features: dict | None) -> None:
    import pandas as pd

    rows = load_rows(p)
    if rows.empty:
        log("No finished comparisons yet; nothing to report.")
        return
    rows.to_csv(Ctx.out / "ablation_results.csv", index=False)
    get = lambda name: rows[rows.cell == name].iloc[0] if (rows.cell == name).any() else None
    ref = get("reference")
    models = []
    for f in sorted((Ctx.out / "models").glob("*/final/metrics.json")):
        models.append(json.loads(f.read_text()))
    models = pd.DataFrame(models)
    if not models.empty:
        models.to_csv(Ctx.out / "finetune_results.csv", index=False)

    lines = [f"# Results ({p.name} preset)", ""]
    if p.name != "main":
        lines += ["> **Wiring check only.** This preset is far too small for any scientific conclusion.", ""]
    n_total = len(build_study(p))
    if len(rows) < n_total:
        lines += [f"> **Partial run:** {len(rows)} of {n_total} comparisons finished. Attach this output and run again to continue.", ""]

    lines += ["## Hypotheses", ""]
    h1 = "NOT RUN"
    if ref is not None:
        h1, detail = verdict_h1(ref)
        lines.append(f"- **H1, change beyond seed noise: {h1}.** {detail}.")
    control = get("control_above_layer")
    if control is not None:
        ok = control.ci_low <= 0.01 and abs(control.delta) < 0.01
        lines.append(f"- **Calibration, LoRA above the hooked layer: {'PASS' if ok else 'FAIL'}.** True change is zero here; "
                     f"measured shift {control.delta:+.4f}, activation cosine {control.act_cosine:.6f}."
                     + ("" if ok else " Treat every other result in this report as unreliable until this is explained."))
    dose_rows = [(0.0, 0.0)] + [(d, get(f"dose_{round(d * 100)}").delta) for d in (0.1, 0.3) if get(f"dose_{round(d * 100)}") is not None]
    if ref is not None and len(dose_rows) == 3:
        from scipy.stats import spearmanr

        dose_rows.append((1.0, float(ref.delta)))
        xs, ys = zip(*dose_rows)
        rho = float(spearmanr(xs, ys).statistic)
        lines.append(f"- **H2, change grows with training: {'SUPPORTED' if rho >= 0.8 and h1 == 'SUPPORTED' else 'NOT SUPPORTED'}.** "
                     f"Spearman {rho:.2f} on {len(xs)} doses; shifts: " + ", ".join(f"{x:.0%}={y:+.4f}" for x, y in dose_rows) + ".")
    placebo = get("control_placebo")
    if ref is not None and placebo is not None and features:
        t = features["tuned"]
        beats_placebo = ref.ci_low > placebo.ci_high
        beats_noise = t["new_rate_one_seed"] > t["new_rate_seed_noise"] if not math.isnan(t["new_rate_seed_noise"]) else False
        selective = t["selectivity_p"] < 0.05 if not math.isnan(t["selectivity_p"]) else False
        ok = beats_placebo and beats_noise and selective
        lines.append(f"- **H3, change is domain-specific: {'SUPPORTED' if ok else 'NOT SUPPORTED'}.** Python shift {ref.delta:+.4f} vs placebo "
                     f"{placebo.delta:+.4f} ({'CIs separate' if beats_placebo else 'CIs overlap'}); new-feature rate "
                     f"{t['new_rate_one_seed']:.2%} vs {t['new_rate_seed_noise']:.2%} from seed noise; new features vs stable "
                     f"selectivity AUC {t['selectivity_auc']:.2f}, one-sided p={t['selectivity_p']:.3g}.")
    lines.append("")

    if not models.empty:
        cols = ["key", "method", "corpus", "lora_r", "lr", "trainable_params", "updates", "base_ppl_python", "tuned_ppl_python",
                "base_ppl_wiki", "tuned_ppl_wiki", "relative_param_change"]
        lines += ["## Fine-tuning", "", to_markdown(models[cols]), ""]
    cols = ["cell", "group", "within", "cross", "delta", "ci_low", "ci_high", "fvu_a", "fvu_b", "l0_b", "dead_frac_b", "ce_recovered_b",
            "hung_cross", "hung_within", "corr_cross", "corr_within", "greedy_many_to_one", "freq_shift_matched", "freq_shift_indexwise"]
    lines += ["## Ablation table", "", "`delta` is the shift: per-feature match within a model's own seeds minus match to the other model.",
              "", to_markdown(rows[cols]), ""]
    if features:
        lines += ["## Features (reference comparison, first seed)", "", to_markdown(pd.DataFrame(features).T.reset_index(names="sae")),
                  "", "See `example_features.md` for the top-activating tokens.", ""]
    figures = make_figures(rows)
    lines += ["## Figures", ""] + [f"![{f}]({f})" for f in figures]
    (Ctx.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_resume_lines(p, ref, rows, models, features, h1)
    log(f"Report written to {Ctx.out / 'report.md'}")


def to_markdown(df) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if math.isnan(v) else f"{v:.4g}"
        return str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, *body])


def write_resume_lines(p, ref, rows, models, features, h1) -> None:
    path = Ctx.out / "resume_lines.md"
    if p.name != "main":
        path.write_text("Not generated: this preset is a wiring check, so its numbers must not be quoted.\n")
        return
    if ref is None:
        path.write_text("Not generated: the reference comparison has not finished.\n")
        return
    ft = models[models.key == "ft"].iloc[0] if not models.empty and (models.key == "ft").any() else None
    outcome = ("shifted features beyond seed noise" if h1 == "SUPPORTED" else "did not shift features measurably beyond seed noise")
    topics = {"design": "corpus", "control": "placebo and zero-change controls", "dose": "training dose", "layer": "layer",
              "lora": "LoRA hyperparameters", "sparsity": "sparsity", "architecture": "architecture", "width": "width",
              "pipeline": "pipeline choices"}
    covered = ", ".join(topics[g] for g in GROUP_ORDER if g in topics and (rows.group == g).any())
    placebo = rows[rows.cell == "control_placebo"]
    versus = f"; a placebo tuned on Wikipedia shifted {placebo.iloc[0].delta:+.3f}" if len(placebo) else ""
    lines = [
        "**Feature Shift in Sparse Autoencoders under LoRA Fine-Tuning** | [GitHub] | AIMS DTU, 2026",
        f"- Built a reproducible PyTorch pipeline to LoRA fine-tune Pythia-160M on Python code"
        + (f" (held-out perplexity {ft.base_ppl_python:.1f} to {ft.tuned_ppl_python:.1f})" if ft is not None else "")
        + f" and train BatchTopK sparse autoencoders on its layer {p.layer} residual stream.",
        f"- Measured that LoRA tuning {outcome} (shift {ref.delta:+.3f}, seed range [{ref.seed_delta_min:+.3f}, {ref.seed_delta_max:+.3f}]) "
        f"using same-token, multi-seed comparisons of decoder directions{versus}.",
        f"- Ran an ablation study over {len(rows)} configurations ({covered}); reference SAE reached FVU {ref.fvu_b:.3f} at L0 "
        f"{ref.l0_b:.0f} with {ref.ce_recovered_b:.0%} of CE loss recovered.",
        "",
        "Tools: PyTorch, Hugging Face Transformers, PEFT, SciPy",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.set_axisbelow(True)


def make_figures(rows) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    saved = []

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(Ctx.out / name)
        plt.close(fig)
        saved.append(name)

    # Every comparison's shift with its interval.
    data = rows.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, max(2.5, 0.27 * len(data) + 1)))
    y = np.arange(len(data))
    ax.hlines(y, data.ci_low, data.ci_high, color=SERIES_1, linewidth=2)
    ax.scatter(data.delta, y, s=28, color=SERIES_1, edgecolor="#fcfcfb", linewidth=1, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, data.cell)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_xlabel("Shift: own-seed match minus cross-model match (95% CI)")
    ax.set_title("Feature shift by comparison", loc="left")
    style_axes(ax)
    finish(fig, "fig_shift_by_comparison.png")

    # Dose-response.
    points = [(0.0, 0.0, 0.0, 0.0)]
    for name, dose in (("dose_10", 0.1), ("dose_30", 0.3), ("reference", 1.0)):
        r = rows[rows.cell == name]
        if len(r):
            r = r.iloc[0]
            points.append((dose, r.delta, r.delta - r.ci_low, r.ci_high - r.delta))
    if len(points) >= 3:
        xs, ys, lo, hi = map(np.array, zip(*points))
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES_1, linewidth=2, marker="o", markersize=6, capsize=0)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_xlabel("Fraction of LoRA updates completed")
        ax.set_ylabel("Shift")
        ax.set_title("Feature shift as training progresses", loc="left")
        style_axes(ax)
        finish(fig, "fig_dose_response.png")

    # Reconstruction vs sparsity.
    sub = rows[rows.group.isin(["reference", "architecture", "sparsity", "width"])]
    if len(sub) >= 2:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.scatter(sub.l0_b, sub.fvu_b, s=36, color=SERIES_1, edgecolor="#fcfcfb", linewidth=1)
        for i, r in enumerate(sub.sort_values("l0_b").itertuples()):
            above = i % 2 == 0  # neighbours on the log axis alternate sides so labels do not collide
            ax.annotate(r.cell, (r.l0_b, r.fvu_b), xytext=(0, 7 if above else -7), textcoords="offset points",
                        fontsize=7, color=MUTED, ha="center", va="bottom" if above else "top")
        ax.set_xscale("log")
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_xlabel("L0 (active features per token, log scale)")
        ax.set_ylabel("Fraction of variance unexplained")
        ax.set_title("SAE quality: lower left is better", loc="left")
        style_axes(ax)
        finish(fig, "fig_sparsity_vs_reconstruction.png")

    # New vs stable selectivity.
    table_path = Ctx.out / "features_tuned.csv"
    if table_path.exists():
        import pandas as pd

        t = pd.read_csv(table_path)
        new, stable = t[t["class"] == "new"], t[t["class"] == "stable"]
        if len(new) and len(stable):
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            bins = np.linspace(-4, 4, 41)
            ax.hist(stable.python_selectivity_log10, bins=bins, density=True, histtype="step", linewidth=2, color=SERIES_1, label=f"Stable (n={len(stable)})")
            ax.hist(new.python_selectivity_log10, bins=bins, density=True, histtype="step", linewidth=2, color=SERIES_2, label=f"New (n={len(new)})")
            ax.axvline(0, color=MUTED, linewidth=1)
            ax.grid(axis="y", color=GRID, linewidth=0.7)
            ax.legend(frameon=False, labelcolor=INK)
            ax.set_xlabel("log10(Python fire rate / Wikipedia fire rate)")
            ax.set_ylabel("Density")
            ax.set_title("Are new features Python-specific?", loc="left")
            style_axes(ax)
            finish(fig, "fig_domain_selectivity.png")
    return saved


# ── Orchestration ────────────────────────────────────────────────────────────
GROUP_ORDER = ["reference", "design", "control", "dose", "layer", "lora", "sparsity", "architecture", "width", "pipeline"]


def models_needed(cells: list[Cell]) -> list[str]:
    seen = []
    for c in cells:
        for spec in (c.a, c.b):
            name = spec.model.partition("@")[0]
            if name != "base" and name not in seen:
                seen.append(name)
    return seen


def write_manifest(p: Preset, argv: list[str]) -> None:
    import datasets
    import peft
    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    Ctx.out.mkdir(parents=True, exist_ok=True)
    (Ctx.out / "run_manifest.json").write_text(json.dumps({
        "preset": asdict(p), "argv": argv, "python": platform.python_version(), "torch": torch.__version__, "gpu": gpu,
        "transformers": transformers.__version__, "peft": peft.__version__, "datasets": datasets.__version__,
    }, indent=2, default=str))


def run(p: Preset, stage: str = "all", groups: list[str] | None = None, argv: list[str] | None = None,
        require_previous: bool = False) -> None:
    Ctx.out.mkdir(parents=True, exist_ok=True)
    Ctx.cache.mkdir(parents=True, exist_ok=True)
    write_manifest(p, argv or [])
    adopted = adopt_previous_outputs()
    if require_previous and not adopted:
        raise RuntimeError("--require-previous: no earlier SAEs found, so finished work would be redone. Stopping.")
    log(f"Preset {p.name} on {Ctx.device}; outputs {Ctx.out}; scratch {Ctx.cache}")
    study = build_study(p)
    wanted = groups or GROUP_ORDER
    cells = sorted((c for c in study if c.group in wanted), key=lambda c: GROUP_ORDER.index(c.group))
    ref = study[0]
    summary_path = Ctx.out / "feature_summary.json"
    features = json.loads(summary_path.read_text()) if summary_path.exists() else None
    finished = True

    def analyze_when_ready():
        if features is None and all(sae_path(s).exists() for s in sides(ref)[0] + sides(ref)[1]):
            return analyze_features(ref, p)
        return features

    try:
        if stage in ("all", "finetune"):
            for key in models_needed(cells):
                finetune(key, p)
        if stage in ("all", "study"):
            for cell in cells:
                for spec in (cell.a, cell.b):
                    name = spec.model.partition("@")[0]
                    if name != "base":
                        finetune(name, p)
                run_cell(cell, p)
                features = analyze_when_ready()  # early, so a session that runs out of time still reports H3
    except OutOfTime as e:
        finished = False
        log(f"Time budget reached before {e}. Saving what is done; attach this output and re-run to continue.")
    ModelSlot.clear()
    write_report(p, features)
    mem_snapshot("end")
    log("Finished." if finished else "Stopped early; results so far are saved.")


def default_paths() -> tuple[Path, Path]:
    if Path("/kaggle/working").exists():
        scratch = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
        return Path("/kaggle/working/outputs"), scratch / "mechinterp_cache"
    return Path("outputs"), Path(".cache_mechinterp")


def main(argv: list[str] | None = None) -> None:
    out, cache = default_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="main", choices=sorted(PRESETS))
    ap.add_argument("--stage", default="all", choices=["all", "finetune", "study", "report"])
    ap.add_argument("--groups", default="", help=f"comma-separated subset of: {','.join(GROUP_ORDER)}")
    ap.add_argument("--out", type=Path, default=out)
    ap.add_argument("--cache", type=Path, default=cache)
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop cleanly before Kaggle's 12h session limit")
    ap.add_argument("--require-previous", action="store_true", help="stop at once if no attached earlier run is found")
    args, _ = ap.parse_known_args(argv)  # tolerate notebook kernels' extra arguments
    ensure_deps()
    Ctx.out, Ctx.cache, Ctx.deadline = args.out, args.cache, time.time() + args.max_hours * 3600
    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    if groups and (unknown := set(groups) - set(GROUP_ORDER)):
        ap.error(f"unknown groups: {sorted(unknown)}")
    if Ctx.device == "cpu" and args.preset == "main":
        print("WARNING: no GPU found. On Kaggle set Accelerator = GPU T4 x1 or x2.", flush=True)
    run(PRESETS[args.preset], args.stage, groups, sys.argv if argv is None else argv, args.require_previous)


if __name__ == "__main__":
    main()
