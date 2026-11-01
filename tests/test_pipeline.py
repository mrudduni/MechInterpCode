"""Fast CPU tests: a tiny random GPT-NeoX and synthetic text stand in for Pythia and the datasets."""
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as rx  # noqa: E402

WORDS = "def return import class self value data list for while print open file name int str if else the of and in".split()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, GPTNeoXConfig, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny")
    vocab = {"<eos>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<eos>", unk_token="<unk>")
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=len(vocab), hidden_size=48, num_hidden_layers=4, num_attention_heads=4,
                        intermediate_size=96, max_position_embeddings=128, eos_token_id=0, bos_token_id=0)
    AutoModelForCausalLM.from_config(cfg).half().save_pretrained(path)  # float16, like Pythia on the Hub
    return path


@pytest.fixture()
def tiny_preset(tiny_model_dir):
    return rx.Preset(
        name="test", base_model=str(tiny_model_dir), layer=1, layer_low=0, layer_high=3, seq_len=24, ft_docs=32,
        ft_eval_docs=8, ft_epochs=1, ft_batch=4, ft_accum=1, ft_eval_every=2, lora_r=2, n_train=1500, n_test=400,
        small_corpus=500, acts_batch=4, d_sae=96, k=6, steps=30, sae_batch=64, widths=(48,), sparsities=(3,),
        l1s=(1.0,), legacy_batch=64, ce_docs=2, corr_tokens=200, bootstraps=20, n_examples=2,
        ft_reserve_hours=0, sae_reserve_hours=0,
    )


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    from datasets import Dataset

    monkeypatch.setattr(rx.Ctx, "out", tmp_path / "out")
    monkeypatch.setattr(rx.Ctx, "cache", tmp_path / "cache")
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    monkeypatch.setattr(rx.Ctx, "deadline", float("inf"))
    rx.Ctx.out.mkdir()
    rx.Ctx.cache.mkdir()

    def fake_table(corpus, _p):
        rng = random.Random(1 if corpus == "python" else 2)
        texts = [" ".join(rng.choices(WORDS, k=rng.randint(10, 30))) * 3 for _ in range(1500)]
        groups = np.array([f"{corpus}{i % 200}" for i in range(1500)], dtype=object)
        return Dataset.from_dict({"text": texts}), "text", groups

    monkeypatch.setattr(rx, "raw_table", fake_table)
    rx.ModelSlot.clear()
    yield tmp_path
    rx.ModelSlot.clear()


def planted_data(n=8000, d=48, m=96, active=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    truth = F.normalize(torch.randn(m, d, generator=g), dim=1)
    which = torch.rand(n, m, generator=g).topk(active, dim=1).indices
    vals = torch.rand(n, active, generator=g) + 0.5
    z = torch.zeros(n, m).scatter(1, which, vals)
    return z @ truth, truth


def test_sae_recovers_planted_dictionary(monkeypatch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    x, truth = planted_data()
    spec = rx.SaeSpec(d_sae=192, k=3, steps=1500, batch=256, lr=1e-3, n_train=len(x))
    sae = rx.fit_sae(spec, x[:7000], verbose_every=0)
    metrics = rx.eval_sae(sae, x[7000:])
    assert metrics["fvu"] < 0.25
    cos = (truth @ F.normalize(sae.W_dec.detach(), dim=1).T).max(1).values
    assert cos.mean() > 0.8


@pytest.mark.parametrize("arch", ["batchtopk", "topk"])
def test_l0_tracks_k(monkeypatch, arch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    x, _ = planted_data()
    sae = rx.fit_sae(rx.SaeSpec(arch=arch, d_sae=128, k=8, steps=300, batch=256, lr=1e-3), x[:7000], verbose_every=0)
    l0 = rx.eval_sae(sae, x[7000:])["l0"]
    assert (l0 <= 8.001) if arch == "topk" else (3 < l0 < 16)


def test_decoder_stays_unit_norm(monkeypatch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    x, _ = planted_data(n=2000)
    sae = rx.fit_sae(rx.SaeSpec(d_sae=64, k=4, steps=20, batch=128), x, verbose_every=0)
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5)


def test_token_norm_reconstruction_returns_to_raw_scale(monkeypatch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    x, _ = planted_data(n=3000)
    x = x * 40
    sae = rx.fit_sae(rx.SaeSpec(arch="relu_l1", l1=1e-4, norm="token", d_sae=128, steps=300, batch=256, lr=1e-3, aux=False), x, verbose_every=0)
    ratio = sae.reconstruct_raw(x[:64]).norm(dim=1).mean() / x[:64].norm(dim=1).mean()
    assert 0.5 < ratio < 1.5


def test_nonfinite_activations_abort(monkeypatch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    x, _ = planted_data(n=1000)
    x[:, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        rx.fit_sae(rx.SaeSpec(d_sae=32, k=2, steps=5, batch=32), x, verbose_every=0)


def test_matching_identical_and_permuted(monkeypatch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    w = F.normalize(torch.randn(50, 48), dim=1)
    perm = torch.randperm(50)
    assert rx.best_cosine(w, w[perm]).min() > 0.9999
    rows, cols, cos = rx.hungarian(w, w[perm])
    assert len(set(cols.tolist())) == 50 and cos.min() > 0.9999
    assert rx.best_cosine(w, F.normalize(torch.randn(50, 48), dim=1)).mean() < 0.7


def test_hungarian_is_one_to_one_where_greedy_is_not(monkeypatch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    v = F.normalize(torch.randn(1, 8), dim=1)
    wa = F.normalize(v.repeat(4, 1) + 0.01 * torch.randn(4, 8), dim=1)
    wb = F.normalize(torch.cat([v, torch.randn(3, 8)]), dim=1)
    assert len(set((wa @ wb.T).argmax(1).tolist())) == 1
    assert len(set(rx.hungarian(wa, wb)[1].tolist())) == 4


def test_activation_correlation_detects_same_features(monkeypatch):
    monkeypatch.setattr(rx.Ctx, "device", "cpu")
    x, _ = planted_data(n=2000)
    sae = rx.fit_sae(rx.SaeSpec(d_sae=64, k=4, steps=50, batch=128), x, verbose_every=0)
    assert rx.activation_correlation(sae, sae, x, x).mean() > 0.999


def test_classify_requires_reproducibility():
    repro = np.array([0.9, 0.9, 0.9, 0.3])
    other = np.array([0.2, 0.95, 0.6, 0.1])
    assert rx.classify(repro, other).tolist() == ["new", "stable", "intermediate", "unreproducible"]


def test_sides_never_share_seeds():
    cell = rx.build_study(rx.Preset())[0]
    a, b = rx.sides(cell)
    assert not {s.seed for s in a} & {s.seed for s in b}


def test_every_study_cell_has_unique_name_and_known_group():
    cells = rx.build_study(rx.Preset())
    assert len({c.name for c in cells}) == len(cells)
    assert {c.group for c in cells} <= set(rx.GROUP_ORDER)
    assert set(rx.models_needed(cells)) <= set(rx.model_registry(rx.Preset()))


def test_layer_tensor_handles_both_return_shapes():
    h = torch.zeros(2, 5, 8)
    assert rx.layer_tensor(h) is h
    assert rx.layer_tensor((h, None)) is h
    with pytest.raises(RuntimeError):
        rx.layer_tensor(h[0])  # the shape the old out[0] produced on tensor-returning versions


def test_base_is_cast_to_float32(tiny_preset, workspace):
    assert next(rx.load_base(tiny_preset).parameters()).dtype == torch.float32


def test_stop_at_layer_matches_full_forward(tiny_preset, workspace):
    model = rx.load_model("base", tiny_preset)
    ids = torch.randint(0, 20, (2, 12))
    caught = []
    handle = model.gpt_neox.layers[1].register_forward_hook(lambda _m, _i, o: caught.append(rx.layer_tensor(o).detach()))
    with torch.no_grad():
        model(ids)
    handle.remove()
    assert torch.allclose(rx.forward_to_layer(model, ids, None, 1), caught[0])


def test_right_padding_does_not_change_activations(tiny_preset, workspace):
    model = rx.load_model("base", tiny_preset)
    short = torch.randint(1, 20, (1, 7))
    padded = torch.cat([short, torch.zeros(1, 5, dtype=torch.long)], 1)
    mask = torch.cat([torch.ones(1, 7), torch.zeros(1, 5)], 1).long()
    a = rx.forward_to_layer(model, short, torch.ones_like(short), 1)
    b = rx.forward_to_layer(model, padded, mask, 1)
    assert torch.allclose(a, b[:, :7], atol=1e-4)


def test_activation_cache_excludes_padding_and_separator(tiny_preset, workspace):
    path = rx.ensure_acts("base", "python", 1, False, tiny_preset)
    ts, layout = rx.interp_tokens("python", False, tiny_preset)
    arr = np.load(path, mmap_mode="r")
    assert arr.dtype == np.float16
    assert arr.shape[0] == len(ts.ids) - len(ts) == layout.total
    assert np.isfinite(np.asarray(arr)).all()
    # The cache must agree with a one-document-at-a-time forward pass.
    model = rx.ModelSlot.get("base", tiny_preset)
    doc = torch.from_numpy(ts.doc(1).astype(np.int64))[None]
    single = rx.forward_to_layer(model, doc, torch.ones_like(doc), 1)[0, 1:].float().numpy()
    start, stop = layout.act_offsets[1], layout.act_offsets[2]
    assert np.allclose(arr[start:stop], single, atol=2e-2)


def test_splice_identity_leaves_loss_unchanged(tiny_preset, workspace):
    model = rx.load_model("base", tiny_preset)
    ts = rx.get_tokens("python", "eval", tiny_preset, max_docs=4)
    docs = np.arange(4)
    plain = rx.ce_with_replacement(model, ts, docs, 0, 1, False)
    same = rx.ce_with_replacement(model, ts, docs, 0, 1, False, lambda h: h)
    other = rx.ce_with_replacement(model, ts, docs, 0, 1, False, lambda h: h * 0)
    assert abs(plain - same) < 1e-5 and abs(plain - other) > 1e-4


def test_splits_are_group_disjoint():
    groups = [f"repo{i}" for i in range(2000)]
    splits = {g: rx.split_of(g) for g in groups}
    assert splits == {g: rx.split_of(g) for g in groups}
    assert set(splits.values()) == {"train", "eval", "interp"}


def test_lora_trains_only_adapter_and_saves_small_file(tiny_preset, workspace):
    rx.finetune("ft", tiny_preset)
    final = rx.model_dir("ft")
    metrics = json.loads((final / "metrics.json").read_text())
    assert metrics["trainable_params"] < 30_000 and metrics["updates"] > 0
    assert metrics["relative_param_change"] > 0
    assert (final / "adapter_model.safetensors").exists()
    assert not (final / "model.safetensors").exists()
    assert rx.model_dir("ft@10").exists() and rx.model_dir("ft@30").exists()
    base, tuned = rx.load_model("base", tiny_preset), rx.load_model("ft", tiny_preset)
    ids = torch.randint(0, 20, (1, 10))
    assert not torch.allclose(base(ids).logits, tuned(ids).logits)


def test_lora_above_hook_leaves_hooked_layer_identical(tiny_preset, workspace):
    rx.finetune("ft_above", tiny_preset)
    base, tuned = rx.load_model("base", tiny_preset), rx.load_model("ft_above", tiny_preset)
    ids = torch.randint(0, 20, (2, 10))
    assert torch.allclose(rx.forward_to_layer(base, ids, None, 1), rx.forward_to_layer(tuned, ids, None, 1), atol=1e-6)
    assert not torch.allclose(base(ids).logits, tuned(ids).logits)


def test_end_to_end_and_resume(tiny_preset, workspace, monkeypatch):
    groups = ["reference", "design", "control", "dose"]
    rx.run(tiny_preset, "all", groups)
    out = rx.Ctx.out
    for name in ("report.md", "ablation_results.csv", "feature_summary.json", "features_tuned.csv",
                 "example_features.md", "resume_lines.md", "run_manifest.json", "memory_log.csv"):
        assert (out / name).exists(), name
    assert "wiring check" in (out / "report.md").read_text(encoding="utf-8").lower()
    assert "must not be quoted" in (out / "resume_lines.md").read_text()
    # Zero true change must look like zero: same activations on both sides.
    rows = rx.load_rows(tiny_preset).set_index("cell")
    assert rows.loc["control_above_layer", "act_cosine"] > 0.9999

    def must_not_run(*_a, **_k):
        raise AssertionError("resume should not retrain")

    monkeypatch.setattr(rx, "fit_sae", must_not_run)
    monkeypatch.setattr(rx, "_finetune", must_not_run)
    rx.run(tiny_preset, "all", groups)


def test_time_budget_stops_cleanly(tiny_preset, workspace, monkeypatch):
    monkeypatch.setattr(rx.Ctx, "deadline", 0.0)
    rx.run(tiny_preset, "all", ["reference"])
    assert "Stopped early" in (rx.Ctx.out / "run.log").read_text(encoding="utf-8")
