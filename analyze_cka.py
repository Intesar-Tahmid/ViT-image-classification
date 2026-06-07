"""
Centered Kernel Alignment (CKA) analysis -- support script for Experiment 8
("Feature Space Analysis") of the WACV proposal.

This script runs three linked sub-studies, all on the SAME fixed probe set
(identical images, identical order -- required for row-aligned CKA) so that
every reported score is directly comparable:

  A. Pipeline transformation vs. accuracy
     For each of the five `situation` checkpoints, compute
     CKA(VE_pooled, Projector_pooled). Tabulated alongside each situation's
     val_acc (from the preliminary results / Experiment 5 logs), this is the
     headline plot explaining *why* `train_proj_ch` (projector-only training,
     89.9% in the preliminary run) beats both `train_proj_out_ch` (frozen
     projector, 87.5%) and `train_all` (joint training, 88.0% -- "interference
     between VE and projector optimization" per the proposal).

  B. Pretrained weights vs. architecture (-> Experiment 2: architecture vs.
     weights). Compares CKA(VE, Proj) across `llava` (pretrained), `llava_rand`
     (same architecture, random init), `gemma` (different pretrained source),
     and `none` (identity, a CKA==1.0 sanity check). If pretrained and
     random-init projectors reshape geometry by a *similar magnitude* yet only
     the pretrained one improves accuracy, that is direct evidence the
     *direction* of the learned transformation -- not just "any nonlinear
     remap" -- is what drives the gain.

  C. Cross-projector-source convergence. Pairwise CKA between the final pooled
     representations of different projector families (LLaVA / Gemma / ...) on
     the same frozen backbone -- do independently pretrained VLM projectors
     converge toward a similar "useful-for-classification" geometry?

Usage:
    python analyze_cka.py --config configs/cka.yaml [--study A B C]
"""
import argparse
import copy
import itertools
import os

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor

from dataset_imagenet import CSVImageDataset
from model_imagenet import ProjectorAblationModel


# ---------------------------------------------------------------------------
# Linear CKA  (Kornblith et al., 2019 -- "Similarity of Neural Network
# Representations Revisited"). Invariant to orthogonal transforms and
# isotropic scaling, and well-defined between feature spaces of different
# width -- exactly the situation here (768/1024-dim VE vs. up to 4096-dim
# projector outputs).
# ---------------------------------------------------------------------------
def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """
    x: (n_samples, d1), y: (n_samples, d2) -- row-aligned to the same inputs,
    in the same order. Returns a similarity score in [0, 1]: 1.0 == identical
    geometry up to rotation/reflection + isotropic scaling, 0.0 == unrelated.
    """
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross  = np.linalg.norm(y.T @ x, ord="fro") ** 2     # HSIC_linear(X, Y) for centered X, Y
    norm_x = np.linalg.norm(x.T @ x, ord="fro")
    norm_y = np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / (norm_x * norm_y))


# ---------------------------------------------------------------------------
# Fixed probe set: sample once from the held-out validation CSV (never used
# for training or checkpoint selection), write to disk, and re-read it for
# every variant. Each variant applies its own backbone-specific
# AutoImageProcessor, but the underlying images and their order are identical
# across all variants and all sub-studies -- the precondition for comparing
# CKA scores against each other.
# ---------------------------------------------------------------------------
def build_probe_csv(source_csv, num_samples, seed, out_path):
    if os.path.exists(out_path):
        print(f"Probe set already exists -> {out_path} (reusing for cross-run comparability)")
        return out_path
    df = pd.read_csv(source_csv)
    rng = np.random.default_rng(seed)
    n = min(num_samples, len(df))
    idx = np.sort(rng.choice(len(df), size=n, replace=False))
    df.iloc[idx].reset_index(drop=True).to_csv(out_path, index=False)
    print(f"Probe set: {n} images sampled from {source_csv} (seed={seed}) -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stage-wise feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_stage_features(model: ProjectorAblationModel, loader, device):
    """
    Manually replays VE -> Bridge -> Projector and mean-pools tokens at each
    stage, mirroring the pooling `ProjectorAblationModel.forward` applies just
    before the classification head.

    This is done by hand rather than by calling `model.forward` because two
    of the five situations (`train_ve_out_ch`, `train_ve_ch`) bypass the
    projector at inference time (`use_projector=False`) even though `self.proj`
    (and `self.bridge`) are still instantiated and loaded with weights --
    sub-study A specifically needs the projector's transformation measured in
    *every* situation, including those two, to compare against accuracy.
    """
    model.to(device).eval()
    ve_feats, bridge_feats, proj_feats = [], [], []

    for batch in tqdm(loader, desc="  extracting", leave=False):
        pixel_values = batch["pixel_values"].to(device)

        ve_tokens = model.ve(pixel_values=pixel_values).last_hidden_state
        ve_feats.append(ve_tokens.mean(dim=1).float().cpu())

        feat = ve_tokens
        if model.bridge is not None:
            feat = model.bridge(feat)
            bridge_feats.append(feat.mean(dim=1).float().cpu())

        feat = model.proj(feat)
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        proj_feats.append(feat.float().cpu())

    out = {
        "ve":   torch.cat(ve_feats).numpy(),
        "proj": torch.cat(proj_feats).numpy(),
    }
    if bridge_feats:
        out["bridge"] = torch.cat(bridge_feats).numpy()
    return out


def load_variant(entry, base_cfg, device):
    """
    Builds the config for one comparison cell by overriding only the field(s)
    that distinguish it from `base_cfg` -- e.g. {'situation': 'train_proj_ch'}
    for Study A, or {'projector_type': 'gemma'} for Study B/C. This mirrors
    how the project already drives its whole ablation grid from a single
    train.yaml by editing `experiment.situation` / `model.projector_type`
    between runs (see configs/train.yaml's inline option comments).

    entry: {'label', 'checkpoint' (optional), 'val_acc' (optional),
            'situation' (optional override), 'projector_type' (optional override),
            'bridge_type' (optional override)}
    """
    cfg = copy.deepcopy(base_cfg)
    if "projector_type" in entry:
        cfg["model"]["projector_type"] = entry["projector_type"]
    if "bridge_type" in entry:
        cfg["model"]["bridge_type"] = entry["bridge_type"]
    if "situation" in entry:
        cfg["experiment"]["situation"] = entry["situation"]

    # Seed immediately before construction so any random-init component
    # (`llava_rand`, or an auto-inserted bridge when a projector's input dim
    # does not match the backbone, e.g. gemma's 1152 vs ViT-Large's 1024) is
    # IDENTICAL on every run and independent of study/variant order -- a
    # precondition for the CKA scores being comparable run-to-run.
    torch.manual_seed(0)
    np.random.seed(0)

    if entry.get("checkpoint"):
        model = ProjectorAblationModel.load_from_checkpoint(
            entry["checkpoint"], config=cfg, map_location=device
        )
    else:
        # No checkpoint -> evaluate the freshly constructed pipeline (pretrained
        # VE/projector weights as loaded, classification head untrained). This is
        # the right reference state for sub-studies B/C, which probe what the
        # *pretrained* projector weights do to geometry, independent of any
        # downstream fine-tuning.
        model = ProjectorAblationModel(cfg)
    model.eval()
    return model, cfg


def get_probe_loader(cfg, probe_csv, batch_size, num_workers):
    processor = AutoImageProcessor.from_pretrained(cfg["model"]["id"])
    return DataLoader(
        CSVImageDataset(probe_csv, processor),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )


def run_variant(entry, base_cfg, probe_csv, batch_size, num_workers, device):
    label = entry["label"]
    print(f"\n=== Loading variant: {label} ===")
    model, cfg = load_variant(entry, base_cfg, device)
    loader = get_probe_loader(cfg, probe_csv, batch_size, num_workers)
    feats = extract_stage_features(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return feats


# ---------------------------------------------------------------------------
# Sub-study A: pipeline transformation vs. accuracy, across the five situations
# ---------------------------------------------------------------------------
def run_study_a(cka_cfg, base_cfg, probe_csv, out_dir, device, batch_size, num_workers):
    entries = cka_cfg.get("study_a_situations", [])
    if not entries:
        print("\n[Study A] No `study_a_situations` entries in config -- skipping.")
        return

    print("\n" + "=" * 70)
    print("STUDY A -- Pipeline transformation (CKA(VE, Projector)) vs. accuracy")
    print("=" * 70)

    rows = []
    for entry in entries:
        feats = run_variant(entry, base_cfg, probe_csv, batch_size, num_workers, device)
        row = {
            "situation": entry["label"],
            "val_acc": entry.get("val_acc"),   # fill in from training logs / checkpoint metrics
            "cka_ve_vs_proj": linear_cka(feats["ve"], feats["proj"]),
        }
        if "bridge" in feats:
            row["cka_ve_vs_bridge"]   = linear_cka(feats["ve"], feats["bridge"])
            row["cka_bridge_vs_proj"] = linear_cka(feats["bridge"], feats["proj"])
        rows.append(row)
        print(f"  {entry['label']:<20} CKA(VE, Proj) = {row['cka_ve_vs_proj']:.4f}"
              f"   (lower => projector reshapes geometry more)"
              f"   val_acc = {row['val_acc']}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "study_a_pipeline_vs_accuracy.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
    print(df.to_string(index=False))

    if df["val_acc"].notna().any():
        corr = df[["cka_ve_vs_proj", "val_acc"]].dropna().corr().iloc[0, 1]
        print(f"\nPearson correlation(CKA(VE,Proj), val_acc) = {corr:.3f}")
        print("  (Fill in `val_acc` for every situation in configs/cka.yaml to make this meaningful --")
        print("   it is the single number that most directly supports or refutes the 'projector")
        print("   reshapes geometry into something more classifiable' narrative for Experiment 8.)")
    else:
        print("\n`val_acc` not provided for any situation -- add it to configs/cka.yaml")
        print("(e.g. copy from the preliminary-results table or your training logs) to enable")
        print("the CKA-vs-accuracy correlation that makes this sub-study useful.")
    return df


# ---------------------------------------------------------------------------
# Sub-study B: pretrained weights vs. architecture (Experiment 2 support)
# ---------------------------------------------------------------------------
def run_study_b(cka_cfg, base_cfg, probe_csv, out_dir, device, batch_size, num_workers):
    entries = cka_cfg.get("study_b_sources", [])
    if not entries:
        print("\n[Study B] No `study_b_sources` entries in config -- skipping.")
        return

    print("\n" + "=" * 70)
    print("STUDY B -- Pretrained weights vs. architecture (projector source ablation)")
    print("=" * 70)

    rows = []
    proj_features = {}
    for entry in entries:
        feats = run_variant(entry, base_cfg, probe_csv, batch_size, num_workers, device)
        proj_features[entry["label"]] = feats["proj"]
        rows.append({
            "label": entry["label"],
            "projector_type": entry.get("projector_type", "?"),
            "cka_ve_vs_proj": linear_cka(feats["ve"], feats["proj"]),
        })
        print(f"  {entry['label']:<20} CKA(VE, Proj) = {rows[-1]['cka_ve_vs_proj']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "study_b_source_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved -> {csv_path}")
    print(df.to_string(index=False))

    # Pretrained vs. random-init of the *same* architecture: do they reshape
    # geometry by a similar magnitude (similar CKA(VE,Proj)) but point in
    # different directions (low CKA between the two projector outputs)?
    labels = list(proj_features.keys())
    if "llava_pretrained" in labels and "llava_random" in labels:
        cross = linear_cka(proj_features["llava_pretrained"], proj_features["llava_random"])
        print(f"\nCKA(LLaVA-pretrained-out, LLaVA-random-out) = {cross:.4f}")
        print("  Low value => pretrained and random-init projectors of the *same* architecture")
        print("  produce geometrically distinct output spaces -- i.e. the pretrained weights'")
        print("  *direction* of transformation (not just its magnitude) is what should explain")
        print("  any accuracy gap between them (pair this with their val_acc numbers).")
    return df


# ---------------------------------------------------------------------------
# Sub-study C: cross-projector-source convergence
# ---------------------------------------------------------------------------
def run_study_c(cka_cfg, base_cfg, probe_csv, out_dir, device, batch_size, num_workers):
    entries = cka_cfg.get("study_c_variants", [])
    if len(entries) < 2:
        print("\n[Study C] Fewer than 2 `study_c_variants` entries -- skipping (need >= 2 to compare).")
        return

    print("\n" + "=" * 70)
    print("STUDY C -- Cross-projector-source convergence (final representations)")
    print("=" * 70)

    final_features = {}
    for entry in entries:
        feats = run_variant(entry, base_cfg, probe_csv, batch_size, num_workers, device)
        final_features[entry["label"]] = feats["proj"]

    labels = list(final_features.keys())
    cross = np.zeros((len(labels), len(labels)))
    for i, j in itertools.combinations_with_replacement(range(len(labels)), 2):
        score = linear_cka(final_features[labels[i]], final_features[labels[j]])
        cross[i, j] = cross[j, i] = score

    cross_df = pd.DataFrame(cross, index=labels, columns=labels)
    csv_path = os.path.join(out_dir, "study_c_cross_source_matrix.csv")
    cross_df.to_csv(csv_path)
    print(f"\nSaved -> {csv_path}")
    print(cross_df.round(3).to_string())

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 2))
        im = ax.imshow(cross, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cross[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_title("Linear CKA -- final pooled representations (pre-classifier-head)")
        fig.colorbar(im, ax=ax, label="CKA similarity")
        fig.tight_layout()
        heatmap_path = os.path.join(out_dir, "study_c_cross_source_heatmap.png")
        fig.savefig(heatmap_path, dpi=150)
        print(f"Saved heatmap -> {heatmap_path}")
    except ImportError:
        print("matplotlib not installed -- skipping heatmap (CSV results are still saved).")
    return cross_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cka.yaml")
    parser.add_argument("--study", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"],
                        help="Which sub-studies to run (default: all).")
    args = parser.parse_args()

    with open(args.config) as f:
        cka_cfg = yaml.safe_load(f)

    with open(cka_cfg["base_config"]) as f:
        base_cfg = yaml.safe_load(f)

    probe_cfg = cka_cfg["probe"]
    out_dir = probe_cfg.get("cache_dir", "output/cka")
    os.makedirs(out_dir, exist_ok=True)

    probe_csv = build_probe_csv(
        probe_cfg["source_csv"],
        probe_cfg.get("num_samples", 500),
        probe_cfg.get("seed", 42),
        os.path.join(out_dir, "probe_set.csv"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = probe_cfg.get("batch_size", 32)
    num_workers = probe_cfg.get("num_workers", 4)

    if "A" in args.study:
        run_study_a(cka_cfg, base_cfg, probe_csv, out_dir, device, batch_size, num_workers)
    if "B" in args.study:
        run_study_b(cka_cfg, base_cfg, probe_csv, out_dir, device, batch_size, num_workers)
    if "C" in args.study:
        run_study_c(cka_cfg, base_cfg, probe_csv, out_dir, device, batch_size, num_workers)

    print(f"\nAll requested sub-studies complete. Results in: {out_dir}/")


if __name__ == "__main__":
    main()
