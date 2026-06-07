# Software Documentation: ViT / VLM Projector Ablation Framework

This document is a comprehensive, implementation-level reference for every source file in this
repository. It explains *what* each component does, *how* it works internally, and *how the
pieces connect* during training and evaluation. For the research motivation and theoretical
background, see [`wacv_research_proposal.md`](wacv_research_proposal.md) and
[`architecture.md`](architecture.md); for setup/usage instructions, see [`README.md`](README.md).

---

## 1. High-Level Overview

The codebase is a **PyTorch Lightning** framework for studying how Vision Transformer (ViT)
backbones interact with **pretrained Vision-Language Model (VLM) projector layers** when
repurposed for pure image classification. A single shared model
(`ProjectorAblationModel` in [`model_imagenet.py`](model_imagenet.py)) can be configured, via
YAML, to:

- Use any HuggingFace ViT-style vision encoder (CLIP vision tower, plain ViT, DINOv2, etc.).
- Optionally graft on a **real, pretrained projector** extracted from LLaVA 1.5, Gemma 3, or
  Gemma 4 (the layer that normally maps vision features into an LLM's embedding space).
- Optionally insert a **Dimension Bridge** module when the vision encoder's output width does not
  match the projector's expected input width.
- Freeze/unfreeze different combinations of {Vision Encoder, Projector+Bridge, Classification
  Head} according to one of five experiment "situations".
- Train and evaluate against either the **Tiny ImageNet** (200-class) dataset or the
  **Flowers102** (102-class) dataset.

The experiment asks: *does inheriting a pretrained cross-modal projector help, hurt, or do
nothing for downstream image classification — and does that answer change depending on which
parts of the pipeline are frozen vs. fine-tuned?*

### Pipeline at a glance

```
pixel_values
   │
   ▼
┌──────────────────────┐
│ Vision Encoder (VE)  │  e.g. CLIP/ViT/DINOv2 vision tower (HuggingFace AutoModel)
└──────────┬───────────┘
           │ last_hidden_state  (B, N, D_in)
           ▼
┌──────────────────────┐
│ Dimension Bridge     │  optional — only when D_in != D_proj (or forced via config)
│ (linear/low_rank/    │
│  mlp/patch_pool/     │
│  cross_attention)    │
└──────────┬───────────┘
           │ (B, N', D_proj)
           ▼
┌──────────────────────┐
│ VLM Projector (proj) │  pretrained weights lifted from LLaVA-1.5 / Gemma-3 / Gemma-4,
│                      │  or nn.Identity() if projector_type == "none"
└──────────┬───────────┘
           │ (B, N', D_out) → mean-pooled over tokens → (B, D_out)
           ▼
┌──────────────────────┐
│ Classification Head  │  single nn.Linear(D_out, num_classes)
│ (ch)                 │
└──────────┬───────────┘
           ▼
        logits
```

Which of these stages receives gradients is controlled entirely by the `experiment.situation`
config key (see §4.5).

---

## 2. Repository Layout

```
.
├── configs/
│   ├── train.yaml          # Tiny ImageNet training configuration
│   ├── train_flowers.yaml  # Flowers102 training configuration
│   └── test.yaml           # Checkpoint evaluation configuration
├── dataset_imagenet.py     # LightningDataModule + Dataset classes for Tiny ImageNet
├── dataset_flowers102.py   # LightningDataModule + Dataset class for Flowers102
├── model_imagenet.py       # Bridge modules + ProjectorAblationModel (LightningModule)
├── train_imagenet.py       # Training entry point (Tiny ImageNet)
├── train_flowers102.py     # Training entry point (Flowers102)
├── test_imagenet.py        # Checkpoint evaluation / inference entry point
├── Dockerfile              # CUDA + PyTorch + dependency image definition
├── requirements.txt        # Python dependency pins
├── README.md               # Usage / setup guide
├── architecture.md         # Research-proposal appendix: backbone/projector/bridge specs
└── wacv_research_proposal.md  # Full research proposal document
```

Two ancillary artifacts referenced by the code but not present as source files in this snapshot:
- `process_imagenet.py` — a data-prep script (mentioned in the README) that downloads the
  Tiny ImageNet class mapping, splits the validation set into Val/Test, and writes
  `data/validation.csv`, `data/test.csv`, and `class_mapping.json` to disk.
- `class_mapping.json` — a WordNet-ID → human-readable-label lookup consumed by
  [`test_imagenet.py`](test_imagenet.py).

---

## 3. Datasets and DataModules

Both dataset modules follow the same Lightning convention: a thin `torch.utils.data.Dataset`
wrapper that applies a HuggingFace `AutoImageProcessor` to raw PIL images, plus a
`LightningDataModule` that wires up `train/val/test` splits and `DataLoader`s.

### 3.1 `dataset_imagenet.py` — Tiny ImageNet

**`HFTinyImageNetDataset`** ([dataset_imagenet.py:12-30](dataset_imagenet.py#L12-L30))
Wraps a HuggingFace `datasets.Dataset` object (used for the *training* split, streamed/loaded
directly from the Hub as `zh-plus/tiny-imagenet`). Each `__getitem__` call:
1. Converts the image to RGB.
2. Runs it through the model's `AutoImageProcessor` (resizing/normalizing to the format the
   chosen ViT backbone expects).
3. Returns a dict with `pixel_values` (squeezed to drop the batch dim the processor adds) and
   `label` as a `torch.long` tensor.

**`CSVImageDataset`** ([dataset_imagenet.py:35-55](dataset_imagenet.py#L35-L55))
Used for the *validation* and *test* splits, which are pre-materialized to local disk (by the
external `process_imagenet.py` prep script) as a CSV of `image_path,label` rows plus physical
JPEG files. Loads each image from disk with `PIL.Image.open`, applies the same processor, and —
critically for the test script — also returns the original `image_path` string so predictions
can be traced back to specific files.

**`TinyImageNetDataModule`** ([dataset_imagenet.py:60-106](dataset_imagenet.py#L60-L106))
The Lightning orchestration layer:
- `__init__`: instantiates `AutoImageProcessor.from_pretrained(model_id)` once, so the
  preprocessing always matches whatever vision backbone the experiment config selects.
- `setup(stage)`:
  - `fit`/`None` → loads the **100k-image train split directly from the HuggingFace Hub**
    (`load_dataset("zh-plus/tiny-imagenet", split="train")`) and wraps it in
    `HFTinyImageNetDataset`; loads the **9k validation split from a local CSV**
    (`config['data']['val_csv']`) via `CSVImageDataset`.
  - `test` → loads the **1k test split** from `config['test']['test_csv']`.
- `train_dataloader` / `val_dataloader` / `test_dataloader`: build `DataLoader`s. Batch sizes are
  resolved with a fallback chain — e.g.
  `config['data'].get('train_batch_size', config['data'].get('batch_size', 256))` — so the same
  DataModule works whether it's driven by `train.yaml` (which specifies
  `train_batch_size`/`val_batch_size`) or `test.yaml` (which only specifies a flat `batch_size`).
  Training shuffles; validation/testing do not.

> **Why a HF stream for train but local CSVs for val/test?** The 100k training images are large
> and only need to be iterated once per epoch in random order, so streaming from the Hub avoids
> duplicating that storage locally. The val/test splits are small, fixed, and need to be
> *reproducible* (same images, same order, same paths) across runs — hence they are
> pre-materialized to disk with stable CSV manifests by `process_imagenet.py`.

### 3.2 `dataset_flowers102.py` — Flowers102

**`HFProcessorFlowersDataset`** ([dataset_flowers102.py:10-27](dataset_flowers102.py#L10-L27))
A minimal adapter around `torchvision.datasets.Flowers102`, which already returns
`(PIL.Image, label)` tuples. The wrapper just runs the image through the same
`AutoImageProcessor` pattern used for Tiny ImageNet, producing the same
`{"pixel_values", "label"}` dict shape — this is what lets both datasets feed the *same*
`ProjectorAblationModel` unmodified.

**`Flowers102DataModule`** ([dataset_flowers102.py:32-83](dataset_flowers102.py#L32-L83))
- `prepare_data()`: downloads all three official Flowers102 splits (`train`, `val`, `test`) into
  `config['data']['data_dir']` (default `./data`) if not already present. This Lightning hook
  runs once, on a single process, before `setup`.
- `setup(stage)`: instantiates the raw `torchvision` datasets (with `download=False`, since
  `prepare_data` already handled it) and wraps each in `HFProcessorFlowersDataset`.
- The three `*_dataloader` methods mirror `TinyImageNetDataModule`'s batch-size fallback logic,
  with an additional default of `4` workers if `num_workers` is omitted from the config.

This dataset exists to test whether conclusions drawn on Tiny ImageNet (200 generic object
classes) generalize to a visually different, fine-grained domain (102 flower species, much
smaller dataset).

---

## 4. The Model — `model_imagenet.py`

This file contains the entire modeling logic: five interchangeable **Dimension Bridge** modules
and the single **`ProjectorAblationModel`** LightningModule that composes
`{Vision Encoder, Bridge, Projector, Classification Head}` according to the active config.

### 4.1 Dimension Bridges

A "Dimension Bridge" is a small learned module inserted between the vision encoder and the VLM
projector whenever their channel dimensions disagree (`D_in != D_proj`) — or whenever the config
explicitly forces one. Five are registered in `_BRIDGE_REGISTRY`
([model_imagenet.py:95-101](model_imagenet.py#L95-L101)):

| Key | Class | Formulation | Notes |
|---|---|---|---|
| `linear` | `LinearBridge` | $Y = XW$ | Single `nn.Linear`, no non-linearity — the "control baseline" bridge ([model_imagenet.py:13-20](model_imagenet.py#L13-L20)). |
| `low_rank` | `LowRankBridge` | $Y = \text{GELU}(XW_{\text{down}})W_{\text{up}}$, rank $r=64$ | LoRA-style bottleneck: down-project to rank 64, GELU, up-project ([model_imagenet.py:23-31](model_imagenet.py#L23-L31)). |
| `mlp` | `MLPBridge` | `Linear → GELU → LayerNorm → Linear`, hidden = midpoint of in/out dims | A heavier 2-layer MLP bridge with normalization ([model_imagenet.py:78-91](model_imagenet.py#L78-L91)). |
| `patch_pool` | `PatchPoolBridge` | reshape `(B,N,C) → (B,N/4,4C)` then `Linear(4C → out)` | Spatial 2×2 token-merge inspired by Qwen2-VL/Gemma-style pixel-shuffle; **falls back to a plain truncated-linear projection when `N` isn't divisible by 4** (e.g. a `[CLS]` token is present) ([model_imagenet.py:34-56](model_imagenet.py#L34-L56)). |
| `cross_attention` | `CrossAttentionBridge` | $Y = \text{MultiHeadAttention}(Q, K, V)$ with $Q$ a fixed bank of learnable queries | Perceiver-style resampler: 256 learnable query vectors (in `out_dim` space) cross-attend to key/value projections of the visual tokens, simultaneously changing both sequence length *and* channel width ([model_imagenet.py:59-75](model_imagenet.py#L59-L75)). |

Key implementation details worth calling out:

- **`PatchPoolBridge` fallback** ([model_imagenet.py:45-56](model_imagenet.py#L45-L56)):
  ```python
  def forward(self, x):
      B, N, C = x.shape
      if N % 4 != 0:
          return nn.functional.linear(x, self.proj.weight[:, :C], self.proj.bias)
      x = x.view(B, N // 4, C * 4)
      return self.proj(x)
  ```
  The bridge's `nn.Linear` is always sized for `in_dim * 4` inputs. When the token count can't be
  evenly grouped into 2×2 blocks (true for ViT/CLIP encoders that prepend a `[CLS]` token to a
  14×14 or 16×16 patch grid, giving e.g. 197 tokens), the module slices the weight matrix down to
  its first `C` input columns and applies it as an ordinary linear layer — degrading gracefully to
  a non-spatial projection rather than crashing on a shape mismatch.

- **`CrossAttentionBridge`** ([model_imagenet.py:59-75](model_imagenet.py#L59-L75)) decouples
  *both* the sequence length (fixed at `num_queries=256`, regardless of the input token count) and
  the channel width from the input. `kv_proj` first maps the `in_dim`-wide visual tokens into
  `out_dim` space; the `MultiheadAttention` (8 heads, `batch_first=True`) then lets the learnable
  queries attend over them, producing a fixed-size `(B, 256, out_dim)` output.

### 4.2 `ProjectorAblationModel.__init__` — Construction Pipeline

The constructor ([model_imagenet.py:104-240](model_imagenet.py#L104-L240)) builds the full model
in five ordered stages. It is intentionally **config-driven**: the same class produces five
structurally different models depending on `config['experiment']['situation']`,
`config['model']['projector_type']`, and `config['model']['bridge_type']`.

**Stage 1 — Load the Vision Encoder (VE)**
([model_imagenet.py:115-126](model_imagenet.py#L115-L126))
```python
base_model = AutoModel.from_pretrained(model_id)
if hasattr(base_model, "vision_model"):
    self.ve = base_model.vision_model   # multimodal checkpoints (e.g. CLIP) → keep only the tower
else:
    self.ve = base_model                # plain vision checkpoints (ViT, DINOv2) used directly
```
It then derives `ve_dim`, the encoder's output channel width, by probing
`self.ve.config.hidden_size` and falling back to `self.ve.config.vision_config` (handling the
case where `hidden_size` itself turns out to be a sub-config object rather than an int).

**Stage 2 — Extract a pretrained VLM projector**
([model_imagenet.py:128-205](model_imagenet.py#L128-L205))
Depending on `projector_type`:
- `"llava"`: loads the full `llava-hf/llava-1.5-7b-hf` model in fp16 on CPU, pulls out
  `vlm.multi_modal_projector` (the 2-layer MLP+GELU that LLaVA uses to map CLIP vision features
  into Vicuna/LLaMA-2 embedding space), casts it back to fp32 (`.float()` — required so it plays
  nicely with PyTorch AMP/`GradScaler`), reads off `proj_in_dim`/`llm_hidden_dim` from its
  `linear_1`/`linear_2` layer shapes, then **deletes the multi-billion-parameter VLM and forces
  garbage collection** (`del vlm; gc.collect()`) so only the small projector survives in memory.
- `"gemma"` / `"gemma4"`: loads `google/gemma-3-4b-it` or `google/gemma-4-E4B-it` respectively via
  `AutoModelForImageTextToText`, then **searches** the model's top-level (and `.model`) children
  for any submodule whose name contains `"multi_modal_projector"`, `"vision_projector"`, or
  `"projector"` — because, unlike LLaVA, these checkpoints don't expose a single, stably-named
  projector attribute. Raises `ValueError` if nothing matches. Reads `proj_in_dim` from
  `vlm.config.vision_config.hidden_size` and `llm_hidden_dim` from `text_config.hidden_size`
  (falling back to the top-level `hidden_size`). Same `.float()` cast and `del`/`gc.collect()`
  cleanup.
- `"none"`: `self.proj = nn.Identity()`; both `proj_in_dim` and `llm_hidden_dim` are simply set to
  `ve_dim` (i.e. the projector stage is a complete no-op, matching dimensions trivially).
- Anything else raises `ValueError("Invalid projector_type. Choose 'llava', 'gemma', 'gemma4', or 'none'.")`.

This is the crux of the experimental design: rather than *training a projector from scratch*, the
model **transplants real, pretrained cross-modal projection weights** from production VLMs and
asks how they behave when repurposed for plain classification.

**Stage 3 — Insert a Dimension Bridge if needed**
([model_imagenet.py:207-229](model_imagenet.py#L207-L229))
```python
self.bridge = None
if projector_type != "none":
    dims_match = (ve_dim == proj_in_dim)
    need_bridge = (not dims_match) or (bridge_type not in ('auto', 'none'))
    if need_bridge:
        effective_bridge = bridge_type if bridge_type not in ('auto', 'none') else 'linear'
        if bridge_type == 'none' and not dims_match:
            raise ValueError(...)
        if effective_bridge not in _BRIDGE_REGISTRY:
            raise ValueError(...)
        self.bridge = _BRIDGE_REGISTRY[effective_bridge](ve_dim, proj_in_dim)
```
The decision logic:
- No bridge is ever created when `projector_type == "none"` (there's nothing to bridge into).
- A bridge is created when dimensions genuinely mismatch, **or** when the user explicitly forces
  a non-`auto`/`none` `bridge_type` (allowing ablation of bridge architecture even when one isn't
  strictly required).
- `bridge_type: "auto"` resolves to `linear` when a bridge is actually needed — the
  lowest-overhead default.
- `bridge_type: "none"` combined with mismatched dimensions is a **hard configuration error**
  (the model cannot be constructed), forcing the user to make an explicit choice rather than
  silently producing a shape-mismatched pipeline.
- An unrecognized `bridge_type` string raises `ValueError` listing the valid registry keys.

**Stage 4 — Route Logic & Output Dimension**
([model_imagenet.py:231-234](model_imagenet.py#L231-L234))
```python
no_proj_situations = ["train_ve_out_ch", "train_ve_ch"]
self.use_projector = self.situation not in no_proj_situations
out_dim = llm_hidden_dim if self.use_projector and projector_type != "none" else ve_dim
```
Two of the five situations are defined to **bypass the projector entirely** regardless of what
`projector_type` is configured to (they study the raw encoder, with or without VE fine-tuning).
For the remaining situations, the classification head's input width is the projector's output
width (`llm_hidden_dim`, e.g. 4096 for LLaVA / Vicuna space) when a real projector is active, or
the encoder's native width otherwise.

**Stage 5 — Classification Head**
([model_imagenet.py:236-238](model_imagenet.py#L236-L238))
A single `nn.Linear(out_dim, num_classes)` plus `nn.CrossEntropyLoss()`. This head is the *only*
component that is unconditionally trained in every situation (see §4.3) — it is the common
"probe" used to read out classification performance from whatever upstream representation the
current situation produces.

Finally, `_apply_situation_rules()` is called to set up the freeze/unfreeze pattern and print a
parameter-count summary.

### 4.3 `_apply_situation_rules` — Freezing Strategy

([model_imagenet.py:242-278](model_imagenet.py#L242-L278))

```python
for param in self.parameters():
    param.requires_grad = False
for param in self.ch.parameters():
    param.requires_grad = True
```
The method starts by freezing **everything**, then unconditionally re-enables gradients for the
classification head — guaranteeing the head is always trainable, in every situation. It then
selectively re-enables additional components based on `self.situation`:

| Situation | VE | Bridge + Projector | CH | Net effect |
|---|:---:|:---:|:---:|---|
| `train_ve_out_ch` | ❄️ frozen | *(bypassed — `use_projector=False`)* | 🔥 trained | Probe of the **raw, frozen** encoder's representation power |
| `train_proj_out_ch` | ❄️ frozen | ❄️ frozen | 🔥 trained | Probe of whether a **frozen pretrained projector** preserves/degrades VE features for classification |
| `train_proj_ch` | ❄️ frozen | 🔥 trained | 🔥 trained | Learn a dedicated translation layer on top of a frozen VE — "adapt the projector to the new task" |
| `train_ve_ch` | 🔥 trained | *(bypassed — `use_projector=False`)* | 🔥 trained | Fine-tune the encoder directly (no projector in the loop) |
| `train_all` | 🔥 trained | 🔥 trained | 🔥 trained | Full end-to-end fine-tune of the entire transplanted pipeline |

Any other string raises `ValueError(f"Unknown situation: {self.situation}")`. After applying the
rules, it prints a parameter accounting line:
```python
trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
total = sum(p.numel() for p in self.parameters())
print(f"Situation: {self.situation} | Trainable Params: {trainable:,} / {total:,}")
```
which is invaluable for sanity-checking that the intended subset of the network is actually
unfrozen before a (potentially expensive) training run starts.

> **Note on the README's "LoRA" framing.** The README describes situations 4–5 as applying
> **LoRA (Low-Rank Adaptation)** adapters to the VE/projector attention matrices. The
> implementation in `_apply_situation_rules` as written performs **full-parameter** unfreezing of
> those modules (`param.requires_grad = True` on every parameter), not PEFT/LoRA adapter
> injection — `peft` is listed in `requirements.txt` but is not imported or used in
> `model_imagenet.py`. If the project intends to use LoRA for these situations, that wiring would
> need to be added (e.g. via `peft.get_peft_model`); as it stands, situations 4 and 5 fully
> fine-tune the relevant submodules.

### 4.4 `forward` — The Computation Graph

([model_imagenet.py:280-297](model_imagenet.py#L280-L297))

```python
def forward(self, pixel_values):
    ve_outputs = self.ve(pixel_values=pixel_values)
    if self.use_projector:
        features = ve_outputs.last_hidden_state          # (B, N, D_in)
        if self.bridge is not None:
            features = self.bridge(features)             # (B, N', D_proj)
        features = self.proj(features)                   # (B, N', D_out)
        if features.dim() == 3:
            features = features.mean(dim=1)              # mean-pool tokens → (B, D_out)
    else:
        if hasattr(ve_outputs, 'pooler_output') and ve_outputs.pooler_output is not None:
            features = ve_outputs.pooler_output
        else:
            features = ve_outputs.last_hidden_state[:, 0, :]   # CLS token fallback
    return self.ch(features)
```

Two distinct feature-extraction paths:
1. **Projector path** (`use_projector=True`): the full token sequence flows through the optional
   bridge and then the projector. Because VLM projectors are designed to operate token-wise (they
   don't reduce sequence length, except `CrossAttentionBridge` which does), the result is mean-
   pooled across the token dimension to yield one feature vector per image before the
   classification head. The `if features.dim() == 3` guard makes this robust to projectors that
   *do* already reduce to a single vector.
2. **Bypass path** (`use_projector=False`): uses the encoder's native pooled representation —
   preferring `pooler_output` when the HF model provides one (e.g. CLIP's vision tower), and
   falling back to the `[CLS]` token (`last_hidden_state[:, 0, :]`) otherwise (e.g. plain ViT/
   DINOv2 encoders that don't expose a pooler).

### 4.5 Training / Validation Steps & Optimizer

([model_imagenet.py:299-316](model_imagenet.py#L299-L316))

`training_step` and `validation_step` are near-identical: run `forward`, compute
`CrossEntropyLoss` against `batch['label']`, compute simple top-1 accuracy
(`(argmax(logits) == label).float().mean()`), and log both `*_loss` and `*_acc` to the Lightning
progress bar/logger (`train_loss`/`train_acc`/`val_loss`/`val_acc` — the latter is what the
checkpoint callback monitors, see §5).

`configure_optimizers` filters `self.parameters()` down to those with `requires_grad=True`
(i.e., exactly the subset enabled by `_apply_situation_rules`) and wraps them in a single
`torch.optim.AdamW(trainable_params, lr=self.lr)`, with `lr` read from
`config['trainer']['learning_rate']` (default `1e-4`).

---

## 5. Training Entry Points

Both training scripts are thin, near-identical CLI wrappers that wire a DataModule + Model into a
`lightning.Trainer`. They differ only in *which dataset* they use and the checkpoint directory
naming convention.

### 5.1 `train_imagenet.py`

([train_imagenet.py](train_imagenet.py))
1. Parses `--config` (default `configs/train.yaml`) and loads the YAML.
2. Prints a one-line run summary: backbone short-name, projector type, situation.
3. Instantiates `TinyImageNetDataModule(config)` and `ProjectorAblationModel(config)`.
4. Configures a `ModelCheckpoint` callback:
   ```python
   ModelCheckpoint(
       dirpath=f"checkpoints/{model_id}_{proj}_{sit}",
       filename=f"{situation}-{{epoch:02d}}",
       save_top_k=1, monitor="val_acc", mode="max", save_last=True
   )
   ```
   Saves only the single best-validation-accuracy checkpoint per run (plus a rolling `last.ckpt`),
   into a directory name that encodes the backbone/projector/situation combination — making it
   easy to run many ablation cells in parallel without checkpoint collisions.
5. Builds an `L.Trainer` with `accelerator="auto"`, `devices=1`, mixed-precision
   (`config['trainer']['precision']`, e.g. `"16-mixed"`), and `max_epochs` from the config.
6. Calls `trainer.fit(model, datamodule=dm)`.

### 5.2 `train_flowers102.py`

([train_flowers102.py](train_flowers102.py))
Structurally identical to §5.1, but:
- Imports `Flowers102DataModule` instead of `TinyImageNetDataModule`.
- Defaults `--config` to `configs/train_flowers.yaml`.
- Suffixes the checkpoint directory with `_flowers102` to avoid collision with Tiny ImageNet runs
  for the same backbone/projector/situation combination.
- Notably **reuses `ProjectorAblationModel` unmodified** — the comment
  `# Model remains identical, just uses config['model']['num_classes']` makes explicit that the
  shared model class is dataset-agnostic; only the `num_classes` value in the config changes the
  classification head's output width.

---

## 6. Evaluation — `test_imagenet.py`

([test_imagenet.py](test_imagenet.py)) loads a trained checkpoint, runs it over the held-out
1k Tiny ImageNet test split, and produces both an aggregate accuracy figure and a per-image CSV
for qualitative/error analysis.

Step by step:
1. **Load config** from `--config` (default `configs/test.yaml`); read `checkpoint_path`,
   `output_csv`.
2. **Load the human-readable class mapping** (`class_mapping.json`, a WordNet-ID → label string
   dict produced by the data-prep stage), and also fetch the Tiny ImageNet HF dataset just to
   read off `features['label'].names` — the integer-index → WordNet-ID lookup needed to bridge
   between the model's numeric predictions and the human-readable mapping.
3. **Set up data**: instantiates `TinyImageNetDataModule(config)`, calls `dm.setup(stage="test")`,
   grabs the `test_dataloader`.
4. **Load the model from checkpoint**:
   ```python
   model = ProjectorAblationModel.load_from_checkpoint(ckpt_path, config=config)
   ```
   This relies on Lightning's `save_hyperparameters()` call in the model constructor (see
   [model_imagenet.py:106](model_imagenet.py#L106)) to restore architecture + weights; `config`
   is passed explicitly to ensure the *evaluation* config (which may point at a different
   checkpoint dir / batch size / test CSV) takes precedence over whatever was saved at train time.
5. **Inference loop**, run under `torch.no_grad()` **and** `torch.autocast`:
   ```python
   autocast_dtype = torch.float16 if device.type == "cuda" else torch.float32
   with torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
       ...
   ```
   The explicit dtype selection (fp16 on CUDA, fp32 on CPU) avoids the runtime error that occurs
   when `torch.autocast` is requested with `float16` on a CPU-only device — a defensive fix the
   author called out inline (`# FIX: Dynamically set the correct dtype...`).
   For each batch: moves `pixel_values`/`label` to device, retrieves `image_path` (falling back to
   `"unknown"` strings if the dataset didn't supply one — defensive against datasets like
   Flowers102 that don't carry paths), runs the forward pass, takes `argmax` for the prediction,
   and accumulates predictions/labels for the global accuracy计算.
6. **Per-image result records**: for every sample, both the true and predicted *integer* label are
   translated through `hf_label_names` (int → WordNet ID) and then `class_mapping`
   (WordNet ID → human string, defaulting to `"Unknown"` for unmapped IDs), producing rows with
   `image_path, true_label_idx, pred_label_idx, true_label_name, pred_label_name, is_correct`.
7. **Persist results**: ensures the output directory exists (`os.makedirs(..., exist_ok=True)`),
   writes the full results table to `output_csv` via pandas — enabling downstream qualitative
   review (e.g. "show me all the misclassified Persian cat images").
8. **Final accuracy**: computes and prints overall top-1 accuracy as a percentage, framed in a
   `"="*40` banner for visibility in long log streams.

---

## 7. Configuration Files (`configs/`)

All three YAML files share the same top-level schema (`model`, `experiment`, `data`, `trainer`,
plus `test` for evaluation), letting the same Python code paths consume any of them via
`config['section']['key']` lookups (with `.get(...)` fallbacks where keys legitimately differ
between training and testing contexts).

### 7.1 `configs/train.yaml` — Tiny ImageNet training
Heavily commented as the canonical reference for all valid option values:
- `model.id`: any HF ViT-family checkpoint (documents the seven backbones from
  `architecture.md` as examples — Vanilla ViT-{B,L,H}, DINOv2-{B,L}).
- `model.projector_type`: `none | llava | gemma | gemma4`.
- `model.bridge_type`: `auto | none | linear | low_rank | mlp | patch_pool | cross_attention`
  — each option's behavior is documented inline.
- `model.num_classes: 200` (Tiny ImageNet).
- `experiment.situation`: one of the five situations from §4.3, documented inline.
- `data`: `train_batch_size: 64`, `val_batch_size: 128`, `num_workers: 4`,
  `val_csv: "data/validation.csv"`.
- `trainer`: `learning_rate: 1.0e-4`, `max_epochs: 5`, `precision: "16-mixed"`.

The active example configures `google/vit-large-patch16-224` + `llava` projector + `bridge_type:
auto` (which, since ViT-Large outputs 1024-dim and LLaVA expects 1024-dim `proj_in_dim`, would in
practice resolve to "no bridge needed" per the dims-match branch) running the `train_all`
situation.

### 7.2 `configs/train_flowers.yaml` — Flowers102 training
Same schema, trimmed comments. Notable differences from `train.yaml`:
- `model.id: "google/vit-large-patch32-384"` (a different ViT variant — patch32 @ 384px).
- `model.num_classes: 102` (Flowers102's class count).
- `data.data_dir: "./data"` instead of `val_csv` (consumed by `Flowers102DataModule.prepare_data`
  / `setup` to locate/download the torchvision dataset).
- No `val_csv`/`test_csv` keys — the Flowers102 DataModule manages its own splits internally via
  `torchvision.datasets.Flowers102`.

### 7.3 `configs/test.yaml` — Checkpoint evaluation
Adds a `test` section (absent from the training configs):
```yaml
test:
  checkpoint_path: "checkpoints/vit-large-patch32-384_llava_train_all/train_all-epoch=00.ckpt"
  mapping_file: "class_mapping.json"
  test_csv: "data/test.csv"
  output_csv: "output/train_all_bestckp.csv"
```
Plus the standard `model`/`experiment`/`data`/`trainer` sections — these must describe the *same
architecture* the checkpoint was trained with (backbone, projector type, situation, num_classes)
so that `load_from_checkpoint` can reconstruct a matching module graph before loading the saved
state dict. `data` here uses a flat `batch_size: 64` (no train/val split distinction, since only
the test loader is used), which the DataModule's `.get(...)` fallback chains handle gracefully.

---

## 8. Environment & Dependencies

### 8.1 `Dockerfile`
Builds on `ghcr.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` (PyTorch 2.8 + CUDA 12.8 +
cuDNN 9), installs `git`/`build-essential`/`gcc`/`g++` (needed for any pip packages that compile
native extensions, e.g. `bitsandbytes`), copies the repo into `/app`, installs
`requirements.txt`, and defaults to an interactive `bash` shell. The README's run commands mount
the working directory (`-v $(pwd):/app`) and pass through `.env` (for `HF_TOKEN` and friends),
so the container always reflects the latest local source and has Hub-download credentials.

### 8.2 `requirements.txt`
Key groups of dependencies and why they matter to this codebase specifically:
- **Core ML**: `torch`, `torchvision`, `lightning` — the training/eval framework.
- **HF stack**: `transformers` (pinned `>=4.52.0,<=4.52.4` — likely to avoid breaking changes in
  `AutoModelForImageTextToText`/projector module layouts across versions), `accelerate`,
  `datasets`, `huggingface-hub`.
- **PEFT/quantization**: `peft`, `bitsandbytes` — present for LoRA-style fine-tuning (referenced
  by the README's description of situations 4–5; see the note in §4.3 about the current
  full-finetune implementation vs. this stated intent).
- **Data/IO**: `pillow`, `pandas`, `opencv-python`, `kagglehub`, `requests`, `python-dotenv`.
- **Experiment tracking**: `wandb`.
- **Misc NLP utilities** (`jieba`, `rouge-chinese`, `nltk`, `qwen-vl-utils`, `einops`,
  `protobuf`): likely inherited from a broader VLM-research template/toolkit rather than used
  directly by the classification pipeline in this snapshot — `qwen-vl-utils` in particular hints
  at planned-but-not-yet-implemented Qwen2-VL projector support (Qwen2-VL appears in
  `architecture.md`'s projector table but has no corresponding branch in
  `ProjectorAblationModel.__init__`).

---

## 9. Cross-Cutting Design Notes

- **One model class, many architectures.** `ProjectorAblationModel` is the single point where
  backbone choice × projector choice × bridge choice × freezing strategy combine. This is what
  makes the YAML-driven ablation grid in `architecture.md` §4 tractable — every cell in that grid
  is just a different config file passed to the same `train_imagenet.py`/`train_flowers102.py`.
- **Memory discipline when extracting projectors.** Loading a 7B-parameter VLM just to steal a
  small submodule is expensive; the explicit `del vlm; gc.collect()` calls
  ([model_imagenet.py:136](model_imagenet.py#L136),
  [model_imagenet.py:167](model_imagenet.py#L167),
  [model_imagenet.py:198](model_imagenet.py#L198)) ensure that memory is reclaimed immediately
  after the projector is copied out, rather than waiting for Python's garbage collector to get
  around to it under memory pressure.
- **`.float()` casts on extracted projectors.** VLMs are loaded in `torch.float16` to keep the
  download/load cheap, but the projector is cast back to `float32` before being kept — this is
  required for compatibility with PyTorch's gradient-scaling-based mixed-precision training
  (`GradScaler`/`"16-mixed"` precision), which expects master weights in fp32.
- **Defensive shape handling.** Both `PatchPoolBridge` (fallback when `N % 4 != 0`) and `forward`
  (the `if features.dim() == 3` pooling guard, and the `pooler_output`-vs-`[CLS]`-token fallback)
  are written to tolerate variation across HuggingFace model families without per-backbone special
  casing in the config.
- **Reproducible evaluation splits.** The deliberate local-CSV materialization of val/test splits
  (vs. on-the-fly HF streaming for train) ensures that accuracy numbers are comparable across runs
  and across different machines/seeds — an important property for an ablation study where the
  whole point is to compare many configurations against each other.
