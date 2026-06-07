import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import gc
from transformers import AutoModel, AutoModelForImageTextToText, LlavaForConditionalGeneration


# ---------------------------------------------------------------------------
# Dimension Bridge modules (used when ve_dim != proj_in_dim)
# ---------------------------------------------------------------------------

class LinearBridge(nn.Module):
    """Single linear projection: Y = XW  (no non-linearity)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


class LowRankBridge(nn.Module):
    """LoRA-style bottleneck: Y = GELU(X W_down) W_up, rank r."""
    def __init__(self, in_dim, out_dim, rank=64):
        super().__init__()
        self.down = nn.Linear(in_dim, rank)
        self.up   = nn.Linear(rank, out_dim)

    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


class PatchPoolBridge(nn.Module):
    """
    Spatial pixel-merge bridge (inspired by Qwen2-VL / Gemma 4).
    Merges every 2x2 neighbourhood in the token sequence, then projects.
    Falls back to a simple linear layer when the sequence length is not
    divisible by 4 (e.g. because a [CLS] token is present).
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim * 4, out_dim)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        if N % 4 != 0:
            # fallback: plain linear without spatial merge
            return nn.functional.linear(
                x,
                self.proj.weight[:, :C],
                self.proj.bias
            )
        x = x.view(B, N // 4, C * 4)
        return self.proj(x)


class CrossAttentionBridge(nn.Module):
    """
    Perceiver-style cross-attention resampler.
    A fixed set of k learnable queries attend to the visual tokens.
    """
    def __init__(self, in_dim, out_dim, num_queries=256, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, out_dim))
        self.attn    = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.kv_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # x: (B, N, in_dim)
        kv = self.kv_proj(x)
        q  = self.queries.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, kv, kv)
        return out


class MLPBridge(nn.Module):
    """Two-layer MLP with a midpoint hidden dim: Linear → GELU → LayerNorm → Linear."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        mid = (in_dim + out_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, out_dim)
        )

    def forward(self, x):
        return self.net(x)



_BRIDGE_REGISTRY = {
    "linear":          LinearBridge,
    "low_rank":        LowRankBridge,
    "mlp":             MLPBridge,
    "patch_pool":      PatchPoolBridge,
    "cross_attention": CrossAttentionBridge,
}

class ProjectorAblationModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.situation = config['experiment']['situation']
        self.lr = config['trainer'].get('learning_rate', 1e-4) 
        
        model_id        = config['model']['id']
        projector_type  = config['model']['projector_type']
        bridge_type     = config['model'].get('bridge_type', 'auto')  # auto | none | linear | low_rank | patch_pool | cross_attention
        
        # --- 1. Load the Base Vision Encoder (VE) ---
        base_model = AutoModel.from_pretrained(model_id)
        
        if hasattr(base_model, "vision_model"):
            self.ve = base_model.vision_model
            print(f"Detected multimodal model ({model_id}). Extracted vision_model only.")
        else:
            self.ve = base_model
            
        ve_dim = getattr(self.ve.config, "hidden_size", 
                         getattr(self.ve.config, "vision_config", 768))
        if not isinstance(ve_dim, int): ve_dim = ve_dim.hidden_size

        # --- 2. Extract Pre-Trained Projectors directly to self.proj ---
        if projector_type == "llava":
            print("Extracting LLaVA 1.5 Projector Weights...")
            vlm = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="cpu")
            self.proj = vlm.multi_modal_projector.float() # Ensures FP32 for GradScaler
            
            proj_in_dim = self.proj.linear_1.in_features
            llm_hidden_dim = self.proj.linear_2.out_features
            del vlm; gc.collect() 

        elif projector_type == "gemma":
            print("Extracting Gemma 3 Projector Weights...")
            vlm = AutoModelForImageTextToText.from_pretrained("google/gemma-3-4b-it", torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
            
            self.proj = None
            possible_names = ["multi_modal_projector", "vision_projector", "projector"]
            
            search_bases = [vlm]
            if hasattr(vlm, "model"):
                search_bases.append(vlm.model)
                
            for base in search_bases:
                for name, module in base.named_children():
                    if any(p in name.lower() for p in possible_names):
                        self.proj = module.float() # Ensures FP32 for GradScaler
                        break
                if self.proj is not None:
                    break
                        
            if self.proj is None:
                raise ValueError("Could not automatically locate the Gemma projector.")
            
            proj_in_dim = vlm.config.vision_config.hidden_size
            
            if hasattr(vlm.config, "text_config"):
                llm_hidden_dim = vlm.config.text_config.hidden_size
            else:
                llm_hidden_dim = vlm.config.hidden_size
                
            del vlm; gc.collect() 

        elif projector_type == "gemma4":
            print("Extracting Gemma 4 Projector Weights...")
            vlm = AutoModelForImageTextToText.from_pretrained("google/gemma-4-E4B-it", torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
            
            self.proj = None
            possible_names = ["multi_modal_projector", "vision_projector", "projector"]
            
            search_bases = [vlm]
            if hasattr(vlm, "model"):
                search_bases.append(vlm.model)
                
            for base in search_bases:
                for name, module in base.named_children():
                    if any(p in name.lower() for p in possible_names):
                        self.proj = module.float() # Ensures FP32 for GradScaler
                        break
                if self.proj is not None:
                    break
                        
            if self.proj is None:
                raise ValueError("Could not automatically locate the Gemma 4 projector.")
            
            proj_in_dim = vlm.config.vision_config.hidden_size
            
            if hasattr(vlm.config, "text_config"):
                llm_hidden_dim = vlm.config.text_config.hidden_size
            else:
                llm_hidden_dim = vlm.config.hidden_size
                
            del vlm; gc.collect()

        elif projector_type == "llava_rand":
            print("Building LLaVA-architecture Projector with random weights...")
            proj_in_dim   = 1024   # matches CLIP ViT-L/14 output (LLaVA 1.5 default)
            llm_hidden_dim = 4096  # matches LLaMA-2 7B hidden dim (LLaVA 1.5 default)
            self.proj = nn.Sequential(
                nn.Linear(proj_in_dim, llm_hidden_dim),
                nn.GELU(),
                nn.Linear(llm_hidden_dim, llm_hidden_dim),
            )

        elif projector_type == "none":
            self.proj = nn.Identity()
            proj_in_dim = ve_dim
            llm_hidden_dim = ve_dim
        else:
            raise ValueError("Invalid projector_type. Choose 'llava', 'llava_rand', 'gemma', 'gemma4', or 'none'.")

        # --- 3. Dimension Bridge (optional) ---
        # The bridge is inserted between the VE and the projector only when
        # the dimensions do not match, or when an explicit bridge_type is given.
        self.bridge = None
        if projector_type != "none":
            dims_match = (ve_dim == proj_in_dim)
            need_bridge = (not dims_match) or (bridge_type not in ('auto', 'none'))

            if need_bridge:
                effective_bridge = bridge_type if bridge_type not in ('auto', 'none') else 'linear'
                if bridge_type == 'none' and not dims_match:
                    raise ValueError(
                        f"Dimension mismatch! VE outputs {ve_dim}, projector expects {proj_in_dim}. "
                        f"Set model.bridge_type to a valid bridge (linear | low_rank | patch_pool | cross_attention) "
                        f"or use a matching backbone."
                    )
                if effective_bridge not in _BRIDGE_REGISTRY:
                    raise ValueError(f"Unknown bridge_type '{effective_bridge}'. "
                                     f"Choose from: {list(_BRIDGE_REGISTRY.keys())} or 'none' / 'auto'.")
                self.bridge = _BRIDGE_REGISTRY[effective_bridge](ve_dim, proj_in_dim)
                print(f"Bridge: {effective_bridge}  ({ve_dim} → {proj_in_dim})")
            else:
                print("Bridge: not needed (dimensions already match)")

        # --- 4. Route Logic & Dimensions based on Situation ---
        no_proj_situations = ["train_ve_out_ch", "train_ve_ch"]
        self.use_projector = self.situation not in no_proj_situations
        out_dim = llm_hidden_dim if self.use_projector and projector_type != "none" else ve_dim

        # --- 5. Classifier Head (CH) ---
        self.ch = nn.Linear(out_dim, config['model']['num_classes'])
        self.loss_fn = nn.CrossEntropyLoss()

        self._apply_situation_rules()

    def _apply_situation_rules(self):
        # Default: Freeze everything
        for param in self.parameters():
            param.requires_grad = False
            
        # Always unfreeze the classification head
        for param in self.ch.parameters():
            param.requires_grad = True

        # Situation-specific unfreezing (Full Finetuning)
        if self.situation == "train_ve_out_ch":
            pass # VE frozen, Proj bypassed
        elif self.situation == "train_proj_out_ch":
            pass # VE frozen, Proj frozen
        elif self.situation == "train_proj_ch":
            for param in self.proj.parameters():
                param.requires_grad = True
            if self.bridge is not None:  # bridge is part of the projector pipeline
                for param in self.bridge.parameters():
                    param.requires_grad = True
        elif self.situation == "train_ve_ch":
            for param in self.ve.parameters():
                param.requires_grad = True # Proj bypassed
        elif self.situation == "train_all":
            for param in self.ve.parameters():
                param.requires_grad = True
            for param in self.proj.parameters():
                param.requires_grad = True
            if self.bridge is not None:
                for param in self.bridge.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown situation: {self.situation}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Situation: {self.situation} | Trainable Params: {trainable:,} / {total:,}")

    def forward(self, pixel_values):
        ve_outputs = self.ve(pixel_values=pixel_values)
        
        if self.use_projector:
            features = ve_outputs.last_hidden_state
            if self.bridge is not None:
                features = self.bridge(features)
            features = self.proj(features)
            
            if features.dim() == 3:
                features = features.mean(dim=1)
        else:
            if hasattr(ve_outputs, 'pooler_output') and ve_outputs.pooler_output is not None:
                features = ve_outputs.pooler_output
            else:
                features = ve_outputs.last_hidden_state[:, 0, :]
            
        return self.ch(features)

    def training_step(self, batch, batch_idx):
        logits = self(batch['pixel_values'])
        loss = self.loss_fn(logits, batch['label'])
        acc = (torch.argmax(logits, dim=1) == batch['label']).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['pixel_values'])
        loss = self.loss_fn(logits, batch['label'])
        acc = (torch.argmax(logits, dim=1) == batch['label']).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(trainable_params, lr=self.lr)