"""Lightweight GPT-2–style temporal forecaster (non-graph LLM sanity baseline).

This is **not** a line-by-line reproduction of Time-LLM; it is a minimal adapter that maps
``[B,T,N,C]`` series to token embeddings, runs a frozen GPT-2 backbone (or a PyTorch
``TransformerEncoder`` fallback), and predicts the horizon with a small head.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import ensure_4d
from basicstfm.registry import MODELS


def _try_load_gpt2_model(
    model_name: str,
    local_files_only: bool,
) -> Tuple[Any, int]:
    """Return (module, hidden_dim). Raises on failure."""
    try:
        from transformers import GPT2Model  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "``backend: gpt2`` requires ``transformers``. Install with "
            "``pip install -e '.[llm]'`` or ``pip install transformers``."
        ) from exc

    torch_dtype = torch.float32
    kwargs = {"torch_dtype": torch_dtype}
    try:
        model = GPT2Model.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            **kwargs,
        )
    except TypeError:
        model = GPT2Model.from_pretrained(model_name, local_files_only=local_files_only)
    hidden = int(model.config.n_embd)
    return model, hidden


@MODELS.register("TimeLLMGPT2Lite")
class TimeLLMGPT2Lite(nn.Module):
    """Non-graph temporal forecaster: per-node time series as a token sequence for a text LM backbone."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        backend: str = "gpt2",
        gpt2_model_name: str = "gpt2",
        local_files_only: bool = False,
        num_prompt_tokens: int = 4,
        freeze_backbone: bool = True,
        transformer_dim: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_ffn_mult: int = 4,
        dropout: float = 0.1,
        max_num_nodes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"TimeLLMGPT2Lite got unexpected keys: {sorted(kwargs)!r}")

        self.backend = str(backend).lower().strip()
        if self.backend not in {"gpt2", "fallback_transformer"}:
            raise ValueError("backend must be 'gpt2' or 'fallback_transformer'")

        self.num_nodes = int(max_num_nodes) if max_num_nodes is not None else int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.num_prompt_tokens = int(num_prompt_tokens)
        self.freeze_backbone = bool(freeze_backbone)

        self.gpt2_core: Optional[nn.Module] = None
        self.fallback_encoder: Optional[nn.Module] = None
        self.pos_embed: Optional[nn.Parameter] = None

        if self.backend == "gpt2":
            self.gpt2_core, d_model = _try_load_gpt2_model(gpt2_model_name, local_files_only)
            if freeze_backbone:
                for p in self.gpt2_core.parameters():
                    p.requires_grad = False
        else:
            d_model = int(transformer_dim)
            if d_model % int(transformer_heads) != 0:
                raise ValueError("transformer_dim must be divisible by transformer_heads")
            ff = int(transformer_ffn_mult) * d_model
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(transformer_heads),
                dim_feedforward=ff,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.fallback_encoder = nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
            max_pos = self.num_prompt_tokens + self.input_len + 8
            self.pos_embed = nn.Parameter(torch.zeros(1, max_pos, d_model))

        self.in_proj = nn.Linear(self.input_dim, d_model)
        self.prompt_tokens = nn.Parameter(torch.zeros(1, self.num_prompt_tokens, d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.output_len * self.output_dim)

        self._apply_init()

    def _apply_init(self) -> None:
        nn.init.normal_(self.prompt_tokens, std=0.02)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)

    def encode_sequence(self, u: torch.Tensor) -> torch.Tensor:
        """u: [BN, T, C_in] -> embeddings [BN, P+T, d]."""
        bns, t, _ = u.shape
        if t != self.input_len:
            raise ValueError(f"Expected T={self.input_len}, got {t}")

        x = self.in_proj(u)
        prompt = self.prompt_tokens.expand(bns, -1, -1)
        seq = torch.cat([prompt, x], dim=1)

        if self.backend == "gpt2":
            assert self.gpt2_core is not None
            out = self.gpt2_core(inputs_embeds=seq, use_cache=False)
            return out.last_hidden_state

        assert self.fallback_encoder is not None and self.pos_embed is not None
        pos = self.pos_embed[:, : seq.size(1), :].expand(bns, -1, -1)
        z = seq + pos
        z = self.fallback_encoder(z)
        return z

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ):
        del graph, mask
        x = ensure_4d(x)
        b, t, n, c = x.shape
        if n > self.num_nodes:
            raise ValueError(
                f"TimeLLMGPT2Lite expects at most num_nodes={self.num_nodes}, got {n}"
            )
        if c < self.input_dim:
            x = F.pad(x, (0, self.input_dim - c))
        elif c > self.input_dim:
            raise ValueError(f"Expected at most input_dim={self.input_dim}, got {c}")

        u = x.permute(0, 2, 1, 3).reshape(b * n, t, self.input_dim)
        h = self.encode_sequence(u)
        summary = h[:, -1, :]
        summary = self.out_norm(summary)
        flat = self.head(summary)
        pred = flat.view(b * n, self.output_len, self.output_dim)
        pred = pred.view(b, n, self.output_len, self.output_dim).permute(0, 2, 1, 3).contiguous()

        out: dict = {"forecast": pred}
        if mode in {"encode", "embedding"}:
            out["embedding"] = h.view(b, n, h.size(1), h.size(-1))
        return out
