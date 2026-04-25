"""UniST-style prompt-empowered spatio-temporal foundation model adapter."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from basicstfm.models.foundation.common import (
    GraphMixingBlock,
    MemoryPool,
    TransformerBlock,
    ensure_4d,
    load_filtered_weights,
    load_weights,
)
from basicstfm.registry import MODELS


class FactorizedSTLayer(nn.Module):
    """Per-node temporal Transformer + dense graph mixing (``[B,T,N,D]``).

    Attention length is **T** (or patched T), not ``N*T``, matching the cost class of
    patch-wise models like FactoST while keeping an explicit encoder stack.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal = TransformerBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.graph = GraphMixingBlock(dim=hidden_dim, dropout=dropout)

    def forward(self, h: torch.Tensor, graph: Optional[torch.Tensor]) -> torch.Tensor:
        batch, steps, nodes, dim = h.shape
        x = h.permute(0, 2, 1, 3).reshape(batch * nodes, steps, dim)
        x = self.temporal(x)
        h = x.reshape(batch, nodes, steps, dim).permute(0, 2, 1, 3).contiguous()
        return self.graph(h, graph)


class UniSTPrompt(nn.Module):
    """Spatial / temporal memory prompts (stage-2), analogous to Prompt_ST fusion."""

    def __init__(self, hidden_dim: int, spatial_slots: int, temporal_slots: int) -> None:
        super().__init__()
        self.spatial_memory = MemoryPool(spatial_slots, hidden_dim)
        self.temporal_memory = MemoryPool(temporal_slots, hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        spatial_query = h.mean(dim=1)
        temporal_query = h.mean(dim=2)
        spatial_prompt = self.spatial_memory(spatial_query)[:, None]
        temporal_prompt = self.temporal_memory(temporal_query)[:, :, None]
        spatial_prompt = spatial_prompt.expand(-1, h.shape[1], -1, -1)
        temporal_prompt = temporal_prompt.expand(-1, -1, h.shape[2], -1)
        return self.fuse(torch.cat([spatial_prompt, temporal_prompt], dim=-1))


@MODELS.register("UniSTFoundationModel")
class UniSTFoundationModel(nn.Module):
    """Masked autoencoder with a **separate decoder** and decoder-side prompts.

    UniST (KDD 2024) structure: embedding → encoder → ``decoder_embed`` → prompt on the
    forecast span → decoder → heads.

    **attention_mode**

    - ``joint``: one Transformer over **all** flattened spatio-temporal tokens (matches
      the original ``(H·W·T')`` joint attention idea, but slow when ``N·T`` is large).
    - ``factorized`` (default): each layer is **temporal self-attention per node** then
      **graph mixing** over nodes at each time step — same complexity class as FactoST /
      OpenCity-style backbones (attention length ~ ``T``, not ``N·T``), much faster on
      large graphs.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        hidden_dim: int = 128,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        t_patch_size: int = 1,
        attention_mode: str = "factorized",
        max_seq_len: int = 4096,
        use_prompt: bool = True,
        num_memory_spatial: int = 128,
        num_memory_temporal: int = 128,
        max_num_nodes: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        strict_load: bool = False,
    ) -> None:
        super().__init__()
        emb_n = int(max_num_nodes) if max_num_nodes is not None else int(num_nodes)
        self.num_nodes = emb_n
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.hidden_dim = int(hidden_dim)
        self.t_patch_size = max(1, int(t_patch_size))
        self.max_seq_len = int(max_seq_len)
        self.use_prompt = bool(use_prompt)
        mode = str(attention_mode).lower().strip()
        if mode not in {"joint", "factorized"}:
            raise ValueError("attention_mode must be 'joint' or 'factorized'")
        self.attention_mode = mode

        self.value_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.node_emb = nn.Embedding(emb_n, self.hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.max_seq_len, 1, self.hidden_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_dim))

        if self.attention_mode == "joint":
            self.encoder_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=self.hidden_dim,
                        num_heads=int(num_heads),
                        ffn_dim=int(ffn_dim),
                        dropout=float(dropout),
                    )
                    for _ in range(int(encoder_layers))
                ]
            )
            self.decoder_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        dim=self.hidden_dim,
                        num_heads=int(num_heads),
                        ffn_dim=int(ffn_dim),
                        dropout=float(dropout),
                    )
                    for _ in range(int(decoder_layers))
                ]
            )
        else:
            self.encoder_blocks = nn.ModuleList(
                [
                    FactorizedSTLayer(
                        hidden_dim=self.hidden_dim,
                        num_heads=int(num_heads),
                        ffn_dim=int(ffn_dim),
                        dropout=float(dropout),
                    )
                    for _ in range(int(encoder_layers))
                ]
            )
            self.decoder_blocks = nn.ModuleList(
                [
                    FactorizedSTLayer(
                        hidden_dim=self.hidden_dim,
                        num_heads=int(num_heads),
                        ffn_dim=int(ffn_dim),
                        dropout=float(dropout),
                    )
                    for _ in range(int(decoder_layers))
                ]
            )

        self.encoder_norm = nn.LayerNorm(self.hidden_dim)

        self.decoder_embed = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.prompt = UniSTPrompt(
            hidden_dim=self.hidden_dim,
            spatial_slots=int(num_memory_spatial),
            temporal_slots=int(num_memory_temporal),
        )

        self.decoder_norm = nn.LayerNorm(self.hidden_dim)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len * self.output_dim),
        )
        self.reset_parameters()

        if pretrained_path:
            load_weights(self, pretrained_path, strict=strict_load)

    def load_backbone_weights(self, path: str, strict: bool = False) -> tuple[list[str], list[str]]:
        """Load the stage-1 backbone while leaving prompt memories stage-local."""

        return load_filtered_weights(
            self,
            path,
            strict=strict,
            exclude_prefixes=("prompt.",),
        )

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.node_emb.weight, std=0.02)

    def _output_len_patched(self) -> int:
        u = self.t_patch_size
        if u <= 1:
            return self.output_len
        if self.output_len % u != 0:
            raise ValueError(
                f"output_len={self.output_len} must be divisible by t_patch_size={u} "
                "when temporal patching is enabled"
            )
        return self.output_len // u

    def _time_patch(self, h: torch.Tensor) -> torch.Tensor:
        u = self.t_patch_size
        if u <= 1:
            return h
        batch, steps, nodes, dim = h.shape
        if steps % u != 0:
            raise ValueError(
                f"Sequence length {steps} must be divisible by t_patch_size={u} "
                "(e.g. use input_len+output_len divisible by u)"
            )
        return h.view(batch, steps // u, u, nodes, dim).mean(dim=2)

    def _time_unpatch_logits(self, h: torch.Tensor, target_steps: int) -> torch.Tensor:
        u = self.t_patch_size
        if u <= 1:
            return h
        h = h.repeat_interleave(u, dim=1)
        return h[:, :target_steps, :, :]

    def _embed_inputs(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = ensure_4d(x)
        batch, steps, nodes, channels = x.shape
        if nodes > self.node_emb.num_embeddings:
            raise ValueError(
                f"Input has {nodes} nodes but node embedding has {self.node_emb.num_embeddings} rows; "
                f"increase num_nodes / max_num_nodes in config."
            )
        if channels > self.input_dim:
            raise ValueError(f"Expected at most input_dim={self.input_dim}, got {channels}")
        if steps > self.max_seq_len:
            raise ValueError(f"Input length {steps} exceeds max_seq_len={self.max_seq_len}")
        if channels < self.input_dim:
            x = F.pad(x, (0, self.input_dim - channels))
            if mask is not None:
                mask = F.pad(mask, (0, self.input_dim - channels))
        h = self.value_proj(x)
        if mask is not None:
            token = self.mask_token.to(dtype=h.dtype, device=h.device)
            pos_mask = mask.any(dim=-1, keepdim=True)
            h = torch.where(pos_mask, token, h)
        node_ids = torch.arange(nodes, device=x.device)
        h = h + self.node_emb(node_ids)[None, None] + self.temporal_pos[:, :steps]
        return h

    def _run_encoder(self, h: torch.Tensor, graph: Optional[torch.Tensor]) -> torch.Tensor:
        if self.attention_mode == "joint":
            batch, steps, nodes, dim = h.shape
            tokens = h.reshape(batch, steps * nodes, dim)
            for block in self.encoder_blocks:
                tokens = block(tokens)
            tokens = self.encoder_norm(tokens)
            return tokens.reshape(batch, steps, nodes, dim)
        out = h
        for block in self.encoder_blocks:
            out = block(out, graph)
        return self.encoder_norm(out)

    def _run_decoder(
        self,
        enc: torch.Tensor,
        graph: Optional[torch.Tensor],
        use_prompt: bool,
    ) -> torch.Tensor:
        batch, steps, nodes, dim = enc.shape
        h = self.decoder_embed(enc)
        ol = self._output_len_patched()
        if use_prompt:
            p = self.prompt(enc)
            h = h.clone()
            h[:, -ol:, :, :] = h[:, -ol:, :, :] + p[:, -ol:, :, :]
        if self.attention_mode == "joint":
            tokens = h.reshape(batch, steps * nodes, dim)
            for block in self.decoder_blocks:
                tokens = block(tokens)
            tokens = self.decoder_norm(tokens)
            return tokens.reshape(batch, steps, nodes, dim)
        out = h
        for block in self.decoder_blocks:
            out = block(out, graph)
        return self.decoder_norm(out)

    def encode(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self._embed_inputs(x, mask=mask)
        h = self._time_patch(h)
        return self._run_encoder(h, graph)

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "forecast",
    ):
        h_in = self._embed_inputs(x, mask=mask)
        t_orig = h_in.shape[1]
        h = self._time_patch(h_in)
        encoded = self._run_encoder(h, graph)
        out: dict = {"embedding": encoded}
        if mode in {"encode", "embedding"}:
            return out

        prompt_enabled = self.use_prompt or mode == "prompt_forecast"
        decoded = self._run_decoder(encoded, graph=graph, use_prompt=prompt_enabled)

        if mode in {"forecast", "zero_shot", "prompt_forecast", "both"}:
            summary = decoded[:, -1]
            forecast = self.forecast_head(summary)
            batch, nodes, _ = forecast.shape
            forecast = forecast.reshape(batch, nodes, self.output_len, self.output_dim)
            out["forecast"] = forecast.permute(0, 2, 1, 3).contiguous()

        if mode in {"reconstruct", "reconstruction", "both"}:
            rec = self.reconstruction_head(decoded)
            rec = self._time_unpatch_logits(rec, t_orig)
            out["reconstruction"] = rec
        return out
