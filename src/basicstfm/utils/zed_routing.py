"""Routing feature construction for ZED (Zero-shot Expert Diffusion)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch


def resolve_zed_routing_key(
    batch: Dict[str, Any],
    routing_key: str,
) -> Optional[torch.Tensor]:
    """Return optional per-sample domain index if metadata exists (for logging / aux)."""

    rk = str(routing_key).lower().strip()
    if rk in ("auto", "none", "off", ""):
        return None
    if rk == "dataset_name":
        names = batch.get("dataset_name")
        if names is None:
            return None
        if isinstance(names, str):
            return None
        # Prefer explicit index tensor if dataloader provides it
        idx = batch.get("dataset_index")
        if isinstance(idx, torch.Tensor) and idx.ndim == 1:
            return idx
        return None
    if rk == "domain_id":
        idx = batch.get("domain_id")
        return idx if isinstance(idx, torch.Tensor) else None
    if rk == "graph_id":
        idx = batch.get("graph_id")
        return idx if isinstance(idx, torch.Tensor) else None
    return None


def build_route_features(
    x: torch.Tensor,
    graph: Optional[torch.Tensor],
    event_latent: torch.Tensor,
    router_inputs: Sequence[str],
    optional_dataset_id: Optional[torch.Tensor] = None,
    max_dataset_embed: int = 64,
) -> torch.Tensor:
    """Concatenate graph / temporal / event signatures (and optional dataset id one-hot).

    Parameters
    ----------
    x:
        Input windows ``[B, T, N, C]``.
    graph:
        Adjacency ``[N, N]`` or ``[B, N, N]`` (float).
    event_latent:
        Residual-event latent ``[B, T, N, D]``.
    router_inputs:
        Subset of ``graph_signature``, ``temporal_statistics``, ``residual_event_signature``,
        ``dataset_id_embed``.
    optional_dataset_id:
        Long tensor ``[B]`` with indices in ``[0, max_dataset_embed)``.
    """

    parts: List[torch.Tensor] = []
    bsz = int(x.shape[0])
    dev = x.device
    dt = x.dtype

    ins = {str(item).lower().strip() for item in router_inputs}

    if "temporal_statistics" in ins:
        m = x.mean(dim=(1, 2))
        s = x.std(dim=(1, 2), unbiased=False)
        parts.extend([m, s])

    if "graph_signature" in ins:
        if graph is None:
            parts.append(torch.zeros(bsz, 8, device=dev, dtype=dt))
        else:
            g = graph
            if g.ndim == 2:
                gg = g.to(device=dev, dtype=dt)
                stats = torch.stack(
                    [
                        gg.mean(),
                        gg.std(unbiased=False),
                        gg.max(),
                        gg.min(),
                        torch.linalg.matrix_norm(gg, ord="fro"),
                    ]
                )
                pad = torch.zeros(8, device=dev, dtype=dt)
                pad[: stats.numel()] = stats.to(dt)
                parts.append(pad.unsqueeze(0).expand(bsz, -1).contiguous())
            elif g.ndim == 3:
                # Per-sample graph: pooled stats per batch item
                stats_list: List[torch.Tensor] = []
                gx = g.to(device=dev, dtype=dt)
                for i in range(bsz):
                    gg = gx[i]
                    vec = torch.stack(
                        [
                            gg.mean(),
                            gg.std(unbiased=False),
                            gg.max(),
                            gg.min(),
                            torch.linalg.matrix_norm(gg, ord="fro"),
                        ]
                    )
                    pad = torch.zeros(8, device=dev, dtype=dt)
                    pad[: vec.numel()] = vec.to(dt)
                    stats_list.append(pad)
                parts.append(torch.stack(stats_list, dim=0))
            else:
                parts.append(torch.zeros(bsz, 8, device=dev, dtype=dt))

    if "residual_event_signature" in ins:
        e = event_latent.mean(dim=(1, 2))
        parts.append(e)

    if "dataset_id_embed" in ins and optional_dataset_id is not None:
        aid = optional_dataset_id.long().clamp(min=0, max=max_dataset_embed - 1)
        one_hot = torch.zeros(bsz, max_dataset_embed, device=dev, dtype=dt)
        one_hot.scatter_(1, aid.unsqueeze(1), 1.0)
        parts.append(one_hot)

    if not parts:
        raise ValueError("router_inputs produced an empty feature vector; check stage2.router_inputs")

    return torch.cat(parts, dim=-1)
