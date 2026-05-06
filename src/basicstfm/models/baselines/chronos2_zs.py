"""Chronos-2 (and compatible) zero-shot bridge — requires ``chronos-forecasting`` optional dep."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from basicstfm.models.foundation.common import ensure_4d
from basicstfm.registry import MODELS


def _resolve_chronos_pipeline_class(pretrained_model_id: str) -> Any:
    """Pick Chronos2 / Bolt / v1 (T5 seq2seq) pipeline from the checkpoint config.

    ``Chronos2Pipeline`` only resolves classes exported on ``chronos.chronos2``; Hub ids like
    ``amazon/chronos-t5-small`` still use ``T5ForConditionalGeneration``, which must be loaded via
    ``ChronosPipeline`` (``AutoModelForSeq2SeqLM``), not Chronos-2's loader.
    """
    try:
        from transformers import AutoConfig  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Chronos zero-shot requires ``transformers`` (pulled in by chronos-forecasting)."
        ) from exc

    try:
        import chronos.chronos2 as chronos2_mod  # type: ignore
        import chronos.chronos_bolt as chronos_bolt_mod  # type: ignore
        from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Chronos zero-shot requires the ``chronos-forecasting`` package. "
            "Install with: pip install 'chronos-forecasting>=2.2' "
            "or: pip install -e '.[chronos]'"
        ) from exc

    config = AutoConfig.from_pretrained(pretrained_model_id)
    archs = getattr(config, "architectures", None) or []
    arch0: Optional[str] = archs[0] if archs else None

    if arch0 and hasattr(chronos2_mod, arch0):
        return Chronos2Pipeline
    if arch0 and hasattr(chronos_bolt_mod, arch0):
        return ChronosBoltPipeline
    return ChronosPipeline


@MODELS.register("Chronos2ZeroShotForecaster")
class Chronos2ZeroShotForecaster(nn.Module):
    """Non-graph FM sanity check; delegates forecast to Chronos-2 (HF id configurable).

    **Contract:** ``forward`` returns ``forecast`` shaped ``[B, output_len, N, C]`` in the **same
    normalized space** as input ``x`` (StandardScaler features). Chronos was pre-trained on raw
    scale — expect a domain gap; this stage is labeled **Chronos2-ZS** for protocol tracking only.

    The HF model is loaded lazily on first ``forward`` to keep ``basicstfm dry-run`` lightweight.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        input_len: int,
        output_len: int,
        pretrained_model_id: str = "amazon/chronos-t5-small",
        quantile_level: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.pretrained_model_id = str(pretrained_model_id)
        self.quantile_level = float(quantile_level)
        self._pipeline = None
        self._pipeline_device: Optional[torch.device] = None
        self.register_buffer("_noop", torch.zeros(1), persistent=False)

    def _ensure_pipeline(self, device: torch.device) -> None:
        if self._pipeline is not None and self._pipeline_device == device:
            return
        pipe_cls = _resolve_chronos_pipeline_class(self.pretrained_model_id)
        device_map = str(device) if device.type == "cuda" else "cpu"
        try:
            self._pipeline = pipe_cls.from_pretrained(self.pretrained_model_id, device_map=device_map)
        except TypeError:
            self._pipeline = pipe_cls.from_pretrained(self.pretrained_model_id)
        self._pipeline_device = device

    def _predict_series(self, context_1d: torch.Tensor, device: torch.device) -> torch.Tensor:
        """context_1d: [T]; returns [H] on ``device``."""
        pipe = self._pipeline
        h = self.output_len
        q = self.quantile_level
        ctx = context_1d[None, :].contiguous().float().to(device)
        # Chronos-1 / Bolt: ``inputs`` is (batch, history). Chronos-2 requires (n_series, n_variates, history).
        if type(pipe).__name__ == "Chronos2Pipeline":
            inputs = ctx.unsqueeze(1)
        else:
            inputs = ctx
        # v1 tokenizer bins live on CPU; bucketize requires context and boundaries on the same device.
        predict_inputs = inputs if type(pipe).__name__ == "Chronos2Pipeline" else inputs.cpu()

        if hasattr(pipe, "predict_quantiles"):
            quant_out = pipe.predict_quantiles(
                predict_inputs,
                prediction_length=h,
                quantile_levels=[q],
            )
            # Chronos2Pipeline: (list[Tensor], list[Tensor]); each quantile tensor is (n_variates, H, Q).
            if isinstance(quant_out[0], list):
                q_tensor = quant_out[0][0]
                median = q_tensor[..., 0]
            else:
                quantiles = quant_out[0]
                median = quantiles[..., 0]
        elif hasattr(pipe, "predict"):
            median = pipe.predict(predict_inputs, prediction_length=h)
            if isinstance(median, torch.Tensor) and median.ndim >= 3:
                median = median.median(dim=1).values
        else:
            raise RuntimeError("Chronos pipeline has no predict_quantiles or predict")

        if isinstance(median, torch.Tensor):
            y = median.float().to(device)
        else:
            y = torch.as_tensor(median, dtype=torch.float32, device=device)
        if y.ndim == 1:
            y = y.view(1, -1)
        if y.shape[-1] != h:
            y = y[..., :h]
        return y.reshape(-1)[:h]

    def forward(
        self,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        mode: str = "zero_shot",
    ):
        del graph, mask, mode
        x = ensure_4d(x)
        b, t, n, c = x.shape
        if c > 1:
            ctx_full = x.mean(dim=-1)
        else:
            ctx_full = x[..., 0]

        device = x.device
        self._ensure_pipeline(device)
        rows = []
        for bi in range(b):
            cols = []
            for ni in range(n):
                y = self._predict_series(ctx_full[bi, :, ni].detach(), device)
                cols.append(y)
            rows.append(torch.stack(cols, dim=0))
        preds = torch.stack(rows, dim=0)
        out = preds.unsqueeze(-1).permute(0, 2, 1, 3).contiguous()
        if self.output_dim > 1:
            out = out.expand(-1, -1, -1, self.output_dim)
        return {"forecast": out}
