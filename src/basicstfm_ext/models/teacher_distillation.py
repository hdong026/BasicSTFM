"""Teacher helpers for matched-length protocol distillation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import torch
from torch import nn

from basicstfm.models.foundation.opencity import OpenCityFoundationModel
from basicstfm.models.foundation.common import load_weights


class OpenCityTeacher(nn.Module):
    """Optional frozen teacher for matched-length distillation."""

    def __init__(
        self,
        *,
        source: str = "backbone",
        pretrained_path: Optional[str] = None,
        strict_load: bool = False,
        backbone_kwargs: Optional[dict[str, Any]] = None,
        backbone: Optional[OpenCityFoundationModel] = None,
    ) -> None:
        super().__init__()
        self.source = str(source)
        self.model: Optional[OpenCityFoundationModel] = None

        if self.source == "external":
            if backbone_kwargs is None:
                raise ValueError("backbone_kwargs are required for an external teacher")
            self.model = OpenCityFoundationModel(**backbone_kwargs)
            if pretrained_path:
                load_weights(self.model, pretrained_path, strict=strict_load)
            elif backbone is not None:
                self.model.load_state_dict(deepcopy(backbone.state_dict()), strict=True)
            self._freeze_model(self.model)

    @staticmethod
    def _freeze_model(model: nn.Module) -> None:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def sync_from_backbone(self, backbone: OpenCityFoundationModel) -> None:
        if self.model is None:
            return
        self.model.load_state_dict(deepcopy(backbone.state_dict()), strict=True)
        self._freeze_model(self.model)

    def forecast(
        self,
        *,
        backbone: OpenCityFoundationModel,
        x: torch.Tensor,
        graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        teacher = backbone if self.model is None else self.model
        teacher.eval()
        with torch.no_grad():
            outputs = teacher(x, graph=graph, mode="forecast")
        return outputs["forecast"].detach()
