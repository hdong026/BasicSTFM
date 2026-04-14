"""Explicit residual-event diffusion mechanism learner."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from basicstfm.models.dataset_modulation import DatasetConditionedFiLM
from basicstfm.models.foundation.common import normalize_adjacency


class DiffusionMechanismLearner(nn.Module):
    """Learn event activation, propagation, attenuation, and spillover dynamics."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_datasets: int = 1,
        propagation_temperature: float = 1.0,
        diffusion_dropout: float = 0.0,
        use_inertia_gate: bool = True,
        use_attenuation_gate: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.temperature = float(propagation_temperature)
        self.use_inertia_gate = bool(use_inertia_gate)
        self.use_attenuation_gate = bool(use_attenuation_gate)

        self.dataset_modulation = DatasetConditionedFiLM(
            hidden_dim=self.hidden_dim,
            num_datasets=int(num_datasets),
            modulation_scale=0.1,
        )

        gate_dim = self.hidden_dim * 2
        self.activation_gate = nn.Linear(gate_dim, 1)
        self.diffusion_gate = nn.Linear(gate_dim, 1)
        self.inertia_gate = nn.Linear(gate_dim, 1)
        self.attenuation_gate = nn.Linear(gate_dim, 1)
        self.spillover_gate = nn.Linear(gate_dim, 1)

        self.source_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.target_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.event_to_state = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.message_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_drive_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.state_norm = nn.LayerNorm(self.hidden_dim)
        self.state_dropout = nn.Dropout(float(diffusion_dropout))
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        self._reset_gate_parameters()

    def _reset_gate_parameters(self) -> None:
        nn.init.xavier_uniform_(self.activation_gate.weight)
        nn.init.xavier_uniform_(self.diffusion_gate.weight)
        nn.init.xavier_uniform_(self.inertia_gate.weight)
        nn.init.xavier_uniform_(self.attenuation_gate.weight)
        nn.init.xavier_uniform_(self.spillover_gate.weight)

        # Gate priors to prevent diffusion-branch death in early optimization.
        self._set_gate_bias(self.activation_gate, 0.73)   # ~= sigmoid(1.0)
        self._set_gate_bias(self.diffusion_gate, 0.40)    # target 0.3~0.5
        self._set_gate_bias(self.inertia_gate, 0.50)      # target 0.5
        self._set_gate_bias(self.attenuation_gate, 0.85)  # target 0.8~0.9
        self._set_gate_bias(self.spillover_gate, 0.40)

    @staticmethod
    def _set_gate_bias(layer: nn.Linear, target_mean: float) -> None:
        target = float(target_mean)
        target = min(max(target, 1e-4), 1.0 - 1e-4)
        bias = math.log(target / (1.0 - target))
        nn.init.constant_(layer.bias, bias)

    def _static_graph(
        self,
        graph: Optional[torch.Tensor],
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if graph is not None and graph.ndim == 3:
            graph = graph.mean(dim=0)
        return normalize_adjacency(
            graph,
            num_nodes=num_nodes,
            device=device,
            dtype=dtype,
            add_self_loops=False,
        )

    def _dynamic_propagation(
        self,
        state: torch.Tensor,
        static_graph: torch.Tensor,
        activation: torch.Tensor,
        diffusion_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.source_proj(state)
        target = self.target_proj(state)
        logits = torch.einsum("bid,bjd->bij", source, target) / math.sqrt(self.hidden_dim)
        logits = logits / max(self.temperature, 1e-6)
        dynamic_graph = torch.sigmoid(logits)

        propagation_strength = dynamic_graph * static_graph.unsqueeze(0)
        edge_gate = diffusion_gate.expand(-1, -1, state.shape[1])
        propagation_strength = propagation_strength * edge_gate
        propagated = torch.einsum(
            "bij,bjd->bid",
            propagation_strength,
            self.message_proj(self.state_norm(state * activation)),
        )
        return propagated, propagation_strength

    def forward(
        self,
        event_latent: torch.Tensor,
        event_score: torch.Tensor,
        graph: Optional[torch.Tensor],
        output_len: int,
        dataset_index: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if event_latent.ndim != 4:
            raise ValueError(
                f"event_latent must be [B, T, N, D], got {tuple(event_latent.shape)}"
            )
        if event_score.ndim != 4:
            raise ValueError(f"event_score must be [B, T, N, 1], got {tuple(event_score.shape)}")

        batch, steps, nodes, hidden = event_latent.shape
        if hidden != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden}")
        if event_score.shape[:3] != (batch, steps, nodes):
            raise ValueError("event_score shape must match event_latent batch/time/node dimensions")

        output_len = int(output_len)
        if output_len <= 0:
            raise ValueError("output_len must be positive")

        weighted_event = event_latent * event_score
        drive = weighted_event.mean(dim=1)
        drive, modulation = self.dataset_modulation(drive, dataset_index=dataset_index)
        state = self.event_to_state(drive)

        static_graph = self._static_graph(graph, nodes, event_latent.device, event_latent.dtype)
        static_graph = static_graph.clamp(min=0.0)

        forecasts = []
        activation_steps = []
        diffusion_steps = []
        inertia_steps = []
        attenuation_steps = []
        spillover_steps = []
        propagation_steps = []

        base_drive = self.self_drive_proj(drive)
        for _ in range(output_len):
            gate_input = torch.cat([state, base_drive], dim=-1)
            activation = torch.sigmoid(self.activation_gate(gate_input))
            diffusion_gate = torch.sigmoid(self.diffusion_gate(gate_input))
            if self.use_inertia_gate:
                inertia = torch.sigmoid(self.inertia_gate(gate_input))
            else:
                inertia = activation.new_full(activation.shape, 0.5)
            if self.use_attenuation_gate:
                attenuation = torch.sigmoid(self.attenuation_gate(gate_input))
            else:
                attenuation = activation.new_zeros(activation.shape)
            spillover = torch.sigmoid(self.spillover_gate(gate_input))

            propagated, propagation_strength = self._dynamic_propagation(
                state,
                static_graph=static_graph,
                activation=activation,
                diffusion_gate=diffusion_gate,
            )

            self_drive = inertia * state + (1.0 - inertia) * base_drive
            diffusion_drive = spillover * propagated
            next_state = (self_drive + diffusion_drive) * (1.0 - attenuation)
            next_state = self.state_dropout(next_state)
            state = next_state

            forecasts.append(self.output_proj(state))
            activation_steps.append(activation)
            diffusion_steps.append(diffusion_gate)
            inertia_steps.append(inertia)
            attenuation_steps.append(attenuation)
            spillover_steps.append(spillover)
            propagation_steps.append(propagation_strength)

        residual_forecast = torch.stack(forecasts, dim=1)
        return {
            "residual_forecast": residual_forecast,
            "event_activation": torch.stack(activation_steps, dim=1),
            "diffusion_gate": torch.stack(diffusion_steps, dim=1),
            "inertia_gate": torch.stack(inertia_steps, dim=1),
            "attenuation_gate": torch.stack(attenuation_steps, dim=1),
            "spillover_gate": torch.stack(spillover_steps, dim=1),
            "propagation_map": torch.stack(propagation_steps, dim=1),
            "dataset_gamma": modulation["gamma"],
            "dataset_beta": modulation["beta"],
        }
