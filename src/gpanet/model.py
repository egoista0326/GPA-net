from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


OUTPUT_NAMES = [
    "UpperAsymmetry",
    "LowerAsymmetry",
    "TotalAsymmetry",
    "Classification",
]


class PhaseSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        projection_dim: int,
        phase_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.feature_weight = nn.Parameter(torch.empty(num_features, projection_dim))
        self.feature_bias = nn.Parameter(torch.empty(num_features, projection_dim))
        nn.init.kaiming_uniform_(self.feature_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(1.0)
        nn.init.uniform_(self.feature_bias, -bound, bound)
        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_features * projection_dim, phase_hidden_dim),
            nn.ReLU(),
        )

    def forward(self, phase_input: torch.Tensor) -> torch.Tensor:
        token_sequence = (
            phase_input.unsqueeze(-1) * self.feature_weight.unsqueeze(0)
            + self.feature_bias.unsqueeze(0)
        )
        attended, _ = self.attention(
            token_sequence,
            token_sequence,
            token_sequence,
            need_weights=False,
        )
        return self.projection(attended)


class TaskSpecificHead(nn.Module):
    def __init__(
        self,
        phase_count: int,
        phase_hidden_dim: int,
        scalar_hidden_dim: int,
        dropout: float,
        classification: bool,
    ) -> None:
        super().__init__()
        self.phase_count = phase_count
        self.classification = classification
        self.phase_weights = nn.Linear(phase_hidden_dim, 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=phase_hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.phase_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(phase_hidden_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                )
                for _ in range(phase_count)
            ]
        )
        merged_dim = phase_count * 16 + scalar_hidden_dim
        if classification:
            self.output_head = nn.Sequential(
                nn.Linear(merged_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(merged_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(
        self,
        phase_sequence: torch.Tensor,
        scalar_embedding: torch.Tensor,
    ) -> torch.Tensor:
        phase_weights = torch.sigmoid(self.phase_weights(phase_sequence))
        weighted_sequence = phase_sequence * phase_weights
        attended, _ = self.attention(
            weighted_sequence,
            weighted_sequence,
            weighted_sequence,
            need_weights=False,
        )
        phase_outputs = [
            phase_mlp(attended[:, phase_index, :])
            for phase_index, phase_mlp in enumerate(self.phase_mlps)
        ]
        merged = torch.cat(phase_outputs + [scalar_embedding], dim=-1)
        return self.output_head(merged)


class DualAttentionBaselineModel(nn.Module):
    def __init__(
        self,
        phase_feature_dim: int,
        phase_count: int,
        feature_projection_dim: int,
        phase_hidden_dim: int,
        scalar_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.phase_count = phase_count
        self.phase_blocks = nn.ModuleList(
            [
                PhaseSelfAttentionBlock(
                    num_features=phase_feature_dim,
                    projection_dim=feature_projection_dim,
                    phase_hidden_dim=phase_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(phase_count)
            ]
        )
        self.phase_norms = nn.ModuleList(
            [nn.LayerNorm(phase_hidden_dim) for _ in range(phase_count)]
        )
        self.phase_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(phase_count)])
        self.scalar_branch = nn.Sequential(
            nn.Linear(1, scalar_hidden_dim),
            nn.ReLU(),
        )
        self.upper_head = TaskSpecificHead(
            phase_count=phase_count,
            phase_hidden_dim=phase_hidden_dim,
            scalar_hidden_dim=scalar_hidden_dim,
            dropout=dropout,
            classification=False,
        )
        self.lower_head = TaskSpecificHead(
            phase_count=phase_count,
            phase_hidden_dim=phase_hidden_dim,
            scalar_hidden_dim=scalar_hidden_dim,
            dropout=dropout,
            classification=False,
        )
        self.total_head = TaskSpecificHead(
            phase_count=phase_count,
            phase_hidden_dim=phase_hidden_dim,
            scalar_hidden_dim=scalar_hidden_dim,
            dropout=dropout,
            classification=False,
        )
        self.classification_head = TaskSpecificHead(
            phase_count=phase_count,
            phase_hidden_dim=phase_hidden_dim,
            scalar_hidden_dim=scalar_hidden_dim,
            dropout=dropout,
            classification=True,
        )

    def forward(
        self,
        phase_inputs: Sequence[torch.Tensor],
        scalar_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if len(phase_inputs) != self.phase_count:
            raise ValueError(
                f"Expected {self.phase_count} phase inputs, got {len(phase_inputs)}"
            )

        phase_embeddings = []
        for phase_input, phase_block, phase_norm, phase_dropout in zip(
            phase_inputs,
            self.phase_blocks,
            self.phase_norms,
            self.phase_dropouts,
        ):
            embedding = phase_block(phase_input)
            embedding = phase_norm(embedding)
            embedding = phase_dropout(embedding)
            phase_embeddings.append(embedding)

        phase_sequence = torch.stack(phase_embeddings, dim=1)
        scalar_embedding = self.scalar_branch(scalar_feature)

        return {
            "UpperAsymmetry": self.upper_head(phase_sequence, scalar_embedding),
            "LowerAsymmetry": self.lower_head(phase_sequence, scalar_embedding),
            "TotalAsymmetry": self.total_head(phase_sequence, scalar_embedding),
            "Classification": self.classification_head(phase_sequence, scalar_embedding),
        }
