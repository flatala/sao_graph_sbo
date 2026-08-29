from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DenseGINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm: str = "layer"):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        if norm == "batch":
            norm_layer = nn.BatchNorm1d(out_dim)
        elif norm == "layer":
            norm_layer = nn.LayerNorm(out_dim)
        else:
            raise ValueError(f"Unsupported GIN norm {norm!r}; expected 'layer' or 'batch'.")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            norm_layer,
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, h: Tensor, A: Tensor) -> Tensor:
        # Check if batched as (B, N, F) or just single sample as (N, F)
        if not (is_batched := h.dim() == 3):
            h = h.unsqueeze(0)
            A = A.unsqueeze(0)

        # Get batch size and num nodes
        B, N, _ = h.shape

        # For each node we add up the feature vals from neighbors: (N x N) @ (N x F) -> (N x F)
        # S_ij = (1 + eps) * X_ij + SUM_{k in N(i)} X_kj
        agg = torch.bmm(A, h)
        out = (1.0 + self.eps) * h + agg

        # Apply MLP per each node's feature vector
        out = out.reshape(B * N, -1)
        out = self.mlp(out)
        out = out.reshape(B, N, -1)

        return out if is_batched else out.squeeze(0)


class GINEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 3, norm: str = "layer"):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList(
            [DenseGINLayer(dims[i], dims[i + 1], norm=norm) for i in range(n_layers)]
        )
        self.output_dim = hidden_dim
        self.norm = norm

    def forward(self, x: Tensor, A: Tensor) -> tuple[Tensor, list[Tensor]]:
        h = x
        all_h = []
        for layer in self.layers:
            h = layer(h, A)
            all_h.append(h)
        return h, all_h


class VAEHeads(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self.fc_mu(h), self.fc_log_var(h)

class ADSGDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, adj_decoder: str = "inner_product"):
        super().__init__()
        self.fc_features = nn.Linear(latent_dim, output_dim)
        if adj_decoder == "mlp":
            # Learned scalar map on the inner product; the bias lets near-zero
            # inner products decode to confident "no edge" logits.
            self.adj_mlp = nn.Sequential(
                nn.Linear(1, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, 1),
            )
        elif adj_decoder == "inner_product":
            self.adj_mlp = None
        else:
            raise ValueError(f"Unsupported adj_decoder {adj_decoder!r}; expected 'inner_product' or 'mlp'.")

    def forward(self, Z: Tensor) -> tuple[Tensor, Tensor]:
        # Inner product decoder for adjacency: (N, Z) x (Z, N) -> (N, N)
        A_logits = torch.bmm(Z, Z.transpose(1, 2))
        if self.adj_mlp is not None:
            A_logits = self.adj_mlp(A_logits.unsqueeze(-1)).squeeze(-1)

        # Get latent shape
        B, N, d = Z.shape

        # Flatten for node-wide decoding
        Z_flat = Z.reshape(B * N, d)

        # Node wise decoder MLP for features: (B, N, Z) -> (B, N, F)
        X_logits = self.fc_features(Z_flat).reshape(B, N, -1)

        return A_logits, X_logits


class GIN_VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_gin_layers: int = 3,
        norm: str = "layer",
        adj_decoder: str = "inner_product",
    ):
        super().__init__()
        self.encoder = GINEncoder(input_dim, hidden_dim, n_layers=n_gin_layers, norm=norm)
        self.vae_heads = VAEHeads(hidden_dim, latent_dim)
        self.decoder = ADSGDecoder(latent_dim, input_dim, adj_decoder=adj_decoder)
        self.norm = norm
        self.adj_decoder = adj_decoder

    def encode(self, x: Tensor, A: Tensor) -> tuple[Tensor, Tensor]:
        h, _ = self.encoder(x, A)
        return self.vae_heads(h)

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    @staticmethod
    def graph_embedding(mu: Tensor) -> Tensor:
        # arch2vec-style: sum the node codes over the whole padded graph.
        return mu.sum(dim=1)

    def forward(
        self,
        x: Tensor,
        A: Tensor,
    ) -> dict[str, Tensor]:
        mu, log_var = self.encode(x, A)
        Z = self.reparametrize(mu, log_var)
        z_G = self.graph_embedding(mu)
        A_logits, X_logits = self.decoder(Z)

        return {
            "z_G":      z_G,
            "mu":       mu,
            "log_var":  log_var,
            "A_logits": A_logits,
            "X_logits": X_logits,
        }
