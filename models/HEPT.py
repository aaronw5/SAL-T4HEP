import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantile_partition(sorted_indices: torch.Tensor, num_regions: torch.Tensor) -> torch.Tensor:
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(
        torch.tensor(float(total_elements), device=sorted_indices.device) / num_regions
    )
    inverse_indices = torch.argsort(sorted_indices, dim=-1)
    base = torch.arange(total_elements, device=sorted_indices.device, dtype=region_size.dtype)[
        None
    ]
    region_indices = torch.floor(base / region_size).long() + 1
    return region_indices[:, inverse_indices]


def get_regions(
    num_regions: int,
    num_or_hashes: int,
    num_heads: int,
    num_and_hashes: int = 2,
) -> torch.Tensor:
    lb = 2.0
    ub = 2.0 * num_regions ** (1.0 / num_and_hashes) - lb
    regions = []
    for _ in range(num_or_hashes * num_heads):
        region = []
        for _ in range(num_and_hashes):
            a = torch.rand(1).item() * (ub - lb) + lb
            region.append(a)
        regions.append(region)
    regions = torch.tensor(regions, dtype=torch.float32)
    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1.0 / num_and_hashes) * regions
    regions = torch.round(regions * 3) / 3
    return regions.view(num_heads, num_or_hashes, num_and_hashes).permute(1, 2, 0).contiguous()


def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    arange = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return torch.empty_like(perm).scatter_(-1, perm, arange)


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    last_dim = values.shape[-1]
    expanded_indices = indices.unsqueeze(-1).expand(*indices.shape, last_dim)
    expanded_values = values.expand(*expanded_indices.shape[:-2], *values.shape[-2:])
    return expanded_values.gather(-2, expanded_indices)


def uniform(a: float, b: float, shape: Tuple[int, ...], device: Optional[torch.device] = None) -> torch.Tensor:
    return (b - a) * torch.rand(shape, device=device) + a


@torch.no_grad()
def lsh_mapping(
    e2lsh: "E2LSH", queries: torch.Tensor, keys: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    max_hash_shift = torch.max(
        queries_hashed.max(-1, keepdim=True).values,
        keys_hashed.max(-1, keepdim=True).values,
    )
    min_hash_shift = torch.min(
        queries_hashed.min(-1, keepdim=True).values,
        keys_hashed.min(-1, keepdim=True).values,
    )
    hash_shift = max_hash_shift - min_hash_shift
    return queries_hashed, keys_hashed, hash_shift


class E2LSH(nn.Module):
    def __init__(self, n_hashes: int, n_heads: int, dim: int, r: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.normal(0, 1, (n_heads, dim, n_hashes)), requires_grad=False)
        self.beta = nn.Parameter(uniform(0, r, shape=(1, n_hashes)), requires_grad=False)

    def forward(self, vecs: torch.Tensor) -> torch.Tensor:
        projection = torch.bmm(vecs, self.alpha)
        return projection.permute(2, 0, 1).contiguous()


def sort_to_buckets(x: torch.Tensor, perm: torch.Tensor, bucket_size: int) -> torch.Tensor:
    selected = batched_index_select(x.unsqueeze(0), perm)
    n_hashes, num_heads, seq_len, dim = selected.shape
    return selected.view(n_hashes, num_heads, seq_len // bucket_size, bucket_size, dim)


def unsort_from_buckets(bucketed_x: torch.Tensor, perm_inverse: torch.Tensor) -> torch.Tensor:
    n_hashes, num_heads, num_buckets, bucket_size, dim = bucketed_x.shape
    flat_x = bucketed_x.view(n_hashes, num_heads, num_buckets * bucket_size, dim)
    return batched_index_select(flat_x, perm_inverse)


def qkv_res(
    sorted_query: torch.Tensor, sorted_key: torch.Tensor, sorted_value: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_sq_05 = -0.5 * (sorted_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (sorted_key**2).sum(dim=-1, keepdim=True)
    clustered_dists = torch.einsum("...id,...jd->...ij", sorted_query, sorted_key)
    clustered_dists = (
        clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)
    ).clamp(max=0.0).exp()
    denom = clustered_dists.sum(dim=-1, keepdim=True) + 1e-20
    so = torch.einsum("...ij,...jd->...id", clustered_dists, sorted_value)
    return denom, so


def prep_qk(
    query: torch.Tensor, key: torch.Tensor, w: torch.Tensor, coords: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    qw_expanded = torch.cat([qw[:, :1], qw], dim=-1)
    sqrt_w_r = torch.sqrt(2.0 * qw_expanded).unsqueeze(0) * coords.unsqueeze(1)
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    return q_hat, k_hat


@torch.no_grad()
def get_geo_shift(
    regions_h: torch.Tensor,
    hash_shift: torch.Tensor,
    region_indices: Tuple[torch.Tensor, torch.Tensor],
    num_or_hashes: int,
) -> torch.Tensor:
    region_indices_eta, region_indices_phi = region_indices
    q_hash_shift_eta = region_indices_eta * hash_shift
    k_hash_shift_eta = region_indices_eta * hash_shift
    eta_scales = torch.ceil(regions_h[0][:, None]) + 1
    q_hash_shift_phi = region_indices_phi * hash_shift * eta_scales
    k_hash_shift_phi = region_indices_phi * hash_shift * eta_scales
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    num_heads = hash_shift.shape[0] // num_or_hashes
    return res.view(2, num_or_hashes, num_heads, -1)


class HEPTAttention(nn.Module):
    def __init__(
        self,
        hash_dim: int,
        hidden_dim: int,
        num_heads: int,
        block_size: int,
        n_hashes: int,
        num_w_per_dist: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.n_hashes = n_hashes
        self.num_w_per_dist = num_w_per_dist
        self.out_linear = nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim)
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def _build_region_indices(
        self, coords: torch.Tensor, regions: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        sorted_eta_idx = torch.argsort(coords[:, 0], dim=-1)
        sorted_phi_idx = torch.argsort(coords[:, 1], dim=-1)
        regions_h = regions.permute(1, 0, 2).reshape(2, -1)
        region_indices_eta = quantile_partition(sorted_eta_idx, regions_h[0][:, None])
        region_indices_phi = quantile_partition(sorted_phi_idx, regions_h[1][:, None])
        return (region_indices_eta, region_indices_phi), regions_h

    def _forward_single(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        raw_size: int,
        regions: torch.Tensor,
        w_rpe: nn.Linear,
    ) -> torch.Tensor:
        seq_len = query.shape[0]
        raw_size = max(int(raw_size), 1)
        query = query[:raw_size].reshape(raw_size, self.num_heads, self.hidden_dim)
        key = key[:raw_size].reshape(raw_size, self.num_heads, self.hidden_dim)
        value = value[:raw_size].reshape(raw_size, self.num_heads, self.hidden_dim)
        coords = coords[:raw_size]

        padded_len = int(math.ceil(raw_size / self.block_size) * self.block_size)
        if padded_len > raw_size:
            pad_tokens = padded_len - raw_size
            query = F.pad(query, (0, 0, 0, 0, 0, pad_tokens))
            key = F.pad(key, (0, 0, 0, 0, 0, pad_tokens))
            value = F.pad(value, (0, 0, 0, 0, 0, pad_tokens))
            coords_pad = torch.full(
                (pad_tokens, coords.shape[-1]),
                float("inf"),
                device=coords.device,
                dtype=coords.dtype,
            )
            coords = torch.cat([coords, coords_pad], dim=0)

        region_indices, regions_h = self._build_region_indices(coords, regions)
        valid_token_mask = (
            torch.arange(padded_len, device=coords.device) < raw_size
        ).to(coords.dtype)
        coords_for_attention = coords * valid_token_mask.unsqueeze(-1)

        w = w_rpe.weight.view(
            self.num_heads,
            self.hidden_dim,
            coords.shape[-1] - 1,
            self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, coords_for_attention)
        q_hat = q_hat.permute(1, 0, 2).contiguous()
        k_hat = k_hat.permute(1, 0, 2).contiguous()
        value = value.permute(1, 0, 2).contiguous()

        valid_head_mask = valid_token_mask.view(1, padded_len, 1)
        q_hat = q_hat * valid_head_mask
        k_hat = k_hat * valid_head_mask
        value = value * valid_head_mask

        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)
        hash_shift = hash_shift.reshape(-1, hash_shift.shape[-1])
        invalid_hash_mask = ~valid_token_mask.bool().view(1, 1, padded_len)
        q_hashed = q_hashed.masked_fill(invalid_hash_mask, float("inf"))
        k_hashed = k_hashed.masked_fill(invalid_hash_mask, float("inf"))

        q_shifts, k_shifts = get_geo_shift(regions_h, hash_shift, region_indices, self.n_hashes)
        q_hashed = q_hashed + q_shifts
        k_hashed = k_hashed + k_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        sorted_query = sort_to_buckets(q_hat, q_positions, self.block_size)
        sorted_key = sort_to_buckets(k_hat, k_positions, self.block_size)
        sorted_value = sort_to_buckets(value, k_positions, self.block_size)

        denom, so = qkv_res(sorted_query, sorted_key, sorted_value)
        q_rev_positions = invert_permutation(q_positions)
        out = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(denom, q_rev_positions)
        out = out.sum(dim=0) / (logits.sum(dim=0) + 1e-9)
        out = self.out_linear(out.permute(1, 0, 2).reshape(padded_len, self.num_heads * self.hidden_dim))
        out = out[:raw_size]

        if seq_len == raw_size:
            return out
        return F.pad(out, (0, 0, 0, seq_len - raw_size))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        raw_sizes: torch.Tensor,
        regions: torch.Tensor,
        w_rpe: nn.Linear,
    ) -> torch.Tensor:
        outputs = []
        for idx in range(query.shape[0]):
            outputs.append(
                self._forward_single(
                    query[idx],
                    key[idx],
                    value[idx],
                    coords[idx],
                    int(raw_sizes[idx].item()),
                    regions,
                    w_rpe,
                )
            )
        return torch.stack(outputs, dim=0)


class HEPTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        block_size: int,
        n_hashes: int,
        num_regions: int,
        num_w_per_dist: int,
        coords_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.coords_dim = coords_dim
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.w_q = nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim * num_heads, bias=False)
        self.w_rpe = nn.Linear(num_w_per_dist * (coords_dim - 1), hidden_dim * num_heads)
        self.attn = HEPTAttention(
            hash_dim=hidden_dim + coords_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            block_size=block_size,
            n_hashes=n_hashes,
            num_w_per_dist=num_w_per_dist,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("regions", get_regions(num_regions, n_hashes, num_heads), persistent=False)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, raw_sizes: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.attn(
            self.w_q(x_norm),
            self.w_k(x_norm),
            self.w_v(x_norm),
            coords=coords,
            raw_sizes=raw_sizes,
            regions=self.regions.to(coords.device),
            w_rpe=self.w_rpe,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class HEPTClassifier(nn.Module):
    def __init__(
        self,
        num_particles: int,
        feature_dim: int,
        hidden_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 1,
        output_dim: int = 5,
        block_size: int = 8,
        n_hashes: int = 4,
        num_regions: int = 16,
        num_w_per_dist: int = 4,
        dropout: float = 0.1,
        aggregation: str = "max",
    ) -> None:
        super().__init__()
        self.num_particles = num_particles
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.coords_dim = 3

        self.feat_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                HEPTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    n_hashes=n_hashes,
                    num_regions=num_regions,
                    num_w_per_dist=num_w_per_dist,
                    coords_dim=self.coords_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        readout_dim = hidden_dim * (num_layers + 1)
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, max(64, hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden_dim), output_dim),
        )

    def _build_coords(self, x: torch.Tensor) -> torch.Tensor:
        eta = x[..., 1]
        phi = x[..., 2]
        delta_r = torch.sqrt(eta.square() + phi.square() + 1e-9)
        return torch.stack([eta, phi, delta_r], dim=-1)

    def _masked_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        if self.aggregation == "mean":
            weights = mask.to(x.dtype)
            denom = weights.sum(dim=1).clamp_min(1.0)
            return (x * weights).sum(dim=1) / denom
        if self.aggregation != "max":
            raise ValueError("aggregation must be 'max' or 'mean'")
        masked = x.masked_fill(~mask, torch.finfo(x.dtype).min)
        pooled = masked.max(dim=1).values
        empty_rows = ~mask.squeeze(-1).any(dim=1)
        return torch.where(empty_rows.unsqueeze(-1), torch.zeros_like(pooled), pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[..., 0] > 0
        raw_sizes = mask.sum(dim=1).clamp_min(1)
        coords = self._build_coords(x)
        encoded = self.feat_encoder(x)
        all_states = [encoded]
        for layer in self.layers:
            encoded = layer(encoded, coords, raw_sizes)
            all_states.append(encoded)
        encoded = torch.cat(all_states, dim=-1)
        pooled = self._masked_pool(encoded, mask)
        return self.readout(pooled)

    def get_config(self) -> Dict[str, int | float | str]:
        return {
            "num_particles": self.num_particles,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "aggregation": self.aggregation,
            "num_layers": len(self.layers),
        }
