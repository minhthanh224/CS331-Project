"""DSCNet (arXiv:2405.04913) training, CAM/LayerCAM extraction, and evaluation utilities.

Implements the Dual-stream Contextual Information Contrastive Learning Network (DSCNet)
from:
  "Weakly-supervised Semantic Segmentation via Dual-stream Contrastive Learning of
   Cross-image Contextual Information" (Lai & Vong, arXiv:2405.04913)

Core components:
1. Backbone encoder (ResNet-101 with optional `layer4 stride=1`) + initial CAM generator O_i (Eq. 1-2).
2. PGCL (Pixel-wise Group Contrastive Learning Module, Eq. 4-7): Pearson-correlation-guided
   disjoint pixel grouping per image + cross-image group-level InfoNCE contrastive loss.
3. SGCL (Semantic-wise Graph Contrastive Learning Module, Eq. 8-9): GCN over semantic class nodes
   and cross-image semantic-group graph contrastive loss.
4. DSCL (Dual-Stream Contrastive Learning refinement, Eq. 10-13): Context-enriched feature fusion
   f_tilde = softmax(f @ C^T) @ S, concatenation f_hat = [f, f_tilde], refined CAM O_hat, and
   joint objective L_DSCL = alpha * L_PGCL + beta * L_SGCL + L_CE (alpha=0.6, beta=0.4).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

from .cam import ensure_cam_size, gen_layercam_resnet
from .constants import IMAGENET_MEAN, IMAGENET_STD, VOC_CLASSES
from .engine import normalize_batch
from .models import build_classifier, get_classifier_head
from .pipeline import (
    BackbonePipeline,
    CamFn,
    MaskFn,
    make_cam_mask_fn,
    make_sam_mask_fn,
)


def forward_resnet_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a torchvision ResNet forward pass returning `(logits, layer4_features)`."""
    if not hasattr(model, "layer4") or not hasattr(model, "fc"):
        raise ValueError(f"forward_resnet_features requires a ResNet model, got {type(model).__name__}")
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    feat = model.layer4(x)
    pooled = model.avgpool(feat)
    logits = model.fc(torch.flatten(pooled, 1))
    return logits, feat


def pearson_center_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Mean-center and L2-normalize `x` along `dim` so dot products equal Pearson correlations."""
    centered = x - x.mean(dim=dim, keepdim=True)
    return F.normalize(centered, dim=dim, eps=eps)


def compute_pixel_context_matrix(feat_flat: torch.Tensor) -> torch.Tensor:
    """Compute pairwise spatial Pearson correlation matrix R_i in [-1, 1]^(V x V) (Eq. 4)."""
    z = pearson_center_normalize(feat_flat, dim=-1)
    return torch.matmul(z, z.T)


def compute_initial_cams(
    feat_raw: torch.Tensor,
    backbone: nn.Module,
    targets: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    pred_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate initial CAM probabilities O_i (Eq. 1) and pseudo-labels y_i (Eq. 2).

    Returns:
        raw_cam: (B, K, H_f, W_f) unnormalized class activation logits.
        cam_probs: (B, K + 1, H_f, W_f) [bg, fg_1, ..., fg_K] normalized in [0, 1].
        pseudo_labels: (B, H_f, W_f) int64 argmax(cam_probs) in {0, ..., K}.
        present_mask: (B, K) float32 binary mask of active classes per image.
    """
    head = get_classifier_head(backbone)
    feat_f32 = feat_raw.float()
    raw_cam = F.conv2d(feat_f32, head.weight.float()[:, :, None, None], bias=None)

    B, K, _, _ = raw_cam.shape
    if targets is not None:
        present = (targets > 0.5).to(raw_cam.dtype)
    elif logits is not None:
        probs = torch.sigmoid(logits.detach().float())
        present = (probs >= pred_threshold).to(raw_cam.dtype)
        # Fallback to top-1 class if no class crosses threshold
        empty_rows = present.sum(dim=1) < 0.5
        if empty_rows.any():
            top1 = probs.argmax(dim=1)
            present[empty_rows, top1[empty_rows]] = 1.0
    else:
        present = torch.ones((B, K), dtype=raw_cam.dtype, device=raw_cam.device)

    present_4d = present[:, :, None, None]
    cam_fg = F.relu(raw_cam) * present_4d
    cam_max = cam_fg.amax(dim=(-2, -1), keepdim=True)
    # If an active class happens to have all-negative logits early in training, fall back to shifted min-max
    zero_active = (present_4d > 0.5) & (cam_max < 1e-5)
    if zero_active.any():
        raw_min = raw_cam.amin(dim=(-2, -1), keepdim=True)
        raw_shifted = (raw_cam - raw_min) * present_4d
        cam_fg = torch.where(zero_active, raw_shifted, cam_fg)
        cam_max = cam_fg.amax(dim=(-2, -1), keepdim=True)
    cam_fg = (cam_fg / cam_max.clamp_min(1e-5)) * present_4d
    cam_bg = (1.0 - cam_fg.amax(dim=1, keepdim=True)).clamp(0.0, 1.0)

    cam_probs = torch.cat([cam_bg, cam_fg], dim=1)
    pseudo_labels = cam_probs.argmax(dim=1)
    return raw_cam, cam_probs, pseudo_labels, present


@torch.no_grad()
def _init_kmeans_centroids(
    z_det: torch.Tensor,
    num_groups: int,
    cam_probs_i: torch.Tensor | None = None,
    present_i: torch.Tensor | None = None,
) -> torch.Tensor:
    """Initialize K-means centroids deterministically using active CAM peaks + farthest-point fill."""
    V = z_det.shape[0]
    chosen: list[int] = []
    used = set()

    if cam_probs_i is not None:
        # cam_probs_i: (K + 1, V) where index 0 is background
        active_channels = [0]
        if present_i is not None:
            fg_idx = torch.where(present_i > 0.5)[0].tolist()
            active_channels.extend([idx + 1 for idx in fg_idx])
        else:
            active_channels.extend(range(1, cam_probs_i.shape[0]))

        for ch in active_channels:
            if len(chosen) >= num_groups:
                break
            order = torch.argsort(cam_probs_i[ch], descending=True)
            for idx_t in order.tolist():
                if idx_t not in used:
                    chosen.append(idx_t)
                    used.add(idx_t)
                    break

    if not chosen:
        first_idx = int(z_det.norm(dim=-1).argmax().item())
        chosen.append(first_idx)
        used.add(first_idx)

    # Farthest-point seeding in Pearson similarity space for any remaining slots
    while len(chosen) < num_groups and len(chosen) < V:
        cur = z_det[chosen]  # (m, D)
        max_sim = torch.matmul(z_det, cur.T).amax(dim=1)
        if len(used) > 0:
            max_sim = max_sim.clone()
            for u_idx in used:
                max_sim[u_idx] = 1e9
        next_idx = int(max_sim.argmin().item())
        if next_idx in used:
            break
        chosen.append(next_idx)
        used.add(next_idx)

    return z_det[chosen].clone()


def cluster_pixel_groups_single(
    feat_flat: torch.Tensor,
    num_groups: int,
    cam_probs_i: torch.Tensor | None = None,
    present_i: torch.Tensor | None = None,
    n_iters: int = 6,
    pos_corr_thr: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition one image's `(V, D)` feature map into disjoint pixel groups (Eq. 4-5).

    Uses Pearson-correlation-guided K-means to form disjoint groups {G_{i,u}}, then computes
    differentiable group prototypes p_{i,u} by averaging positively correlated pixels in each group.

    Returns:
        prototypes: (M_i, D) tensor (differentiable w.r.t. `feat_flat`).
        assign: (V,) int64 cluster IDs in [0, M_i - 1].
        group_masks: (M_i, V) float32 one-hot cluster membership matrix.
    """
    V, D = feat_flat.shape
    k = max(2, min(int(num_groups), V))
    feat_f32 = feat_flat.float()

    with torch.no_grad():
        z_det = pearson_center_normalize(feat_f32.detach(), dim=-1)
        centroids = _init_kmeans_centroids(
            z_det, k, cam_probs_i=cam_probs_i, present_i=present_i
        )
        k_eff = centroids.shape[0]
        assign = torch.zeros(V, dtype=torch.long, device=feat_flat.device)

        for _ in range(max(1, n_iters)):
            sims = torch.matmul(z_det, centroids.T)  # (V, k_eff)
            assign = sims.argmax(dim=1)

            counts = torch.bincount(assign, minlength=k_eff)
            # Re-seed any empty cluster to the pixel with weakest assignment similarity
            if (counts == 0).any():
                best_sim = sims.amax(dim=1)
                worst_pixels = torch.argsort(best_sim, descending=False).tolist()
                ptr = 0
                for cid in range(k_eff):
                    if counts[cid] == 0 and ptr < len(worst_pixels):
                        p_idx = worst_pixels[ptr]
                        assign[p_idx] = cid
                        ptr += 1

            sums = torch.zeros((k_eff, D), dtype=z_det.dtype, device=z_det.device)
            sums.index_add_(0, assign, z_det)
            counts_f = torch.bincount(assign, minlength=k_eff).clamp_min(1).to(z_det.dtype).unsqueeze(1)
            centroids = F.normalize(sums / counts_f, dim=-1, eps=1e-6)

        # Compact non-empty clusters to contiguous IDs [0, M_i - 1]
        unique_ids, inv_assign = torch.unique(assign, sorted=True, return_inverse=True)
        M_i = int(unique_ids.numel())
        assign = inv_assign

        # Eq. (4)-(5): identify positively correlated pixels (Pearson corr >= pos_corr_thr) per group
        group_onehot = F.one_hot(assign, num_classes=M_i).to(feat_f32.dtype)  # (V, M_i)
        counts_m = group_onehot.sum(dim=0).clamp_min(1.0).unsqueeze(1)  # (M_i, 1)
        z_means = F.normalize(torch.matmul(group_onehot.T, z_det) / counts_m, dim=-1, eps=1e-6)
        pixel_group_corr = (z_det * z_means[assign]).sum(dim=-1)  # (V,)
        pos_mask = pixel_group_corr >= pos_corr_thr  # (V,)

        # Fall back to all group pixels for any group with zero pixels passing pos_corr_thr
        pos_weights = group_onehot * pos_mask.to(feat_f32.dtype).unsqueeze(1)  # (V, M_i)
        pos_counts = pos_weights.sum(dim=0)  # (M_i,)
        use_all = pos_counts < 1.0
        if use_all.any():
            pos_weights = torch.where(use_all.unsqueeze(0), group_onehot, pos_weights)

        weight_norm = pos_weights / pos_weights.sum(dim=0, keepdim=True).clamp_min(1.0)  # (V, M_i)

    # Differentiable pooling from `feat_f32` -> prototypes (M_i, D)
    prototypes = torch.matmul(weight_norm.T, feat_f32)
    return prototypes, assign, group_onehot.T


def build_voc_cooccurrence_matrix(
    labels_df: pd.DataFrame,
    classes: tuple[str, ...] = VOC_CLASSES,
) -> torch.Tensor:
    """Build a `(K + 1, K + 1)` symmetric semantic co-occurrence prior from image-level labels."""
    mat = labels_df[list(classes)].to_numpy(dtype=np.float32)
    mat = (mat > 0.5).astype(np.float32)
    K = mat.shape[1]
    adj = np.zeros((K + 1, K + 1), dtype=np.float32)

    if len(mat) > 0:
        co = mat.T @ mat  # (K, K)
        freq = np.maximum(mat.sum(axis=0), 1.0)
        norm_co = co / np.sqrt(np.outer(freq, freq))
        adj[1:, 1:] = norm_co
        # Background node weakly connects to all foreground classes
        fg_share = np.clip(freq / float(len(mat)), 0.05, 1.0)
        adj[0, 1:] = 0.25 * fg_share
        adj[1:, 0] = 0.25 * fg_share

    np.fill_diagonal(adj, 1.0)
    return torch.from_numpy(adj)


class SemanticGraphModule(nn.Module):
    """SGCL Semantic Graph Convolutional Network over `(K + 1)` category nodes (Fig. 2 & Eq. 8)."""

    def __init__(
        self,
        num_classes: int = 20,
        embed_dim: int = 256,
        graph_temp: float = 0.2,
        cooccur_adj: torch.Tensor | None = None,
        cooccur_momentum: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_nodes = self.num_classes + 1  # 0: background, 1..K: foreground classes
        self.embed_dim = int(embed_dim)
        self.graph_temp = max(float(graph_temp), 1e-3)
        self.cooccur_momentum = float(cooccur_momentum)

        self.class_nodes = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim) * 0.02)
        nn.init.orthogonal_(self.class_nodes)

        self.gcn1 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.gcn2 = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.node_norm = nn.LayerNorm(self.embed_dim)

        if cooccur_adj is not None:
            adj_init = cooccur_adj.float().clone()
            if adj_init.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(f"Expected cooccur_adj shape {(self.num_nodes, self.num_nodes)}, got {adj_init.shape}")
        else:
            adj_init = torch.eye(self.num_nodes, dtype=torch.float32)
        self.register_buffer("cooccur_adj", adj_init)

    @torch.no_grad()
    def update_cooccurrence(self, targets: torch.Tensor) -> None:
        """Update cross-image semantic co-occurrence graph using batch image-level labels."""
        if targets is None or targets.numel() == 0:
            return
        y = (targets.detach().float() > 0.5).float()  # (B, K)
        B = float(y.shape[0])
        co = torch.matmul(y.T, y)
        freq = y.sum(dim=0).clamp_min(1.0)
        batch_adj = torch.zeros_like(self.cooccur_adj)
        batch_adj[1:, 1:] = co / torch.sqrt(torch.outer(freq, freq))
        bg_edge = (y.sum(dim=0) / max(B, 1.0)).clamp(0.05, 1.0) * 0.25
        batch_adj[0, 1:] = bg_edge
        batch_adj[1:, 0] = bg_edge
        batch_adj.fill_diagonal_(1.0)

        m = self.cooccur_momentum
        self.cooccur_adj.mul_(1.0 - m).add_(m * batch_adj)

    def get_graph_nodes(self) -> torch.Tensor:
        """Run 2-layer residual GCN over semantic nodes returning normalized `(K + 1, D)` nodes."""
        nodes_norm = F.normalize(self.class_nodes.float(), dim=-1, eps=1e-6)
        dyn_adj = F.softmax(torch.matmul(nodes_norm, nodes_norm.T) / 0.5, dim=-1)

        prior = self.cooccur_adj.float()
        deg_inv_sqrt = prior.sum(dim=-1).clamp_min(1e-6).pow(-0.5)
        norm_prior = deg_inv_sqrt[:, None] * prior * deg_inv_sqrt[None, :]

        adj = 0.5 * norm_prior + 0.5 * dyn_adj
        h = self.act(self.gcn1(torch.matmul(adj, self.class_nodes.float())))
        h = self.gcn2(torch.matmul(adj, h))
        out = self.node_norm(self.class_nodes.float() + h)
        return F.normalize(out, dim=-1, eps=1e-6)

    def forward(
        self,
        group_prototypes: torch.Tensor,
        graph_nodes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute group-to-category similarity M_i and semantic context S_i (Eq. 8).

        Args:
            group_prototypes: (M_i, D) pixel group vectors p_{i,u}.
        Returns:
            sim_matrix: (M_i, K + 1) probabilities M_{i,u}^k.
            sem_context: (M_i, D) semantic-wise context representations S_i.
            graph_nodes: (K + 1, D) GCN-refined category prototypes.
        """
        if graph_nodes is None:
            graph_nodes = self.get_graph_nodes()
        p_norm = F.normalize(group_prototypes.float(), dim=-1, eps=1e-6)
        logits = torch.matmul(p_norm, graph_nodes.T) / self.graph_temp
        sim_matrix = F.softmax(logits, dim=-1)  # (M_i, K + 1)
        sem_context = torch.matmul(sim_matrix, graph_nodes)  # (M_i, D)
        return sim_matrix, sem_context, graph_nodes


class DualStreamContextRefinement(nn.Module):
    """DSCNet Dual-Stream Contrastive Context Module (PGCL + SGCL + Eq. 10-11 CAM refinement)."""

    def __init__(
        self,
        in_channels: int = 2048,
        embed_dim: int = 256,
        num_classes: int = 20,
        kmeans_iters: int = 6,
        pos_corr_thr: float = 0.5,
        graph_temp: float = 0.2,
        max_groups: int = 12,
        cooccur_adj: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.kmeans_iters = int(kmeans_iters)
        self.pos_corr_thr = float(pos_corr_thr)
        self.max_groups = int(max_groups)

        # Dense feature projection (B, in_channels, H_f, W_f) -> (B, D, H_f, W_f)
        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
        )

        self.semantic_graph = SemanticGraphModule(
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            graph_temp=graph_temp,
            cooccur_adj=cooccur_adj,
        )

        self.enriched_norm = nn.BatchNorm2d(self.embed_dim)
        # Eq. (11): f_hat = [f_i, f_tilde] -> O_hat = F_CAM(f_hat)
        self.refined_cam_conv = nn.Conv2d(self.embed_dim * 2, self.num_classes, kernel_size=1, bias=True)

    def forward(
        self,
        feat_raw: torch.Tensor,
        cam_probs: torch.Tensor,
        present_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Run PGCL grouping, SGCL graph mapping, and DSCL feature/CAM refinement (Eq. 10-11)."""
        with torch.amp.autocast(device_type=feat_raw.device.type, enabled=False):
            feat_raw_f32 = feat_raw.float()
            cam_probs_f32 = cam_probs.float()
            present_f32 = present_mask.float()

            if self.training and targets is not None:
                self.semantic_graph.update_cooccurrence(targets)

            feat_embed = self.proj(feat_raw_f32).float()  # (B, D, H_f, W_f)
            B, D, H_f, W_f = feat_embed.shape
            V = H_f * W_f
            scale = D ** 0.5

            graph_nodes = self.semantic_graph.get_graph_nodes()  # (K + 1, D)
            cam_flat = cam_probs_f32.detach().view(B, self.num_classes + 1, V)

            enriched_list: list[torch.Tensor] = []
            groups_list: list[torch.Tensor] = []
            assigns_list: list[torch.Tensor] = []
            sim_matrices: list[torch.Tensor] = []
            sem_contexts: list[torch.Tensor] = []

            for b in range(B):
                f_b = feat_embed[b].permute(1, 2, 0).reshape(V, D)  # (V, D)
                n_active = int((present_f32[b] > 0.5).sum().item())
                num_groups = min(max(2, n_active + 1), self.max_groups, V)

                prototypes_b, assign_b, _ = cluster_pixel_groups_single(
                    f_b,
                    num_groups=num_groups,
                    cam_probs_i=cam_flat[b],
                    present_i=present_f32[b],
                    n_iters=self.kmeans_iters,
                    pos_corr_thr=self.pos_corr_thr,
                )
                sim_b, sem_b, _ = self.semantic_graph(prototypes_b, graph_nodes=graph_nodes)

                # Eq. (10): f_tilde = softmax(f_i @ C_i^T / sqrt(D)) @ S_i
                attn_b = F.softmax(torch.matmul(f_b, prototypes_b.T) / scale, dim=-1)  # (V, M_b)
                f_tilde_b = torch.matmul(attn_b, sem_b)  # (V, D)
                f_tilde_b = f_tilde_b.T.reshape(D, H_f, W_f)

                enriched_list.append(f_tilde_b)
                groups_list.append(prototypes_b)
                assigns_list.append(assign_b)
                sim_matrices.append(sim_b)
                sem_contexts.append(sem_b)

            feat_enriched = self.enriched_norm(torch.stack(enriched_list, dim=0))  # (B, D, H_f, W_f)
            feat_concat = torch.cat([feat_embed, feat_enriched], dim=1)  # Eq. (11): f_hat = [f_i, f_tilde]
            cam_refined = self.refined_cam_conv(feat_concat)  # (B, K, H_f, W_f)
            logits_refined = F.adaptive_avg_pool2d(cam_refined, (1, 1)).flatten(1)  # (B, K)

            return {
                "feat_embed": feat_embed,
                "feat_enriched": feat_enriched,
                "feat_concat": feat_concat,
                "cam_refined": cam_refined,
                "logits_refined": logits_refined,
                "groups": groups_list,
                "assigns": assigns_list,
                "sim_matrices": sim_matrices,
                "sem_contexts": sem_contexts,
                "graph_nodes": graph_nodes,
            }


class DSCNetClassifier(nn.Module):
    """End-to-end DSCNet model wrapping a ResNet-101 backbone and Dual-Stream Context Refinement."""

    def __init__(
        self,
        backbone_name: str = "resnet101",
        num_classes: int = 20,
        embed_dim: int = 256,
        layer4_stride1: bool = True,
        pretrained: bool = True,
        kmeans_iters: int = 6,
        pos_corr_thr: float = 0.5,
        graph_temp: float = 0.2,
        max_groups: int = 12,
        pred_cls_threshold: float = 0.5,
        cooccur_adj: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = str(backbone_name)
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.layer4_stride1 = bool(layer4_stride1)
        self.pred_cls_threshold = float(pred_cls_threshold)

        self.backbone = build_classifier(
            model_name=self.backbone_name,
            num_classes=self.num_classes,
            layer4_stride1=self.layer4_stride1,
            pretrained=pretrained,
        )
        in_channels = get_classifier_head(self.backbone).in_features
        self.dscl = DualStreamContextRefinement(
            in_channels=in_channels,
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            kmeans_iters=kmeans_iters,
            pos_corr_thr=pos_corr_thr,
            graph_temp=graph_temp,
            max_groups=max_groups,
            cooccur_adj=cooccur_adj,
        )

    def forward_dscnet(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Full DSCNet forward pass returning initial/refined logits, CAMs, and DSCL intermediates."""
        logits_init, feat_raw = forward_resnet_features(self.backbone, x)
        raw_cam_init, cam_probs_init, pseudo_init, present_mask = compute_initial_cams(
            feat_raw,
            self.backbone,
            targets=targets,
            logits=logits_init,
            pred_threshold=self.pred_cls_threshold,
        )
        dscl_out = self.dscl(
            feat_raw,
            cam_probs=cam_probs_init,
            present_mask=present_mask,
            targets=targets,
        )
        logits = 0.5 * logits_init.float() + 0.5 * dscl_out["logits_refined"].float()
        return {
            "logits": logits,
            "logits_init": logits_init,
            "feat_raw": feat_raw,
            "raw_cam_init": raw_cam_init,
            "cam_probs_init": cam_probs_init,
            "pseudo_init": pseudo_init,
            "present_mask": present_mask,
            **dscl_out,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard classifier forward pass returning `(B, num_classes)` logits for eval & LayerCAM."""
        return self.forward_dscnet(x, targets=None)["logits"]

    def forward_cam_maps(self, x: torch.Tensor, use_refined: bool = True) -> torch.Tensor:
        """Return `(B, num_classes, H_f, W_f)` CAM logits (refined O_hat by default, or initial O)."""
        out = self.forward_dscnet(x, targets=None)
        return out["cam_refined"] if use_refined else out["raw_cam_init"]


def _multi_positive_infonce(
    anchors: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute numerically stable multi-positive InfoNCE across a batch of normalized vectors."""
    N = anchors.shape[0]
    if N < 2:
        return anchors.sum() * 0.0

    temp = max(float(temperature), 1e-4)
    norm_vecs = F.normalize(anchors.float(), dim=-1, eps=1e-6)
    sim_logits = torch.matmul(norm_vecs, norm_vecs.T) / temp  # (N, N)

    valid_other = (pos_mask | neg_mask) & (~torch.eye(N, dtype=torch.bool, device=anchors.device))
    has_pos = pos_mask.any(dim=1)
    has_neg = neg_mask.any(dim=1)
    valid_anchors = has_pos & has_neg
    if not valid_anchors.any():
        return anchors.sum() * 0.0

    # Mask out self and excluded columns before log_softmax
    masked_logits = sim_logits.masked_fill(~valid_other, -1e4)
    # Subtract row-max for stability
    row_max = masked_logits.amax(dim=1, keepdim=True).detach()
    shifted = masked_logits - row_max
    exp_logits = torch.exp(shifted) * valid_other.to(shifted.dtype)
    log_denom = torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    log_prob = shifted - log_denom  # (N, N)

    pos_f = pos_mask[valid_anchors].to(log_prob.dtype)
    pos_counts = pos_f.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (pos_f * log_prob[valid_anchors]).sum(dim=1) / pos_counts
    return -mean_log_prob_pos.mean()


@torch.no_grad()
def _assign_group_pseudo_labels(
    assign_b: torch.Tensor,
    cam_probs_b: torch.Tensor,
    num_groups: int,
    present_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assign each disjoint pixel group its semantic category label in `{0, ..., K}`.

    Uses combined cluster-mean and top-quartile foreground CAM affinity so sparse object CAMs
    are not drowned out by `cam_bg` majority pixels inside a cluster, and ensures background
    retains its lowest-foreground cluster while active classes claim their strongest matching groups.
    """
    cam_flat = cam_probs_b.detach().flatten(1)  # (K + 1, V)
    K_plus_1, _ = cam_flat.shape
    labels = torch.zeros(num_groups, dtype=torch.long, device=cam_flat.device)
    if num_groups <= 1:
        return labels

    fg_flat = cam_flat[1:]  # (K, V)
    scores = torch.zeros((num_groups, K_plus_1 - 1), dtype=cam_flat.dtype, device=cam_flat.device)
    for u in range(num_groups):
        mask_u = assign_b == u
        if not mask_u.any():
            continue
        vals = fg_flat[:, mask_u]  # (K, n_u)
        mean_u = vals.mean(dim=1)
        k_top = max(1, vals.shape[1] // 4)
        top_u = vals.topk(k_top, dim=1).values.mean(dim=1)
        scores[u] = 0.5 * mean_u + 0.5 * top_u

    if present_b is not None:
        active_fg = torch.where(present_b > 0.5)[0].tolist()
    else:
        active_fg = [c for c in range(K_plus_1 - 1) if float(fg_flat[c].max().item()) > 1e-4]

    if not active_fg:
        return labels

    # Reserve the group with the weakest overall foreground activation as background (label 0)
    max_fg_per_group = scores[:, active_fg].amax(dim=1)
    bg_group = int(max_fg_per_group.argmin().item())
    used_groups = {bg_group}

    # Greedily match active foreground classes to remaining clusters by highest CAM affinity
    pairs = []
    for u in range(num_groups):
        if u == bg_group:
            continue
        for c in active_fg:
            pairs.append((float(scores[u, c].item()), u, c + 1))
    pairs.sort(key=lambda item: item[0], reverse=True)

    used_classes = set()
    for _, u, cls_lbl in pairs:
        if u in used_groups or cls_lbl in used_classes:
            continue
        labels[u] = cls_lbl
        used_groups.add(u)
        used_classes.add(cls_lbl)

    # Any extra clusters beyond `len(active_fg) + 1` take their strongest active class if >= 0.35 else bg
    for u in range(num_groups):
        if u in used_groups:
            continue
        best_c = max(active_fg, key=lambda c: float(scores[u, c].item()))
        if float(scores[u, best_c].item()) >= 0.35:
            labels[u] = best_c + 1
    return labels


def pgcl_contrastive_loss(
    groups_list: list[torch.Tensor],
    assigns_list: list[torch.Tensor],
    cam_probs_init: torch.Tensor,
    present_mask: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    """PGCL loss (Eq. 6-7): cross-image InfoNCE contrasting disjoint pixel groups across the batch."""
    if not groups_list:
        return cam_probs_init.sum() * 0.0

    all_groups: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for b, (p_b, assign_b) in enumerate(zip(groups_list, assigns_list)):
        if p_b.numel() == 0:
            continue
        pres_b = present_mask[b] if present_mask is not None else None
        lbl_b = _assign_group_pseudo_labels(
            assign_b, cam_probs_init[b], num_groups=p_b.shape[0], present_b=pres_b
        )
        all_groups.append(p_b)
        all_labels.append(lbl_b)

    if len(all_groups) < 2:
        return cam_probs_init.sum() * 0.0

    P_all = torch.cat(all_groups, dim=0)  # (G_tot, D)
    L_all = torch.cat(all_labels, dim=0)  # (G_tot,)
    G_tot = P_all.shape[0]
    if G_tot < 2:
        return P_all.sum() * 0.0

    not_self = ~torch.eye(G_tot, dtype=torch.bool, device=P_all.device)
    same_cls = (L_all.unsqueeze(1) == L_all.unsqueeze(0)) & not_self
    diff_cls = (L_all.unsqueeze(1) != L_all.unsqueeze(0)) & not_self

    # For foreground anchors (L > 0): all same-class groups across the batch are positives.
    # For background anchors (L == 0): positively correlated background groups (cosine sim >= 0)
    # act as positives so visually distinct background textures (sky vs indoor floor) do not collapse,
    # while ALL foreground groups act as negatives.
    with torch.no_grad():
        cos_sim = torch.matmul(
            F.normalize(P_all.detach(), dim=-1, eps=1e-6),
            F.normalize(P_all.detach(), dim=-1, eps=1e-6).T,
        )
    is_bg = L_all == 0
    pos_mask = same_cls.clone()
    if is_bg.any():
        bg_pos_sim = same_cls & (cos_sim >= 0.0)
        # Keep fallback to all same-bg groups if an anchor has no >=0 cosine bg partner
        bg_has_sim = bg_pos_sim.any(dim=1)
        use_filtered_bg = is_bg & bg_has_sim
        pos_mask[use_filtered_bg] = bg_pos_sim[use_filtered_bg]

    return _multi_positive_infonce(P_all, pos_mask=pos_mask, neg_mask=diff_cls, temperature=temperature)


def sgcl_contrastive_loss(
    groups_list: list[torch.Tensor],
    assigns_list: list[torch.Tensor],
    sim_matrices: list[torch.Tensor],
    graph_nodes: torch.Tensor,
    cam_probs_init: torch.Tensor,
    pseudo_init: torch.Tensor,
    present_mask: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    """SGCL loss (Eq. 8-9): semantic-wise graph contrast between pixel groups, graph nodes, and images."""
    temp = max(float(temperature), 1e-4)
    g_norm = F.normalize(graph_nodes.float(), dim=-1, eps=1e-6)  # (K + 1, D)

    # 1. Group-to-semantic-graph node alignment loss
    group_vecs: list[torch.Tensor] = []
    group_lbls: list[torch.Tensor] = []
    sem_cls_vecs: list[torch.Tensor] = []
    sem_cls_lbls: list[torch.Tensor] = []

    for b, (p_b, assign_b, m_b) in enumerate(zip(groups_list, assigns_list, sim_matrices)):
        if p_b.numel() == 0:
            continue
        pres_b = present_mask[b] if present_mask is not None else None
        lbl_b = _assign_group_pseudo_labels(
            assign_b, cam_probs_init[b], num_groups=p_b.shape[0], present_b=pres_b
        )
        group_vecs.append(p_b)
        group_lbls.append(lbl_b)

        # Eq. (8): S_{i,k} = sum_u M_{i,u}^k * p_{i,u} for categories present in group labels / pseudo_init[b]
        present_cats = torch.unique(torch.cat([lbl_b, pseudo_init[b].reshape(-1)]))
        for cat_t in present_cats.tolist():
            weights_k = m_b[:, cat_t].unsqueeze(1)  # (M_b, 1)
            if float(weights_k.sum().item()) > 1e-6:
                s_ik = (weights_k * p_b).sum(dim=0) / weights_k.sum().clamp_min(1e-6)
                sem_cls_vecs.append(s_ik)
                sem_cls_lbls.append(cat_t)

    if not group_vecs:
        return graph_nodes.sum() * 0.0

    P_all = F.normalize(torch.cat(group_vecs, dim=0).float(), dim=-1, eps=1e-6)
    L_all = torch.cat(group_lbls, dim=0).long()

    # Pull pixel groups toward their corresponding GCN semantic node and away from other classes
    node_logits = torch.matmul(P_all, g_norm.T) / temp  # (G_tot, K + 1)
    loss_node = F.cross_entropy(node_logits, L_all)

    # 2. Cross-image semantic-wise contrast over S_{i,k} (Eq. 9)
    loss_cross_sem = P_all.sum() * 0.0
    if len(sem_cls_vecs) >= 2:
        S_all = torch.stack(sem_cls_vecs, dim=0)  # (N_sem, D)
        S_lbls = torch.tensor(sem_cls_lbls, dtype=torch.long, device=S_all.device)
        M_tot = S_all.shape[0]
        not_self = ~torch.eye(M_tot, dtype=torch.bool, device=S_all.device)
        pos_mask = (S_lbls.unsqueeze(1) == S_lbls.unsqueeze(0)) & not_self
        neg_mask = (S_lbls.unsqueeze(1) != S_lbls.unsqueeze(0)) & not_self
        loss_cross_sem = _multi_positive_infonce(S_all, pos_mask=pos_mask, neg_mask=neg_mask, temperature=temp)

    return 0.5 * loss_node + 0.5 * loss_cross_sem


def dscnet_total_loss(
    outputs: dict[str, Any],
    targets: torch.Tensor,
    cls_criterion: nn.Module,
    alpha_pgcl: float = 0.6,
    beta_sgcl: float = 0.4,
    temperature: float = 0.2,
    contrast_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute overall DSCNet objective L_DSCL = alpha * L_PGCL + beta * L_SGCL + L_CE (Eq. 13)."""
    loss_ce_init = cls_criterion(outputs["logits_init"].float(), targets.float())
    loss_ce_refined = cls_criterion(outputs["logits_refined"].float(), targets.float())
    loss_ce = 0.5 * loss_ce_init + 0.5 * loss_ce_refined

    eff_alpha = float(alpha_pgcl) * float(contrast_scale)
    eff_beta = float(beta_sgcl) * float(contrast_scale)

    loss_pgcl = outputs["logits_init"].new_zeros((), dtype=torch.float32)
    if eff_alpha > 0.0:
        loss_pgcl = pgcl_contrastive_loss(
            outputs["groups"],
            outputs["assigns"],
            outputs["cam_probs_init"],
            present_mask=outputs.get("present_mask"),
            temperature=temperature,
        )

    loss_sgcl = outputs["logits_init"].new_zeros((), dtype=torch.float32)
    if eff_beta > 0.0:
        loss_sgcl = sgcl_contrastive_loss(
            outputs["groups"],
            outputs["assigns"],
            outputs["sim_matrices"],
            outputs["graph_nodes"],
            outputs["cam_probs_init"],
            outputs["pseudo_init"],
            present_mask=outputs.get("present_mask"),
            temperature=temperature,
        )

    total = loss_ce + eff_alpha * loss_pgcl + eff_beta * loss_sgcl
    stats = {
        "loss": float(total.item()),
        "ce_loss": float(loss_ce.item()),
        "pgcl_loss": float(loss_pgcl.item()),
        "sgcl_loss": float(loss_sgcl.item()),
    }
    return total, stats


def build_dscnet_optimizer(
    model: DSCNetClassifier,
    backbone_lr: float,
    head_lr: float | None = None,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with separate backbone and head/DSCL learning rates."""
    if head_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=backbone_lr, weight_decay=weight_decay)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.") and not name.startswith("backbone.fc."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def train_dscnet_one_epoch(
    model: DSCNetClassifier,
    loader,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    device: torch.device,
    scheduler=None,
    alpha_pgcl: float = 0.6,
    beta_sgcl: float = 0.4,
    temperature: float = 0.2,
    contrast_scale: float = 1.0,
    use_amp: bool = True,
    scaler=None,
    desc: str = "DSCNet Train",
) -> dict[str, float]:
    """Run one epoch of DSCNet training (`L_CE + alpha * L_PGCL + beta * L_SGCL`)."""
    model.train()
    amp_enabled = bool(use_amp and device.type == "cuda")
    if scaler is None and amp_enabled:
        scaler = torch.amp.GradScaler("cuda")

    tot_loss, tot_ce, tot_pgcl, tot_sgcl, tot_n = 0.0, 0.0, 0.0, 0.0, 0
    progress = tqdm(loader, desc=desc)
    for images, targets in progress:
        inputs = normalize_batch(images).to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model.forward_dscnet(inputs, targets=targets)

        loss, stats = dscnet_total_loss(
            outputs,
            targets,
            cls_criterion=cls_criterion,
            alpha_pgcl=alpha_pgcl,
            beta_sgcl=beta_sgcl,
            temperature=temperature,
            contrast_scale=contrast_scale,
        )

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bs = images.size(0)
        tot_n += bs
        tot_loss += stats["loss"] * bs
        tot_ce += stats["ce_loss"] * bs
        tot_pgcl += stats["pgcl_loss"] * bs
        tot_sgcl += stats["sgcl_loss"] * bs
        progress.set_postfix(
            loss=f"{tot_loss / tot_n:.4f}",
            ce=f"{tot_ce / tot_n:.4f}",
            pgcl=f"{tot_pgcl / tot_n:.4f}",
            sgcl=f"{tot_sgcl / tot_n:.4f}",
        )

    denom = max(tot_n, 1)
    return {
        "loss": tot_loss / denom,
        "ce_loss": tot_ce / denom,
        "pgcl_loss": tot_pgcl / denom,
        "sgcl_loss": tot_sgcl / denom,
    }


def default_dscnet_layercam_layers(model: DSCNetClassifier) -> tuple[str, ...]:
    """Default multi-stage ResNet LayerCAM hook paths for a `DSCNetClassifier`."""
    return ("backbone.layer2", "backbone.layer3", "backbone.layer4")


def gen_dscnet_cam(
    model: DSCNetClassifier,
    input_tensor: torch.Tensor,
    class_indices,
    target_size: tuple[int, int] | None = None,
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = False,
    use_refined: bool = True,
) -> torch.Tensor:
    """Generate DSCNet refined CAM heatmaps (Eq. 11) with optional Multi-Scale + HFlip TTA.

    Matches `gen_cam_resnet` broadcasting semantics:
    1. Multi-class on one image: input (3, H, W) or (1, 3, H, W), class_indices (K,) -> (K, H, W)
    2. Single-class per image on batch: input (B, 3, H, W), class_indices (B,) -> (B, H, W)
    """
    model.eval()
    device = next(model.parameters()).device
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device, non_blocking=True)
    class_idx = torch.as_tensor(class_indices, dtype=torch.long, device=device).view(-1)

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    image = normalize(input_tensor)

    orig_h, orig_w = image.shape[-2:]
    if target_size is None:
        target_size = (orig_h, orig_w)

    def _extract_single(img_t: torch.Tensor, flip_back: bool = False) -> torch.Tensor:
        cam_maps = model.forward_cam_maps(img_t, use_refined=use_refined)  # (B, C, H_f, W_f)
        if img_t.shape[0] == 1:
            selected = cam_maps[0, class_idx].unsqueeze(0)  # (1, K, H_f, W_f)
            cam_relu = torch.relu(selected).squeeze(0).unsqueeze(1)  # (K, 1, H_f, W_f)
        else:
            b_idx = torch.arange(img_t.shape[0], device=device)
            cam_relu = torch.relu(cam_maps[b_idx, class_idx]).unsqueeze(1)  # (B, 1, H_f, W_f)
        if flip_back:
            cam_relu = torch.flip(cam_relu, dims=[-1])
        cam_up = F.interpolate(cam_relu, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
        return cam_up / cam_up.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    with torch.no_grad():
        accum_cam = None
        for s in scales:
            if s == 1.0:
                scaled_img = image
            else:
                sh = max(32, int(round(orig_h * s)))
                sw = max(32, int(round(orig_w * s)))
                scaled_img = F.interpolate(image, size=(sh, sw), mode="bilinear", align_corners=False)

            cam_s = _extract_single(scaled_img, flip_back=False)
            accum_cam = cam_s if accum_cam is None else (accum_cam + cam_s)

            if hflip:
                flipped_img = torch.flip(scaled_img, dims=[-1])
                cam_f = _extract_single(flipped_img, flip_back=True)
                accum_cam = accum_cam + cam_f

        cam = accum_cam / accum_cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return cam


def compute_dscnet_cams(
    model: DSCNetClassifier,
    img_tensor: torch.Tensor,
    class_indices,
    method: str = "cam",
    target_size: tuple[int, int] | None = None,
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = False,
    layers: tuple[str, ...] | None = None,
    shallow_gamma: float = 2.0,
    use_refined: bool = True,
) -> np.ndarray:
    """Compute DSCNet CAM (`method='cam'`) or LayerCAM (`method='layercam'`) as NumPy `(K, H, W)` in `[0, 1]`."""
    device = next(model.parameters()).device
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device, non_blocking=True)
    class_idx = torch.as_tensor(class_indices, dtype=torch.long, device=device).view(-1)

    if method == "cam":
        cams = gen_dscnet_cam(
            model,
            img_tensor,
            class_idx,
            target_size=target_size,
            scales=scales,
            hflip=hflip,
            use_refined=use_refined,
        )
    elif method == "layercam":
        resolved_layers = tuple(layers) if layers is not None else default_dscnet_layercam_layers(model)
        # Accept either ("layer2", ...) or ("backbone.layer2", ...) seamlessly
        resolved_layers = tuple(
            l if l.startswith("backbone.") else f"backbone.{l}" for l in resolved_layers
        )
        cams = gen_layercam_resnet(
            model,
            img_tensor,
            class_idx,
            target_size=target_size,
            layers=resolved_layers,
            scales=scales,
            hflip=hflip,
            shallow_gamma=shallow_gamma,
        )
    else:
        raise ValueError(f"method must be 'cam' or 'layercam', got: {method}")
    return cams.detach().cpu().numpy()


def make_dscnet_cam_fn(
    model: DSCNetClassifier,
    method: str = "cam",
    scales: tuple[float, ...] = (1.0,),
    hflip: bool = False,
    layers: tuple[str, ...] | None = None,
    shallow_gamma: float = 2.0,
    use_refined: bool = True,
) -> CamFn:
    """Build a pipeline-compatible CAM / LayerCAM function for `DSCNetClassifier`."""
    def cam_fn(
        img_tensor: torch.Tensor,
        active_classes,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        cams = compute_dscnet_cams(
            model,
            img_tensor,
            active_classes,
            method=method,
            target_size=target_size,
            scales=scales,
            hflip=hflip,
            layers=layers,
            shallow_gamma=shallow_gamma,
            use_refined=use_refined,
        )
        return ensure_cam_size(cams, target_size)

    return cam_fn


def build_dscnet_optuna_pipeline(
    model: DSCNetClassifier,
    probs: np.ndarray,
    best_params: dict,
    predictor=None,
    encode_image: bool = False,
    scales: tuple[float, ...] | None = None,
    hflip: bool | None = None,
    layercam_layers: tuple[str, ...] | None = None,
    shallow_gamma: float | None = None,
    use_refined_cam: bool = True,
) -> BackbonePipeline:
    """Build a `BackbonePipeline` for DSCNet supporting the 4 methods:
    `Vanilla CAM`, `CAM + SAM`, `LayerCAM`, and `LayerCAM + SAM`.
    """
    scales = tuple(scales if scales is not None else best_params.get("cam_scales", (1.0,)))
    hflip = bool(hflip if hflip is not None else best_params.get("cam_hflip", False))
    if layercam_layers is None:
        layercam_layers = best_params.get("layercam_layers")
    layercam_layers = tuple(layercam_layers) if layercam_layers is not None else None
    shallow_gamma = float(
        shallow_gamma if shallow_gamma is not None else best_params.get("layercam_shallow_gamma", 2.0)
    )

    cam_fns: dict[str, CamFn] = {
        "cam": make_dscnet_cam_fn(
            model, method="cam", scales=scales, hflip=hflip, use_refined=use_refined_cam
        ),
    }
    if "best_layercam_params" in best_params or "best_layercam_sam_params" in best_params:
        cam_fns["layercam"] = make_dscnet_cam_fn(
            model,
            method="layercam",
            scales=scales,
            hflip=hflip,
            layers=layercam_layers,
            shallow_gamma=shallow_gamma,
        )

    methods: dict[str, tuple[str, MaskFn]] = {}
    if "best_cam_params" in best_params:
        methods["Vanilla CAM"] = ("cam", make_cam_mask_fn(best_params["best_cam_params"]["cam_threshold"]))
    if predictor is not None and "best_sam_params" in best_params:
        methods["CAM + SAM"] = (
            "cam",
            make_sam_mask_fn(predictor, encode_image=encode_image, **best_params["best_sam_params"]),
        )
    if "best_layercam_params" in best_params:
        methods["LayerCAM"] = (
            "layercam",
            make_cam_mask_fn(best_params["best_layercam_params"]["cam_threshold"]),
        )
    if predictor is not None and "best_layercam_sam_params" in best_params:
        methods["LayerCAM + SAM"] = (
            "layercam",
            make_sam_mask_fn(predictor, encode_image=encode_image, **best_params["best_layercam_sam_params"]),
        )

    return BackbonePipeline(
        probs=probs,
        cls_threshold=float(best_params.get("best_cls_threshold", 0.5)),
        cam_fns=cam_fns,
        methods=methods,
    )


def load_dscnet_classifier(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    backbone_name: str = "resnet101",
    num_classes: int | None = None,
    embed_dim: int | None = None,
    layer4_stride1: bool | None = None,
) -> DSCNetClassifier:
    """Load a saved `DSCNetClassifier` checkpoint in eval mode on `device`."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg: dict[str, Any] = {}
    if isinstance(state, dict) and "model_state_dict" in state:
        cfg = state.get("config") or {}
        if num_classes is None:
            if "num_classes" in state and state["num_classes"] is not None:
                num_classes = int(state["num_classes"])
            elif isinstance(state.get("classes"), (list, tuple)):
                num_classes = len(state["classes"])
        if isinstance(cfg, dict):
            backbone_name = str(cfg.get("model_name", cfg.get("backbone", backbone_name)))
            if embed_dim is None and "dscnet_embed_dim" in cfg:
                embed_dim = int(cfg["dscnet_embed_dim"])
            if layer4_stride1 is None and "layer4_stride1" in cfg:
                layer4_stride1 = bool(cfg["layer4_stride1"])
        state = state["model_state_dict"]

    if isinstance(state, dict):
        if num_classes is None and "backbone.fc.weight" in state:
            num_classes = int(state["backbone.fc.weight"].shape[0])
        if embed_dim is None and "dscl.proj.0.weight" in state:
            embed_dim = int(state["dscl.proj.0.weight"].shape[0])

    model = DSCNetClassifier(
        backbone_name=backbone_name,
        num_classes=int(num_classes if num_classes is not None else 20),
        embed_dim=int(embed_dim if embed_dim is not None else 256),
        layer4_stride1=bool(True if layer4_stride1 is None else layer4_stride1),
        pretrained=False,
    )
    model.load_state_dict(state)
    return model.to(device).eval()
