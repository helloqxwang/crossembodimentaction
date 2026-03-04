from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class RealMaskStats:
    point_prob: torch.Tensor
    count_values: torch.Tensor


def build_real_mask_stats(
    real_masks: torch.Tensor,
    eps: float = 1e-8,
) -> RealMaskStats:
    masks = real_masks.detach().bool().cpu()
    if masks.ndim != 2:
        raise ValueError(f"Expected (M,N) masks, got {tuple(masks.shape)}")
    point_prob = masks.float().mean(dim=0)
    point_prob = point_prob.clamp_min(float(eps))
    point_prob = point_prob / point_prob.sum().clamp_min(float(eps))
    count_values = masks.sum(dim=1).long()
    return RealMaskStats(point_prob=point_prob, count_values=count_values)


def sample_anchor_indices(
    n_anchor: int,
    num_points: int,
    anchor_source: str,
    generator: Optional[torch.Generator] = None,
    real_point_prob: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    uniform_mix: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    n = int(max(1, min(int(n_anchor), int(num_points))))
    if str(anchor_source) == "real_distribution":
        if real_point_prob is None:
            raise ValueError("anchor_source=real_distribution requires real_point_prob")
        p = real_point_prob.to(device=device, dtype=torch.float32).clamp_min(1e-8)
        t = float(max(1e-3, float(temperature)))
        if abs(t - 1.0) > 1e-6:
            p = p.pow(1.0 / t)
        um = float(max(0.0, min(1.0, float(uniform_mix))))
        if um > 0.0:
            p = (1.0 - um) * p + um * torch.ones_like(p)
        p = p / p.sum().clamp_min(1e-8)
        return torch.multinomial(p, num_samples=n, replacement=False, generator=generator)
    if str(anchor_source) == "uniform":
        return torch.randperm(int(num_points), device=device, generator=generator)[:n]
    raise ValueError(f"Unknown anchor_source: {anchor_source}")


def sample_target_count(
    target_count_source: str,
    count_values_real: Optional[torch.Tensor],
    num_points: int,
    generator: Optional[torch.Generator] = None,
    device: torch.device = torch.device("cpu"),
    fallback_min: int = 1,
    fallback_max: Optional[int] = None,
    jitter_frac: float = 0.0,
) -> int:
    if str(target_count_source) == "real":
        if count_values_real is None or int(count_values_real.numel()) == 0:
            raise ValueError("target_count_source=real requires non-empty count_values_real")
        idx = int(torch.randint(0, int(count_values_real.numel()), (1,), device=device, generator=generator).item())
        base = int(max(1, min(int(num_points), int(count_values_real[idx].item()))))
        jf = float(max(0.0, float(jitter_frac)))
        if jf > 0.0:
            lo = max(0.1, 1.0 - jf)
            hi = 1.0 + jf
            scale = float(torch.empty(1, device=device).uniform_(lo, hi, generator=generator).item())
            base = int(round(float(base) * scale))
        return int(max(1, min(int(num_points), base)))
    if str(target_count_source) == "none":
        hi = int(num_points if fallback_max is None else max(1, min(int(num_points), int(fallback_max))))
        lo = int(max(1, min(hi, int(fallback_min))))
        return int(torch.randint(lo, hi + 1, (1,), device=device, generator=generator).item())
    raise ValueError(f"Unknown target_count_source: {target_count_source}")


def _alloc_component_sizes(total: int, n_comp: int, generator: Optional[torch.Generator], device: torch.device) -> List[int]:
    total = int(max(1, total))
    n_comp = int(max(1, min(total, n_comp)))
    if n_comp == 1:
        return [total]
    w = torch.rand((n_comp,), device=device, generator=generator).clamp_min(1e-6)
    w = w / w.sum()
    raw = torch.floor(w * float(total)).long()
    raw = raw.clamp_min(1)
    s = int(raw.sum().item())
    while s > total:
        i = int(torch.argmax(raw).item())
        if int(raw[i].item()) > 1:
            raw[i] -= 1
            s -= 1
        else:
            break
    while s < total:
        i = int(torch.randint(0, n_comp, (1,), device=device, generator=generator).item())
        raw[i] += 1
        s += 1
    return [int(v.item()) for v in raw]


def _grow_compact(
    neighbors: Sequence[Sequence[int]],
    seed: int,
    target_size: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    branch_min: int = 1,
    branch_max: int = 3,
) -> List[int]:
    selected = {int(seed)}
    frontier = [int(seed)]
    max_iter = int(30 * max(1, target_size))
    it = 0
    while len(selected) < target_size and it < max_iter:
        it += 1
        if len(frontier) == 0:
            frontier = list(selected)
        fi = int(torch.randint(0, len(frontier), (1,), device=device, generator=generator).item())
        cur = int(frontier[fi])
        nb = list(neighbors[cur])
        if len(nb) == 0:
            continue
        b = int(torch.randint(int(branch_min), int(branch_max) + 1, (1,), device=device, generator=generator).item())
        for _ in range(b):
            j = int(torch.randint(0, len(nb), (1,), device=device, generator=generator).item())
            nxt = int(nb[j])
            if nxt not in selected:
                selected.add(nxt)
                frontier.append(nxt)
            if len(selected) >= target_size:
                break
    return list(selected)


def _grow_filament(
    neighbors: Sequence[Sequence[int]],
    seed: int,
    target_size: int,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> List[int]:
    selected = {int(seed)}
    chain = [int(seed)]
    cur = int(seed)
    max_iter = int(60 * max(1, target_size))
    it = 0
    while len(selected) < target_size and it < max_iter:
        it += 1
        nb = list(neighbors[cur])
        if len(nb) == 0:
            cur = chain[int(torch.randint(0, len(chain), (1,), device=device, generator=generator).item())]
            continue
        j = int(torch.randint(0, len(nb), (1,), device=device, generator=generator).item())
        nxt = int(nb[j])
        cur = nxt
        if nxt not in selected:
            selected.add(nxt)
            chain.append(nxt)
        if bool(torch.rand(1, device=device, generator=generator).item() < 0.25):
            cur = chain[int(torch.randint(0, len(chain), (1,), device=device, generator=generator).item())]
    return list(selected)


def sample_direct_surface_mask(
    neighbors: Sequence[Sequence[int]],
    num_points: int,
    anchor_indices: torch.Tensor,
    target_count: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    compact_prob: float = 0.55,
    filament_prob: float = 0.25,
) -> torch.Tensor:
    n_pts = int(num_points)
    target = int(max(1, min(n_pts, int(target_count))))
    anchors = [int(v) for v in anchor_indices.view(-1).tolist()]
    if len(anchors) == 0:
        anchors = [int(torch.randint(0, n_pts, (1,), device=device, generator=generator).item())]
    comp_sizes = _alloc_component_sizes(total=target, n_comp=len(anchors), generator=generator, device=device)

    selected = set()
    for seed, csz in zip(anchors, comp_sizes):
        u = float(torch.rand(1, device=device, generator=generator).item())
        if u < float(compact_prob):
            comp = _grow_compact(neighbors=neighbors, seed=seed, target_size=csz, generator=generator, device=device)
        elif u < float(compact_prob + filament_prob):
            comp = _grow_filament(neighbors=neighbors, seed=seed, target_size=csz, generator=generator, device=device)
        else:
            c0 = max(1, int(round(0.6 * csz)))
            c1 = max(1, csz - c0)
            comp = _grow_compact(neighbors=neighbors, seed=seed, target_size=c0, generator=generator, device=device)
            tail_seed = int(comp[-1]) if len(comp) > 0 else int(seed)
            comp_tail = _grow_filament(neighbors=neighbors, seed=tail_seed, target_size=c1, generator=generator, device=device)
            comp = comp + comp_tail
        selected.update(int(v) for v in comp)

    # Adjust to target count.
    if len(selected) < target:
        active = list(selected)
        if len(active) == 0:
            active = anchors.copy()
            selected.update(active)
        while len(selected) < target:
            base = int(active[int(torch.randint(0, len(active), (1,), device=device, generator=generator).item())])
            nb = list(neighbors[base])
            if len(nb) == 0:
                nxt = int(torch.randint(0, n_pts, (1,), device=device, generator=generator).item())
            else:
                nxt = int(nb[int(torch.randint(0, len(nb), (1,), device=device, generator=generator).item())])
            if nxt not in selected:
                selected.add(nxt)
                active.append(nxt)
    elif len(selected) > target:
        idx = torch.tensor(list(selected), dtype=torch.long, device=device)
        keep = idx[torch.randperm(int(idx.numel()), device=device, generator=generator)[:target]]
        selected = set(int(v) for v in keep.tolist())

    mask = torch.zeros((n_pts,), dtype=torch.bool, device=device)
    if len(selected) > 0:
        mask[torch.tensor(list(selected), dtype=torch.long, device=device)] = True
    return mask


def sample_direct_bernoulli_mask(
    num_points: int,
    target_count: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    base_prob: Optional[torch.Tensor] = None,
    neighbors: Optional[Sequence[Sequence[int]]] = None,
    noise_scale: float = 1.0,
    smooth_steps: int = 1,
    smooth_add_thr: float = 0.55,
    smooth_keep_thr: float = 0.35,
) -> torch.Tensor:
    n = int(num_points)
    k = int(max(1, min(n, int(target_count))))
    if base_prob is None:
        p = torch.full((n,), 1.0 / float(n), dtype=torch.float32, device=device)
    else:
        p = base_prob.to(device=device, dtype=torch.float32).clamp_min(1e-8)
        p = p / p.sum().clamp_min(1e-8)

    u = torch.rand((n,), device=device, generator=generator).clamp_(1e-8, 1.0 - 1e-8)
    gumbel = -torch.log(-torch.log(u))
    logits = torch.log(p)
    score = logits + float(noise_scale) * gumbel
    topk = torch.topk(score, k=k, dim=0).indices
    mask = torch.zeros((n,), dtype=torch.bool, device=device)
    mask[topk] = True

    if neighbors is not None and int(smooth_steps) > 0:
        for _ in range(int(smooth_steps)):
            ratio = torch.zeros((n,), dtype=torch.float32, device=device)
            for i in range(n):
                nb = neighbors[i]
                if len(nb) == 0:
                    ratio[i] = mask[i].float()
                else:
                    nb_idx = torch.tensor(nb, dtype=torch.long, device=device)
                    ratio[i] = mask[nb_idx].float().mean()
            grown = ratio >= float(smooth_add_thr)
            kept = mask & (ratio >= float(smooth_keep_thr))
            mask = grown | kept
            if int(mask.sum().item()) > k:
                idx = torch.where(mask)[0]
                sub_score = score[idx]
                keep_local = torch.topk(sub_score, k=k, dim=0).indices
                keep_idx = idx[keep_local]
                new_mask = torch.zeros_like(mask)
                new_mask[keep_idx] = True
                mask = new_mask
            elif int(mask.sum().item()) < k:
                missing = int(k - mask.sum().item())
                idx = torch.where(~mask)[0]
                sub_score = score[idx]
                add_local = torch.topk(sub_score, k=missing, dim=0).indices
                add_idx = idx[add_local]
                mask[add_idx] = True

    return mask


def swap_jitter_mask(
    mask: torch.Tensor,
    neighbors: Sequence[Sequence[int]],
    steps: int,
    generator: Optional[torch.Generator],
    device: torch.device,
    global_jump_prob: float = 0.2,
) -> torch.Tensor:
    out = mask.clone().bool()
    n = int(out.numel())
    s = int(max(0, int(steps)))
    if s <= 0:
        return out
    for _ in range(s):
        active = torch.where(out)[0]
        if int(active.numel()) == 0:
            break
        i = int(active[int(torch.randint(0, int(active.numel()), (1,), device=device, generator=generator).item())].item())
        if bool(torch.rand(1, device=device, generator=generator).item() < float(global_jump_prob)):
            inactive = torch.where(~out)[0]
            if int(inactive.numel()) == 0:
                continue
            j = int(inactive[int(torch.randint(0, int(inactive.numel()), (1,), device=device, generator=generator).item())].item())
            out[i] = False
            out[j] = True
            continue
        nb = neighbors[i]
        if len(nb) == 0:
            continue
        cand = [int(v) for v in nb if not bool(out[int(v)].item())]
        if len(cand) == 0:
            continue
        j = int(cand[int(torch.randint(0, len(cand), (1,), device=device, generator=generator).item())])
        out[i] = False
        out[j] = True
    return out
