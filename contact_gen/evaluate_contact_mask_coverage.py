from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from typing import Dict, Optional, Tuple

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate coverage of sampled contact-mask distribution against real contact masks."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "contact_gen", "generated_masks"),
    )
    parser.add_argument(
        "--real-masks-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "contact_gen", "contact_points_validate.pt"),
        help=(
            "Path to extract_contact_points.py payload. "
            "Used when generated payload does not contain 'real_masks'."
        ),
    )
    parser.add_argument(
        "--robot-names",
        type=str,
        nargs="*",
        default=["allegro", "barrett", "ezgripper", "robotiq_3finger", "shadowhand"],
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "contact_gen", "coverage_eval"))
    parser.add_argument("--iou-ref-max", type=int, default=20000)
    parser.add_argument("--iou-chunk", type=int, default=256)
    parser.add_argument("--pca-samples-per-set", type=int, default=2000)
    parser.add_argument(
        "--compute-tsne",
        action="store_true",
        help="Also render t-SNE scatter for real vs sampled masks.",
    )
    parser.add_argument(
        "--tsne-samples-per-set",
        type=int,
        default=1000,
        help="Per-set sample count for t-SNE. <=0 means use all.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=40.0)
    parser.add_argument("--tsne-n-iter", type=int, default=1000)
    parser.add_argument("--tsne-learning-rate", type=float, default=200.0)
    parser.add_argument("--span-samples-per-set", type=int, default=2000)
    parser.add_argument("--span-embed-dim", type=int, default=8)
    parser.add_argument("--chamfer-real", type=int, default=100)
    parser.add_argument("--chamfer-sampled", type=int, default=1000)
    parser.add_argument("--compute-chamfer", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _js_divergence(prob_a: torch.Tensor, prob_b: torch.Tensor, eps: float = 1e-8) -> float:
    pa = prob_a / prob_a.sum().clamp_min(eps)
    pb = prob_b / prob_b.sum().clamp_min(eps)
    m = 0.5 * (pa + pb)
    kl_a = (pa * (torch.log(pa + eps) - torch.log(m + eps))).sum()
    kl_b = (pb * (torch.log(pb + eps) - torch.log(m + eps))).sum()
    return float(0.5 * (kl_a + kl_b))


def _wasserstein_hist(counts_a: torch.Tensor, counts_b: torch.Tensor, max_bin: int) -> float:
    h_a = torch.bincount(counts_a.long(), minlength=max_bin + 1).float()
    h_b = torch.bincount(counts_b.long(), minlength=max_bin + 1).float()
    p = h_a / h_a.sum().clamp_min(1e-8)
    q = h_b / h_b.sum().clamp_min(1e-8)
    return float(torch.abs(torch.cumsum(p, dim=0) - torch.cumsum(q, dim=0)).sum().item())


def _max_iou_coverage(
    real_masks: torch.Tensor,
    sampled_masks: torch.Tensor,
    ref_max: int,
    chunk: int,
    seed: int,
) -> torch.Tensor:
    if sampled_masks.shape[0] > ref_max:
        g = torch.Generator().manual_seed(int(seed))
        keep = torch.randperm(sampled_masks.shape[0], generator=g)[:ref_max]
        sampled_masks = sampled_masks[keep]

    sampled_f = sampled_masks.float()
    sampled_sum = sampled_f.sum(dim=1).unsqueeze(0)

    max_iou = torch.zeros((real_masks.shape[0],), dtype=torch.float32)
    real_f = real_masks.float()
    for i in range(0, real_masks.shape[0], chunk):
        a = real_f[i : i + chunk]
        inter = a @ sampled_f.T
        union = a.sum(dim=1, keepdim=True) + sampled_sum - inter
        iou = inter / union.clamp_min(1e-8)
        max_iou[i : i + chunk] = iou.max(dim=1).values
    return max_iou


def _pca_2d(x: torch.Tensor, q: int = 2) -> torch.Tensor:
    x_center = x - x.mean(dim=0, keepdim=True)
    u, s, _ = torch.pca_lowrank(x_center, q=q)
    return u[:, :q] * s[:q]


def _resolve_draw_count(total: int, requested: int) -> int:
    if int(requested) <= 0:
        return int(total)
    return int(min(int(total), int(requested)))


def _tsne_2d(
    x: torch.Tensor,
    seed: int,
    perplexity: float,
    n_iter: int,
    learning_rate: float,
) -> Optional[torch.Tensor]:
    try:
        from sklearn.manifold import TSNE  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "t-SNE requested but scikit-learn is not available in the current environment."
        ) from e

    x_np = x.detach().cpu().numpy()
    n = int(x_np.shape[0])
    if n < 6:
        return None
    perp_max = max(5.0, float((n - 1) // 3))
    perp = float(max(2.0, min(float(perplexity), perp_max)))

    sig = inspect.signature(TSNE.__init__)
    kwargs = {
        "n_components": 2,
        "perplexity": perp,
        "learning_rate": float(learning_rate),
        "init": "pca",
        "random_state": int(seed),
    }
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = int(max(250, int(n_iter)))
    else:
        kwargs["n_iter"] = int(max(250, int(n_iter)))
    if "n_jobs" in sig.parameters:
        kwargs["n_jobs"] = -1

    emb = TSNE(**kwargs).fit_transform(x_np)
    return torch.from_numpy(emb).float()


def _one_nn_accuracy(emb: torch.Tensor, labels: torch.Tensor) -> float:
    d = torch.cdist(emb, emb, p=2)
    d.fill_diagonal_(1e9)
    nn = d.argmin(dim=1)
    pred = labels[nn]
    return float((pred == labels).float().mean().item())


def _covariance(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tuple(x.shape)}")
    xc = x - x.mean(dim=0, keepdim=True)
    den = max(1, int(x.shape[0]) - 1)
    return (xc.T @ xc) / float(den)


def _span_distribution_metrics(
    real_masks: torch.Tensor,
    sampled_masks: torch.Tensor,
    samples_per_set: int,
    embed_dim: int,
    seed: int,
) -> Dict[str, float]:
    g = torch.Generator().manual_seed(int(seed) + 123)
    nr = min(int(samples_per_set), int(real_masks.shape[0]))
    ns = min(int(samples_per_set), int(sampled_masks.shape[0]))
    ir = torch.randperm(int(real_masks.shape[0]), generator=g)[:nr]
    is_ = torch.randperm(int(sampled_masks.shape[0]), generator=g)[:ns]
    xr = real_masks[ir].float()
    xs = sampled_masks[is_].float()

    mu = xr.mean(dim=0, keepdim=True)
    xr0 = xr - mu
    xs0 = xs - mu
    q = int(max(2, min(int(embed_dim), int(xr0.shape[0]) - 1, int(xr0.shape[1]) - 1)))
    if q < 2:
        return {
            "pc2d_span_ratio_mean": 0.0,
            "pc2d_intersection_cover_mean": 0.0,
            "span_trace_ratio": 0.0,
            "span_logdet_ratio": 0.0,
            "real_to_sampled_nn_p95": 0.0,
            "real_to_real_nn_p95": 0.0,
            "nn_p95_ratio": 0.0,
        }

    _, _, v = torch.pca_lowrank(xr0, q=q)
    emb_r = xr0 @ v[:, :q]
    emb_s = xs0 @ v[:, :q]

    eps = 1e-6
    cov_r = _covariance(emb_r) + eps * torch.eye(q)
    cov_s = _covariance(emb_s) + eps * torch.eye(q)
    tr_ratio = float((torch.trace(cov_s) / torch.trace(cov_r).clamp_min(eps)).item())
    sign_r, ldr = torch.linalg.slogdet(cov_r)
    sign_s, lds = torch.linalg.slogdet(cov_s)
    if float(sign_r.item()) > 0.0 and float(sign_s.item()) > 0.0:
        ld_ratio = float(torch.exp((lds - ldr) / float(q)).item())
    else:
        ld_ratio = 0.0

    d_rs = torch.cdist(emb_r, emb_s, p=2).min(dim=1).values
    d_rr = torch.cdist(emb_r, emb_r, p=2)
    d_rr.fill_diagonal_(1e9)
    d_rr = d_rr.min(dim=1).values
    p95_rs = float(torch.quantile(d_rs, q=0.95).item())
    p95_rr = float(torch.quantile(d_rr, q=0.95).item())
    nn_ratio = float(p95_rs / max(1e-8, p95_rr))

    def _span_cover_1d(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
        a_lo = float(torch.quantile(a, q=0.01).item())
        a_hi = float(torch.quantile(a, q=0.99).item())
        b_lo = float(torch.quantile(b, q=0.01).item())
        b_hi = float(torch.quantile(b, q=0.99).item())
        a_span = max(1e-8, a_hi - a_lo)
        b_span = max(0.0, b_hi - b_lo)
        span_ratio = float(b_span / a_span)
        inter = max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))
        cover = float(inter / a_span)
        return span_ratio, cover

    span_ratios = []
    covers = []
    for j in range(min(2, q)):
        sr, cv = _span_cover_1d(emb_r[:, j], emb_s[:, j])
        span_ratios.append(sr)
        covers.append(cv)

    return {
        "pc2d_span_ratio_mean": float(sum(span_ratios) / max(1, len(span_ratios))),
        "pc2d_intersection_cover_mean": float(sum(covers) / max(1, len(covers))),
        "span_trace_ratio": tr_ratio,
        "span_logdet_ratio": ld_ratio,
        "real_to_sampled_nn_p95": p95_rs,
        "real_to_real_nn_p95": p95_rr,
        "nn_p95_ratio": nn_ratio,
    }


def _mask_to_points(mask: torch.Tensor, template_points: torch.Tensor) -> torch.Tensor:
    pts = template_points[mask.bool()]
    if pts.shape[0] == 0:
        return template_points[:1] * 0.0
    return pts


def _chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    d = torch.cdist(a, b, p=2)
    return float((d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item())


def _draw_figures(
    robot_name: str,
    real_masks: torch.Tensor,
    sampled_masks: torch.Tensor,
    template_points: torch.Tensor,
    out_dir: str,
    pca_samples_per_set: int,
    compute_tsne: bool,
    tsne_samples_per_set: int,
    tsne_perplexity: float,
    tsne_n_iter: int,
    tsne_learning_rate: float,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    real_counts = real_masks.sum(dim=1).numpy()
    sampled_counts = sampled_masks.sum(dim=1).numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(real_counts, bins=40, alpha=0.6, label="real", density=True)
    plt.hist(sampled_counts, bins=40, alpha=0.6, label="sampled", density=True)
    plt.xlabel("contact count")
    plt.ylabel("density")
    plt.title(f"{robot_name}: contact-count distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{robot_name}_count_hist.png"), dpi=180)
    plt.close()

    occ_real = real_masks.float().mean(dim=0).numpy()
    occ_sample = sampled_masks.float().mean(dim=0).numpy()
    occ_diff = abs(occ_real - occ_sample)
    pts = template_points.numpy()

    fig = plt.figure(figsize=(14, 4.5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    s = 16
    p1 = ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=occ_real, s=s, cmap="viridis")
    p2 = ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=occ_sample, s=s, cmap="viridis")
    p3 = ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=occ_diff, s=s, cmap="magma")
    ax1.set_title("real occupancy")
    ax2.set_title("sampled occupancy")
    ax3.set_title("|real - sampled|")
    for ax in (ax1, ax2, ax3):
        ax.set_axis_off()
    fig.colorbar(p1, ax=ax1, shrink=0.6)
    fig.colorbar(p2, ax=ax2, shrink=0.6)
    fig.colorbar(p3, ax=ax3, shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{robot_name}_surface_occupancy.png"), dpi=180)
    plt.close()

    g = torch.Generator().manual_seed(seed)
    nr = _resolve_draw_count(real_masks.shape[0], int(pca_samples_per_set))
    ns = _resolve_draw_count(sampled_masks.shape[0], int(pca_samples_per_set))
    ir = torch.randperm(real_masks.shape[0], generator=g)[:nr]
    is_ = torch.randperm(sampled_masks.shape[0], generator=g)[:ns]
    xr = real_masks[ir].float()
    xs = sampled_masks[is_].float()
    x = torch.cat([xr, xs], dim=0)
    y = torch.cat([torch.zeros(nr), torch.ones(ns)], dim=0)
    emb = _pca_2d(x, q=2).numpy()
    y_np = y.numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(emb[y_np == 0, 0], emb[y_np == 0, 1], s=8, alpha=0.5, label=f"real (n={nr})")
    plt.scatter(emb[y_np == 1, 0], emb[y_np == 1, 1], s=8, alpha=0.5, label=f"sampled (n={ns})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{robot_name}: mask PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{robot_name}_pca.png"), dpi=180)
    plt.close()

    if bool(compute_tsne):
        nr_t = _resolve_draw_count(real_masks.shape[0], int(tsne_samples_per_set))
        ns_t = _resolve_draw_count(sampled_masks.shape[0], int(tsne_samples_per_set))
        ir_t = torch.randperm(real_masks.shape[0], generator=g)[:nr_t]
        is_t = torch.randperm(sampled_masks.shape[0], generator=g)[:ns_t]
        xt = torch.cat([real_masks[ir_t].float(), sampled_masks[is_t].float()], dim=0)
        yt = torch.cat([torch.zeros(nr_t), torch.ones(ns_t)], dim=0)
        emb_t = _tsne_2d(
            x=xt,
            seed=int(seed),
            perplexity=float(tsne_perplexity),
            n_iter=int(tsne_n_iter),
            learning_rate=float(tsne_learning_rate),
        )
        if emb_t is not None:
            emb_t_np = emb_t.numpy()
            yt_np = yt.numpy()
            plt.figure(figsize=(6, 5))
            plt.scatter(
                emb_t_np[yt_np == 0, 0],
                emb_t_np[yt_np == 0, 1],
                s=8,
                alpha=0.5,
                label=f"real (n={nr_t})",
            )
            plt.scatter(
                emb_t_np[yt_np == 1, 0],
                emb_t_np[yt_np == 1, 1],
                s=8,
                alpha=0.5,
                label=f"sampled (n={ns_t})",
            )
            plt.xlabel("tSNE-1")
            plt.ylabel("tSNE-2")
            plt.title(f"{robot_name}: mask t-SNE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{robot_name}_tsne.png"), dpi=180)
            plt.close()
        else:
            print(
                f"[{robot_name}] skip t-SNE: too few points."
            )


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    os.makedirs(args.output_dir, exist_ok=True)

    sys.path.append(PROJECT_ROOT)
    from robot_model.robot_model import create_robot_model  # type: ignore

    real_payload = None
    real_by_robot: Dict[str, list] = {}
    real_masks_path = os.path.abspath(args.real_masks_path)
    if os.path.exists(real_masks_path):
        real_payload = torch.load(real_masks_path, map_location="cpu")
        if isinstance(real_payload, dict) and "samples" in real_payload:
            for s in list(real_payload["samples"]):
                rn = str(s.get("robot_name", ""))
                if rn:
                    real_by_robot.setdefault(rn, []).append(s)

    all_metrics: Dict[str, Dict[str, float]] = {}
    for robot_name in args.robot_names:
        in_path = os.path.join(args.input_dir, f"{robot_name}_random_masks.pt")
        if not os.path.exists(in_path):
            print(f"[skip] missing: {in_path}")
            continue
        payload = torch.load(in_path, map_location="cpu")
        sampled_masks = payload["sampled_masks"].bool()
        num_points = int(payload["meta"]["num_surface_points"])
        resolved_robot_name = str(payload["meta"]["robot_model_name"])
        if "real_masks" in payload:
            real_masks = payload["real_masks"].bool()
        else:
            if len(real_by_robot.get(robot_name, [])) == 0:
                raise RuntimeError(
                    f"{robot_name}: generated payload has no 'real_masks', and no matching real samples "
                    f"were found in --real-masks-path={real_masks_path}"
                )
            rm_list = []
            for s in real_by_robot[robot_name]:
                m = torch.as_tensor(s["hand_contact_mask"]).bool().view(-1)
                if int(m.numel()) == int(num_points):
                    rm_list.append(m)
            if len(rm_list) == 0:
                raise RuntimeError(
                    f"{robot_name}: no real masks in {real_masks_path} with num_surface_points={num_points}"
                )
            real_masks = torch.stack(rm_list, dim=0).bool()

        model = create_robot_model(
            robot_name=resolved_robot_name,
            device=torch.device("cpu"),
            num_points=num_points,
        )
        q0 = model.get_canonical_q().detach().cpu()
        template_points, _ = model.get_surface_points_normals(q0)
        template_points = template_points.detach().cpu()

        real_counts = real_masks.sum(dim=1).long()
        sampled_counts = sampled_masks.sum(dim=1).long()

        count_w = _wasserstein_hist(real_counts, sampled_counts, max_bin=num_points)
        occ_real = real_masks.float().mean(dim=0)
        occ_sample = sampled_masks.float().mean(dim=0)
        js_occ = _js_divergence(occ_real, occ_sample)

        max_iou = _max_iou_coverage(
            real_masks=real_masks,
            sampled_masks=sampled_masks,
            ref_max=int(args.iou_ref_max),
            chunk=int(args.iou_chunk),
            seed=int(args.seed),
        )
        cov_03 = float((max_iou >= 0.3).float().mean().item())
        cov_05 = float((max_iou >= 0.5).float().mean().item())
        cov_07 = float((max_iou >= 0.7).float().mean().item())
        med_iou = float(max_iou.median().item())

        g = torch.Generator().manual_seed(int(args.seed))
        nr = _resolve_draw_count(real_masks.shape[0], int(args.pca_samples_per_set))
        ns = _resolve_draw_count(sampled_masks.shape[0], int(args.pca_samples_per_set))
        ir = torch.randperm(real_masks.shape[0], generator=g)[:nr]
        is_ = torch.randperm(sampled_masks.shape[0], generator=g)[:ns]
        x = torch.cat([real_masks[ir].float(), sampled_masks[is_].float()], dim=0)
        labels = torch.cat([torch.zeros(nr), torch.ones(ns)], dim=0)
        emb = _pca_2d(x, q=2)
        one_nn_acc = _one_nn_accuracy(emb, labels)
        span_metrics = _span_distribution_metrics(
            real_masks=real_masks,
            sampled_masks=sampled_masks,
            samples_per_set=int(args.span_samples_per_set),
            embed_dim=int(args.span_embed_dim),
            seed=int(args.seed),
        )

        chamfer_median = None
        chamfer_p90 = None
        if bool(args.compute_chamfer):
            nr_c = min(int(args.chamfer_real), real_masks.shape[0])
            ns_c = min(int(args.chamfer_sampled), sampled_masks.shape[0])
            ir_c = torch.randperm(real_masks.shape[0], generator=g)[:nr_c]
            is_c = torch.randperm(sampled_masks.shape[0], generator=g)[:ns_c]
            sampled_point_sets = [_mask_to_points(sampled_masks[i], template_points) for i in is_c.tolist()]
            best_vals = []
            for i in ir_c.tolist():
                rp = _mask_to_points(real_masks[i], template_points)
                best = float("inf")
                for sp in sampled_point_sets:
                    c = _chamfer(rp, sp)
                    if c < best:
                        best = c
                best_vals.append(best)
            best_t = torch.tensor(best_vals, dtype=torch.float32)
            chamfer_median = float(best_t.median().item())
            chamfer_p90 = float(torch.quantile(best_t, q=0.9).item())

        robot_metrics: Dict[str, float] = {
            "count_wasserstein": float(count_w),
            "occupancy_js_divergence": float(js_occ),
            "coverage_iou_ge_03": cov_03,
            "coverage_iou_ge_05": cov_05,
            "coverage_iou_ge_07": cov_07,
            "median_best_iou": med_iou,
            "one_nn_separability_acc": float(one_nn_acc),
            "real_count_mean": float(real_counts.float().mean().item()),
            "sampled_count_mean": float(sampled_counts.float().mean().item()),
        }
        robot_metrics.update(span_metrics)
        if chamfer_median is not None and chamfer_p90 is not None:
            robot_metrics["chamfer_median"] = chamfer_median
            robot_metrics["chamfer_p90"] = chamfer_p90

        out_robot = os.path.join(args.output_dir, robot_name)
        os.makedirs(out_robot, exist_ok=True)
        with open(os.path.join(out_robot, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(robot_metrics, f, indent=2)
        _draw_figures(
            robot_name=robot_name,
            real_masks=real_masks,
            sampled_masks=sampled_masks,
            template_points=template_points,
            out_dir=out_robot,
            pca_samples_per_set=int(args.pca_samples_per_set),
            compute_tsne=bool(args.compute_tsne),
            tsne_samples_per_set=int(args.tsne_samples_per_set),
            tsne_perplexity=float(args.tsne_perplexity),
            tsne_n_iter=int(args.tsne_n_iter),
            tsne_learning_rate=float(args.tsne_learning_rate),
            seed=int(args.seed),
        )
        all_metrics[robot_name] = robot_metrics
        print(f"[{robot_name}] metrics: {robot_metrics}")

    with open(os.path.join(args.output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[saved] {os.path.join(args.output_dir, 'summary_metrics.json')}")


if __name__ == "__main__":
    main()
