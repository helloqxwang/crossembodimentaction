# Patch-v3 Refinement (Round 2)

This folder contains a clean comparison of recent `patch_v3` variants after:
- penetration post-process change in `robot_model._sample_virtual_object_patches`
- explore sampler update with `q_source=mixed` support

## Evaluated variants

1. `patch_v3_default_10k`:
`q_source=random`
2. `patch_v3_qreal_fix_10k`:
`q_source=real`
3. `patch_v3_mix_default_qreal_10k`:
50/50 merge of the two 10k variants above

## Mean metrics across 5 robots

| variant | cov@0.3 | cov@0.5 | median best IoU | span ratio (PC2D) | span trace ratio | nn p95 ratio | count wasserstein |
|---|---:|---:|---:|---:|---:|---:|---:|
| patch_v3_default_10k | 0.9384 | 0.2603 | 0.4458 | 0.6891 | 0.5349 | 1.4736 | 0.3121 |
| patch_v3_qreal_fix_10k | 0.9432 | 0.2439 | 0.4405 | 0.6821 | 0.5361 | 1.4564 | 0.2901 |
| patch_v3_mix_default_qreal_10k | 0.9405 | 0.2535 | 0.4454 | 0.6931 | 0.5395 | 1.4657 | 0.2802 |

Interpretation:
- `patch_v3_default_10k` is strongest on IoU balance.
- `patch_v3_mix_default_qreal_10k` gives the best span coverage in this round.
- `patch_v3_qreal_fix_10k` gives strongest `cov@0.3` and best OOD-tail ratio among these three.

## Commands used

Generate default 10k:

```bash
conda run -n repr python3 contact_gen/generate_random_contact_masks_explore.py \
  sample_count=10000 sample_chunk=1000 \
  output_dir=/tmp/patch_v3_default_10k_new \
  q_source=random sampler_family=patch_distance \
  anchor_source=real_distribution target_count_source=real
```

Generate qreal 10k:

```bash
conda run -n repr python3 contact_gen/generate_random_contact_masks_explore.py \
  sample_count=10000 sample_chunk=1000 \
  output_dir=/tmp/patch_v3_qreal_fix_10k \
  q_source=real sampler_family=patch_distance \
  anchor_source=real_distribution target_count_source=real
```

Evaluate:

```bash
conda run -n repr python3 contact_gen/evaluate_contact_mask_coverage.py \
  --input-dir /tmp/patch_v3_default_10k_new \
  --real-masks-path contact_gen/contact_points_validate.pt \
  --output-dir /tmp/patch_v3_default_10k_new_eval --seed 42
```

```bash
conda run -n repr python3 contact_gen/evaluate_contact_mask_coverage.py \
  --input-dir /tmp/patch_v3_qreal_fix_10k \
  --real-masks-path contact_gen/contact_points_validate.pt \
  --output-dir /tmp/patch_v3_qreal_fix_10k_eval --seed 42
```

Rank all experiments in this folder:

```bash
python3 contact_gen/rank_sampler_experiments.py \
  --eval-root contact_gen/coverage_eval_explore_round2 \
  --output-json contact_gen/coverage_eval_explore_round2/ranking_blended.json
```
