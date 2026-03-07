# contact_gen

Clean pipeline for:
1. extracting real hand contact masks from DRO-Grasp,
2. fitting sampler hyperparameters,
3. generating synthetic contact masks,
4. evaluating distribution coverage,
5. visualizing both real and sampled contacts.

All commands below assume:

```bash
cd /home/qianxu/Project/crossembodimentaction
conda activate repr
```

---

## 1) Extract real contact masks

Script: `contact_gen/extract_contact_points.py`

```bash
python3 contact_gen/extract_contact_points.py \
  --dro-root /home/qianxu/Project/DRO-Grasp \
  --split validate \
  --threshold 0.4 \
  --output-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt
```

Output:
- `contact_points_validate.pt` (per-sample contact mask, contact points, hand points, q, etc.)

---

## 2) Visualize extracted real contacts

Script: `contact_gen/visualize_contact_points.py`

```bash
python3 contact_gen/visualize_contact_points.py \
  --contact-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt \
  --dro-root /home/qianxu/Project/DRO-Grasp \
  --host 127.0.0.1 \
  --port 8080
```

Viewer sliders:
- sample index
- vis mode:
  - `0`: points only
  - `1`: points + hand mesh
  - `2`: points + hand mesh + object mesh

---

## 3) Infer sampler hyperparameters

Script: `contact_gen/infer_contact_sampler_hparams.py`

```bash
python3 contact_gen/infer_contact_sampler_hparams.py \
  --real-masks-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt \
  --output-yaml /home/qianxu/Project/crossembodimentaction/contact_gen/contact_sampler_hparams.yaml
```

Optional component visualization:

```bash
python3 contact_gen/infer_contact_sampler_hparams.py \
  --real-masks-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt \
  --output-yaml /home/qianxu/Project/crossembodimentaction/contact_gen/contact_sampler_hparams.yaml \
  --vis-components \
  --vis-port 8094
```

Output:
- `contact_sampler_hparams.yaml`

---

## 4) Visualize surface graph

Script: `contact_gen/visualize_surface_graph.py`

```bash
python3 contact_gen/visualize_surface_graph.py \
  --num-surface-points 1024 \
  --vis-port 8092
```

---

## 5) Generate synthetic contact masks (best clean sampler)

Script: `contact_gen/generate_random_contact_masks.py`

Method:
- uses fitted component range from `contact_sampler_hparams.yaml`,
- uses real mask count statistics from `contact_points_validate.pt`,
- anchor sampling can be `point_prob` (real occupancy prior) or `uniform` (no point prior),
- samples random robot `q`,
- samples virtual object patches,
- computes GenDex-style hand contact value,
- thresholds and count-adjusts masks.

Default config file:
- `contact_gen/config_generate_random_contact_masks.yaml`

Run:

```bash
python3 contact_gen/generate_random_contact_masks.py \
  hparams_path=contact_gen/contact_sampler_hparams.yaml \
  real_masks_path=contact_gen/contact_points_validate.pt \
  anchor_sampling_mode=uniform \
  sample_count=10000 \
  output_dir=contact_gen/generated_masks
```

Output:
- one file per robot: `*_random_masks.pt`

---

## 6) Visualize sampled masks

Script: `contact_gen/visualize_sampled_contact_masks.py`

```bash
python3 contact_gen/visualize_sampled_contact_masks.py \
  --sampled-path /home/qianxu/Project/crossembodimentaction/contact_gen/generated_masks/allegro_random_masks.pt \
  --host 127.0.0.1 \
  --port 8080
```

Viewer vis modes:
- `0`: contact points only
- `1`: contact points + hand
- `2`: contact points + sampled object patches
- `3`: contact points + hand + sampled object patches

---

## 7) Evaluate coverage (metrics + PCA + t-SNE)

Script: `contact_gen/evaluate_contact_mask_coverage.py`

```bash
python3 contact_gen/evaluate_contact_mask_coverage.py \
  --input-dir /home/qianxu/Project/crossembodimentaction/contact_gen/generated_masks \
  --real-masks-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt \
  --output-dir /home/qianxu/Project/crossembodimentaction/contact_gen/coverage_eval \
  --compute-tsne \
  --pca-samples-per-set 2000 \
  --tsne-samples-per-set 1000
```

Notes:
- PCA and t-SNE plots sample up to `*_samples_per_set` points **per set** (real and sampled), so both colors can appear with similar point counts even when raw dataset sizes differ.
- Use `--pca-samples-per-set -1` to draw all masks for each set.

Output:
- per-robot metrics: `coverage_eval/<robot>/metrics.json`
- plots: count histogram, occupancy, PCA, t-SNE
- summary: `coverage_eval/summary_metrics.json`
