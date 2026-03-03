# contact_gen

Extract and visualize manipulator contact point clouds from DRO-Grasp grasps using a GenDex-style continuous contact map.
Hand geometry is loaded from `crossembodimentaction/robot_model/robot_model.py`.

## 1) Extract contact points

```bash
cd /home/qianxu/Project/crossembodimentaction
python3 contact_gen/extract_contact_points.py \
  --dro-root /home/qianxu/Project/DRO-Grasp \
  --split validate \
  --threshold 0.4 \
  --num-hand-points 2048 \
  --output-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt
```

Notes:
- Contact values are computed **directly on hand surface points**.
- Hand surface points are sampled uniformly on the full hand mesh with exact count `--num-hand-points`.
- Surface templates are deterministic per robot/point-count and stored with `surface_template_hash`.
- The contact mask is binarized by `--threshold` (default `0.4`).
- Default method matches GenDex align-distance style:
  - `align_exp_scale=2.0`
  - `sigmoid_scale=10.0`

## 2) Visualize with sliders

```bash
cd /home/qianxu/Project/crossembodimentaction
python3 contact_gen/visualize_contact_points.py \
  --contact-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt \
  --dro-root /home/qianxu/Project/DRO-Grasp \
  --host 127.0.0.1 \
  --port 8080
```

Sliders:
- `sample_index`: choose grasp sample
- `vis_mode`:
  - `0`: point cloud only
  - `1`: point cloud + hand mesh
  - `2`: point cloud + hand mesh + object mesh

## 3) Infer Sampler Hyperparameters From Real Masks (YAML only)

```bash
cd /home/qianxu/Project/crossembodimentaction
python3 contact_gen/infer_contact_sampler_hparams.py \
  --real-masks-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt \
  --output-yaml /home/qianxu/Project/crossembodimentaction/contact_gen/contact_sampler_hparams.yaml
```

This step is the only place that uses real masks for fitting hyperparameters.

## 4) Visualize Surface Connectivity Graph

```bash
cd /home/qianxu/Project/crossembodimentaction
python3 contact_gen/visualize_surface_graph.py \
  --num-surface-points 512 \
  --vis-port 8092
```

## 5) Generate Random Contact Masks (Hydra, data-less runtime)

```bash
cd /home/qianxu/Project/crossembodimentaction
python3 contact_gen/generate_random_contact_masks.py \
  hparams_path=contact_gen/contact_sampler_hparams.yaml \
  sample_count=10000 \
  output_dir=contact_gen/generated_masks
```

Notes:
- Random generation reads only the YAML hyperparameters.
- Runtime sampling is data-less: random joint limits + virtual object patches.
- Real masks are only used in step 3 to infer component range.
- No target contact-count (`k`) constraints or trial optimization are used.
- Base translation joints (`base/root/world xyz` when present) are forced to zero during random sampling.

## 6) Evaluate Coverage

```bash
cd /home/qianxu/Project/crossembodimentaction
python3 contact_gen/evaluate_contact_mask_coverage.py \
  --input-dir /home/qianxu/Project/crossembodimentaction/contact_gen/generated_masks \
  --real-masks-path /home/qianxu/Project/crossembodimentaction/contact_gen/contact_points_validate.pt \
  --output-dir /home/qianxu/Project/crossembodimentaction/contact_gen/coverage_eval
```

## 7) Visualize sampled masks + virtual patches

```bash
cd /home/qianxu/Project/crossembodimentaction
python3 contact_gen/visualize_sampled_contact_masks.py \
  --sampled-path /home/qianxu/Project/crossembodimentaction/contact_gen/generated_masks/allegro_random_masks.pt \
  --host 127.0.0.1 --port 8080
```

`vis_mode` options:
- `0`: contact points only
- `1`: contact points + hand
- `2`: contact points + sampled object patches
- `3`: contact points + hand + sampled object patches
