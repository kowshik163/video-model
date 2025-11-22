Phase 6 — Training & Optimization (Pivot Strategy)

This README summarizes the starting point for Phase 6 and how to run the scaffolded tools in this repo.

Goals:
- Generate millions of synthetic physics clips using Kubric (or local placeholders for debugging).
- Train the Physics Expert on delta-state prediction using synthetic data.
- Fine-tune Visual Expert and Supervisor on real clips using self-supervised consistency loss (Physics is frozen).
- Optimize models for deployment (TensorRT, quantization) and measure OOD generalization.

Files added:
- `scripts/generate_synthetic_kubric.py`: Generates synthetic clips (uses Kubric if installed; otherwise creates placeholders).
- `scripts/train_physics.py`: Minimal training harness for `PhysicsExpert` using synthetic dataset.
- `scripts/finetune_visual_supervisor.py`: Scaffold for fine-tuning Visual + Supervisor with frozen Physics Expert.

Quick start (CPU or single GPU):

1) Create a small synthetic dataset for debugging:

```powershell
python scripts/generate_synthetic_kubric.py --out_dir data/synthetic --num_clips 20 --frames 32
```

2) Train Physics Expert (debug run):

```powershell
python scripts/train_physics.py --data_dir data/synthetic --epochs 3 --batch_size 2 --clip_frames 8
```

3) (Optional) Fine-tune Visual & Supervisor on short clips.

```powershell
python scripts/finetune_visual_supervisor.py --data_dir data/synthetic --epochs 2 --clip_frames 8
```

Kubric notes:
- Kubric requires additional dependencies and is recommended to run on a cloud VM with Docker or on a workstation with an up-to-date Python environment.
- See https://github.com/google-research/kubric for installation and dataset generation recipes.

Next steps:
- Replace placeholder dataset loader with actual Kubric scene parser and ground-truth physics/state targets.
- Implement proper loss functions described in `implementation_plan.md` (`L_vis`, `L_phys`, `L_consistency`).
- Add distributed training + Ray integration for horizontal scaling.

