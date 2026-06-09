# BioReason-Pro Baseline

Current baseline model:

- W&B artifact ref: `wandb-healthcare/bioreasoning-pro/bioreason-pro-rl-paper:production`
- Data bundle: `main_production`
- Benchmark alias: `213.221.225.228`
- Split used for Senpai steering: validation
- Primary metric: `overall_mean_fmax`
- Direction: higher is better

Baseline metric values are generated with the same validation budget as the candidate run:

```bash
python train.py --mode baseline --max_val_samples 100 --wandb_name baseline/bioreason-pro-rl-paper
```

The default full Senpai run compares the 5-step candidate against this baseline. If the candidate does not strictly improve `overall_mean_fmax`, the run stops and reports a terminal negative result. If it improves, training continues to 20 total steps and reports the best validation checkpoint.
