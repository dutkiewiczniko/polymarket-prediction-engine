# Deterministic Diverse Strategy Suite

This folder contains generated non-random strategy variants for simulator training.

The suite combines the strongest previous families:

- down-biased deterministic entries, replacing random down-biased buys
- price momentum chasers
- price momentum faders
- cheap long-shot entries when a side is near 0.00-0.05
- take-profit exits when held position price reaches 0.90-0.95

Generated ranges:

- take-profit thresholds: `0.90`, `0.925`, `0.95`
- cheap-entry thresholds: `0.01`, `0.025`, `0.05`
- token amounts: `1`, `5`, `10`, `20`
- branches: `down_bias_chaser`, `down_bias_fader`, `hybrid_price_signal`

Run the whole suite with:

```powershell
python run_batch.py --config configs/simulation_deterministic_diverse.yaml
```

Regenerate the configs with:

```powershell
python scripts/simulation/generate_deterministic_strategy_configs.py
```
