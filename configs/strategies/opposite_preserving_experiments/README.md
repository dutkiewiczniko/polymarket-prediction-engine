# Opposite-Preserving Experiments

These strategies test `sell_opposite_first: false`, allowing lottery tickets and
directional positions to coexist instead of automatically closing the opposite
side on every buy.

Use the batch config:

```powershell
python run_batch.py --batch-config configs/simulation_opposite_preserving_experiments.yaml
```

Suggested comparison points:

- Whether `sell_opposite_first` stays `False` on buy rows in trajectories.
- Whether strategies hold both UP and DOWN token balances after cheap lottery
  rules fire.
- Whether `max_market_spend_usd` prevents aggressive layered rules from
  exhausting the simulated market balance.

