# Live Experimental Mix

Fifteen strategy candidates for live paper comparison. Use `--strategy-pattern "[0-9]*.yaml"` so helper configs are not loaded as standalone strategies.

Suggested lightweight live command:

```powershell
python scripts\live\live_strategy_suite.py `
  --strategy-folder configs\strategies\live_experimental_mix `
  --strategy-pattern "[0-9]*.yaml" `
  --balance-config configs\live_start120_reserve360_topup_below5_drawdown60_window20_balance_bands.yaml `
  --run-id paper_live_experimental_mix_v1 `
  --execution-mode paper `
  --liquidity-aware-execution `
  --liquidity-depth-window-cents 2 `
  --liquidity-fill-fraction 1.0 `
  --liquidity-missing-depth-policy skip `
  --orderbook-depth-interval 2.5 `
  --min-order-usd 1.0 `
  --allow-target-fallback `
  --max-target-fallback-elapsed 15 `
  --trajectory-log-mode none `
  --port 5070
```
