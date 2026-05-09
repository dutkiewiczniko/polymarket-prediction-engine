# BTC Simulator Starter

Simulator and training tools now live under `scripts/`, with small compatibility
wrappers left at the repo root for common commands.

## Install dependency

```bash
pip install -r requirements.txt
```

## Run one replay

Example:

```bash
python run_replay.py ^
  --market-csv data/btc-updown-5m-1234567890.csv ^
  --strategy-config configs/strategies/momentum_basic.yaml ^
  --output-csv runs/momentum_basic_market_001.csv
```

Equivalent direct script path:

```bash
python scripts/simulation/run_replay.py ^
  --market-csv data/btc-updown-5m-1234567890.csv ^
  --strategy-config configs/strategies/momentum_basic.yaml ^
  --output-csv runs/momentum_basic_market_001.csv
```

On macOS/Linux/WSL:

```bash
python run_replay.py \
  --market-csv data/btc-updown-5m-1234567890.csv \
  --strategy-config configs/strategies/momentum_basic.yaml \
  --output-csv runs/momentum_basic_market_001.csv
```

## What this does

It replays a recorded market CSV tick by tick, lets a strategy decide actions, simulates a fake portfolio, resolves the final position, and writes a trajectory CSV.

## Why this comes before ML

The ML dataset should come from many resolved simulated trajectories.

Each trajectory row includes:

- market state
- portfolio state
- action
- final outcome
- final balance
- reward_to_go

`reward_to_go = final_balance - balance_before`

That is more useful than giving every row only the final balance.

## Important limitations in this first version

- assumes instant fills
- no fees
- no slippage
- no partial fills
- final outcome is inferred from final BTC versus price_to_beat unless manually supplied
- strategy logic is intentionally basic

These are acceptable for the first version because the goal is to get deterministic replay working.
