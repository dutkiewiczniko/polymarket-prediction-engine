# Project Script Summary

This file summarizes the main scripts in this project by stage, with how to run them and their key CLI parameters.

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Data preparation

### `filter_full_market_csvs_v3.py`
- Purpose: Scan `data_uncleaned/` for BTC market CSVs, validate schema and timing, and copy valid files into `data/`.
- Recommended over `filter_full_market_csvs_v2.py` because it handles corrupted CSVs, NUL bytes, and writes a reject log.
- Run:

```bash
python filter_full_market_csvs_v3.py
```

- There are no CLI options in the wrapper script, but the script itself uses these hard-coded defaults:
  - `data_uncleaned/` → `data/`
  - `btc-updown-5m-*.csv`

### `filter_full_market_csvs_v2.py`
- Purpose: Similar to `v3`, but older and simpler.
- Run:

```bash
python filter_full_market_csvs_v2.py
```

### `find_simulator_ready_markets.py`
- Purpose: Copy ready market CSVs into `simulator_ready_markets/` after checking schema, elapsed, prices, and file age.
- Use this to gather a flat folder of replayable, validated market files.
- Run:

```bash
python find_simulator_ready_markets.py
```

- Overrides available via CLI (direct script arguments):
  - `--input-dir` (can repeat)
  - `--output-dir` (default: `simulator_ready_markets`)
  - `--reject-log` (default: `simulator_ready_market_rejections.csv`)
  - `--max-first-elapsed` (default: `15.0` seconds)
  - `--min-file-age-seconds` (default: `60.0` seconds)

### `build_training_market_data.py`
- Purpose: Convert raw market replay CSVs into feature-engineered market CSVs used for training input.
- Run:

```bash
python build_training_market_data.py
```

- CLI options:
  - `--input-dir` (default: `data`)
  - `--output-dir` (default: `market_data`)
  - `--pattern` (default: `btc-updown-5m-*.csv`)

### `build_training_dataset.py`
- Purpose: Build the training dataset CSV from market data feature files.
- Run:

```bash
python build_training_dataset.py
```

- CLI options:
  - `--market-data-dir` (default: `market_data`)
  - `--output-csv` (default: `training_data/reward_model_dataset.csv`)

---

## 3. Simulation and replay

### `run_replay.py`
- Purpose: Replay one market CSV through one strategy config.
- Run:

```bash
python run_replay.py \
  --market-csv data/btc-updown-5m-1234567890.csv \
  --strategy-config configs/strategies/momentum_basic.yaml \
  --output-csv runs/momentum_basic_market_001.csv
```

- CLI options:
  - `--market-csv` (required)
  - `--strategy-config` (required)
  - `--output-csv` (required)
  - `--final-outcome up|down` (optional override)

### `run_batch.py`
- Purpose: Run a batch of simulated markets over many strategy configs.
- Run:

```bash
python run_batch.py --config configs/simulation_batch.yaml
```

- CLI options:
  - `--config` (default: `configs/simulation_batch.yaml`)
  - `--markets-folder`
  - `--market-pattern`
  - `--output-root`
  - `--batch-id`
  - `--max-markets`
  - `--compound-balance`
  - `--no-compound-balance`
  - `--no-input`

### `ml_inference_replay.py`
- Purpose: Simulate offline ML inference using a saved reward model on a recorded market CSV.
- Run:

```bash
python ml_inference_replay.py --market-csv data/btc-updown-5m-1234567890.csv
```

- CLI options:
  - `--market-csv`
  - `--model` (default: `models/reward_model.pkl`)
  - `--output` (default: `runs/ml_inference_test.csv`)
  - `--minimum-edge` (default: `0.10`)
  - `--starting-balance` (default: `100.0`)
  - `--order-usd` (default: `1.0`)
  - `--final-outcome up|down`
  - `--log-every` (default: `1`)

### `plot_simulated_run.py`
- Purpose: Plot buy/sell decisions and optional balance curves from simulated trajectory CSVs.
- Run a single CSV:

```bash
python plot_simulated_run.py runs/momentum_basic_market_001.csv
```

- Run a folder of trajectories:

```bash
python plot_simulated_run.py --trajectory-folder runs/my_batch/trajectories --sample 5
```

- CLI options:
  - `trajectory_csv` positional (default: latest sample path)
  - `--trajectory-folder`
  - `--sample`
  - `--seed`
  - `--no-recursive`
  - `--no-balance`

### `strategy_report.py`
- Purpose: Generate an HTML performance report from a batch `summary.csv`.
- Run:

```bash
python strategy_report.py --run-folder runs/deterministic_diverse_sizing_suite --no-sample-plots
```

- CLI options:
  - `--run-folder`
  - `--output`
  - `--examples-per-strategy`
  - `--max-trajectory-files-per-strategy`
  - `--sample-plot-seed`
  - `--max-sample-plots`
  - `--no-sample-plots`

---

## 4. Training

### `train_reward_model.py`
- Purpose: Train the reward-to-go regression model from a prepared dataset.
- Run:

```bash
python train_reward_model.py --dataset training_data/reward_model_dataset.csv --output-dir models/reward_model_v1
```

- CLI options:
  - `--dataset` (default: `training_data/reward_model_dataset.csv`)
  - `--output-dir` (default: `models/reward_model`)
  - `--seed` (default: `42`)
  - `--batch-size` (default: `1024`)
  - `--epochs` (default: `100`)
  - `--early-stopping-patience` (default: `25`)
  - `--min-epochs-before-stopping` (default: `25`)
  - `--reduce-lr-patience` (default: `8`)
  - `--no-early-stopping`
  - `--balance-by-strategy`

---

## 5. Analysis and live trading

### `analyze_market_end_swings.py`
- Purpose: Analyze late price swings in replayable market CSVs and generate summary outputs.
- Run:

```bash
python analyze_market_end_swings.py
```

- CLI options:
  - `--input-dir` (default: `market_data`)
  - `--output-dir` (default: `analysis/market_end_swings`)
  - `--late-window-seconds` (default: `15.0`)

### `live_ml_trader.py`
- Purpose: Run a live paper-trading ML inference dashboard for BTC 5-minute markets.
- Run:

```bash
python live_ml_trader.py
```

- CLI options:
  - `--model` (defaults to newest model in `models/`)
  - `--minimum-edge` (default: `0.10`)
  - `--starting-balance` (default: `100.0`)
  - `--order-usd` (default: `1.0`)
  - `--max-orders-per-market` (default: `20`)
  - `--cooldown-s` (default: `5.0`)
  - `--max-position-exposure-pct` (default: `0.75`)
  - `--port` (default: `5050`)

### `btc-trader.py`
- Purpose: Wrapper for the browser-based demo/live market simulator in `scripts/live/btc_trader.py`.
- Run:

```bash
python btc-trader.py
```

---

## 6. Configuration generation helpers

These scripts generate strategy/configuration YAMLs for batch simulation.

- `scripts/simulation/generate_deterministic_strategy_configs.py`
- `scripts/simulation/generate_deterministic_sizing_strategy_configs.py`
- `scripts/simulation/generate_down_bias_chaser_balance_scaled_configs.py`

Run directly with Python if you need to produce new config sets.

---

## 7. Notes on wrappers vs direct script invocation

Most top-level root scripts are simple wrappers that delegate to the equivalent `scripts/` module path. For example:
- `python run_replay.py` ↔ `python scripts/simulation/run_replay.py`
- `python train_reward_model.py` ↔ `python scripts/training/train_reward_model.py`
- `python live_ml_trader.py` ↔ `python scripts/live/live_ml_trader.py`

If you need full package-relative imports, use the `scripts/` path directly.

---

## 8. Suggested chronological workflow

1. `python filter_full_market_csvs_v3.py`
2. `python find_simulator_ready_markets.py`
3. `python build_training_market_data.py`
4. `python build_training_dataset.py`
5. `python run_replay.py ...` for single tests or `python run_batch.py --config ...` for batch runs
6. `python ml_inference_replay.py ...` for offline ML replay
7. `python strategy_report.py --run-folder ...` for batch summary
8. `python train_reward_model.py ...` to train the ML model
9. `python analyze_market_end_swings.py` for analysis
10. `python live_ml_trader.py` or `python btc-trader.py` for live/demo usage
