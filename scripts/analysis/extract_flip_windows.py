"""Extract "down-state windows" as per-window CSVs for flip-prediction modelling.

A window = one episode of a side being priced nearly-dead:
  * ENTRY: first tick with 0 < side price <= --entry-threshold (default 0.20)
           and seconds_left >= --min-seconds-left.
  * END:   price recovers to >= --exit-price (default 0.50, the "state has
           flipped back to even" point) or the market ends. After a recovery
           the side re-arms; a later dip below the entry threshold starts a
           NEW window (episode_num increments).

Each window becomes windows/<slug>_<side>_w<ep>_<TAG>.csv of per-tick features
(trailing-only, no lookahead), TAG = FLIP (side truly won at resolution),
REC (recovered to >= 0.50 but lost), DEAD (neither). manifest.csv holds one
row per window with market/cohort metadata and ALL labels so the training
target can be changed without re-extracting:
  recovered_50/75/90/95, won_resolution, max_price_after, time_to_50_s.

Outcome labels come ONLY from the true-outcomes CSV (real Polymarket
resolutions) -- never inferred from ticks (42/320 inference errors).

Long momentum (600s/900s) uses BTC history stitched from predecessor markets
(slug timestamps step by 300s: T-300, T-600, T-900 when present in the same
folder). Missing chain => NaN momentum (GBMs handle it), never a dropped
market; manifest has has_hist_600s/has_hist_900s flags.

Feature notes:
  * All BTC-derived series (momentum, vol) are computed on a 1s grid built
    over history+market, then mapped onto ticks; %s are relative to strike.
  * z_deficit = deficit_pct / (rv_60s * sqrt(seconds_left)) -- deficit in
    units of plausible BTC movement given time left ("moneyness").
  * btc_staleness_s tracks time since the merged BTC value last changed
    (Chainlink outage stretches exist in the July live data -- filter at
    training time, windows are never excluded here).

Usage:
    python scripts/analysis/extract_flip_windows.py \
        --markets-folder simulator_ready_official_clean_plus_live \
        --true-outcomes-csv simulator_ready_official_clean_plus_live_true_outcomes.csv \
        --out-dir flip_windows_v1
"""

import argparse
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd

MOM_WINDOWS_S = (15, 60, 120, 240, 600, 900)
RV_WINDOWS_S = (30, 60, 120, 300)
RANGE_WINDOWS_S = (60, 120)
EWMA_HALFLIFE_S = 30
TOKEN_VOL_S = 60
ACCEL_LAG_S = 30
CRASH_SPEED_S = 30
LABEL_LEVELS = (0.50, 0.75, 0.90, 0.95)
PRED_OFFSETS_S = (300, 600, 900)  # predecessor markets for long momentum
HIST_GAP_TOL_S = 45  # max staleness of the history sample used for momentum

BOOK_COLS = ["best_bid", "best_ask", "spread",
             "ask_usd_within_1c", "bid_usd_within_1c",
             "ask_usd_within_2c", "bid_usd_within_2c",
             "ask_usd_within_5c", "bid_usd_within_5c"]


def load_truth(csv_path: str) -> dict:
    truth = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            outcome = (row.get("true_outcome") or "").strip().lower()
            if outcome in ("up", "down"):
                truth[row["slug"]] = outcome
    return truth


def merged_btc(df: pd.DataFrame) -> pd.Series:
    return df["btc_chainlink"].where(df["btc_chainlink"].notna(),
                                     df["btc_binance"]).ffill()


def load_history(folder: Path, slug: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """BTC series of up to 3 predecessor markets (walk back, stop at a gap)."""
    ts = int(slug.rsplit("-", 1)[1])
    times, btcs = [], []
    flags = {}
    for off in PRED_OFFSETS_S:
        flags[f"has_hist_{off}s"] = False
    for off in PRED_OFFSETS_S:
        path = folder / f"{slug.rsplit('-', 1)[0]}-{ts - off}.csv"
        if not path.exists():
            break
        h = pd.read_csv(path, usecols=["unix_time", "btc_binance",
                                       "btc_chainlink"])
        b = merged_btc(h)
        ok = b.notna().to_numpy()
        times.append(h["unix_time"].to_numpy(dtype=float)[ok])
        btcs.append(b.to_numpy()[ok])
        flags[f"has_hist_{off}s"] = True
    if times:
        order = np.argsort([t[0] for t in times])
        times = np.concatenate([times[i] for i in order])
        btcs = np.concatenate([btcs[i] for i in order])
        return times, btcs, flags
    return np.array([]), np.array([]), flags


def grid_lookup(grid_t: np.ndarray, arr: np.ndarray,
                at: np.ndarray) -> np.ndarray:
    """Value of a 1s-grid series at (or just before) each time in `at`."""
    j = np.searchsorted(grid_t, at, side="right") - 1
    out = np.full(len(at), np.nan)
    ok = j >= 0
    out[ok] = arr[j[ok]]
    return out


def extract_market(csv_path: Path, folder: Path, true_outcome: str,
                   args) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    df = pd.read_csv(csv_path)
    slug = csv_path.stem
    ptb = df["price_to_beat"].dropna()
    if ptb.empty:
        return [], {}
    strike = float(ptb.iloc[-1])
    times = df["unix_time"].to_numpy(dtype=float)
    seconds_left = df["seconds_left"].to_numpy(dtype=float)
    btc_now = merged_btc(df)
    ok_btc = btc_now.notna().to_numpy()

    # ---- stitched history + current BTC series ----
    h_times, h_btc, hist_flags = load_history(folder, slug)
    all_t = np.concatenate([h_times, times[ok_btc]])
    all_b = np.concatenate([h_btc, btc_now.to_numpy()[ok_btc]])
    if all_t.size < 2:
        return [], {}

    # ---- 1s grid over history + market ----
    g_t = np.arange(np.floor(all_t[0]), np.ceil(all_t[-1]) + 1.0)
    j = np.searchsorted(all_t, g_t, side="right") - 1
    g_valid = j >= 0
    g_b = np.where(g_valid, all_b[np.clip(j, 0, None)], np.nan)
    g = pd.Series(g_b)

    # momentum on the grid: btc(t) - btc(t-W), % of strike; NaN when the
    # lookback sample is missing or the series had a gap > HIST_GAP_TOL_S
    sample_age = g_t - np.where(g_valid, all_t[np.clip(j, 0, None)], -np.inf)
    mom_grid = {}
    for w in MOM_WINDOWS_S:
        past = g.shift(w)
        past_age = pd.Series(sample_age).shift(w)
        m = (g - past) / strike * 100.0
        m[past_age > HIST_GAP_TOL_S] = np.nan
        mom_grid[w] = m.to_numpy()

    # volatility on the grid (1s returns, % of strike)
    r = g.diff() / strike * 100.0
    rv_grid = {w: r.rolling(w, min_periods=max(10, w // 3)).std().to_numpy()
               for w in RV_WINDOWS_S}
    range_grid = {
        w: ((g.rolling(w, min_periods=max(10, w // 3)).max()
             - g.rolling(w, min_periods=max(10, w // 3)).min())
            / strike * 100.0).to_numpy()
        for w in RANGE_WINDOWS_S}
    ewma_grid = np.sqrt(
        (r ** 2).ewm(halflife=EWMA_HALFLIFE_S, min_periods=10).mean()
    ).to_numpy()

    # BTC staleness: time since merged value last changed (over full series)
    changed = np.empty(all_t.size, dtype=bool)
    changed[0] = True
    changed[1:] = all_b[1:] != all_b[:-1]
    last_change = pd.Series(np.where(changed, all_t, np.nan)).ffill().to_numpy()
    # staleness measured at grid time = grid time - last change before it
    lc_at_grid = grid_lookup(all_t, last_change, g_t)
    stale_grid = g_t - lc_at_grid

    # basis (binance - chainlink), % of strike, only where both are fresh
    both = df["btc_binance"].notna() & df["btc_chainlink"].notna()
    basis = np.where(
        both.to_numpy(),
        (df["btc_binance"] - df["btc_chainlink"]).to_numpy() / strike * 100.0,
        np.nan)

    # per-tick lookups into the grid
    def at_ticks(arr: np.ndarray) -> np.ndarray:
        return grid_lookup(g_t, arr, times)

    tick = {"stale": at_ticks(stale_grid), "ewma": at_ticks(ewma_grid)}
    for w in MOM_WINDOWS_S:
        tick[f"mom_{w}"] = at_ticks(mom_grid[w])
    for w in RV_WINDOWS_S:
        tick[f"rv_{w}"] = at_ticks(rv_grid[w])
    for w in RANGE_WINDOWS_S:
        tick[f"range_{w}"] = at_ticks(range_grid[w])
    # momentum acceleration: mom_15s now vs ACCEL_LAG_S ago
    mom15_lag = pd.Series(mom_grid[15]).shift(ACCEL_LAG_S).to_numpy()
    tick["mom_accel"] = tick["mom_15"] - at_ticks(mom15_lag)
    vol_ratio = np.where(tick["rv_300"] > 0,
                         tick["rv_30"] / tick["rv_300"], np.nan)

    # token-price vol per side on a 1s in-market grid
    mkt_g_t = np.arange(np.floor(times[0]), np.ceil(times[-1]) + 1.0)
    token_vol = {}
    for side in ("up", "down"):
        p = pd.Series(grid_lookup(times, df[f"{side}_price"].to_numpy(),
                                  mkt_g_t))
        tv = p.diff().rolling(TOKEN_VOL_S,
                              min_periods=TOKEN_VOL_S // 3).std().to_numpy()
        token_vol[side] = grid_lookup(mkt_g_t, tv, times)

    btc_arr = btc_now.to_numpy()
    manifest_rows, window_frames = [], {}
    cohort = ("july_live" if int(slug.rsplit("-", 1)[1]) >= 1780000000
              else "may_bench")

    for side in ("up", "down"):
        opp = "down" if side == "up" else "up"
        prices = df[f"{side}_price"].to_numpy()
        sign = 1.0 if side == "up" else -1.0
        deficit = sign * (strike - btc_arr) / strike * 100.0
        won = side == true_outcome

        eligible = ((seconds_left >= args.min_seconds_left)
                    & (prices > 0) & (prices <= args.entry_threshold))
        episode = 0
        i = 0
        n = len(df)
        while i < n:
            if not eligible[i]:
                i += 1
                continue
            episode += 1
            i0 = i
            # window ends at recovery (inclusive of the crossing tick) or EOM
            i1 = n - 1
            reason = "market_end"
            for k in range(i0 + 1, n):
                if prices[k] >= args.exit_price:
                    i1, reason = k, "recovered"
                    break
            sl = slice(i0, i1 + 1)

            # labels from everything at/after entry (full market tail)
            tail = prices[i0:]
            tail = tail[~np.isnan(tail)]
            max_after = float(tail.max()) if tail.size else np.nan
            labels = {f"recovered_{int(x * 100)}":
                      bool(max_after >= x) for x in LABEL_LEVELS}
            t50_idx = np.flatnonzero(prices[i0:] >= 0.50)
            time_to_50 = (float(times[i0 + t50_idx[0]] - times[i0])
                          if t50_idx.size else np.nan)

            d_entry = deficit[i0]
            run_min = np.minimum.accumulate(
                np.where(np.isnan(prices[sl]), np.inf, prices[sl]))
            rv60 = tick["rv_60"][sl]
            sec = seconds_left[sl]
            with np.errstate(divide="ignore", invalid="ignore"):
                z_def = np.where(rv60 > 0,
                                 deficit[sl] / (rv60 * np.sqrt(np.maximum(sec, 1e-9))),
                                 np.nan)
                retrace = (1.0 - deficit[sl] / d_entry
                           if d_entry and d_entry > 0 else np.full(i1 - i0 + 1, np.nan))

            def book(side_name: str, prefix: str) -> dict:
                cols = {}
                for c in BOOK_COLS:
                    cols[f"{prefix}{c.replace('_within', '')}"] = \
                        df[f"{side_name}_{c}"].to_numpy()[sl]
                bid1 = df[f"{side_name}_bid_usd_within_1c"].to_numpy()[sl]
                ask1 = df[f"{side_name}_ask_usd_within_1c"].to_numpy()[sl]
                tot = bid1 + ask1
                with np.errstate(divide="ignore", invalid="ignore"):
                    cols[f"{prefix}imbalance_1c"] = np.where(tot > 0,
                                                             bid1 / tot, np.nan)
                return cols

            data = {
                "unix_time": times[sl], "seconds_left": sec,
                "time_in_window_s": times[sl] - times[i0],
                "price": prices[sl],
                "other_price": df[f"{opp}_price"].to_numpy()[sl],
                "price_vs_min": prices[sl] - run_min,
                "btc": btc_arr[sl], "deficit_pct": deficit[sl],
                "z_deficit": z_def, "retrace_frac": retrace,
                "mom_accel": tick["mom_accel"][sl],
                "ewma_vol_30s": tick["ewma"][sl],
                "vol_ratio": vol_ratio[sl],
                "token_vol_60s": token_vol[side][sl],
                "btc_staleness_s": tick["stale"][sl],
                "basis_pct": basis[sl],
            }
            for w in MOM_WINDOWS_S:
                data[f"mom_toward_{w}s"] = sign * tick[f"mom_{w}"][sl]
            for w in RV_WINDOWS_S:
                data[f"rv_{w}s"] = tick[f"rv_{w}"][sl]
            for w in RANGE_WINDOWS_S:
                data[f"range_{w}s"] = tick[f"range_{w}"][sl]
            data.update(book(side, ""))
            data.update(book(opp, "opp_"))

            tag = "FLIP" if won else ("REC" if labels["recovered_50"]
                                      else "DEAD")
            fname = f"{slug}_{side}_w{episode}_{tag}.csv"
            window_frames[fname] = pd.DataFrame(data)

            # crash speed: how much the side price fell over the 30s pre-entry
            pre = np.flatnonzero(times < times[i0] - CRASH_SPEED_S)
            crash = (float(prices[pre[-1]] - prices[i0])
                     if pre.size and not np.isnan(prices[pre[-1]]) else np.nan)

            manifest_rows.append({
                "window_file": fname, "market": slug, "side": side,
                "cohort": cohort, "episode_num": episode,
                "entry_price": float(prices[i0]),
                "entry_seconds_left": float(seconds_left[i0]),
                "entry_unix_time": float(times[i0]),
                "entry_deficit_pct": float(d_entry),
                "crash_speed_30s": crash,
                "hour_utc": int((times[i0] // 3600) % 24),
                "n_ticks": i1 - i0 + 1,
                "window_end_reason": reason,
                "won_resolution": won,
                "max_price_after": max_after,
                "time_to_50_s": time_to_50,
                **labels, **hist_flags,
            })

            # resume scanning after this window; re-arm requires the price to
            # have recovered (windows ending at market end terminate the side)
            i = i1 + 1
        # end while
    return manifest_rows, window_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets-folder", required=True)
    ap.add_argument("--true-outcomes-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--entry-threshold", type=float, default=0.20)
    ap.add_argument("--exit-price", type=float, default=0.50)
    ap.add_argument("--min-seconds-left", type=float, default=5.0)
    args = ap.parse_args()

    truth = load_truth(args.true_outcomes_csv)
    folder = Path(args.markets_folder)
    out = Path(args.out_dir)
    win_dir = out / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)

    manifest, skipped = [], 0
    files = sorted(folder.glob("*.csv"))
    for k, path in enumerate(files):
        outcome = truth.get(path.stem)
        if outcome is None:
            skipped += 1
            continue
        rows, frames = extract_market(path, folder, outcome, args)
        manifest.extend(rows)
        for fname, frame in frames.items():
            frame.to_csv(win_dir / fname, index=False)
        if (k + 1) % 50 == 0:
            print(f"  ...{k + 1}/{len(files)} markets, "
                  f"{len(manifest)} windows so far")

    mdf = pd.DataFrame(manifest)
    mdf.to_csv(out / "manifest.csv", index=False)
    print(f"\nwrote {len(mdf)} windows from {len(files) - skipped} markets "
          f"(skipped {skipped} without truth) -> {out}")
    if mdf.empty:
        return
    print(f"cohorts: {mdf.groupby('cohort').size().to_dict()}")
    print(f"tags: FLIP={int(mdf['won_resolution'].sum())}, "
          f"REC(no win)={int((mdf['recovered_50'] & ~mdf['won_resolution']).sum())}, "
          f"DEAD={int((~mdf['recovered_50'] & ~mdf['won_resolution']).sum())}")
    print("label rates:")
    for x in LABEL_LEVELS:
        c = f"recovered_{int(x * 100)}"
        print(f"  {c}: {int(mdf[c].sum())} ({100 * mdf[c].mean():.1f}%)")
    print(f"  won_resolution: {int(mdf['won_resolution'].sum())} "
          f"({100 * mdf['won_resolution'].mean():.1f}%)")
    print(f"full 900s history: {int(mdf['has_hist_900s'].sum())} windows "
          f"({100 * mdf['has_hist_900s'].mean():.1f}%)")
    by = mdf.groupby("cohort")["won_resolution"].agg(["count", "sum", "mean"])
    print(f"\nby cohort:\n{by.to_string()}")


if __name__ == "__main__":
    main()
