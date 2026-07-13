"""Quantify per-feed staleness in recorded market CSVs (Binance vs Chainlink).

Recorded rows carry the LAST received feed value forward on every snapshot tick,
so "how fresh is the feed" cannot be read off non-null sample counts alone: a
dead feed shows up either as nulls (never seen this market) or as a frozen,
repeating value. This script therefore measures *value-change events* per feed:

  * coverage: rows with a value present, markets with zero values
  * time-to-first-value from market open (initial null run)
  * inter-change gaps: seconds between consecutive price CHANGES
  * longest frozen run per market (value present but identical)
  * stall episodes: no new price (null or frozen) for > threshold seconds,
    including session-level episodes stitched across consecutive market files

Caveat: a frozen value can be a genuinely unchanged price, not just a dead
feed. Comparing the two feeds' frozen-run distributions side by side is the
point -- BTC essentially never sits still for 30s+, so long frozen runs on one
feed while the other keeps moving indicate relay staleness, not market calm.

Usage:
    python scripts/analysis/analyze_feed_staleness.py <market_folder> [--stall-threshold 30]
"""

import argparse
import csv
import statistics
from datetime import datetime, timezone
from pathlib import Path

FEEDS = ("btc_binance", "btc_chainlink")


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def analyze_market(path: Path):
    rows = []  # (unix_time, {feed: value})
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = parse_float(row.get("unix_time"))
            if t is None:
                continue
            rows.append((t, {feed: parse_float(row.get(feed)) for feed in FEEDS}))
    if not rows:
        return None

    t_open, t_close = rows[0][0], rows[-1][0]
    result = {"slug": path.stem, "t_open": t_open, "t_close": t_close, "n_rows": len(rows)}

    for feed in FEEDS:
        present = [(t, vals[feed]) for t, vals in rows if vals[feed] is not None]
        stats = {
            "n_present": len(present),
            "first_value_delay": (present[0][0] - t_open) if present else None,
            "change_events": [],   # timestamps where the price changed
            "frozen_runs": [],     # (start, end) of maximal identical-value runs
        }
        if present:
            last_price = present[0][1]
            run_start = present[0][0]
            stats["change_events"].append(present[0][0])
            for t, price in present[1:]:
                if price != last_price:
                    stats["frozen_runs"].append((run_start, t))
                    stats["change_events"].append(t)
                    last_price = price
                    run_start = t
            stats["frozen_runs"].append((run_start, present[-1][0]))
        result[feed] = stats
    return result


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def fmt_ts(unix_time):
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime("%m-%d %H:%M:%SZ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--stall-threshold", type=float, default=30.0,
                        help="seconds without a NEW price that counts as a stall")
    args = parser.parse_args()

    results = [analyze_market(p) for p in sorted(args.folder.glob("*.csv"))]
    results = [r for r in results if r]
    print(f"folder: {args.folder}")
    print(f"markets analyzed: {len(results)}\n")

    for feed in FEEDS:
        label = feed.replace("btc_", "")
        missing = [r for r in results if r[feed]["n_present"] == 0]
        delays = [r[feed]["first_value_delay"] for r in results if r[feed]["first_value_delay"] is not None]

        change_gaps = []           # per-market gaps between consecutive price changes
        frozen_maxes = []          # longest frozen run per market
        stall_markets = []         # (slug, worst no-new-price stretch)
        for r in results:
            events = r[feed]["change_events"]
            gaps = [b - a for a, b in zip(events, events[1:])]
            change_gaps.extend(gaps)
            frozen = [(end - start) for start, end in r[feed]["frozen_runs"]]
            frozen_maxes.append(max(frozen) if frozen else None)
            # worst stretch with no new price, counting leading/trailing silence
            stretches = list(gaps)
            if events:
                stretches.append(events[0] - r["t_open"])
                stretches.append(r["t_close"] - events[-1])
            else:
                stretches.append(r["t_close"] - r["t_open"])
            worst = max(stretches) if stretches else 0.0
            if worst > args.stall_threshold:
                stall_markets.append((r["slug"], worst))

        print(f"=== {label} ===")
        print(f"  markets with zero samples: {len(missing)}")
        for r in missing[:12]:
            print(f"    missing: {r['slug']}")
        if delays:
            print(f"  time to first value after open (s): median={statistics.median(delays):.1f} "
                  f"p95={pct(delays, 0.95):.1f} max={max(delays):.1f}")
        if change_gaps:
            print(f"  gap between price CHANGES (s): median={statistics.median(change_gaps):.2f} "
                  f"p90={pct(change_gaps, 0.90):.2f} p99={pct(change_gaps, 0.99):.2f} max={max(change_gaps):.1f}")
            for threshold in (5, 10, 30, 60):
                n = sum(1 for g in change_gaps if g > threshold)
                print(f"    change-gaps > {threshold}s: {n}")
        valid_frozen = [v for v in frozen_maxes if v is not None]
        if valid_frozen:
            print(f"  longest frozen run per market (s): median={statistics.median(valid_frozen):.1f} "
                  f"p95={pct(valid_frozen, 0.95):.1f} max={max(valid_frozen):.1f}")
        print(f"  markets with a no-new-price stretch > {args.stall_threshold:.0f}s: {len(stall_markets)}")
        for slug, worst in sorted(stall_markets, key=lambda x: -x[1])[:12]:
            print(f"    {slug}: {worst:.0f}s")
        print()

    # Session-level stalls: stitch all markets in slug order and find stretches
    # where a feed produced no NEW price across market boundaries. Folders can
    # have holes between recordings (the bench is non-contiguous), so an episode
    # only counts by its RECORDED portion -- time actually covered by market
    # files -- otherwise recording gaps masquerade as feed stalls.
    print(f"=== session-level stalls (> {args.stall_threshold:.0f}s recorded, stitched across markets) ===")
    session_start = min(r["t_open"] for r in results)
    session_end = max(r["t_close"] for r in results)
    intervals = sorted((r["t_open"], r["t_close"]) for r in results)
    total_recorded = sum(e - s for s, e in intervals)

    def recorded_within(start, end):
        return sum(max(0.0, min(e, end) - max(s, start)) for s, e in intervals if s < end and e > start)

    for feed in FEEDS:
        label = feed.replace("btc_", "")
        events = sorted(t for r in results for t in r[feed]["change_events"])
        episodes = []
        prev = session_start
        for t in events + [session_end]:
            recorded = recorded_within(prev, t)
            if recorded > args.stall_threshold:
                episodes.append((prev, t, recorded))
            prev = t
        total_stalled = sum(rec for _, _, rec in episodes)
        print(f"  {label}: {len(episodes)} episodes, {total_stalled:.0f}s stalled "
              f"of {total_recorded:.0f}s recorded ({100 * total_stalled / total_recorded:.2f}%)")
        for s, e, rec in sorted(episodes, key=lambda x: -x[2])[:10]:
            print(f"    {fmt_ts(s)} -> {fmt_ts(e)}  ({rec:.0f}s recorded)")


if __name__ == "__main__":
    main()
