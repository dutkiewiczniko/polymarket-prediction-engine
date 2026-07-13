"""Deep-dive the markets where Binance- and Chainlink-referenced outcomes disagree.

For each market, reports:
  * when each feed's FIRST sample arrived relative to market open (from the
    slug's unix start time) -- a late first sample means that source's "start
    price" is really the price N seconds in, which can be wrong in fast opens
  * the recorded price_to_beat and its source column (gamma vs chainlink
    fallback vs binance fallback -- the bench was recorded with fallbacks on)
  * a better start-price estimate per feed: the LAST sample of the PREVIOUS
    market's file (markets are contiguous 5-min slots, so that sample is
    ~0.1s before this market's open)
  * closing margins per source, so we can see how thin the flip was.

Usage:
    python scripts/analysis/inspect_source_disagreement_markets.py <folder> <slug> [<slug> ...]
    (no slugs -> auto-detect all disagreement markets in the folder)
"""

import csv
import sys
from pathlib import Path


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


def load_series(path: Path):
    binance, chainlink = [], []
    ptb = None
    ptb_source = None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = parse_float(row.get("unix_time"))
            if t is None:
                continue
            b = parse_float(row.get("btc_binance"))
            c = parse_float(row.get("btc_chainlink"))
            if b is not None:
                binance.append((t, b))
            if c is not None:
                chainlink.append((t, c))
            if ptb is None:
                ptb = parse_float(row.get("price_to_beat"))
                src = (row.get("price_to_beat_source") or "").strip()
                if src:
                    ptb_source = src
    return binance, chainlink, ptb, ptb_source


def implied(series):
    if len(series) < 2:
        return None
    return "up" if series[-1][1] >= series[0][1] else "down"


def main():
    folder = Path(sys.argv[1])
    slugs = sys.argv[2:]

    files = {p.stem: p for p in folder.glob("*.csv")}

    if not slugs:
        slugs = []
        for stem, path in sorted(files.items()):
            b, c, _, _ = load_series(path)
            ob, oc = implied(b), implied(c)
            if ob and oc and ob != oc:
                slugs.append(stem)
        print(f"auto-detected {len(slugs)} disagreement markets\n")

    for slug in slugs:
        path = files[slug]
        start_t = int(slug.rsplit("-", 1)[-1])
        b, c, ptb, ptb_source = load_series(path)

        prev_slug = f"btc-updown-5m-{start_t - 300}"
        prev_b_last = prev_c_last = None
        if prev_slug in files:
            pb, pc, _, _ = load_series(files[prev_slug])
            if pb:
                prev_b_last = pb[-1]
            if pc:
                prev_c_last = pc[-1]

        print(f"=== {slug} (start={start_t}) ===")
        print(f"  recorded ptb = {ptb}  source = {ptb_source}")
        for name, series, prev_last in (("binance", b, prev_b_last), ("chainlink", c, prev_c_last)):
            if not series:
                print(f"  {name}: NO SAMPLES")
                continue
            first_t, first_p = series[0]
            last_t, last_p = series[-1]
            line = (f"  {name}: first sample at +{first_t - start_t:.1f}s = {first_p:.2f}, "
                    f"last at +{last_t - start_t:.1f}s = {last_p:.2f}, "
                    f"move first->last = {last_p - first_p:+.2f} ({implied(series)})")
            if prev_last is not None:
                pt, pp = prev_last
                line += f"\n           prev-market last sample at {pt - start_t:+.1f}s = {pp:.2f} (true-start est; first-sample drift {first_p - pp:+.2f})"
            print(line)
        print()


if __name__ == "__main__":
    main()
