import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).resolve().parent / "scripts" / "live" / "live_strategy_suite.py",
        run_name="__main__",
    )
