from pathlib import Path

from simulator.batch import run_batch


DEFAULT_CONFIG = Path("configs/simulation_batch.yaml")


def main():
    print("BTC simulator batch runner")
    print()

    config_input = input(f"Batch config path [{DEFAULT_CONFIG}]: ").strip()
    config_path = Path(config_input) if config_input else DEFAULT_CONFIG

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    run_batch(config_path)


if __name__ == "__main__":
    main()
