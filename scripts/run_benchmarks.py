"""CLI script to execute full comparative research benchmarks and output metrics and figures."""

import sys
import logging
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agri_drift.evaluation.benchmark import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmarks")


def main():
    logger.info("Initializing Agri-Drift Comparative Benchmark Runner...")
    runner = BenchmarkRunner(config_path="configs/experiment.yaml")

    logger.info("Running multi-seed evaluation across all 4 detector systems and 5 fault types...")
    results = runner.run_full_evaluation()

    logger.info("=== Overall System Performance ===")
    print(results["overall"].to_string(index=False))

    logger.info("=== Performance by Fault Class ===")
    print(results["by_fault"].to_string(index=False))

    logger.info("All metrics written to 'results/metrics/'")
    logger.info("Publication figures saved to 'results/figures/'")


if __name__ == "__main__":
    main()

