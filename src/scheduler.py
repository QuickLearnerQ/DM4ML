"""Simple reproducible scheduler for the complete recommendation workflow."""

import argparse
import logging
import math
import time

from pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval-hours",
        type=float,
        help="Repeat the complete pipeline at this interval; omit to run once.",
    )
    args = parser.parse_args()
    if args.interval_hours is not None and (
        not math.isfinite(args.interval_hours) or args.interval_hours <= 0
    ):
        parser.error("--interval-hours must be a finite positive number.")
    interval_seconds = args.interval_hours * 60 * 60 if args.interval_hours is not None else None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    while True:
        started = time.monotonic()
        run()
        if interval_seconds is None:
            break
        time.sleep(max(0.0, interval_seconds - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
