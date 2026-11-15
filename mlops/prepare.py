"""Generate a deterministic demonstration dataset with no real incident records."""

import csv
import math
import random
from pathlib import Path


def main():
    rng = random.Random(489141)
    output = Path("data/monthly_counts.csv")
    output.parent.mkdir(exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("area", "period", "month", "reported_count"))
        for area in range(1, 9):
            for period in range(36):
                month = period % 12 + 1
                seasonal = 7 * math.sin(2 * math.pi * month / 12)
                mean = 20 + 3 * area + 0.3 * period + seasonal
                count = max(0, round(rng.gauss(mean, 5)))
                writer.writerow((f"demo_{area:02}", period, month, count))


if __name__ == "__main__":
    main()
