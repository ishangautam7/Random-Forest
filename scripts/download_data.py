import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from random_forest_scratch.data import download_wine_data


if __name__ == "__main__":
    destination = Path("data/winequality-red.csv")
    result = download_wine_data(destination)
    print(f"Downloaded and verified: {result} ({result.stat().st_size:,} bytes)")
