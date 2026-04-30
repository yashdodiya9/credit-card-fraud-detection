"""Send one known-fraud and one known-legit row from the dataset to the running API.

Usage:  python scripts/sample_request.py data/creditcard.csv
"""
import csv
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8000/predict"


def post(row: dict) -> dict:
    payload = {k: float(v) for k, v in row.items() if k != "Class"}
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main(path: str) -> None:
    fraud = legit = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Class"] == "1" and fraud is None:
                fraud = row
            if row["Class"] == "0" and legit is None:
                legit = row
            if fraud and legit:
                break
    print("known FRAUD ->", post(fraud))
    print("known LEGIT ->", post(legit))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv")
