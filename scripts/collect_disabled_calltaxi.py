"""Collector for Seoul Open Data Plaza disabled call taxi (disabledCalltaxi) trip records.

Usage:
    python scripts/collect_disabled_calltaxi.py --start 2026-05-01 --end 2026-06-30 \
        --out data/raw_data_2026.parquet

- Reads the API key from SEOUL_OPENAPI_KEY in the project root .env.
- This API is XML-only; requesting start/end index 1/1000 returns a full
  day's data in one call (verified empirically; pages beyond 1001 are empty).
- Provided columns: vehicle ID (no), vehicle type (cartype), receipt time (receipttime),
  dispatch time (settime), pickup time (ridetime), origin gu/dong, destination gu/dong.
  Drop-off/cancellation times, fare, and trip distance are not provided by the API.
"""

import argparse
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVICE = "disabledCalltaxi"


def load_api_key() -> str:
    env_path = PROJECT_ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("SEOUL_OPENAPI_KEY"):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f"SEOUL_OPENAPI_KEY not found in .env: {env_path}")


def parse_korean_datetime(text: str):
    """Convert strings like '2026-07-28 오전 12:06:00' (Korean AM/PM) to datetime."""
    if not text or not text.strip():
        return pd.NaT
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s+(오전|오후)\s+(\d{1,2}):(\d{2}):(\d{2})", text.strip())
    if not m:
        return pd.NaT
    day, ampm, hh, mm, ss = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
    if ampm == "오전":
        hh = 0 if hh == 12 else hh
    else:
        hh = 12 if hh == 12 else hh + 12
    return datetime.strptime(f"{day} {hh:02d}:{mm:02d}:{ss:02d}", "%Y-%m-%d %H:%M:%S")


def fetch_day(key: str, day: date, retries: int = 3) -> list[dict]:
    url = f"http://openapi.seoul.go.kr:8088/{key}/xml/{SERVICE}/1/1000/{day:%Y%m%d}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
            root = ET.fromstring(raw)
            code = root.findtext(".//RESULT/CODE", default="")
            if code == "INFO-200":  # no data
                return []
            if code != "INFO-000":
                raise RuntimeError(f"API error {code}: {root.findtext('.//RESULT/MESSAGE', default='')}")
            rows = []
            for item in root.iter("item"):
                rows.append({
                    "차량고유번호": (item.findtext("no") or "").strip(),
                    "차량타입": (item.findtext("cartype") or "").strip(),
                    "접수일시": parse_korean_datetime(item.findtext("receipttime") or ""),
                    "배차일시": parse_korean_datetime(item.findtext("settime") or ""),
                    "승차일시": parse_korean_datetime(item.findtext("ridetime") or ""),
                    "출발지구": (item.findtext("startpos1") or "").strip(),
                    "출발지동": (item.findtext("startpos2") or "").strip(),
                    "목적지구": (item.findtext("endpos1") or "").strip(),
                    "목적지동": (item.findtext("endpos2") or "").strip(),
                })
            total = int(root.findtext(".//list_total_count", default="0"))
            if len(rows) != total:
                print(f"  [warning] {day}: list_total_count={total} vs parsed rows={len(rows)}")
            return rows
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  [retry {attempt}/{retries}] {day}: {e}")
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="output parquet path")
    args = ap.parse_args()

    key = load_api_key()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    all_rows = []
    day = start
    while day <= end:
        rows = fetch_day(key, day)
        all_rows.extend(rows)
        print(f"{day}: {len(rows):,} rows (cumulative {len(all_rows):,})")
        day += timedelta(days=1)
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nSaved: {out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
