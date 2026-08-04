"""서울 열린데이터광장 장애인콜택시(disabledCalltaxi) 운행이력 수집 스크립트.

사용법:
    python scripts/collect_disabled_calltaxi.py --start 2026-05-01 --end 2026-06-30 \
        --out data/raw_data_2026.parquet

- API 키는 프로젝트 루트의 .env 에서 서울시APIKEY 값을 읽는다.
- 이 API는 XML 전용이며, 시작/종료 인덱스를 1/1000 으로 주면 해당 일자의
  전체 데이터가 한 번에 반환된다 (실측 기준; 1001 이후 페이지는 빈 응답).
- 제공 컬럼: 차량고유번호(no), 차량타입(cartype), 접수일시(receipttime),
  배차일시(settime), 승차일시(ridetime), 출발지 구/동, 목적지 구/동.
  하차·취소일시, 요금, 승차거리 등은 API에서 제공하지 않는다.
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
        if line.strip().startswith("서울시APIKEY"):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f".env에서 서울시APIKEY를 찾지 못했습니다: {env_path}")


def parse_korean_datetime(text: str):
    """'2026-07-28 오전 12:06:00' 형식을 datetime으로 변환."""
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
            if code == "INFO-200":  # 데이터 없음
                return []
            if code != "INFO-000":
                raise RuntimeError(f"API 오류 {code}: {root.findtext('.//RESULT/MESSAGE', default='')}")
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
                print(f"  [경고] {day}: list_total_count={total} vs 파싱된 행={len(rows)}")
            return rows
        except Exception as e:
            if attempt == retries:
                raise
            print(f"  [재시도 {attempt}/{retries}] {day}: {e}")
            time.sleep(2 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="저장할 parquet 경로")
    args = ap.parse_args()

    key = load_api_key()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    all_rows = []
    day = start
    while day <= end:
        rows = fetch_day(key, day)
        all_rows.extend(rows)
        print(f"{day}: {len(rows):,}건 (누적 {len(all_rows):,})")
        day += timedelta(days=1)
        time.sleep(0.3)

    df = pd.DataFrame(all_rows)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n저장 완료: {out} ({len(df):,}행)")


if __name__ == "__main__":
    main()
