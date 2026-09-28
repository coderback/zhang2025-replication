"""F1 data: GDELT 1.0 daily event exports, filtered to CAMEO base codes 100-199 and reduced to each file's top
events by NumArticles (preregistration/PREREGISTRATION.md). Resumable: one small parquet per day in
data/gdelt_events/. Nothing else from the ~12 MB daily files is kept."""
from __future__ import annotations

import io
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from zhang_replication.config import DATA_DIR

OUT = DATA_DIR / "gdelt_events"
URL = "https://data.gdeltproject.org/events/{d}.export.CSV.zip"
COLS = {0: "event_id", 1: "sqldate", 27: "base_code", 30: "goldstein", 33: "num_articles", 56: "dateadded", 57: "url"}
MIN_FREE_BYTES = 1_000_000_000
KEEP_PER_FILE = 1500             # >> 100: events dated t can arrive in later files, and each variant needs a top 100


def filter_day(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(z.open(z.namelist()[0]), sep="\t", header=None, usecols=list(COLS), dtype=str,
                     quoting=3, on_bad_lines="skip")
    df = df.rename(columns=COLS)
    code = pd.to_numeric(df["base_code"], errors="coerce")
    df = df[(code >= 100) & (code <= 199)].copy()
    df["num_articles"] = pd.to_numeric(df["num_articles"], errors="coerce")
    df["goldstein"] = pd.to_numeric(df["goldstein"], errors="coerce")
    return df.nlargest(KEEP_PER_FILE, "num_articles")


def fetch_day(d: pd.Timestamp, retries: int = 4) -> str:
    path = OUT / f"{d:%Y%m%d}.parquet"
    if path.exists():
        return "cached"
    for k in range(retries):
        try:
            raw = urllib.request.urlopen(URL.format(d=f"{d:%Y%m%d}"), timeout=120).read()
            filter_day(raw).to_parquet(path, index=False)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(5 * (k + 1))
    return "failed"


def download(start: str = "2015-01-01", end: str = "2024-10-02", workers: int = 4, log=print) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(start, end, freq="D")
    counts: dict[str, int] = {}
    todo = [d for d in days if not (OUT / f"{d:%Y%m%d}.parquet").exists()]
    counts["cached"] = len(days) - len(todo)
    for k in range(0, len(todo), 100):                       # batches, with a free-disk guard between them
        if shutil.disk_usage(OUT).free < MIN_FREE_BYTES:
            log(f"STOPPED: less than {MIN_FREE_BYTES / 1e9:.0f} GB free on disk", flush=True)
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for status in ex.map(fetch_day, todo[k:k + 100]):
                counts[status] = counts.get(status, 0) + 1
        log(f"{counts['cached'] + k + len(todo[k:k + 100])}/{len(days)} days {counts}", flush=True)
    log(f"done {counts}", flush=True)



def load(start=None, end=None) -> pd.DataFrame:
    files = sorted(OUT.glob("*.parquet"))
    if start:
        files = [f for f in files if f.stem >= pd.Timestamp(start).strftime("%Y%m%d")]
    if end:
        files = [f for f in files if f.stem <= pd.Timestamp(end).strftime("%Y%m%d")]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    download()
