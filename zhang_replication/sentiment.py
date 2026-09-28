"""F1 headlines and FinBERT scores (preregistration/PREREGISTRATION.md).

Headlines come from URL slugs (the paper scraped each page; see the pre-registration). FinBERT = ProsusAI/finbert,
score s = P(positive) - P(negative). Scores are cached per unique text in data/gdelt_events/_finbert.parquet so the
long CPU run can resume. The public model is fetched without the machine's stored Hugging Face token (token=False).
"""
from __future__ import annotations

import re
import urllib.parse

import numpy as np
import pandas as pd

from zhang_replication.config import DATA_DIR

MODEL = "ProsusAI/finbert"
CACHE = DATA_DIR / "gdelt_events" / "_finbert.parquet"
_ID = re.compile(r"^(?:[0-9a-f]{6,}|\d+|[a-z]{0,3}\d{4,}[a-z0-9]*)$", re.I)
_EXT = re.compile(r"\.(s?html?|php|aspx?|cms|jsp|ece|amp)$", re.I)


def slug_text(url: str) -> str | None:
    """Headline words from a URL slug: the path segment with the most words (>= 3), ids and extensions dropped."""
    if not isinstance(url, str):
        return None
    path = urllib.parse.urlparse(url).path
    best: list[str] = []
    for seg in path.split("/"):
        seg = re.sub(r"^\d+\.", "", _EXT.sub("", urllib.parse.unquote(seg)))      # "17493833.ethiopia-..."
        words = [w for w in re.split(r"[-_+]+", seg) if w and not _ID.match(w)]
        words = [re.sub(r"[^A-Za-z0-9'%$.]", "", w) for w in words]
        words = [w for w in words if w]
        if len(words) > len(best):
            best = words
    return " ".join(best).lower() if len(best) >= 3 else None


class FinBert:
    def __init__(self, batch: int = 64, max_len: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MODEL, token=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, token=False).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.pos, self.neg = lab["positive"], lab["negative"]
        self.batch, self.max_len = batch, max_len
        torch.set_num_threads(max(1, torch.get_num_threads()))

    def score(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch):
                enc = self.tok(texts[i:i + self.batch], padding=True, truncation=True, max_length=self.max_len,
                               return_tensors="pt")
                p = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
                out.append(p[:, self.pos] - p[:, self.neg])
        return np.concatenate(out) if out else np.zeros(0)


SHIPPED = DATA_DIR.parent / "results" / "finbert_scores.parquet"   # scores published with this repository


def load_cache() -> pd.Series:
    src = CACHE if CACHE.exists() else SHIPPED
    if src.exists():
        c = pd.read_parquet(src)
        return pd.Series(c["score"].to_numpy(), index=c["text"].to_numpy())
    return pd.Series(dtype=float)


def score_all(texts: list[str], chunk: int = 5000, log=print) -> pd.Series:
    """FinBERT score per unique text, resumable (the cache is rewritten after every chunk)."""
    cache = load_cache()
    todo = sorted(set(t for t in texts if isinstance(t, str)) - set(cache.index))
    log(f"FinBERT: {len(cache)} cached, {len(todo)} to score", flush=True)
    if todo:
        fb = FinBert()
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            cache = pd.concat([cache, pd.Series(fb.score(part), index=part)])
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"text": cache.index, "score": cache.to_numpy()}).to_parquet(CACHE, index=False)
            log(f"  scored {i + len(part)}/{len(todo)}", flush=True)
    return cache
