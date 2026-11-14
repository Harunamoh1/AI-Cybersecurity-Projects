"""Build a corpus of legitimate *deep* URLs (URLs with real paths).

Why: in the public phishing datasets almost every legitimate URL is a bare home
page, so a model learns "has a path => phishing" and flags ordinary links such
as GitHub repositories or documentation pages. This script samples real,
crawled page URLs from popular domains to correct that bias.

Method (fully deterministic given the pinned inputs):
  1. Pinned Tranco list (permanent list id) -> top N registrable domains,
     excluding URL shorteners and free/instant hosting platforms (which host
     user content, including phishing).
  2. Seeded random sample of those domains.
  3. For each domain, locate its block in the Common Crawl CDX index of a
     pinned crawl via ``cluster.idx`` and fetch only that block with an HTTP
     range request.
  4. Keep captures with HTTP 200, text/html, a non-trivial path, no
     credentials/e-mail/token-looking query data, length <= 300.
  5. Sample up to K URLs per domain (seeded) and write a CSV.

Inputs are public: Tranco (https://tranco-list.eu, Le Pochat et al., NDSS 2019)
and the Common Crawl index (https://commoncrawl.org/terms-of-use). Output:
``datasets/legit_deep_urls.csv``. The resulting file is committed with its
SHA-256 so training is reproducible even if the upstream services are slow.

Usage::

    python scripts/build_legit_corpus.py --out datasets/legit_deep_urls.csv
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import io
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phishlens.features import FREE_HOSTING_SUFFIXES, URL_SHORTENERS, InvalidURLError, parse_url  # noqa: E402

TRANCO_LIST_ID = "Y897G"
CRAWL = "CC-MAIN-2026-39"
CC_BASE = f"https://data.commoncrawl.org/cc-index/collections/{CRAWL}/indexes/"
UA = {"User-Agent": "phishlens-corpus-builder/1.0 (research)"}
SEED = 42
_BAD_QUERY = re.compile(r"(@|%40|email|token|session|sid=|password|passwd|auth|key=|code=)", re.I)


def fetch(url: str, byte_range: tuple[int, int] | None = None, retries: int = 5) -> bytes:
    headers = dict(UA)
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[0] + byte_range[1] - 1}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:  # noqa: S310
                return r.read()
        except Exception:  # noqa: BLE001 - network retry
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def surt_prefix(domain: str) -> str:
    return ",".join(reversed(domain.split("."))) + ")"


def load_tranco(cache: Path, top_n: int) -> list[tuple[int, str]]:
    path = cache / f"tranco_{TRANCO_LIST_ID}.csv.zip"
    if not path.exists():
        path.write_bytes(fetch(f"https://tranco-list.eu/download/{TRANCO_LIST_ID}/1000000"))
    raw = path.read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            raw = zf.read(zf.namelist()[0])
    out = []
    for line in raw.decode().splitlines():
        rank, dom = line.strip().split(",", 1)
        if int(rank) > top_n:
            break
        out.append((int(rank), dom.lower()))
    return out


def load_cluster_index(cache: Path) -> tuple[list[str], list[tuple[str, int, int]]]:
    path = cache / f"{CRAWL}_cluster.idx"
    if not path.exists():
        path.write_bytes(fetch(CC_BASE + "cluster.idx"))
    keys, blocks = [], []
    with path.open() as fh:
        for line in fh:
            key_ts, fname, offset, length, _ = line.rstrip("\n").split("\t")
            keys.append(key_ts.split(" ")[0])
            blocks.append((fname, int(offset), int(length)))
    return keys, blocks


def eligible(domain: str) -> bool:
    if domain in URL_SHORTENERS or domain in FREE_HOSTING_SUFFIXES:
        return False
    try:
        p = parse_url(domain)
    except InvalidURLError:
        return False
    # skip domains that are themselves a hosting platform suffix (user content)
    return not p.private_suffix and p.registered_domain == domain


def harvest(domain: str, block: tuple[str, int, int]) -> list[dict]:
    fname, offset, length = block
    data = gzip.decompress(fetch(CC_BASE + fname, (offset, length)))
    prefix = surt_prefix(domain)
    sub_prefix = prefix[:-1] + ","  # subdomains: com,example,www)
    rows = []
    for line in data.decode("utf-8", "replace").splitlines():
        if not (line.startswith(prefix) or line.startswith(sub_prefix)):
            continue
        try:
            meta = json.loads(line.split(" ", 2)[2])
        except (IndexError, json.JSONDecodeError):
            continue
        url = meta.get("url", "")
        if meta.get("status") != "200" or "html" not in meta.get("mime", "") or len(url) > 300:
            continue
        try:
            p = parse_url(url)
        except InvalidURLError:
            continue
        if p.registered_domain != domain or len(p.path) < 2 or p.path.endswith("robots.txt"):
            continue
        if p.query and _BAD_QUERY.search(p.query):
            continue
        rows.append({"url": url, "domain": domain, "timestamp": line.split(" ")[1]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("datasets/legit_deep_urls.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--top-n", type=int, default=50000)
    ap.add_argument("--domains", type=int, default=3000)
    ap.add_argument("--per-domain", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    tranco = [(r, d) for r, d in load_tranco(args.cache, args.top_n) if eligible(d)]
    rng = random.Random(SEED)
    sample = sorted(rng.sample(tranco, min(args.domains, len(tranco))))
    keys, blocks = load_cluster_index(args.cache)
    print(f"{len(tranco)} eligible domains, sampled {len(sample)}, {len(keys)} index blocks", flush=True)

    jobs = []
    for rank, dom in sample:
        i = max(bisect.bisect_left(keys, surt_prefix(dom)) - 1, 0)
        jobs.append((rank, dom, blocks[i]))

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(harvest, dom, blk): (rank, dom) for rank, dom, blk in jobs}
        for n, fut in enumerate(futures, 1):
            rank, dom = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {dom}: {exc}", flush=True)
                continue
            seen, uniq = set(), []
            for r in rows:
                key = parse_url(r["url"]).canonical
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            local = random.Random(f"{SEED}-{dom}")
            for r in local.sample(uniq, min(args.per_domain, len(uniq))):
                results.append({**r, "tranco_rank": rank})
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)} domains, {len(results)} urls", flush=True)

    results.sort(key=lambda r: (r["tranco_rank"], r["url"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["url", "domain", "tranco_rank", "timestamp"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {len(results)} URLs from {len({r['domain'] for r in results})} domains to {args.out}")


if __name__ == "__main__":
    main()
