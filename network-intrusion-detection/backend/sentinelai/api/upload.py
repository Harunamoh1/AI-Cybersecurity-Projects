"""Defensive parsing of uploaded network-flow CSV files.

The upload is treated as untrusted data and never executed, evaluated or
written to disk:

* size is enforced while streaming (the body is never fully buffered if too big)
* only ``.csv`` filenames; content must decode as text and contain no NUL bytes
* parsed with pandas' C parser into strings, with row and column limits
* every cell is later coerced to a number; nothing is interpreted as a formula
* the client-supplied filename is only kept, sanitised, as a display label
"""

from __future__ import annotations

import csv
import io
import re
from typing import BinaryIO

import pandas as pd

ALLOWED_CONTENT_TYPES = frozenset({
    "text/csv", "application/csv", "text/plain", "application/vnd.ms-excel", "application/octet-stream", "",
})


class UploadError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def safe_filename(name: str | None) -> str:
    """Display-only label: basename, printable safe characters, bounded length."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return (base or "upload.csv")[:100]


def read_limited(stream: BinaryIO, max_bytes: int, chunk: int = 1 << 16) -> bytes:
    buf = bytearray()
    while True:
        block = stream.read(chunk)
        if not block:
            break
        buf.extend(block)
        if len(buf) > max_bytes:
            raise UploadError(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit", 413)
    return bytes(buf)


def validate_upload_meta(filename: str | None, content_type: str | None) -> None:
    if not filename or not filename.lower().endswith(".csv"):
        raise UploadError("Only .csv files are accepted", 415)
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype not in ALLOWED_CONTENT_TYPES:
        raise UploadError(f"Unsupported content type: {ctype}", 415)


def decode(data: bytes) -> str:
    if not data.strip():
        raise UploadError("The file is empty")
    if b"\x00" in data[:1_000_000]:
        raise UploadError("The file looks binary, not CSV text", 415)
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")  # CIC-IDS2017 exports are latin-1


def parse_csv(data: bytes, max_rows: int, max_columns: int) -> pd.DataFrame:
    text = decode(data)
    first_line = text.split("\n", 1)[0]
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    if first_line.count(sep) + 1 > max_columns:
        raise UploadError(f"Too many columns (limit {max_columns})")
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, nrows=max_rows + 1, engine="c",
                         skipinitialspace=True, on_bad_lines="error", low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, ValueError) as exc:
        raise UploadError(f"Could not parse CSV: {str(exc).splitlines()[0][:200]}") from exc
    if len(df) > max_rows:
        raise UploadError(f"Too many rows (limit {max_rows:,} flows per file)", 413)
    if df.empty:
        raise UploadError("The file contains a header but no flow records")
    df.columns = [str(c).strip() for c in df.columns]
    # CIC-IDS2018 exports repeat the header row inside the file.
    first = df.columns[0]
    df = df[df[first].astype(str).str.strip() != first]
    return df.reset_index(drop=True)
