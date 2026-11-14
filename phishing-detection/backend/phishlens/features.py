"""URL parsing and lexical feature extraction.

This module is the single source of truth for turning a raw URL string into
model features. It is imported by the training pipeline *and* the API so that
training and serving can never drift apart (no training/serving skew).

Design notes
------------
* Features are computed from the URL string only. The URL is never fetched:
  scanning a link must not touch attacker infrastructure.
* URLs are canonicalised before feature extraction (lower-cased, scheme,
  fragment and a leading ``www.`` removed, a bare trailing ``/`` dropped).
  The training corpora contain presentation artefacts that correlate with the
  label but carry no security meaning (for example every legitimate URL in
  PhiUSIIL is ``https://www.<domain>`` with no trailing slash). Canonicalising
  removes those shortcuts so the model has to learn from the structure of the
  host and path instead. See ``docs/MODEL_CARD.md`` for the full rationale.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import unquote, urlsplit

import tldextract

# --------------------------------------------------------------------------- #
# Domain knowledge lists. These are inputs to features, not hard-coded verdicts:
# the model learns how much weight (if any) each signal deserves.
# --------------------------------------------------------------------------- #

SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login", "log-in", "signin", "sign-in", "logon", "verify", "verification",
    "account", "update", "secure", "security", "confirm", "banking", "password",
    "passwd", "credential", "auth", "authenticate", "unlock", "suspend",
    "suspended", "restore", "recover", "billing", "invoice", "payment", "wallet",
    "webscr", "cmd=_", "ebayisapi", "support", "helpdesk", "alert", "limited",
    "validate", "reactivate", "claim", "reward", "bonus", "free", "gift",
    "lucky", "airdrop", "seed", "kyc", "token",
)

BRANDS: tuple[str, ...] = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "onedrive",
    "sharepoint", "google", "gmail", "facebook", "instagram", "whatsapp",
    "amazon", "netflix", "ebay", "chase", "wellsfargo", "bankofamerica",
    "citibank", "hsbc", "santander", "barclays", "dhl", "fedex", "usps",
    "adobe", "dropbox", "docusign", "linkedin", "twitter", "coinbase", "binance",
    "metamask", "blockchain", "steam", "roblox", "yahoo", "wetransfer",
    "americanexpress", "amex", "mastercard", "telegram", "tiktok", "spotify",
    "walmart", "booking", "airbnb", "binance", "trustwallet", "ledger",
)

# Registrable labels (second-level domains) officially used by the brands above.
OFFICIAL_SLDS: frozenset[str] = frozenset(BRANDS) | frozenset(
    {
        "microsoftonline", "live", "office", "msn", "windows", "apple-cloudkit",
        "googleusercontent", "gstatic", "youtube", "fb", "fbcdn", "paypalobjects",
        "amazonaws", "awsstatic", "ebaystatic", "chasebank", "wf", "bofa",
        "linkedin", "twimg", "x", "steampowered", "steamcommunity",
        "americanexpress", "trustwallet", "ledger",
    }
)

URL_SHORTENERS: frozenset[str] = frozenset(
    {
        "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
        "rebrand.ly", "cutt.ly", "shorturl.at", "rb.gy", "tiny.cc", "bit.do",
        "s.id", "t.ly", "lnkd.in", "shorte.st", "adf.ly", "v.gd", "qrco.de",
    }
)

# Hosting / site-builder platforms frequently abused for phishing because a
# subdomain can be created for free in seconds. Plenty of legitimate sites live
# here too, so this is one weak signal among many.
FREE_HOSTING_SUFFIXES: frozenset[str] = frozenset(
    {
        "000webhostapp.com", "weebly.com", "wixsite.com", "godaddysites.com",
        "firebaseapp.com", "web.app", "appspot.com", "herokuapp.com",
        "netlify.app", "vercel.app", "pages.dev", "workers.dev", "glitch.me",
        "repl.co", "replit.dev", "surge.sh", "github.io", "gitlab.io",
        "blogspot.com", "wordpress.com", "webflow.io", "square.site",
        "myshopify.com", "sites.google.com", "ipfs.io", "infura-ipfs.io",
        "cloudflare-ipfs.com", "dweb.link", "ngrok.io", "ngrok-free.app",
        "azurewebsites.net", "web.core.windows.net", "r2.dev", "framer.app",
        "webcindario.com", "my-free.website", "yolasite.com", "jimdosite.com",
        "carrd.co", "duckdns.org", "no-ip.org", "hopto.org", "zapto.org",
    }
)

RISKY_EXTENSIONS: tuple[str, ...] = (
    ".exe", ".scr", ".zip", ".rar", ".apk", ".msi", ".bat", ".js", ".vbs",
    ".jar", ".iso", ".dll",
)
SCRIPT_EXTENSIONS: tuple[str, ...] = (".php", ".html", ".htm", ".asp", ".aspx", ".cgi")

_SPECIAL_CHARS = set("@~_%=&?!*,;$+|")
_TOKEN_SPLIT = re.compile(r"[./\-_=?&:%+~@]+")
_VOWELS = set("aeiou")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_PERCENT_RE = re.compile(r"%[0-9a-fA-F]{2}")

# The offline (bundled) Public Suffix List snapshot is used so that feature
# extraction is deterministic and never performs network I/O.
_EXTRACT_ICANN = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_EXTRACT_PRIVATE = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True
)

FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "host_length",
    "path_length",
    "query_length",
    "num_dots_host",
    "num_subdomains",
    "num_hyphens_host",
    "num_hyphens_path",
    "num_digits_host",
    "digit_ratio",
    "num_special_chars",
    "has_at_symbol",
    "has_double_slash_path",
    "is_ip_host",
    "has_port",
    "domain_entropy",
    "url_entropy",
    "path_depth",
    "num_query_params",
    "num_percent_encoded",
    "has_punycode",
    "suspicious_keyword_count",
    "brand_outside_domain",
    "brand_in_domain_mismatch",
    "is_shortener",
    "is_free_hosting",
    "has_risky_extension",
    "has_script_extension",
    "longest_token_length",
    "max_consonant_run",
    "domain_label_length",
    "embedded_http_token",
    "max_char_repeat",
    "tld_length",
    "num_path_tokens",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("tld",)
ALL_FEATURES: tuple[str, ...] = FEATURE_NAMES + CATEGORICAL_FEATURES


class InvalidURLError(ValueError):
    """Raised when a string cannot be interpreted as an http(s) URL."""


@dataclass(frozen=True)
class ParsedURL:
    """Structured view of a URL used for feature extraction and the UI."""

    original: str
    canonical: str
    scheme: str
    host: str
    subdomain: str
    domain: str
    suffix: str
    registered_domain: str
    private_suffix: str
    group_domain: str
    port: int | None
    path: str
    query: str
    is_ip: bool
    has_explicit_scheme: bool
    userinfo: str = ""
    highlights: list[dict[str, str]] = field(default_factory=list)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _max_run(text: str, predicate) -> int:
    best = run = 0
    for ch in text:
        run = run + 1 if predicate(ch) else 0
        best = max(best, run)
    return best


def _max_repeat(text: str) -> int:
    best = run = 0
    prev = ""
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return best


def _is_ip(host: str) -> bool:
    candidate = host.strip("[]")
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        pass
    # Integer / hex encoded IPv4 (e.g. http://3232235777/ or 0xC0A80001)
    if re.fullmatch(r"(0x[0-9a-f]+|\d{8,10})", candidate):
        return True
    return False


@lru_cache(maxsize=65536)
def parse_url(raw: str) -> ParsedURL:
    """Parse and canonicalise ``raw``. Raises :class:`InvalidURLError`."""
    if not isinstance(raw, str):
        raise InvalidURLError("URL must be a string")
    text = raw.strip()
    if not text:
        raise InvalidURLError("URL is empty")
    if any(ord(c) < 32 or ord(c) == 127 for c in text) or " " in text:
        raise InvalidURLError("URL contains whitespace or control characters")

    has_scheme = bool(_SCHEME_RE.match(text))
    if has_scheme:
        scheme = text.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            raise InvalidURLError(f"Unsupported scheme '{scheme}'; only http and https are analysed")
    else:
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:(?!\d)", text) and not text.lower().startswith(
            ("localhost:",)
        ):
            # e.g. javascript:alert(1), mailto:x, data:text/html
            scheme = text.split(":", 1)[0].lower()
            raise InvalidURLError(f"Unsupported scheme '{scheme}'; only http and https are analysed")
        text = "http://" + text.lstrip("/")
        scheme = ""

    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError as exc:  # invalid port, bad IPv6 literal, ...
        raise InvalidURLError(f"Malformed URL: {exc}") from exc

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise InvalidURLError("URL has no host")
    if len(host) > 253:
        raise InvalidURLError("Host name is invalid")
    if ".." in host or host.startswith("."):
        raise InvalidURLError("Host name is invalid")

    is_ip = _is_ip(host)
    host_no_www = host[4:] if host.startswith("www.") and not is_ip else host

    path = parts.path or ""
    if path == "/":
        path = ""
    path = path.lower()
    query = (parts.query or "").lower()

    # Userinfo ("user:pass@") is kept: "https://paypal.com@evil.xyz/" is a classic
    # trick where everything before "@" is ignored by the browser.
    userinfo = parts.netloc.rsplit("@", 1)[0].lower() + "@" if "@" in parts.netloc else ""

    canonical = userinfo + host_no_www
    if port is not None and port not in (80, 443):
        canonical += f":{port}"
    canonical += path
    if query:
        canonical += "?" + query

    if is_ip:
        subdomain, domain, suffix, registered, private_suffix, group = "", host, "", host, "", host
    else:
        icann = _EXTRACT_ICANN(host_no_www)
        priv = _EXTRACT_PRIVATE(host_no_www)
        subdomain = icann.subdomain
        domain = icann.domain
        suffix = icann.suffix
        registered = ".".join(p for p in (icann.domain, icann.suffix) if p)
        private_suffix = priv.suffix if priv.suffix != icann.suffix else ""
        group = ".".join(p for p in (priv.domain, priv.suffix) if p) or host_no_www

    return ParsedURL(
        original=raw.strip(),
        canonical=canonical,
        scheme=scheme,
        host=host_no_www,
        subdomain=subdomain,
        domain=domain,
        suffix=suffix,
        registered_domain=registered,
        private_suffix=private_suffix,
        group_domain=group,
        port=port if port not in (80, 443) else None,
        path=path,
        query=query,
        is_ip=is_ip,
        has_explicit_scheme=has_scheme,
        userinfo=userinfo.rstrip("@"),
    )


def keyword_matches(text: str) -> list[str]:
    """Suspicious keywords present in ``text``.

    Long keywords match as substrings (``secure-login`` -> login); short or
    ambiguous ones (``auth``, ``free``, ``seed``...) must be a whole token so that
    ``/author/`` or ``freedom`` do not count.
    """
    tokens = set(_TOKEN_SPLIT.split(text))
    hits = []
    for k in SUSPICIOUS_KEYWORDS:
        if len(k) >= 6 or not k.isalpha():
            if k in text:
                hits.append(k)
        elif k in tokens:
            hits.append(k)
    return hits


def _brand_hits(parsed: ParsedURL) -> tuple[list[str], list[str]]:
    """Return (brands outside the registered domain, brand look-alike domains)."""
    outside_text = " ".join((parsed.subdomain, parsed.path, parsed.query))
    sld = parsed.domain
    outside: list[str] = []
    mismatch: list[str] = []
    for brand in dict.fromkeys(BRANDS):
        if brand in outside_text and sld not in OFFICIAL_SLDS:
            outside.append(brand)
        if brand in sld and sld not in OFFICIAL_SLDS:
            mismatch.append(brand)
    return outside, mismatch


def extract_features(url: str | ParsedURL) -> dict[str, float | str]:
    """Return the full feature dictionary (numeric features + ``tld``)."""
    p = url if isinstance(url, ParsedURL) else parse_url(url)
    c = p.canonical
    host = p.host
    path = p.path
    query = p.query
    decoded = unquote(c)

    subdomain_labels = [s for s in p.subdomain.split(".") if s] if p.subdomain else []
    keyword_hits = keyword_matches(decoded)
    brands_out, brands_mismatch = _brand_hits(p)
    tokens = [t for t in _TOKEN_SPLIT.split(host + path) if t]
    path_tokens = [t for t in _TOKEN_SPLIT.split(path) if t]
    letters = sum(ch.isalpha() for ch in c)
    digits = sum(ch.isdigit() for ch in c)
    hosting_suffix = p.private_suffix or p.registered_domain
    after_host = c[len(p.userinfo) + 1 + len(host):] if p.userinfo else c[len(host):]

    feats: dict[str, float | str] = {
        "url_length": len(c),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots_host": host.count("."),
        "num_subdomains": len(subdomain_labels),
        "num_hyphens_host": host.count("-"),
        "num_hyphens_path": path.count("-"),
        "num_digits_host": sum(ch.isdigit() for ch in host) if not p.is_ip else 0,
        "digit_ratio": digits / max(letters + digits, 1),
        "num_special_chars": sum(ch in _SPECIAL_CHARS for ch in c),
        "has_at_symbol": int("@" in c),
        "has_double_slash_path": int("//" in path),
        "is_ip_host": int(p.is_ip),
        "has_port": int(p.port is not None),
        "domain_entropy": round(_entropy(p.domain), 4),
        "url_entropy": round(_entropy(c), 4),
        "path_depth": len([s for s in path.split("/") if s]),
        "num_query_params": len([q for q in query.split("&") if q]) if query else 0,
        "num_percent_encoded": len(_PERCENT_RE.findall(c)),
        "has_punycode": int("xn--" in host),
        "suspicious_keyword_count": len(keyword_hits),
        "brand_outside_domain": int(bool(brands_out)),
        "brand_in_domain_mismatch": int(bool(brands_mismatch)),
        "is_shortener": int(p.registered_domain in URL_SHORTENERS),
        "is_free_hosting": int(
            hosting_suffix in FREE_HOSTING_SUFFIXES
            or any(host.endswith("." + s) for s in FREE_HOSTING_SUFFIXES)
        ),
        "has_risky_extension": int(path.endswith(RISKY_EXTENSIONS)),
        "has_script_extension": int(any(path.endswith(e) or (e + "?") in c for e in SCRIPT_EXTENSIONS)),
        "longest_token_length": max((len(t) for t in tokens), default=0),
        "max_consonant_run": _max_run(p.domain, lambda ch: ch.isalpha() and ch not in _VOWELS),
        "domain_label_length": len(p.domain),
        "embedded_http_token": int("http" in after_host or "https" in host or "http" in p.subdomain),
        "max_char_repeat": _max_repeat(c),
        "tld_length": len(p.suffix),
        "num_path_tokens": len(path_tokens),
        "tld": (p.suffix.rsplit(".", 1)[-1] if p.suffix else ("<ip>" if p.is_ip else "<none>")),
    }
    return feats


def highlights(url: str | ParsedURL) -> list[dict[str, str]]:
    """Human-readable risk indicators found in the URL, with the offending token.

    These are *observations* shown to the analyst alongside the model output.
    They do not change the score: the score comes from the trained model.
    """
    p = url if isinstance(url, ParsedURL) else parse_url(url)
    f = extract_features(p)
    out: list[dict[str, str]] = []
    decoded = unquote(p.canonical)
    brands_out, brands_mismatch = _brand_hits(p)

    for b in brands_out:
        out.append({"token": b, "kind": "brand",
                    "reason": f"Brand name '{b}' appears outside the registered domain ({p.registered_domain})"})
    for b in brands_mismatch:
        out.append({"token": p.domain, "kind": "lookalike",
                    "reason": f"Domain '{p.registered_domain}' contains brand '{b}' but is not an official {b} domain"})
    for k in keyword_matches(decoded):
        out.append({"token": k, "kind": "keyword", "reason": f"Credential/urgency keyword '{k}'"})
    if f["is_ip_host"]:
        out.append({"token": p.host, "kind": "ip", "reason": "Host is a raw IP address instead of a domain name"})
    if f["has_at_symbol"]:
        out.append({"token": "@", "kind": "obfuscation", "reason": "'@' in URL can hide the real destination host"})
    if f["has_punycode"]:
        out.append({"token": "xn--", "kind": "obfuscation", "reason": "Punycode host may impersonate another domain (homograph)"})
    if f["is_shortener"]:
        out.append({"token": p.registered_domain, "kind": "shortener", "reason": "URL shortener hides the final destination"})
    if f["is_free_hosting"]:
        out.append({"token": p.private_suffix or p.registered_domain, "kind": "hosting",
                    "reason": "Hosted on a free / instant site platform often abused for phishing"})
    if f["num_subdomains"] >= 3:
        out.append({"token": p.subdomain, "kind": "structure", "reason": f"Deeply nested subdomains ({f['num_subdomains']} levels)"})
    if f["embedded_http_token"]:
        out.append({"token": "http", "kind": "obfuscation", "reason": "'http' token embedded in host or path"})
    if f["has_port"]:
        out.append({"token": f":{p.port}", "kind": "structure", "reason": f"Non-standard port {p.port}"})
    if f["has_risky_extension"]:
        out.append({"token": path_ext(p.path), "kind": "payload", "reason": "Path points to a downloadable executable/archive"})
    if f["num_hyphens_host"] >= 3:
        out.append({"token": "-", "kind": "structure", "reason": f"{f['num_hyphens_host']} hyphens in host name"})
    if int(f["longest_token_length"]) >= 40:
        out.append({"token": "", "kind": "structure", "reason": "Very long random-looking token (possible tracking/session payload)"})
    return out


def path_ext(path: str) -> str:
    tail = path.rsplit("/", 1)[-1]
    return "." + tail.rsplit(".", 1)[-1] if "." in tail else tail


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "url_length": "Length of the URL",
    "host_length": "Length of the host name",
    "path_length": "Length of the path",
    "query_length": "Length of the query string",
    "num_dots_host": "Dots in the host name",
    "num_subdomains": "Number of subdomain levels",
    "num_hyphens_host": "Hyphens in the host name",
    "num_hyphens_path": "Hyphens in the path",
    "num_digits_host": "Digits in the host name",
    "digit_ratio": "Share of digits among alphanumerics",
    "num_special_chars": "Special characters (@ ~ _ % = & ? ! etc.)",
    "has_at_symbol": "Contains '@'",
    "has_double_slash_path": "Contains '//' inside the path",
    "is_ip_host": "Host is an IP address",
    "has_port": "Uses a non-standard port",
    "domain_entropy": "Randomness (entropy) of the domain label",
    "url_entropy": "Randomness (entropy) of the whole URL",
    "path_depth": "Number of path segments",
    "num_query_params": "Number of query parameters",
    "num_percent_encoded": "Percent-encoded characters",
    "has_punycode": "Punycode (xn--) host",
    "suspicious_keyword_count": "Credential / urgency keywords",
    "brand_outside_domain": "Brand name outside the registered domain",
    "brand_in_domain_mismatch": "Brand look-alike registered domain",
    "is_shortener": "Known URL shortener",
    "is_free_hosting": "Free / instant hosting platform",
    "has_risky_extension": "Executable or archive download",
    "has_script_extension": "Server-side page extension (.php/.html/...)",
    "longest_token_length": "Longest token in host + path",
    "max_consonant_run": "Longest consonant run in the domain",
    "domain_label_length": "Length of the domain label",
    "embedded_http_token": "'http' embedded in host or path",
    "max_char_repeat": "Longest repeated-character run",
    "tld_length": "Length of the public suffix",
    "num_path_tokens": "Tokens in the path",
    "tld": "Top-level domain",
}
