"""Unit tests for URL parsing and feature extraction."""

from __future__ import annotations

import math

import pytest

from phishlens.features import (
    ALL_FEATURES,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    InvalidURLError,
    extract_features,
    highlights,
    keyword_matches,
    parse_url,
)


class TestCanonicalisation:
    @pytest.mark.parametrize(
        "variant",
        ["https://www.example.com", "http://example.com/", "example.com", "HTTPS://WWW.EXAMPLE.COM/", "  example.com  ",
         "https://example.com/#section"],
    )
    def test_presentation_variants_collapse(self, variant):
        assert parse_url(variant).canonical == "example.com"

    def test_path_and_query_kept_lowercased(self):
        p = parse_url("https://Example.com/Login/Index.PHP?User=A")
        assert p.canonical == "example.com/login/index.php?user=a"
        assert p.path == "/login/index.php"
        assert p.query == "user=a"

    def test_non_default_port_kept_default_dropped(self):
        assert parse_url("http://example.com:8080/x").canonical == "example.com:8080/x"
        assert parse_url("https://example.com:443/x").canonical == "example.com/x"

    def test_public_suffix_aware_domain_parts(self):
        p = parse_url("https://login.secure.bbc.co.uk/news")
        assert (p.subdomain, p.domain, p.suffix) == ("login.secure", "bbc", "co.uk")
        assert p.registered_domain == "bbc.co.uk"

    def test_private_suffix_tenant_is_its_own_group(self):
        a = parse_url("https://evil-login.firebaseapp.com/")
        b = parse_url("https://other-app.firebaseapp.com/")
        assert a.private_suffix == "firebaseapp.com"
        assert a.group_domain != b.group_domain

    def test_scheme_recorded_but_absent_when_missing(self):
        assert parse_url("https://a.com").scheme == "https"
        assert parse_url("a.com").scheme == ""


class TestInvalidInput:
    @pytest.mark.parametrize(
        "bad",
        [
            "", "   ", "javascript:alert(1)", "data:text/html,<script>", "mailto:a@b.com", "ftp://example.com/file",
            "file:///etc/passwd", "http://", "http://exa mple.com", "http://example.com/\x00", "http://example.com:99999/",
            "http://[::1", "https://..com", "http://.example.com",
        ],
    )
    def test_rejected(self, bad):
        with pytest.raises(InvalidURLError):
            parse_url(bad)

    def test_non_string_rejected(self):
        with pytest.raises(InvalidURLError):
            parse_url(None)  # type: ignore[arg-type]

    def test_extremely_long_host_rejected(self):
        with pytest.raises(InvalidURLError):
            parse_url("http://" + "a" * 300 + ".com")


class TestFeatures:
    def test_feature_vector_complete_and_finite(self):
        f = extract_features("https://www.github.com/scikit-learn/scikit-learn")
        assert list(f) == list(ALL_FEATURES)
        for name in FEATURE_NAMES:
            assert isinstance(f[name], (int, float)) and math.isfinite(f[name]), name
        assert isinstance(f["tld"], str)

    def test_every_feature_documented(self):
        assert set(ALL_FEATURES) <= set(FEATURE_DESCRIPTIONS)

    def test_deterministic(self):
        u = "http://paypal.com.verify-account.xyz/webscr?cmd=_login"
        assert extract_features(u) == extract_features(u)

    @pytest.mark.parametrize(
        "url", ["http://192.168.0.1/login", "http://10.0.0.1:8080/", "http://3232235777/", "http://[2001:db8::1]/x"]
    )
    def test_ip_hosts(self, url):
        f = extract_features(url)
        assert f["is_ip_host"] == 1
        assert f["tld"] == "<ip>"

    def test_brand_outside_registered_domain(self):
        f = extract_features("http://paypal.com.account-update.example.xyz/signin")
        assert f["brand_outside_domain"] == 1
        assert f["num_subdomains"] == 3

    def test_brand_on_official_domain_is_not_flagged(self):
        for u in ("https://www.paypal.com/signin", "https://login.microsoftonline.com/common/oauth2"):
            f = extract_features(u)
            assert f["brand_outside_domain"] == 0
            assert f["brand_in_domain_mismatch"] == 0

    def test_lookalike_domain(self):
        assert extract_features("https://paypal-secure-help.com/")["brand_in_domain_mismatch"] == 1

    def test_keyword_token_matching_avoids_substring_false_hits(self):
        assert "auth" not in keyword_matches("github.com/author/repo")
        assert "free" not in keyword_matches("freedom.org/about")
        assert "auth" in keyword_matches("example.com/auth/start")
        assert "login" in keyword_matches("example.com/secure-login")

    def test_shortener_and_free_hosting(self):
        assert extract_features("https://bit.ly/3abc")["is_shortener"] == 1
        assert extract_features("https://office-verify.web.app/")["is_free_hosting"] == 1
        assert extract_features("https://example.com/")["is_free_hosting"] == 0

    def test_obfuscation_signals(self):
        f = extract_features("http://example.com@evil.xyz/https-login//x%2F%2F")
        assert f["has_at_symbol"] == 1
        assert f["has_double_slash_path"] == 1
        assert f["num_percent_encoded"] == 2
        assert f["embedded_http_token"] == 1

    def test_punycode_and_risky_extension(self):
        assert extract_features("http://xn--pple-43d.com/")["has_punycode"] == 1
        assert extract_features("http://example.com/files/invoice.exe")["has_risky_extension"] == 1

    def test_scheme_does_not_change_features(self):
        assert extract_features("http://www.example.com/a") == extract_features("https://example.com/a")


class TestHighlights:
    def test_clean_url_has_no_indicators(self):
        assert highlights("https://en.wikipedia.org/wiki/Phishing") == []

    def test_phishing_style_url_has_indicators(self):
        kinds = {h["kind"] for h in highlights("http://paypal.com.secure-login.xyz/webscr?cmd=_login")}
        assert {"brand", "keyword"} <= kinds
