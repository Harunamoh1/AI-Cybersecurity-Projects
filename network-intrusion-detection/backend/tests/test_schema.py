import pytest

from sentinelai.schema import (
    CLASS_INFO,
    CLASSES,
    FEATURE_NAMES,
    FEATURES,
    SEVERITY_ORDER,
    canonical_name,
    map_label,
    normalize_key,
    try_map_label,
)


def test_feature_names_are_unique_across_conventions():
    # 78 CSV columns in CIC-IDS2017, one of which duplicates "Fwd Header Length"
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 77
    keys_2017 = [normalize_key(f.cic2017) for f in FEATURES]
    keys_2018 = [normalize_key(f.cic2018) for f in FEATURES]
    assert len(set(keys_2017)) == 77 and len(set(keys_2018)) == 77


@pytest.mark.parametrize("raw,expected", [
    ("Total Length of Fwd Packets", "total_fwd_bytes"),
    (" Total Length of Fwd Packets ", "total_fwd_bytes"),  # CIC-IDS2017 headers have stray spaces
    ("TotLen Fwd Pkts", "total_fwd_bytes"),  # CSE-CIC-IDS2018
    ("total_fwd_bytes", "total_fwd_bytes"),  # canonical
    ("Init_Win_bytes_backward", "init_win_bytes_bwd"),
    ("Init Bwd Win Byts", "init_win_bytes_bwd"),
    ("Flow Bytes/s", "flow_bytes_per_s"),
    ("FLOW BYTS/S", "flow_bytes_per_s"),
    ("Fwd Header Length.1", "__duplicate__"),
    ("Source IP", None),
    ("not a feature", None),
])
def test_canonical_name(raw, expected):
    assert canonical_name(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("BENIGN", "BENIGN"), ("Benign", "BENIGN"),
    ("DoS Hulk", "DoS"), ("DoS slowloris", "DoS"), ("DoS attacks-SlowHTTPTest", "DoS"),
    ("DDoS", "DDoS"), ("DDOS attack-HOIC", "DDoS"), ("DDOS attack-LOIC-UDP", "DDoS"),
    ("PortScan", "PortScan"),
    ("FTP-Patator", "BruteForce"), ("SSH-Bruteforce", "BruteForce"),
    ("Web Attack � Brute Force", "WebAttack"),  # mis-encoded en dash in CIC-IDS2017
    ("Web Attack ï¿½ XSS", "WebAttack"),  # same, decoded as latin-1
    ("Web Attack - Sql Injection", "WebAttack"), ("SQL Injection", "WebAttack"), ("Brute Force -XSS", "WebAttack"),
    ("Bot", "Bot"),
    ("Infiltration", "Infiltration"), ("Infilteration", "Infiltration"),
    ("Heartbleed", None),
    ("WebAttack", "WebAttack"), ("BruteForce", "BruteForce"),  # taxonomy names themselves
])
def test_map_label(raw, expected):
    assert map_label(raw) == expected


def test_unknown_label():
    with pytest.raises(KeyError):
        map_label("Totally New Attack")
    assert try_map_label("Totally New Attack") is None
    assert try_map_label(float("nan")) is None


def test_taxonomy_is_complete():
    assert set(CLASS_INFO) == set(CLASSES)
    assert CLASSES[0] == "BENIGN"
    for info in CLASS_INFO.values():
        assert info["severity"] in SEVERITY_ORDER
        assert info["label"] and info["description"]
