"""Canonical feature schema and attack taxonomy shared by training and inference.

CICFlowMeter exports the same bidirectional flow statistics under two naming
conventions: the long names used in CIC-IDS2017 (``Total Length of Fwd Packets``)
and the abbreviated names used in CSE-CIC-IDS2018 (``TotLen Fwd Pkts``). Every
column is mapped to one canonical snake_case name so that files produced by
either release can be analysed by the same model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- features


@dataclass(frozen=True)
class FeatureDef:
    name: str  # canonical snake_case name
    cic2017: str  # CIC-IDS2017 column name
    cic2018: str  # CSE-CIC-IDS2018 column name
    description: str
    unit: str = ""


# Order follows the CICFlowMeter export. Times are in microseconds.
FEATURES: tuple[FeatureDef, ...] = (
    FeatureDef("destination_port", "Destination Port", "Dst Port", "Destination transport port"),
    FeatureDef("flow_duration", "Flow Duration", "Flow Duration", "Duration of the flow", "µs"),
    FeatureDef("total_fwd_packets", "Total Fwd Packets", "Tot Fwd Pkts", "Packets sent client→server"),
    FeatureDef("total_bwd_packets", "Total Backward Packets", "Tot Bwd Pkts", "Packets sent server→client"),
    FeatureDef("total_fwd_bytes", "Total Length of Fwd Packets", "TotLen Fwd Pkts", "Payload bytes client→server", "B"),
    FeatureDef("total_bwd_bytes", "Total Length of Bwd Packets", "TotLen Bwd Pkts", "Payload bytes server→client", "B"),
    FeatureDef("fwd_pkt_len_max", "Fwd Packet Length Max", "Fwd Pkt Len Max", "Largest forward packet", "B"),
    FeatureDef("fwd_pkt_len_min", "Fwd Packet Length Min", "Fwd Pkt Len Min", "Smallest forward packet", "B"),
    FeatureDef("fwd_pkt_len_mean", "Fwd Packet Length Mean", "Fwd Pkt Len Mean", "Mean forward packet size", "B"),
    FeatureDef("fwd_pkt_len_std", "Fwd Packet Length Std", "Fwd Pkt Len Std", "Std. dev. of forward packet size", "B"),
    FeatureDef("bwd_pkt_len_max", "Bwd Packet Length Max", "Bwd Pkt Len Max", "Largest backward packet", "B"),
    FeatureDef("bwd_pkt_len_min", "Bwd Packet Length Min", "Bwd Pkt Len Min", "Smallest backward packet", "B"),
    FeatureDef("bwd_pkt_len_mean", "Bwd Packet Length Mean", "Bwd Pkt Len Mean", "Mean backward packet size", "B"),
    FeatureDef("bwd_pkt_len_std", "Bwd Packet Length Std", "Bwd Pkt Len Std", "Std. dev. of backward packet size", "B"),
    FeatureDef("flow_bytes_per_s", "Flow Bytes/s", "Flow Byts/s", "Throughput of the flow", "B/s"),
    FeatureDef("flow_packets_per_s", "Flow Packets/s", "Flow Pkts/s", "Packet rate of the flow", "pkt/s"),
    FeatureDef("flow_iat_mean", "Flow IAT Mean", "Flow IAT Mean", "Mean inter-arrival time", "µs"),
    FeatureDef("flow_iat_std", "Flow IAT Std", "Flow IAT Std", "Std. dev. of inter-arrival time", "µs"),
    FeatureDef("flow_iat_max", "Flow IAT Max", "Flow IAT Max", "Longest gap between packets", "µs"),
    FeatureDef("flow_iat_min", "Flow IAT Min", "Flow IAT Min", "Shortest gap between packets", "µs"),
    FeatureDef("fwd_iat_total", "Fwd IAT Total", "Fwd IAT Tot", "Total forward inter-arrival time", "µs"),
    FeatureDef("fwd_iat_mean", "Fwd IAT Mean", "Fwd IAT Mean", "Mean forward inter-arrival time", "µs"),
    FeatureDef("fwd_iat_std", "Fwd IAT Std", "Fwd IAT Std", "Std. dev. of forward inter-arrival time", "µs"),
    FeatureDef("fwd_iat_max", "Fwd IAT Max", "Fwd IAT Max", "Longest forward gap", "µs"),
    FeatureDef("fwd_iat_min", "Fwd IAT Min", "Fwd IAT Min", "Shortest forward gap", "µs"),
    FeatureDef("bwd_iat_total", "Bwd IAT Total", "Bwd IAT Tot", "Total backward inter-arrival time", "µs"),
    FeatureDef("bwd_iat_mean", "Bwd IAT Mean", "Bwd IAT Mean", "Mean backward inter-arrival time", "µs"),
    FeatureDef("bwd_iat_std", "Bwd IAT Std", "Bwd IAT Std", "Std. dev. of backward inter-arrival time", "µs"),
    FeatureDef("bwd_iat_max", "Bwd IAT Max", "Bwd IAT Max", "Longest backward gap", "µs"),
    FeatureDef("bwd_iat_min", "Bwd IAT Min", "Bwd IAT Min", "Shortest backward gap", "µs"),
    FeatureDef("fwd_psh_flags", "Fwd PSH Flags", "Fwd PSH Flags", "Forward packets with PSH set"),
    FeatureDef("bwd_psh_flags", "Bwd PSH Flags", "Bwd PSH Flags", "Backward packets with PSH set"),
    FeatureDef("fwd_urg_flags", "Fwd URG Flags", "Fwd URG Flags", "Forward packets with URG set"),
    FeatureDef("bwd_urg_flags", "Bwd URG Flags", "Bwd URG Flags", "Backward packets with URG set"),
    FeatureDef("fwd_header_bytes", "Fwd Header Length", "Fwd Header Len", "Header bytes client→server", "B"),
    FeatureDef("bwd_header_bytes", "Bwd Header Length", "Bwd Header Len", "Header bytes server→client", "B"),
    FeatureDef("fwd_packets_per_s", "Fwd Packets/s", "Fwd Pkts/s", "Forward packet rate", "pkt/s"),
    FeatureDef("bwd_packets_per_s", "Bwd Packets/s", "Bwd Pkts/s", "Backward packet rate", "pkt/s"),
    FeatureDef("pkt_len_min", "Min Packet Length", "Pkt Len Min", "Smallest packet in the flow", "B"),
    FeatureDef("pkt_len_max", "Max Packet Length", "Pkt Len Max", "Largest packet in the flow", "B"),
    FeatureDef("pkt_len_mean", "Packet Length Mean", "Pkt Len Mean", "Mean packet size", "B"),
    FeatureDef("pkt_len_std", "Packet Length Std", "Pkt Len Std", "Std. dev. of packet size", "B"),
    FeatureDef("pkt_len_var", "Packet Length Variance", "Pkt Len Var", "Variance of packet size", "B²"),
    FeatureDef("fin_flag_count", "FIN Flag Count", "FIN Flag Cnt", "Packets with FIN set"),
    FeatureDef("syn_flag_count", "SYN Flag Count", "SYN Flag Cnt", "Packets with SYN set"),
    FeatureDef("rst_flag_count", "RST Flag Count", "RST Flag Cnt", "Packets with RST set"),
    FeatureDef("psh_flag_count", "PSH Flag Count", "PSH Flag Cnt", "Packets with PSH set"),
    FeatureDef("ack_flag_count", "ACK Flag Count", "ACK Flag Cnt", "Packets with ACK set"),
    FeatureDef("urg_flag_count", "URG Flag Count", "URG Flag Cnt", "Packets with URG set"),
    FeatureDef("cwe_flag_count", "CWE Flag Count", "CWE Flag Count", "Packets with CWR set"),
    FeatureDef("ece_flag_count", "ECE Flag Count", "ECE Flag Cnt", "Packets with ECE set"),
    FeatureDef("down_up_ratio", "Down/Up Ratio", "Down/Up Ratio", "Backward/forward packet ratio"),
    FeatureDef("avg_packet_size", "Average Packet Size", "Pkt Size Avg", "Average packet size", "B"),
    FeatureDef("avg_fwd_segment_size", "Avg Fwd Segment Size", "Fwd Seg Size Avg", "Average forward segment size", "B"),
    FeatureDef("avg_bwd_segment_size", "Avg Bwd Segment Size", "Bwd Seg Size Avg", "Average backward segment size", "B"),
    FeatureDef("fwd_avg_bytes_bulk", "Fwd Avg Bytes/Bulk", "Fwd Byts/b Avg", "Forward bulk bytes (avg)", "B"),
    FeatureDef("fwd_avg_packets_bulk", "Fwd Avg Packets/Bulk", "Fwd Pkts/b Avg", "Forward bulk packets (avg)"),
    FeatureDef("fwd_avg_bulk_rate", "Fwd Avg Bulk Rate", "Fwd Blk Rate Avg", "Forward bulk rate (avg)"),
    FeatureDef("bwd_avg_bytes_bulk", "Bwd Avg Bytes/Bulk", "Bwd Byts/b Avg", "Backward bulk bytes (avg)", "B"),
    FeatureDef("bwd_avg_packets_bulk", "Bwd Avg Packets/Bulk", "Bwd Pkts/b Avg", "Backward bulk packets (avg)"),
    FeatureDef("bwd_avg_bulk_rate", "Bwd Avg Bulk Rate", "Bwd Blk Rate Avg", "Backward bulk rate (avg)"),
    FeatureDef("subflow_fwd_packets", "Subflow Fwd Packets", "Subflow Fwd Pkts", "Forward packets per subflow"),
    FeatureDef("subflow_fwd_bytes", "Subflow Fwd Bytes", "Subflow Fwd Byts", "Forward bytes per subflow", "B"),
    FeatureDef("subflow_bwd_packets", "Subflow Bwd Packets", "Subflow Bwd Pkts", "Backward packets per subflow"),
    FeatureDef("subflow_bwd_bytes", "Subflow Bwd Bytes", "Subflow Bwd Byts", "Backward bytes per subflow", "B"),
    FeatureDef("init_win_bytes_fwd", "Init_Win_bytes_forward", "Init Fwd Win Byts", "Initial TCP window client→server (-1 = not observed)", "B"),
    FeatureDef("init_win_bytes_bwd", "Init_Win_bytes_backward", "Init Bwd Win Byts", "Initial TCP window server→client (-1 = not observed)", "B"),
    FeatureDef("act_data_pkt_fwd", "act_data_pkt_fwd", "Fwd Act Data Pkts", "Forward packets carrying payload"),
    FeatureDef("min_seg_size_fwd", "min_seg_size_forward", "Fwd Seg Size Min", "Minimum forward segment size", "B"),
    FeatureDef("active_mean", "Active Mean", "Active Mean", "Mean time the flow was active", "µs"),
    FeatureDef("active_std", "Active Std", "Active Std", "Std. dev. of active time", "µs"),
    FeatureDef("active_max", "Active Max", "Active Max", "Longest active period", "µs"),
    FeatureDef("active_min", "Active Min", "Active Min", "Shortest active period", "µs"),
    FeatureDef("idle_mean", "Idle Mean", "Idle Mean", "Mean idle time", "µs"),
    FeatureDef("idle_std", "Idle Std", "Idle Std", "Std. dev. of idle time", "µs"),
    FeatureDef("idle_max", "Idle Max", "Idle Max", "Longest idle period", "µs"),
    FeatureDef("idle_min", "Idle Min", "Idle Min", "Shortest idle period", "µs"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_BY_NAME: dict[str, FeatureDef] = {f.name: f for f in FEATURES}

# Features whose value -1 is a CICFlowMeter sentinel ("not observed"), not an error.
SENTINEL_FEATURES: frozenset[str] = frozenset({"init_win_bytes_fwd", "init_win_bytes_bwd"})

# Excluded from the model by policy (see docs/ARCHITECTURE.md):
#  * destination_port: lets a model shortcut on lab-specific port usage (e.g. all
#    FTP-Patator traffic on port 21) instead of learning flow behaviour.
POLICY_EXCLUDED: dict[str, str] = {
    "destination_port": "port-number shortcut; attacks can run on any port",
}

# Columns that identify hosts/flows. They are never used as features and are
# discarded on ingestion (also keeps IP addresses out of the history store).
IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {"flowid", "sourceip", "srcip", "sourceport", "srcport", "destinationip", "dstip", "timestamp", "protocol"}
)
LABEL_KEYS: frozenset[str] = frozenset({"label"})


def normalize_key(column: str) -> str:
    """Case/spacing/punctuation-insensitive key for matching column names."""
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


_ALIASES: dict[str, str] = {}
for _f in FEATURES:
    for _alias in (_f.name, _f.cic2017, _f.cic2018):
        _ALIASES[normalize_key(_alias)] = _f.name
# CIC-IDS2017 exports "Fwd Header Length" twice; pandas renames the second copy.
_ALIASES[normalize_key("Fwd Header Length.1")] = "__duplicate__"


def canonical_name(column: str) -> str | None:
    """Return the canonical feature name for a raw column, ``"__duplicate__"`` or None."""
    return _ALIASES.get(normalize_key(column))


# --------------------------------------------------------------------------- taxonomy

BENIGN = "BENIGN"
CLASSES: tuple[str, ...] = (
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration",
)

CLASS_INFO: dict[str, dict[str, str]] = {
    "BENIGN": {"label": "Benign", "severity": "none", "description": "Normal traffic."},
    "DoS": {
        "label": "Denial of Service",
        "severity": "high",
        "description": "Single-source floods and slow-rate exhaustion (Hulk, GoldenEye, Slowloris, SlowHTTPTest).",
    },
    "DDoS": {
        "label": "Distributed DoS",
        "severity": "critical",
        "description": "Multi-source volumetric HTTP flood (LOIC).",
    },
    "PortScan": {
        "label": "Port Scan",
        "severity": "medium",
        "description": "Reconnaissance probing of many ports (Nmap scan types).",
    },
    "BruteForce": {
        "label": "Brute Force",
        "severity": "high",
        "description": "Credential guessing against FTP and SSH (Patator).",
    },
    "WebAttack": {
        "label": "Web Attack",
        "severity": "high",
        "description": "Application-layer attacks: login brute force, XSS and SQL injection.",
    },
    "Bot": {
        "label": "Botnet",
        "severity": "critical",
        "description": "Command-and-control traffic of an infected host (Ares botnet).",
    },
    "Infiltration": {
        "label": "Infiltration",
        "severity": "critical",
        "description": "Post-exploitation activity from an internal compromised host.",
    },
}

SEVERITY_ORDER: tuple[str, ...] = ("none", "low", "medium", "high", "critical")

# Raw label -> taxonomy class, for both datasets. ``None`` = excluded from training.
_RAW_LABELS: dict[str, str | None] = {
    # CIC-IDS2017
    "benign": "BENIGN",
    "doshulk": "DoS",
    "dosgoldeneye": "DoS",
    "dosslowloris": "DoS",
    "dosslowhttptest": "DoS",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "ftppatator": "BruteForce",
    "sshpatator": "BruteForce",
    "webattackbruteforce": "WebAttack",
    "webattackxss": "WebAttack",
    "webattacksqlinjection": "WebAttack",
    "bot": "Bot",
    "infiltration": "Infiltration",
    "heartbleed": None,  # 11 flows: held out as an unseen-attack probe
    # taxonomy names themselves (files labelled with SentinelAI classes)
    "dos": "DoS",
    "bruteforce": "BruteForce",
    "webattack": "WebAttack",
    # CSE-CIC-IDS2018
    "dosattackshulk": "DoS",
    "dosattacksgoldeneye": "DoS",
    "dosattacksslowloris": "DoS",
    "dosattacksslowhttptest": "DoS",
    "ddosattackhoic": "DDoS",
    "ddosattackloicudp": "DDoS",
    "ddosattackshoic": "DDoS",
    "ddosattacksloichttp": "DDoS",
    "ftpbruteforce": "BruteForce",
    "sshbruteforce": "BruteForce",
    "bruteforceweb": "WebAttack",
    "bruteforcexss": "WebAttack",
    "sqlinjection": "WebAttack",
    "infilteration": "Infiltration",
}


def _label_key(raw: str) -> str:
    # The CIC-IDS2017 web-attack labels contain a mis-encoded en dash.
    return re.sub(r"[^a-z]", "", str(raw).lower())


def map_label(raw: str) -> str | None:
    """Map a raw dataset label to a taxonomy class.

    Returns None for labels deliberately excluded from training (Heartbleed).
    Raises KeyError for labels this taxonomy does not know.
    """
    key = _label_key(raw)
    if key not in _RAW_LABELS:
        raise KeyError(f"Unknown label: {raw!r}")
    return _RAW_LABELS[key]


def try_map_label(raw: object) -> str | None:
    """Like :func:`map_label` but returns None for unknown labels (used on uploads)."""
    try:
        return map_label(str(raw))
    except KeyError:
        return None
