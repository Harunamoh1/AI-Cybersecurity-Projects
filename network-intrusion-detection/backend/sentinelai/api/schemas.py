"""Request / response models (the OpenAPI schema at /docs is generated from these)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ClassName = Literal["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration"]
Severity = Literal["none", "low", "medium", "high", "critical"]


class ErrorResponse(BaseModel):
    detail: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "flows": [{"Flow Duration": 112740690, "Total Fwd Packets": 32, "Total Backward Packets": 16, "...": 0}],
        "explain": True, "save": False,
    }]})

    flows: list[dict[str, float | int | str | None]] = Field(
        ..., min_length=1,
        description="Flow feature records. Keys may use CIC-IDS2017, CSE-CIC-IDS2018 or canonical names.")
    explain: bool = Field(False, description="Attach a SHAP explanation to each flow (limited count)")
    save: bool = Field(False, description="Store the batch in detection history")
    name: str = Field("API batch", max_length=100)

    @field_validator("flows")
    @classmethod
    def bounded_records(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for rec in v:
            if len(rec) > 300:
                raise ValueError("a flow record may contain at most 300 fields")
            for k, val in rec.items():
                if len(k) > 100:
                    raise ValueError("field names are limited to 100 characters")
                if isinstance(val, str) and len(val) > 64:
                    raise ValueError(f"value of {k[:40]!r} is too long")
        return v


class Reason(BaseModel):
    feature: str
    value: float
    impact: float


class FlowPrediction(BaseModel):
    index: int
    predicted_class: ClassName
    confidence: float = Field(..., ge=0, le=1, description="Calibrated probability of the predicted class")
    attack_probability: float = Field(..., ge=0, le=1, description="Calibrated 1 - P(BENIGN)")
    severity: Severity
    true_label: str | None = None
    imputed_features: int = 0
    reasons: list[Reason] = []
    probabilities: dict[str, float] | None = None
    explanation: dict[str, Any] | None = None


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[FlowPrediction]
    rejected_rows: list[int]
    summary: dict[str, Any]
    analysis_id: str | None = None


class Analysis(BaseModel):
    id: str
    created_at: str
    filename: str
    source: str
    total_flows: int
    threats: int
    rejected_rows: int
    duration_ms: float
    model_version: str
    summary: dict[str, Any]


class AnalysisPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Analysis]


class FlowRecord(BaseModel):
    analysis_id: str
    idx: int
    created_at: str
    predicted: ClassName
    confidence: float
    attack_prob: float
    severity: Severity
    true_label: str | None
    imputed: int
    reasons: list[Reason]
    filename: str


class FlowPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[FlowRecord]


class FlowDetail(FlowRecord):
    explanation: dict[str, Any]
    features: dict[str, float]


class AnalysisCreated(BaseModel):
    analysis: Analysis
    top_threats: list[FlowRecord]
