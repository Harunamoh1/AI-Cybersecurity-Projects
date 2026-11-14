"""Request / response models (OpenAPI is generated from these)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_URL_LENGTH = 2048


class ScanRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"url": "http://paypal.com.account-verify.example.xyz/login"}]})

    url: str = Field(..., min_length=1, max_length=MAX_URL_LENGTH, description="URL to analyse (http/https). Never fetched.")
    save: bool = Field(True, description="Store the result in scan history")

    @field_validator("url")
    @classmethod
    def strip_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL must not be blank")
        return v


class BatchScanRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="Up to PHISHLENS_MAX_BATCH URLs")
    save: bool = True


class Contribution(BaseModel):
    feature: str
    description: str
    value: float | str
    reference: float | str
    impact: float = Field(..., description="Shift in log-odds of phishing attributable to this feature (occlusion vs. a typical legitimate value)")


class Indicator(BaseModel):
    token: str
    kind: str
    reason: str


class Components(BaseModel):
    scheme: str | None
    host: str
    subdomain: str
    domain: str
    suffix: str
    registered_domain: str
    hosting_platform: str | None
    port: int | None
    path: str
    query: str
    is_ip: bool


class ModelRef(BaseModel):
    name: str
    version: str


class ScanResult(BaseModel):
    id: str | None = None
    created_at: str | None = None
    url: str
    canonical_url: str
    verdict: Literal["phishing", "legitimate"]
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: Literal["low", "medium", "high", "critical"]
    probability: float
    threshold: float
    components: Components
    contributions: list[Contribution]
    indicators: list[Indicator]
    features: dict[str, float | str]
    model: ModelRef
    latency_ms: float


class BatchItemError(BaseModel):
    url: str
    error: str


class BatchScanResponse(BaseModel):
    results: list[ScanResult]
    errors: list[BatchItemError]


class HistoryItem(BaseModel):
    id: str
    created_at: str
    url: str
    registered_domain: str | None
    verdict: str
    risk_level: str
    risk_score: float


class HistoryPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[HistoryItem]


class ErrorResponse(BaseModel):
    detail: str
