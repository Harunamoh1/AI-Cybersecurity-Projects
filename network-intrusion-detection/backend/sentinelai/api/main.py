"""SentinelAI REST API.

Note: no ``from __future__ import annotations`` here. FastAPI resolves string
annotations against module globals, and the dependency aliases below are
defined inside ``create_app``.
"""

import hmac
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from sentinelai import __version__
from sentinelai.api.config import Settings, get_settings
from sentinelai.api.ratelimit import RateLimiter
from sentinelai.api.schemas import (
    Analysis,
    AnalysisCreated,
    AnalysisPage,
    ErrorResponse,
    FlowDetail,
    FlowPage,
    PredictRequest,
    PredictResponse,
)
from sentinelai.api.service import analyse_frame
from sentinelai.api.storage import Store
from sentinelai.api.upload import UploadError, parse_csv, read_limited, safe_filename, validate_upload_meta
from sentinelai.features import SchemaError
from sentinelai.inference.detector import Detector
from sentinelai.schema import CLASS_INFO, CLASSES, FEATURE_BY_NAME, SEVERITY_ORDER

log = logging.getLogger("sentinelai.api")
SAMPLE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}\.csv$")
ClassFilter = Annotated[str | None, Query(pattern="^(" + "|".join(CLASSES) + ")$")]
SeverityFilter = Annotated[str | None, Query(pattern="^(" + "|".join(SEVERITY_ORDER) + ")$")]


def discover_samples(samples_dir: Path) -> dict[str, dict[str, Any]]:
    """Whitelist of bundled sample files. Requests can only name an entry of this
    dict; no user input is ever joined into a filesystem path."""
    out: dict[str, dict[str, Any]] = {}
    manifest = samples_dir / "manifest.json"
    meta = json.loads(manifest.read_text()) if manifest.exists() else {}
    if samples_dir.is_dir():
        for p in sorted(samples_dir.glob("*.csv")):
            if SAMPLE_NAME.match(p.name) and p.is_file() and not p.is_symlink():
                out[p.name] = {"name": p.name, "path": p.resolve(), "size_bytes": p.stat().st_size,
                               **meta.get(p.name, {})}
    return out


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        det = detector or Detector.load(settings.model_dir, settings.verify_model_checksum)
        det.booster.set_param({"nthread": settings.inference_threads})
        app.state.detector = det
        app.state.store = Store(settings.database_path, retention=settings.retention_analyses)
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
        app.state.samples = discover_samples(settings.samples_dir)
        metrics_file = settings.model_dir / "metrics.json"
        app.state.metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
        log.info("SentinelAI API ready: model %s, %d samples", det.version, len(app.state.samples))
        yield
        app.state.store.close()

    app = FastAPI(
        title="SentinelAI API",
        version=__version__,
        description=("Explainable machine-learning network intrusion detection over CICFlowMeter flow features. "
                     "Uploaded files are parsed as data only and never executed or stored on disk."),
        lifespan=lifespan,
        responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        start = time.perf_counter()
        # Reject oversized bodies before multipart parsing spools them.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 64 * 1024:
            return JSONResponse(status_code=413, content={"detail": f"Request exceeds {settings.max_upload_mb:g} MB"})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        log.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - start) * 1000)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p not in ("body", "query"))
        msg = first.get("msg", "Invalid request")
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            content={"detail": f"{loc}: {msg}" if loc else msg})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):  # pragma: no cover - defensive
        log.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ----------------------------------------------------------- dependencies
    def authorised(request: Request) -> None:
        if settings.api_key:
            supplied = request.headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied.encode(), settings.api_key.encode()):
                raise HTTPException(status_code=401, detail="Missing or invalid API key")

    def rate_limited(request: Request) -> None:
        client = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
        if not request.app.state.limiter.allow(client):
            raise HTTPException(status_code=429, detail="Rate limit exceeded; try again in a minute")

    def get_detector(request: Request) -> Detector:
        return request.app.state.detector

    def get_store(request: Request) -> Store:
        return request.app.state.store

    DetDep = Annotated[Detector, Depends(get_detector)]
    StoreDep = Annotated[Store, Depends(get_store)]
    secured = [Depends(authorised)]
    heavy = [Depends(authorised), Depends(rate_limited)]

    def _run_and_store(det: Detector, store: Store, df: pd.DataFrame, filename: str, source: str) -> dict[str, Any]:
        try:
            result = analyse_frame(det, df)
        except SchemaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        analysis = store.add_analysis(
            filename=filename, source=source, summary=result["summary"], flows=result["flows"],
            features=result["features"], rejected_rows=len(result["rejected_rows"]),
            duration_ms=result["duration_ms"], model_version=det.version)
        _, top = store.list_flows(analysis_id=analysis["id"], threats_only=True, order="severity", limit=10)
        return {"analysis": analysis, "top_threats": top}

    # ------------------------------------------------------------------ system
    @app.get("/health", tags=["system"])
    def health(det: DetDep) -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "model_version": det.version}

    # ------------------------------------------------------------- prediction
    @app.post("/api/v1/predict", response_model=PredictResponse, tags=["detection"], dependencies=heavy,
              responses={413: {"model": ErrorResponse}})
    def predict(req: PredictRequest, det: DetDep, store: StoreDep) -> dict[str, Any]:
        """Classify one or more flows supplied as JSON feature records."""
        if len(req.flows) > settings.max_batch:
            raise HTTPException(status_code=413, detail=f"Batch exceeds {settings.max_batch} flows")
        df = pd.DataFrame.from_records(req.flows)
        try:
            result = analyse_frame(det, df)
        except SchemaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        flows = result["flows"]
        X = pd.DataFrame(result["features"], columns=det.features)
        proba = det.predict(X).proba
        for i, f in enumerate(flows):
            f["probabilities"] = {c: round(float(p), 6) for c, p in zip(det.classes, proba[i])}
        if req.explain:
            n = min(len(flows), settings.max_explain)
            for f, e in zip(flows[:n], det.explain(X.iloc[:n])):
                f["explanation"] = e
        analysis_id = None
        if req.save:
            analysis_id = store.add_analysis(
                filename=safe_filename(req.name), source="api", summary=result["summary"], flows=flows,
                features=result["features"], rejected_rows=len(result["rejected_rows"]),
                duration_ms=result["duration_ms"], model_version=det.version)["id"]
        return {"model_version": det.version, "predictions": flows, "rejected_rows": result["rejected_rows"],
                "summary": result["summary"], "analysis_id": analysis_id}

    @app.post("/api/v1/analyses", response_model=AnalysisCreated, status_code=201, tags=["detection"],
              dependencies=heavy, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                                             415: {"model": ErrorResponse}})
    def upload(det: DetDep, store: StoreDep, file: Annotated[UploadFile, File(description="CICFlowMeter CSV")]) -> dict[str, Any]:
        """Upload a CICFlowMeter flow CSV (CIC-IDS2017 or CSE-CIC-IDS2018 column names) for batch analysis."""
        try:
            validate_upload_meta(file.filename, file.content_type)
            data = read_limited(file.file, settings.max_upload_bytes)
            df = parse_csv(data, settings.max_rows, settings.max_columns)
        except UploadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        return _run_and_store(det, store, df, safe_filename(file.filename), "upload")

    # ----------------------------------------------------------------- samples
    @app.get("/api/v1/samples", tags=["samples"], dependencies=secured)
    def samples(request: Request) -> list[dict[str, Any]]:
        """Bundled sample files (held-out CIC-IDS2017 test flows)."""
        return [{k: v for k, v in s.items() if k != "path"} for s in request.app.state.samples.values()]

    def _sample(request: Request, name: str) -> dict[str, Any]:
        entry = request.app.state.samples.get(name)
        if entry is None:
            raise HTTPException(status_code=404, detail="Sample not found")
        return entry

    @app.get("/api/v1/samples/{name}", tags=["samples"], dependencies=secured, response_class=FileResponse)
    def sample_file(request: Request, name: str) -> FileResponse:
        entry = _sample(request, name)
        return FileResponse(entry["path"], media_type="text/csv", filename=entry["name"])

    @app.post("/api/v1/samples/{name}/analyze", response_model=AnalysisCreated, status_code=201, tags=["samples"],
              dependencies=heavy)
    def analyse_sample(request: Request, name: str, det: DetDep, store: StoreDep) -> dict[str, Any]:
        entry = _sample(request, name)
        df = parse_csv(entry["path"].read_bytes(), settings.max_rows, settings.max_columns)
        return _run_and_store(det, store, df, entry["name"], "sample")

    # ----------------------------------------------------------------- history
    @app.get("/api/v1/analyses", response_model=AnalysisPage, tags=["history"], dependencies=secured)
    def list_analyses(store: StoreDep, limit: Annotated[int, Query(ge=1, le=100)] = 20,
                      offset: Annotated[int, Query(ge=0)] = 0) -> dict[str, Any]:
        total, items = store.list_analyses(limit, offset)
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    @app.get("/api/v1/analyses/{analysis_id}", response_model=Analysis, tags=["history"], dependencies=secured)
    def get_analysis(analysis_id: str, store: StoreDep) -> dict[str, Any]:
        a = store.get_analysis(analysis_id)
        if a is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return a

    @app.delete("/api/v1/analyses/{analysis_id}", tags=["history"], dependencies=secured)
    def delete_analysis(analysis_id: str, store: StoreDep) -> dict[str, bool]:
        if not store.delete_analysis(analysis_id):
            raise HTTPException(status_code=404, detail="Analysis not found")
        return {"deleted": True}

    @app.delete("/api/v1/analyses", tags=["history"], dependencies=secured)
    def clear_history(store: StoreDep) -> dict[str, int]:
        return {"deleted": store.clear()}

    @app.get("/api/v1/analyses/{analysis_id}/flows", response_model=FlowPage, tags=["history"], dependencies=secured)
    def analysis_flows(
        analysis_id: str, store: StoreDep, cls: ClassFilter = None, severity: SeverityFilter = None,
        threats_only: bool = False, order: Annotated[str, Query(pattern="^(index|severity)$")] = "index",
        limit: Annotated[int, Query(ge=1, le=500)] = 50, offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        if store.get_analysis(analysis_id) is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        total, items = store.list_flows(analysis_id=analysis_id, cls=cls, severity=severity,
                                        threats_only=threats_only, order=order, limit=limit, offset=offset)
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    @app.get("/api/v1/analyses/{analysis_id}/flows/{idx}", response_model=FlowDetail, tags=["explainability"],
             dependencies=secured)
    def flow_detail(analysis_id: str, idx: int, det: DetDep, store: StoreDep) -> dict[str, Any]:
        """A stored flow with its SHAP explanation (computed on demand)."""
        flow = store.get_flow(analysis_id, idx)
        if flow is None:
            raise HTTPException(status_code=404, detail="Flow not found")
        vec = flow.pop("features")
        if len(vec) != len(det.features):
            raise HTTPException(status_code=409, detail="Flow was analysed by an incompatible model version")
        X = pd.DataFrame([vec], columns=det.features)
        flow["explanation"] = det.explain(X)[0]
        flow["features"] = {f: float(v) for f, v in zip(det.features, vec)}
        return flow

    @app.get("/api/v1/threats", response_model=FlowPage, tags=["threat intelligence"], dependencies=secured)
    def threats(
        store: StoreDep, cls: ClassFilter = None, severity: SeverityFilter = None,
        min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
        order: Annotated[str, Query(pattern="^(recent|severity)$")] = "recent",
        limit: Annotated[int, Query(ge=1, le=200)] = 50, offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        """Suspicious flows across all stored analyses."""
        total, items = store.list_flows(threats_only=True, cls=cls, severity=severity, min_confidence=min_confidence,
                                        order=order, limit=limit, offset=offset)
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    @app.get("/api/v1/stats", tags=["threat intelligence"], dependencies=secured)
    def stats(store: StoreDep, days: Annotated[int, Query(ge=1, le=90)] = 14) -> dict[str, Any]:
        return store.stats(days)

    # ------------------------------------------------------------------- model
    def _metrics(request: Request) -> dict[str, Any]:
        m = request.app.state.metrics
        if not m:
            raise HTTPException(status_code=404, detail="No evaluation report found")
        return m

    @app.get("/api/v1/model", tags=["model"], dependencies=secured)
    def model_info(request: Request, det: DetDep) -> dict[str, Any]:
        """Model metadata and the full evaluation report produced by the training pipeline."""
        m = _metrics(request)
        return {
            "name": det.meta.get("name"), "version": det.version, "trained_at": det.meta.get("trained_at"),
            "temperature": det.temperature, "classes": [{"name": c, **CLASS_INFO[c]} for c in det.classes],
            "feature_count": len(det.features),
            **{k: m.get(k) for k in ("deployed_model", "selection", "dataset", "features", "models", "calibration",
                                     "unseen_attack_probe", "experiments", "environment")},
        }

    @app.get("/api/v1/model/explainability", tags=["model", "explainability"], dependencies=secured)
    def global_explainability(request: Request) -> dict[str, Any]:
        """Global SHAP analysis of the deployed model (class-balanced held-out sample)."""
        return _metrics(request).get("explainability", {})

    @app.get("/api/v1/model/features", tags=["model"], dependencies=secured)
    def feature_schema(det: DetDep) -> dict[str, Any]:
        """Columns an uploaded CSV must contain (either naming convention is accepted)."""
        return {
            "required": [{"name": f, "cic2017": FEATURE_BY_NAME[f].cic2017, "cic2018": FEATURE_BY_NAME[f].cic2018,
                          "description": FEATURE_BY_NAME[f].description, "unit": FEATURE_BY_NAME[f].unit}
                         for f in det.features],
            "limits": {"max_upload_mb": settings.max_upload_mb, "max_rows": settings.max_rows,
                       "max_batch": settings.max_batch},
        }

    return app


def __getattr__(name: str):  # pragma: no cover - lazy module-level app for `uvicorn sentinelai.api.main:app`
    if name == "app":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        return create_app()
    raise AttributeError(name)
