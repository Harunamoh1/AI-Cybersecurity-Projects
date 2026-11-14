"""FastAPI application exposing the PhishLens detector."""

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from phishlens import __version__
from phishlens.api.config import Settings, get_settings
from phishlens.api.ratelimit import RateLimiter
from phishlens.api.schemas import (
    BatchItemError,
    BatchScanRequest,
    BatchScanResponse,
    ErrorResponse,
    HistoryPage,
    ScanRequest,
    ScanResult,
)
from phishlens.api.storage import ScanStore
from phishlens.features import InvalidURLError
from phishlens.ml.model import PhishingModel

log = logging.getLogger("phishlens.api")


def create_app(settings: Settings | None = None, model: PhishingModel | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or PhishingModel.load(settings.model_dir, settings.verify_model_checksum)
        app.state.store = ScanStore(settings.database_path)
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
        yield
        app.state.store.close()

    app = FastAPI(
        title="PhishLens API",
        version=__version__,
        description=(
            "Explainable machine-learning phishing URL detection. URLs are analysed lexically "
            "and are **never fetched**."
        ),
        lifespan=lifespan,
        responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = first.get("msg", "Invalid request")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": f"{loc}: {msg}" if loc else msg},
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):  # pragma: no cover - defensive
        log.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ----------------------------------------------------------- dependencies
    def get_model(request: Request) -> PhishingModel:
        return request.app.state.model

    def get_store(request: Request) -> ScanStore:
        return request.app.state.store

    def rate_limited(request: Request) -> None:
        client = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
        if not request.app.state.limiter.allow(client):
            raise HTTPException(status_code=429, detail="Rate limit exceeded; try again in a minute")

    ModelDep = Annotated[PhishingModel, Depends(get_model)]
    StoreDep = Annotated[ScanStore, Depends(get_store)]

    def _analyze(model: PhishingModel, url: str) -> dict[str, Any]:
        if len(url) > settings.max_url_length:
            raise InvalidURLError(f"URL exceeds {settings.max_url_length} characters")
        return model.analyze(url)

    # ------------------------------------------------------------------ routes
    @app.get("/health", tags=["system"])
    def health(model: ModelDep) -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "model": model.model_name, "model_version": model.version}

    @app.post(
        "/api/v1/scan",
        response_model=ScanResult,
        tags=["scan"],
        dependencies=[Depends(rate_limited)],
        responses={400: {"model": ErrorResponse}},
    )
    def scan(req: ScanRequest, model: ModelDep, store: StoreDep) -> dict[str, Any]:
        """Analyse one URL and return verdict, risk score and explanation."""
        try:
            result = _analyze(model, req.url)
        except InvalidURLError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return store.add(result) if req.save else result

    @app.post("/api/v1/scan/batch", response_model=BatchScanResponse, tags=["scan"], dependencies=[Depends(rate_limited)])
    def scan_batch(req: BatchScanRequest, model: ModelDep, store: StoreDep) -> dict[str, Any]:
        """Analyse up to `PHISHLENS_MAX_BATCH` URLs. Invalid URLs are reported per item."""
        if len(req.urls) > settings.max_batch_size:
            raise HTTPException(status_code=400, detail=f"Batch exceeds {settings.max_batch_size} URLs")
        results, errors = [], []
        for url in req.urls:
            try:
                r = _analyze(model, url.strip())
                results.append(store.add(r) if req.save else r)
            except InvalidURLError as exc:
                errors.append(BatchItemError(url=url[:300], error=str(exc)))
        return {"results": results, "errors": errors}

    @app.get("/api/v1/history", response_model=HistoryPage, tags=["history"])
    def history(
        store: StoreDep,
        limit: Annotated[int, Query(ge=1, le=200)] = 25,
        offset: Annotated[int, Query(ge=0)] = 0,
        verdict: Annotated[str | None, Query(pattern="^(phishing|legitimate)$")] = None,
        q: Annotated[str | None, Query(max_length=200)] = None,
    ) -> dict[str, Any]:
        total, items = store.list(limit, offset, verdict, q)
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    @app.get("/api/v1/history/{scan_id}", response_model=ScanResult, tags=["history"])
    def history_item(scan_id: str, store: StoreDep) -> dict[str, Any]:
        item = store.get(scan_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Scan not found")
        return item

    @app.delete("/api/v1/history/{scan_id}", tags=["history"])
    def delete_history_item(scan_id: str, store: StoreDep) -> dict[str, bool]:
        if not store.delete(scan_id):
            raise HTTPException(status_code=404, detail="Scan not found")
        return {"deleted": True}

    @app.delete("/api/v1/history", tags=["history"])
    def clear_history(store: StoreDep) -> dict[str, int]:
        return {"deleted": store.clear()}

    @app.get("/api/v1/stats", tags=["analytics"])
    def stats(store: StoreDep, days: Annotated[int, Query(ge=1, le=90)] = 14) -> dict[str, Any]:
        """Aggregates over stored scans for the analytics dashboard."""
        return store.stats(days)

    @app.get("/api/v1/model", tags=["model"])
    def model_card(model: ModelDep) -> dict[str, Any]:
        """Model metadata and the evaluation report produced by the training pipeline."""
        m = model.metrics
        return {
            "name": model.model_name,
            "version": model.version,
            "trained_at": model.trained_at,
            "threshold": model.threshold,
            "selection": m.get("selection"),
            "datasets": m.get("datasets"),
            "cleaning": m.get("cleaning"),
            "splits": m.get("splits"),
            "split_strategy": m.get("split_strategy"),
            "models": m.get("models"),
            "per_source_test": m.get("per_source_test"),
            "experiments": m.get("experiments"),
            "curves": m.get("curves"),
            "score_distribution": m.get("score_distribution"),
            "feature_importance": m.get("feature_importance"),
            "environment": m.get("environment"),
        }

    return app


app = create_app()  # uvicorn phishlens.api.main:app
