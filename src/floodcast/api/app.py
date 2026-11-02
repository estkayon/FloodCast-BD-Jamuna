# src/floodcast/api/app.py

"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from floodcast.api.dependencies import model_store
from floodcast.api.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup."""
    logger.info("Loading model...")
    model_store.load()
    logger.info("Model loaded: %s (loaded=%s)", model_store.model_type, model_store.is_loaded)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="FloodCast BD API",
    description="Physics-constrained flood nowcasting for Jamuna River Basin",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
