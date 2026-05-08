"""HTTP serving layer — FastAPI app factory + Pydantic transport schemas."""

from energy_forecaster.serving.app import create_app

__all__ = ["create_app"]
