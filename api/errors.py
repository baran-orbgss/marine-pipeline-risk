"""Typed exceptions mapped to clean HTTP responses.

Expected failure conditions (missing project, out-of-bounds tile, disallowed path, non-allowlisted
capability) must never surface as an unhandled 500 -- each gets its own status code here.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class ApiError(Exception):
    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(ApiError):
    status_code = 404


class UnsafePathRequestError(ApiError):
    status_code = 400


class CapabilityNotAllowlistedError(ApiError):
    status_code = 403


class CapabilityNotAvailableError(ApiError):
    status_code = 409


class TileOutOfBoundsError(ApiError):
    status_code = 404


class UnsupportedSpatialFileError(ApiError):
    status_code = 422


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
