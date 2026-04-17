# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared GCP error classification utilities.

Mirrors the oracle's ``aws/common/errors.py`` so stub error reporting is
consistent across NCPs. Maps ``google.api_core`` and
``google.auth`` exceptions to the provider-agnostic error types documented
in ``docs/existing-patterns/common.md``.
"""

import functools
import json
from collections.abc import Callable

from google.api_core import exceptions as gax_exc
from google.auth import exceptions as gauth_exc


def classify_gcp_error(e: Exception) -> tuple[str, str]:
    """Classify a GCP exception into (error_type, error_message).

    Standard error types (matching the oracle):
      - credentials_missing: ADC / service-account credentials not configured
      - credentials_expired: OAuth refresh failed
      - credentials_invalid: Invalid signature / bad key material
      - access_denied: Authenticated but lacks IAM permission
      - api_error: Other GCP API errors
      - unknown_error: Non-GCP exceptions
    """
    if isinstance(e, gauth_exc.DefaultCredentialsError):
        return "credentials_missing", "GCP application-default credentials not found"
    if isinstance(e, gauth_exc.RefreshError):
        return "credentials_expired", "GCP credentials expired - refresh required"
    if isinstance(e, gauth_exc.GoogleAuthError):
        return "credentials_invalid", f"GCP credential error: {e}"
    if isinstance(e, gax_exc.Unauthenticated):
        return "credentials_invalid", f"GCP authentication failed: {e}"
    if isinstance(e, gax_exc.PermissionDenied | gax_exc.Forbidden):
        return "access_denied", f"Access denied: {e}"
    if isinstance(e, gax_exc.GoogleAPIError):
        return "api_error", str(e)
    return "unknown_error", str(e)


def handle_gcp_errors[**P](func: Callable[P, int]) -> Callable[P, int]:
    """Decorator that catches uncaught GCP errors and emits structured JSON.

    Stubs still print their own JSON for the success path; this decorator
    only kicks in if the stub body raises an exception that escaped its own
    try/except (e.g. client construction fails before main's try block).
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> int:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type, error_msg = classify_gcp_error(e)
            print(
                json.dumps(
                    {"success": False, "error_type": error_type, "error": error_msg},
                    indent=2,
                )
            )
            return 1

    return wrapper
