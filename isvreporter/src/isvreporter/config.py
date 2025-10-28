"""Configuration for ISV Lab Service."""

import os


def get_endpoint() -> str:
    """Get ISV Lab Service endpoint from environment.

    Returns:
        The endpoint URL from ISV_SERVICE_ENDPOINT env var, or empty string if not set.
    """
    return os.environ.get("ISV_SERVICE_ENDPOINT", "")


def get_ssa_issuer() -> str:
    """Get SSA issuer URL from environment.

    Returns:
        The SSA issuer URL from ISV_SSA_ISSUER env var, or empty string if not set.
    """
    return os.environ.get("ISV_SSA_ISSUER", "")
