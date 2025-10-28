"""ISV Lab Test Results Reporter - report validation test results to ISV Lab Service."""

from isvreporter.config import get_endpoint, get_ssa_issuer
from isvreporter.version import get_version

__all__ = [
    "__version__",
    "get_endpoint",
    "get_ssa_issuer",
]

__version__ = get_version("isvreporter")
