"""Version detection utilities.

Version resolution works as follows:

1. CI/CD build: The CI pipeline generates `_version.py` files with the
   release version (e.g., `__version__ = "1.2.3"` or `__version__ = "dev-abc1234"`).
   These files are baked into the wheel and take priority at runtime.

2. Git tag: If `_version.py` doesn't exist (e.g., running from a cloned repo),
   we use `git describe --tags` to resolve the version from tags. This covers:
   - Exact tagged release: `v1.2.3` -> `1.2.3`
   - Commits after a tag: `v1.2.3-5-gabc1234` -> `1.2.3-5-gabc1234`

3. Git SHA: If no tags are found, we fall back to the commit SHA for a
   meaningful dev version (e.g., `dev-c42ee70`).

4. Fallback: If git isn't available, we use `dev-local`.

Note: `_version.py` files are in `.gitignore` and should never be committed.
"""

import importlib
import subprocess


def get_version(package_name: str) -> str:
    """Get package version with fallback chain.

    Args:
        package_name: Name of the package (e.g., 'isvctl', 'isvtest', 'isvreporter')

    Returns:
        Version string (e.g., '1.2.3', '1.2.3-5-gabc1234', 'dev-c42ee70', or 'dev-local')
    """
    # CI/CD baked version
    try:
        version_module = importlib.import_module(f"{package_name}._version")
        return version_module.__version__
    except (ModuleNotFoundError, AttributeError):
        # Fall back to git/local when the baked version module is missing
        pass

    # Git tag: try git describe for tagged versions
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--match", "v*"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip().lstrip("v")
            return version
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Git SHA: fall back to commit hash when no tags exist
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return f"dev-{result.stdout.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        # Fall back to dev-local if git is unavailable or too slow
        pass

    return "dev-local"
