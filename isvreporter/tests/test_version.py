"""Tests for version module."""

import subprocess
from unittest.mock import MagicMock, call, patch


class TestGetVersion:
    """Tests for get_version function."""

    def test_returns_baked_version_when_available(self) -> None:
        """Test that baked _version module takes priority."""
        mock_module = MagicMock()
        mock_module.__version__ = "1.2.3"

        with patch("isvreporter.version.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = mock_module
            from isvreporter.version import get_version

            result = get_version("isvreporter")
            assert result == "1.2.3"

    def test_returns_tagged_version_on_exact_tag(self) -> None:
        """Test that exact git tag is returned with v prefix stripped."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="v1.2.3\n")
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "1.2.3"
                # Should only call git describe, not rev-parse
                mock_subprocess.run.assert_called_once()

    def test_returns_tagged_version_with_distance(self) -> None:
        """Test version after a tag includes commit distance and SHA."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="v1.2.3-5-gabc1234\n")
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "1.2.3-5-gabc1234"

    def test_falls_back_to_git_sha_when_no_tags(self) -> None:
        """Test fallback to git SHA when no tags exist."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                # git describe fails (no tags), git rev-parse succeeds
                mock_subprocess.run.side_effect = [
                    MagicMock(returncode=128, stdout=""),
                    MagicMock(returncode=0, stdout="abc1234\n"),
                ]
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "dev-abc1234"
                assert mock_subprocess.run.call_count == 2

    def test_falls_back_to_git_sha_when_version_attr_missing(self) -> None:
        """Test fallback to git when __version__ attribute is missing."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = AttributeError
                # git describe fails, git rev-parse succeeds
                mock_subprocess.run.side_effect = [
                    MagicMock(returncode=128, stdout=""),
                    MagicMock(returncode=0, stdout="def5678\n"),
                ]
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "dev-def5678"

    def test_returns_dev_local_when_all_git_fails(self) -> None:
        """Test fallback to dev-local when all git commands fail."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                # Both git describe and git rev-parse fail
                mock_subprocess.run.side_effect = [
                    MagicMock(returncode=128, stdout=""),
                    MagicMock(returncode=1, stdout=""),
                ]
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "dev-local"

    def test_returns_dev_local_when_git_not_found(self) -> None:
        """Test fallback to dev-local when git is not installed."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                mock_subprocess.run.side_effect = FileNotFoundError
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "dev-local"

    def test_returns_dev_local_when_git_times_out(self) -> None:
        """Test fallback to dev-local when git command times out."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                mock_subprocess.run.side_effect = subprocess.TimeoutExpired("git", 5)
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "dev-local"

    def test_returns_dev_local_on_os_error(self) -> None:
        """Test fallback to dev-local on OSError."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                mock_subprocess.run.side_effect = OSError("Permission denied")
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "dev-local"

    def test_git_tag_version_is_stripped(self) -> None:
        """Test that git tag output whitespace is stripped."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="  v2.0.0  \n")
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "2.0.0"

    def test_git_sha_is_stripped(self) -> None:
        """Test that git SHA whitespace is stripped."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                # git describe fails, git rev-parse returns SHA with whitespace
                mock_subprocess.run.side_effect = [
                    MagicMock(returncode=128, stdout=""),
                    MagicMock(returncode=0, stdout="  abc1234  \n"),
                ]
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                result = get_version("isvreporter")
                assert result == "dev-abc1234"

    def test_git_describe_uses_v_tag_pattern(self) -> None:
        """Test that git describe only matches v* tags."""
        with patch("isvreporter.version.importlib") as mock_importlib:
            with patch("isvreporter.version.subprocess") as mock_subprocess:
                mock_importlib.import_module.side_effect = ModuleNotFoundError
                mock_subprocess.run.return_value = MagicMock(returncode=0, stdout="v3.1.0\n")
                mock_subprocess.TimeoutExpired = subprocess.TimeoutExpired

                from isvreporter.version import get_version

                get_version("isvreporter")

                # Verify git describe is called with --match v*
                describe_call = mock_subprocess.run.call_args_list[0]
                assert describe_call == call(
                    ["git", "describe", "--tags", "--match", "v*"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
