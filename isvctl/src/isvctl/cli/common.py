# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared constants and helpers for CLI subcommands."""

from pathlib import Path

OUTPUT_DIR_NAME = "_output"


def get_output_dir(root: Path | None = None) -> Path:
    """Return the output directory, creating it if needed.

    Args:
        root: Base directory. Defaults to cwd when None.

    Returns:
        Path to the output directory (already created on disk).
    """
    base = root or Path.cwd()
    output_dir = base / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
