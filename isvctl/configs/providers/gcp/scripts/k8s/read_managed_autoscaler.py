#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit current GKE managed-autoscaler evidence for K8sClusterAutoscalerCheck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k8s_lib as k8s


def parse_args() -> argparse.Namespace:
    """Parse the immutable identity of the GKE system pool to inspect."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-name", required=True)
    parser.add_argument("--node-pool", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--project", required=True)
    return parser.parse_args()


def main() -> int:
    """Read live autoscaling state and emit the provider-neutral evidence object."""
    args = parse_args()
    evidence = k8s.read_managed_autoscaler_evidence(
        args.cluster_name,
        args.node_pool,
        args.location,
        args.project,
    )
    print(json.dumps(evidence))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
