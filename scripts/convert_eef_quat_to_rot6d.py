#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert quaternion columns to rot6d in a LeRobot v2.1 dataset.

Creates a full copy of the dataset with three new parquet columns:

  <action-eef-key>_rot6d      (18,): [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6)]
  <state-eef-key>_rot6d       (18,): same layout from the EEF observation column
  <base-pose-key>_rot6d        (9,): [base_xyz(3), base_rot6d(6)]

rot6d = first two rows of the 3×3 rotation matrix, row-major flattened → 6 values.
It is unconstrained (no unit-norm requirement) and can be normalized freely with min-max.

New named slices are registered in meta/modality.json and meta/info.json so the
GR00T data loader can find them via modality_keys.

Expected column layouts:
  EEF (14 dims, two arms): [left_pos(3), left_quat(4), right_pos(3), right_quat(4)]
    Quaternion order: wxyz (default for G1 sim dataset)
  Base pose (7 dims):      [x, y, z, qx, qy, qz, qw]
    Quaternion order: xyzw (standard in observation.robot_base_pose)

Usage:
  python scripts/convert_eef_quat_to_rot6d.py \\
      --dataset-path  /path/to/source/dataset \\
      --output-path   /path/to/output/dataset \\
      [--action-eef-key  action.eef] \\
      [--state-eef-key   observation.eef_state] \\
      [--base-pose-key   observation.robot_base_pose] \\
      [--eef-quat-order  wxyz]

After conversion, generate stats before training:
  python gr00t/data/stats.py \\
      --dataset-path  /path/to/output/dataset \\
      --embodiment-tag new_embodiment \\
      --modality-config-path examples/G1-LocoManip/g1_locomanip_config.py
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Core conversion helpers
# ---------------------------------------------------------------------------


def quat_to_rot6d(quat: np.ndarray, order: str) -> np.ndarray:
    """
    Convert a single quaternion to a 6-dim rot6d vector (first two rows of R).

    Args:
        quat:  (4,) quaternion.
        order: "wxyz" or "xyzw" — scipy always takes xyzw internally.

    Returns:
        (6,) rot6d.
    """
    if order == "wxyz":
        w, x, y, z = quat
        quat_xyzw = np.array([x, y, z, w])
    elif order == "xyzw":
        quat_xyzw = quat
    else:
        raise ValueError(f"Unknown quat order: {order!r}. Use 'wxyz' or 'xyzw'.")

    mat = Rotation.from_quat(quat_xyzw).as_matrix()  # (3, 3)
    return mat[:2, :].flatten()                        # first two rows → (6,)


def eef14_to_rot6d18(eef: np.ndarray, quat_order: str) -> np.ndarray:
    """
    14-dim dual-arm EEF → 18-dim rot6d.

    Input : [left_pos(3),  left_quat(4),  right_pos(3),  right_quat(4)]
    Output: [left_pos(3),  left_rot6d(6), right_pos(3),  right_rot6d(6)]
    """
    return np.concatenate([
        eef[0:3],
        quat_to_rot6d(eef[3:7],   order=quat_order),
        eef[7:10],
        quat_to_rot6d(eef[10:14], order=quat_order),
    ])


def base7_to_rot6d9(base: np.ndarray) -> np.ndarray:
    """
    7-dim base pose → 9-dim rot6d.

    Input : [x, y, z, qx, qy, qz, qw]   (xyzw quaternion order)
    Output: [x, y, z, rot6d(6)]
    """
    return np.concatenate([
        base[0:3],
        quat_to_rot6d(base[3:7], order="xyzw"),
    ])


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------


def process_parquet(
    src: Path,
    dst: Path,
    action_eef_key: str,
    state_eef_key: str,
    base_pose_key: str,
    eef_quat_order: str,
) -> None:
    df = pd.read_parquet(src)

    df[action_eef_key + "_rot6d"] = df[action_eef_key].apply(
        lambda v: eef14_to_rot6d18(v, eef_quat_order)
    )
    df[state_eef_key + "_rot6d"] = df[state_eef_key].apply(
        lambda v: eef14_to_rot6d18(v, eef_quat_order)
    )
    df[base_pose_key + "_rot6d"] = df[base_pose_key].apply(base7_to_rot6d9)

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)


def update_modality_json(
    meta_dir: Path,
    action_eef_key: str,
    state_eef_key: str,
    base_pose_key: str,
) -> None:
    path = meta_dir / "modality.json"
    with open(path) as f:
        meta = json.load(f)

    new_state = {
        "left_eef_rot6d":     {"start": 0, "end": 9,  "original_key": state_eef_key  + "_rot6d"},
        "right_eef_rot6d":    {"start": 9, "end": 18, "original_key": state_eef_key  + "_rot6d"},
        "robot_base_rot6d":   {"start": 0, "end": 9,  "original_key": base_pose_key  + "_rot6d"},
    }
    new_action = {
        "left_eef_rot6d":     {"start": 0, "end": 9,  "original_key": action_eef_key + "_rot6d"},
        "right_eef_rot6d":    {"start": 9, "end": 18, "original_key": action_eef_key + "_rot6d"},
    }

    for k, v in new_state.items():
        meta["state"].setdefault(k, v)
    for k, v in new_action.items():
        meta["action"].setdefault(k, v)

    with open(path, "w") as f:
        json.dump(meta, f, indent=4)


def update_info_json(
    meta_dir: Path,
    action_eef_key: str,
    state_eef_key: str,
    base_pose_key: str,
) -> None:
    path = meta_dir / "info.json"
    with open(path) as f:
        info = json.load(f)

    eef_names = [
        "left_x", "left_y", "left_z",
        "left_r00", "left_r01", "left_r02",
        "left_r10", "left_r11", "left_r12",
        "right_x", "right_y", "right_z",
        "right_r00", "right_r01", "right_r02",
        "right_r10", "right_r11", "right_r12",
    ]
    base_names = ["x", "y", "z", "r00", "r01", "r02", "r10", "r11", "r12"]

    for key in [action_eef_key + "_rot6d", state_eef_key + "_rot6d"]:
        info["features"].setdefault(key, {"dtype": "float64", "shape": [18], "names": eef_names})

    info["features"].setdefault(
        base_pose_key + "_rot6d",
        {"dtype": "float64", "shape": [9], "names": base_names},
    )

    with open(path, "w") as f:
        json.dump(info, f, indent=4)


def copy_non_parquet_files(src_root: Path, dst_root: Path) -> None:
    for src_file in src_root.rglob("*"):
        if src_file.is_dir() or src_file.suffix == ".parquet":
            continue
        rel = src_file.relative_to(src_root)
        dst_file = dst_root / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-path",    required=True,  help="Source LeRobot dataset root")
    parser.add_argument("--output-path",     required=True,  help="Destination for the converted copy")
    parser.add_argument("--action-eef-key",  default="action.eef",                 help="Parquet column for dual-arm EEF action (14-dim)")
    parser.add_argument("--state-eef-key",   default="observation.eef_state",       help="Parquet column for dual-arm EEF state (14-dim)")
    parser.add_argument("--base-pose-key",   default="observation.robot_base_pose", help="Parquet column for base pose (7-dim, xyzw quat)")
    parser.add_argument("--eef-quat-order",  default="wxyz", choices=["wxyz", "xyzw"], help="Quaternion order for the EEF columns (default: wxyz)")
    args = parser.parse_args()

    src_root = Path(args.dataset_path)
    dst_root = Path(args.output_path)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")
    if dst_root.exists():
        raise FileExistsError(
            f"Output path already exists: {dst_root}\n"
            "Delete it first or choose a different --output-path."
        )

    print(f"Source : {src_root}")
    print(f"Output : {dst_root}")
    print(f"EEF action  : {args.action_eef_key} (quat {args.eef_quat_order})  →  _rot6d (18-dim)")
    print(f"EEF state   : {args.state_eef_key}  (quat {args.eef_quat_order})  →  _rot6d (18-dim)")
    print(f"Base pose   : {args.base_pose_key}   (quat xyzw)              →  _rot6d  (9-dim)")
    print()

    print("Copying non-parquet files...")
    copy_non_parquet_files(src_root, dst_root)

    parquet_files = sorted(src_root.glob("data/**/*.parquet"))
    print(f"Processing {len(parquet_files)} parquet files...")
    for src_pq in parquet_files:
        rel = src_pq.relative_to(src_root)
        process_parquet(
            src_pq, dst_root / rel,
            args.action_eef_key, args.state_eef_key, args.base_pose_key,
            args.eef_quat_order,
        )
        print(f"  {rel}")

    print("\nUpdating meta/modality.json and meta/info.json...")
    update_modality_json(dst_root / "meta", args.action_eef_key, args.state_eef_key, args.base_pose_key)
    update_info_json(dst_root / "meta", args.action_eef_key, args.state_eef_key, args.base_pose_key)

    print("\nConversion complete.")
    print("\nNext — generate stats before training:")
    print(
        f"  python gr00t/data/stats.py \\\n"
        f"      --dataset-path {dst_root} \\\n"
        f"      --embodiment-tag new_embodiment \\\n"
        f"      --modality-config-path examples/G1-LocoManip/g1_locomanip_config.py"
    )


if __name__ == "__main__":
    main()
