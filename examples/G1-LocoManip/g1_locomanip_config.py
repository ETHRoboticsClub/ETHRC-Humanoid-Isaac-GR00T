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

# Modality config for G1 locomanipulation dataset (LeRobot v2.1).
#
# Requires scripts/convert_eef_quat_to_rot6d.py to have been run first — it adds
# action.eef_rot6d, observation.eef_state_rot6d, and observation.robot_base_pose_rot6d
# columns to each parquet and registers the named slices in meta/modality.json.
#
# KEY LAYOUT after preprocessing (18 dims each):
#   [left_xyz(3), left_rot6d(6), right_xyz(3), right_rot6d(6)]
#
# EEF action representation:
#   ActionType.EEF + ActionFormat.XYZ_ROT6D + ActionRepresentation.RELATIVE
#   → the processor computes q_relative = q_target ⊗ q_current⁻¹ via SE(3)
#     frame composition (EndEffectorActionChunk / EndEffectorPose).
#   → rot6d is freely normalizable (no unit-norm constraint to break).
#
# Navigate and height commands stay ABSOLUTE (velocity / height targets).
#
# Workflow:
#   1. python examples/G1-LocoManip/preprocess_eef_rot6d.py --dataset-path <path>
#   2. python gr00t/data/stats.py --dataset-path <path> \
#        --embodiment-tag new_embodiment \
#        --modality-config-path examples/G1-LocoManip/g1_locomanip_config.py
#   3. bash examples/G1-LocoManip/finetune_g1_locomanip.sh \
#        --base-model-path <ckpt> --dataset-path <path> --output-dir <out>

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


g1_locomanip_config = {
    # Single ego-view RGB camera at the current timestep.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    # Proprioceptive state: EEF poses only — no global base position (unavailable on real robots).
    # Keys must match meta/modality.json "state" entries added by convert_eef_quat_to_rot6d.py.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_eef_rot6d",  # (9,)  left xyz(3) + rot6d(6)
            "right_eef_rot6d", # (9,)  right xyz(3) + rot6d(6)
            # robot_base_rot6d excluded: requires global localization, unavailable on real robots
        ],
    ),
    # Action: 32-step prediction horizon (1.6 s at 20 fps).
    #
    # EEF keys: RELATIVE + EEF + XYZ_ROT6D
    #   The processor calls EndEffectorActionChunk.from_array / EndEffectorPose.from_action_format
    #   which parses each 9-dim row as xyz(3)+rot6d(6), builds a full SE(3) pose, then
    #   computes the relative transform:  T_relative = T_target * T_current⁻¹
    #   This is the proper q_target ⊗ q_current⁻¹ for orientations.
    #   state_key defaults to the same key name, so left_eef_rot6d action uses
    #   left_eef_rot6d state as the reference frame automatically.
    #
    # Navigate / height: ABSOLUTE NON_EEF (velocity and height targets, no conversion needed).
    "action": ModalityConfig(
        delta_indices=list(range(32)),
        modality_keys=[
            "left_eef_rot6d",       # (9,)
            "right_eef_rot6d",      # (9,)
            "navigate_command",     # (3,)  lin_x, lin_y, ang_z
            "base_height_command",  # (1,)
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF,     format=ActionFormat.XYZ_ROT6D),  # left_eef_rot6d
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF,     format=ActionFormat.XYZ_ROT6D),  # right_eef_rot6d
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),    # navigate_command
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),    # base_height_command
        ],
    ),
    # Language: task description from annotation field.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_locomanip_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
