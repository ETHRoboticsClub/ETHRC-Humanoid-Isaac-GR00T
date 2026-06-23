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

"""G1 + SONIC modality config that uses the ego view AND both wrist cameras.

This is a copy of the built-in ``unitree_g1_sonic`` config
(gr00t/configs/data/embodiment_configs.py) with the *only* change being the
``video`` modality: instead of the single ``ego_view`` camera, it loads all
three views (ego + left wrist + right wrist). The SONIC state/action space
(motion-token latents + hand joints) is preserved exactly.

Usage:
    bash examples/G1-SONIC-3cam/finetune_g1_sonic_3cam.sh \
        --base-model-path <sonic_checkpoint_or_GR00T-N1.7-3B> \
        --dataset-path <your_g1_sonic_dataset> \
        --output-dir <output_dir>
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

unitree_g1_sonic_3cam = {
    # >>> The ONLY change vs. the built-in unitree_g1_sonic config <<<
    # These three keys match the keys under "video" in your dataset's
    # meta/modality.json (ego_view + left_wrist + right_wrist).
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "ego_view",    # head / egocentric camera
            "left_wrist",  # left wrist camera
            "right_wrist", # right wrist camera
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "projected_gravity",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=[
            "motion_token",       # SONIC whole-body latents
            "left_hand_joints",
            "right_hand_joints",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# "unitree_g1_sonic" is already registered in MODALITY_CONFIGS, and
# register_modality_config() asserts the tag is *new*. So we override the entry
# in place instead of re-registering. Keep --embodiment-tag unitree_g1_sonic.
MODALITY_CONFIGS["unitree_g1_sonic"] = unitree_g1_sonic_3cam
