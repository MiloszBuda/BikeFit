"""Halpe26 keypoints specifications and skeleton topologies for bike fitting."""

from typing import Dict, List, Tuple

HALPE26_KEYPOINTS: List[str] = [
    "nose",             # 0
    "left_eye",         # 1
    "right_eye",        # 2
    "left_ear",         # 3
    "right_ear",        # 4
    "left_shoulder",    # 5
    "right_shoulder",   # 6
    "left_elbow",       # 7
    "right_elbow",      # 8
    "left_wrist",       # 9
    "right_wrist",      # 10
    "left_hip",         # 11
    "right_hip",        # 12
    "left_knee",        # 13
    "right_knee",       # 14
    "left_ankle",       # 15
    "right_ankle",      # 16
    "head",             # 17
    "neck",             # 18
    "hip_center",       # 19
    "left_big_toe",     # 20
    "right_big_toe",    # 21
    "left_small_toe",   # 22
    "right_small_toe",  # 23
    "left_heel",        # 24
    "right_heel",       # 25
]

HALPE26_NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(HALPE26_KEYPOINTS)}

# Skeleton pairs for rendering (parent_name, child_name)
HALPE26_SKELETON_PAIRS: List[Tuple[str, str]] = [
    # Torso and head
    ("head", "neck"),
    ("neck", "hip_center"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "hip_center"),
    ("right_hip", "hip_center"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    # Arms
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Legs
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # Feet
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_big_toe"),
    ("left_heel", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_big_toe"),
    ("right_heel", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
]

# Side specific keypoints for bike fitting analysis
SIDE_KEYPOINTS: Dict[str, Dict[str, str]] = {
    "right": {
        "ear": "right_ear",
        "shoulder": "right_shoulder",
        "elbow": "right_elbow",
        "wrist": "right_wrist",
        "hip": "right_hip",
        "knee": "right_knee",
        "ankle": "right_ankle",
        "heel": "right_heel",
        "big_toe": "right_big_toe",
        "small_toe": "right_small_toe",
    },
    "left": {
        "ear": "left_ear",
        "shoulder": "left_shoulder",
        "elbow": "left_elbow",
        "wrist": "left_wrist",
        "hip": "left_hip",
        "knee": "left_knee",
        "ankle": "left_ankle",
        "heel": "left_heel",
        "big_toe": "left_big_toe",
        "small_toe": "left_small_toe",
    },
}
