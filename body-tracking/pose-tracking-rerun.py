import mediapipe as mp
import numpy as np
from typing import Any
import numpy.typing as npt

# Reading 2d landmarks position from mediapipe pose results.

"""
    Args:
        results (any): mediapipe pose results
        image_width (int): width of the input image
        image_height (int): height of the input image.
        
    Return:
        np.array | None: Array of 2D landmark position 
"""

def read_landmarks_positions_2d(
        results: Any,
        image_width: int,
        image_height: int,
)-> npt.NDArray[np.float32] | None:
    if results.pose_landmarks is None:
        return None
    else:
        normalized_landmarks = [result.pose_landmarks.landmark[lm]]

