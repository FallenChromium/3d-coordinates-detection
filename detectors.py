from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from simulation import CameraParams


@dataclass
class Detection:
    position_2d: np.ndarray
    confidence: float
    camera_id: int

class DetectionAlgorithm(ABC):
    @abstractmethod
    def detect(
        self,
        camera: CameraParams,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> Optional[Detection]:
        pass

class BlobDetector(DetectionAlgorithm):
    def __init__(self, min_area: float = 20, max_area: float = 350):
        params = cv2.SimpleBlobDetector.Params()
        params.maxArea = max_area
        params.minArea = min_area

        self.detector = cv2.SimpleBlobDetector.create(params)

    def detect(self, frame: np.ndarray, camera: CameraParams) -> Optional[Detection]:
        keypoints = self.detector.detect(frame)
        if keypoints:
            position = np.array([keypoints[0].pt[0], keypoints[0].pt[1]])
            confidence = keypoints[0].size / 100  # Normalize size to confidence
            return Detection(position, confidence, camera.id)
        return None

