
# Data structures
from dataclasses import dataclass
from math import floor
import os
from typing import List, Optional
from scipy.spatial.transform import Rotation

import cv2
import numpy as np
import pandas as pd


@dataclass
class CameraParams:
    id: str
    position: np.ndarray
    azimuth: float
    focal_length: float
    matrix_width: float
    matrix_height: float
    z_rotation: float = 0

    @property
    def rotation_matrix(self) -> np.ndarray:
        # Create rotation matrix from azimuth
        r = Rotation.from_euler("z", self.azimuth, degrees=True)
        return r.as_matrix()

# Simulation environment
class SimulationEnvironment:
    def __init__(
        self,
        ground_truth_data: pd.DataFrame,
        cameras: List[CameraParams],
        object_diameter: float,
    ):
        self.ground_truth: pd.DataFrame | None = ground_truth_data
        self.cameras: List[CameraParams] = cameras
        self.current_timestamp = 0
        self.sphere_diameter = object_diameter  # from settings
        self.captures: list[cv2.VideoCapture] = []

    def get_camera_views(self) -> List[cv2.typing.MatLike]:
        """Generate simulated camera views based on ground truth position"""
        # current_pos = self.ground_truth.loc[self.current_timestamp][
        #     ["X,m", "Y,m", "Z,m"]
        # ].values
        views = []
        videos_prefix = "./data/step1/videoset1/"
        if len(self.captures) == 0:
            for camera in self.cameras:
                capture = cv2.VideoCapture(os.path.join(videos_prefix, f"Seq1_{camera.id}.mov"))
                self.captures.append(capture)
        for cap in self.captures:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = floor(self.current_timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            views.append(frame)

        return views

    def project_point_to_camera(
        self, point_3d: np.ndarray, camera: CameraParams
    ) -> Optional[np.ndarray]:
        """Project 3D point to camera image plane"""
        # Transform point to camera coordinates
        point_rel = point_3d - camera.position
        point_cam = camera.rotation_matrix.T @ point_rel

        # Check if point is in front of camera
        if point_cam[2] <= 0:
            return None

        # Project to image plane
        x = (point_cam[0] * camera.focal_length) / point_cam[2]
        y = (point_cam[1] * camera.focal_length) / point_cam[2]

        # Convert to pixel coordinates
        px = x * 1920 / camera.matrix_width + 320
        py = y * 1080 / camera.matrix_height + 240

        if 0 <= px <= 1920 and 0 <= py <= 1080:
            return np.array([px, py])
        return None

    def get_projected_radius(self, point_3d: np.ndarray, camera: CameraParams) -> float:
        """Calculate projected radius of sphere in pixels"""
        distance = np.linalg.norm(point_3d - camera.position)
        return (self.sphere_diameter / 2 * camera.focal_length / distance) * (
            640 / camera.matrix_width
        )

    def step(self) -> bool:
        """Advance simulation by one frame"""
        self.current_timestamp += 0.5
        return self.current_timestamp < len(self.ground_truth)
