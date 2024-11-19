
# Data structures
from dataclasses import dataclass
from math import floor
import os
from typing import List, Optional
from scipy.spatial.transform import Rotation

import cv2
import numpy as np
import pandas as pd

from utils import fuse_rgb_ir


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
        cameras: List[CameraParams],
        object_diameter: float,
        ground_truth_data: pd.DataFrame | None = None,
        folder_prefix: str = "./data/step1/videoset1",
    ):
        self.ground_truth: pd.DataFrame | None = ground_truth_data
        self.cameras: List[CameraParams] = cameras
        self.current_timestamp = 0
        self.sphere_diameter = object_diameter  # from settings
        self.captures: list[dict[str, cv2.VideoCapture]] = [{"rgb": cv2.VideoCapture(f"{folder_prefix}/Seq{folder_prefix[-1]}_{camera.id}.mov"),
                                                             "ir": cv2.VideoCapture(f"{folder_prefix}/Seq{folder_prefix[-1]}_{camera.id}T.mov")}
                                                            for camera in cameras]

    def get_camera_views(self) -> List[dict[str, List[cv2.typing.MatLike]]]:
        """Generate simulated camera views based on ground truth position"""
        # current_pos = self.ground_truth.loc[self.current_timestamp][
        #     ["X,m", "Y,m", "Z,m"]
        # ].values
        views = []
        for cap in self.captures:
            fps = cap["rgb"].get(cv2.CAP_PROP_FPS)
            frame_number = floor(self.current_timestamp * fps)
            cap["rgb"].set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            cap["ir"].set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame_rgb = cap["rgb"].read()
            ret, frame_ir = cap["ir"].read()
            frame = fuse_rgb_ir(frame_rgb, frame_ir)
            views.append({"fused": frame, "rgb": frame_rgb, "ir": frame_ir})


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
