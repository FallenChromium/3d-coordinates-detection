from math import floor
import os
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
from abc import ABC, abstractmethod
import json
from PIL import Image


# Data structures
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


@dataclass
class Detection:
    position_2d: np.ndarray
    confidence: float
    camera_id: int


@dataclass
class Track3D:
    position: np.ndarray
    velocity: np.ndarray
    uncertainty: float
    timestamp: float


# Abstract base classes for algorithms
class DetectionAlgorithm(ABC):
    @abstractmethod
    def detect(
        self,
        camera: CameraParams,
        rgb_frame: np.ndarray | None = None,
        ir_frame: np.ndarray | None = None,
    ) -> Optional[Detection]:
        pass


class TrackingAlgorithm(ABC):
    @abstractmethod
    def update(self, detections: List[Detection], timestamp: float) -> Track3D:
        pass


# Concrete implementations
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


class KalmanTracker(TrackingAlgorithm):
    def __init__(self):
        self.kf = cv2.KalmanFilter(
            6, 3
        )  # 6 state variables (x,y,z + velocities), 3 measurements
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], np.float32
        )
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

    def update(self, detections: List[Detection], timestamp: float) -> Track3D:
        if detections:
            # Use triangulation to get 3D position
            pos_3d = self.triangulate_positions(detections)
            st.write(pos_3d)
            measurement = np.array(pos_3d, np.float32).reshape(-1, 1)
            self.kf.correct(measurement)

        prediction = self.kf.predict()
        position = prediction[:3].flatten()
        velocity = prediction[3:].flatten()
        uncertainty = np.trace(self.kf.errorCovPost)

        return Track3D(position, velocity, uncertainty, timestamp)

    def triangulate_positions(self, detections: List[Detection]) -> np.ndarray:
        # Implement triangulation logic here
        # For now, return average of detected positions
        positions = [d.position_2d for d in detections]
        # TODO: stop using 0 as z coordinate
        positions = [np.append(pos, 0) for pos in positions]
        return np.mean(positions, axis=0)


class ParticleTracker(TrackingAlgorithm):
    def __init__(self, num_particles: int = 100):
        self.num_particles = num_particles
        self.particles = None
        self.weights = np.ones(num_particles) / num_particles

    def update(self, detections: List[Detection], timestamp: float) -> Track3D:
        if self.particles is None:
            # Initialize particles around first detection
            pos_3d = self.triangulate_positions(detections)
            self.particles = np.random.normal(pos_3d, 1.0, (self.num_particles, 3))

        # Predict particle positions
        self.particles += np.random.normal(0, 0.1, self.particles.shape)

        if detections:
            # Update weights based on detections
            pos_3d = self.triangulate_positions(detections)
            distances = np.linalg.norm(self.particles - pos_3d, axis=1)
            self.weights = np.exp(-distances)
            self.weights /= np.sum(self.weights)

            # Resample particles
            indices = np.random.choice(
                self.num_particles, self.num_particles, p=self.weights
            )
            self.particles = self.particles[indices]

        # Estimate state
        position = np.average(self.particles, weights=self.weights, axis=0)
        velocity = np.zeros(3)  # Could be computed from consecutive positions
        uncertainty = np.std(self.particles, axis=0).mean()

        return Track3D(position, velocity, uncertainty, timestamp)

    def triangulate_positions(self, detections: List[Detection]) -> np.ndarray:
        # Similar to KalmanTracker implementation
        positions = [d.position_2d for d in detections]
        # TODO: stop using 0 as z coordinate
        for pos in positions:
            pos.append(0)
        return np.mean(positions, axis=0)


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


# Evaluation metrics
def calculate_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    error = np.linalg.norm(predicted - ground_truth)
    return {
        "euclidean_error": error,
        "x_error": abs(predicted[0] - ground_truth[0]),
        "y_error": abs(predicted[1] - ground_truth[1]),
        "z_error": abs(predicted[2] - ground_truth[2]),
    }


# Streamlit app
def main():
    st.title("3D Object Tracking Simulation")
    # streamlit tabs for each Seq of /data/videoset{1..8}/Seq[Number]_settings
    # Load data
    folder_prefix = "./data/step1/videoset1/"
    ground_truth_data = pd.read_csv(
        os.path.join(folder_prefix, "Seq1_settings.csv"), sep=",", header=0, index_col=0
    )

    # Algorithm selection
    detection_algorithm = st.selectbox(
        "Detection Algorithm", ["Blob Detection", "Deep Learning (Simulated)"]
    )

    tracking_algorithm = st.selectbox(
        "Tracking Algorithm", ["Kalman Filter", "Particle Filter"]
    )

    # Create detector and tracker instances
    st.session_state["detector"] = BlobDetector()
    st.session_state["tracker"] = (
        KalmanTracker() if tracking_algorithm == "Kalman Filter" else ParticleTracker()
    )

    if st.session_state.get("simulation") is None:
        settings_data = json.load(open(os.path.join(folder_prefix, "Seq1_settings.json")))
        cameras = [
            CameraParams(
                idx,
                np.array(
                    [
                        camera["position"]["x_m"],
                        camera["position"]["y_m"],
                        camera["position"]["z_m"],
                    ]  # camera coordinates
                ),
                camera["position"]["azimuth_deg"],
                camera["matrix"]["focal_length_mm"],
                camera["matrix"]["sensor_width_mm"],
                camera["matrix"]["sensor_height_mm"],
            )
            for idx, camera in settings_data["cameras"].items()
        ]
        object_diameter = settings_data["object"]["object_diameter_m"]

        # Initialize simulation
        print("Initialize simulation")
        sim = SimulationEnvironment(ground_truth_data, cameras, object_diameter)
        st.session_state["simulation"] = sim

    # Simulation controls
    if st.button("Step Simulation"):
        sim: SimulationEnvironment = st.session_state["simulation"]
        detector: DetectionAlgorithm = st.session_state["detector"]
        tracker: TrackingAlgorithm = st.session_state["tracker"]
        # Get camera views
        views = sim.get_camera_views()

        # Perform detection
        detections = []
        for view, camera in zip(views, sim.cameras):
            detection = detector.detect(view, camera)
            if detection is not None:
                detections.append(detection)
        # Update tracking
        st.write(detections)
        track = tracker.update(detections, sim.current_timestamp)  # 0.5s timestep

        # Get ground truth position
        true_pos = ground_truth_data.loc[sim.current_timestamp][
            ["X,m", "Y,m", "Z,m"]
        ].values

        # Calculate metrics
        metrics = calculate_metrics(track.position, true_pos)

        # Display results
        st.subheader("Camera Views")
        for i, view in enumerate(views):
            st.image(Image.fromarray(view), caption=f"Camera {i+1}", width=480)

        st.subheader("3D Trajectory")
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=[true_pos[0]],
                    y=[true_pos[1]],
                    z=[true_pos[2]],
                    mode="markers",
                    name="Ground Truth",
                    marker=dict(size=5, color="green"),
                ),
                go.Scatter3d(
                    x=[track.position[0]],
                    y=[track.position[1]],
                    z=[track.position[2]],
                    mode="markers",
                    name="Predicted",
                    marker=dict(size=5, color="red"),
                ),
            ]
        )
        st.plotly_chart(fig)

  
        st.subheader("Metrics")
        st.write(f"Euclidean Error: {metrics['euclidean_error']:.2f}m")
        st.write(f"X Error: {metrics['x_error']:.2f}m")
        st.write(f"Y Error: {metrics['y_error']:.2f}m")
        st.write(f"Z Error: {metrics['z_error']:.2f}m")
        st.write(f"Tracking Uncertainty: {track.uncertainty:.2f}")

        # Advance simulation
        sim.step()


if __name__ == "__main__":
    main()
