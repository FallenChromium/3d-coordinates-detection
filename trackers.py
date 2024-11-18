from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
import cv2
import numpy as np

from detectors import Detection


@dataclass
class Track3D:
    position: np.ndarray
    velocity: np.ndarray
    uncertainty: float
    timestamp: float


class TrackingAlgorithm(ABC):
    @abstractmethod
    def update(self, detections: List[Detection], timestamp: float) -> Track3D:
        pass

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

