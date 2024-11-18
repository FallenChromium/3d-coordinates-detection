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
import plotly.graph_objects as go
import json
from PIL import Image

from detectors import BlobDetector, DetectionAlgorithm
from simulation import CameraParams, SimulationEnvironment
from trackers import KalmanTracker, ParticleTracker, TrackingAlgorithm


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
        settings_data = json.load(
            open(os.path.join(folder_prefix, "Seq1_settings.json"))
        )
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
