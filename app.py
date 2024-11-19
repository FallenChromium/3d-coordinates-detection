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
    folder_prefix = "./data/step1/videoset1"
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
        sim = SimulationEnvironment(cameras, object_diameter, ground_truth_data, folder_prefix)
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
            detection = detector.detect(view["fused"], camera)
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
            st.image(Image.fromarray(view["rgb"]), caption=f"Camera {i+1}")
            st.image(Image.fromarray(view["ir"]), caption=f"Camera {i+1} IR")
            st.image(Image.fromarray(view["fused"]), caption=f"Camera {i+1} Fuse")

        st.subheader("3D Trajectory")
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
        data += [
            go.Scatter3d(
                x=[camera.position[0]],
                y=[camera.position[1]],
                z=[camera.position[2]],
                mode="markers",
                name=f"{camera.id}",
                marker=dict(size=5, color="blue")
            ) 
            for camera in sim.cameras
            ]
        fig = go.Figure(
            data=data,
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

@dataclass
class TrackingPoint:
    x: float
    y: float
    confidence: float
    frame_idx: int

class SAMPointTracker:
    def __init__(self):
        # Initialize SAM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        
        # Initialize CoTracker
        self.cotracker = CoTrackerPredictor(checkpoint="cotracker_v3.pth")
        
        # Tracking state
        self.tracked_points = []
        self.reference_embedding = None
        self.last_mask = None
        self.tracking_initialized = False
        
    def set_reference_point(self, frame: np.ndarray, point: np.ndarray):
        """Initialize tracking with a reference point"""
        self.predictor.set_image(frame)
        
        # Get SAM mask for the point
        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # Store the best mask
        best_mask_idx = scores.argmax()
        self.last_mask = masks[best_mask_idx]
        
        # Get embedding for the region
        features = self.predictor.get_image_embedding()
        mask_tensor = torch.from_numpy(self.last_mask).to(self.device)
        self.reference_embedding = features * mask_tensor[None, None, :, :]
        
        self.tracking_initialized = True
        self.tracked_points = [TrackingPoint(point[0], point[1], 1.0, 0)]
    
    def track(self, frame: np.ndarray, frame_idx: int) -> Optional[Tuple[np.ndarray, float]]:
        """Track object in new frame"""
        if not self.tracking_initialized:
            return None
        
        # Prepare frame for CoTracker
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        # Get last tracked point
        last_point = self.tracked_points[-1]
        query_points = torch.tensor([[last_point.x / frame.shape[1], 
                                    last_point.y / frame.shape[0]]])
        
        # Track point with CoTracker
        tracks = self.cotracker.track_points(
            frames=frame_tensor,
            points=query_points,
            num_points=1
        )
        
        # Get predicted position
        pred_point = tracks[0, -1] * torch.tensor([frame.shape[1], frame.shape[0]])
        pred_point = pred_point.cpu().numpy()
        
        # Verify prediction with SAM
        self.predictor.set_image(frame)
        input_point = np.array([[pred_point[0], pred_point[1]]])
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=np.array([1]),
            multimask_output=True
        )
        
        # Compare mask with reference
        best_mask_idx = scores.argmax()
        current_mask = masks[best_mask_idx]
        current_features = self.predictor.get_image_embedding()
        mask_tensor = torch.from_numpy(current_mask).to(self.device)
        current_embedding = current_features * mask_tensor[None, None, :, :]
        
        # Calculate similarity score
        similarity = F.cosine_similarity(
            self.reference_embedding.flatten(),
            current_embedding.flatten(),
            dim=0
        ).item()
        
        # Update tracking state
        self.last_mask = current_mask
        self.tracked_points.append(
            TrackingPoint(pred_point[0], pred_point[1], similarity, frame_idx)
        )
        
        return pred_point, similarity

class InteractiveTracker:
    def __init__(self):
        self.tracker = SAMPointTracker()
        self.selected_point = None
    
    def initialize_ui(self):
        """Create Streamlit UI for interactive tracking"""
        st.sidebar.header("Interactive Tracking Controls")
        
        # Add point selection mode
        self.point_mode = st.sidebar.radio(
            "Point Selection Mode",
            ["Click", "Brush"]
        )
        
        # Add confidence threshold
        self.conf_threshold = st.sidebar.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.5
        )
        
        # Add visualization options
        self.show_trajectory = st.sidebar.checkbox("Show Trajectory", True)
        self.show_mask = st.sidebar.checkbox("Show Segmentation Mask", True)
    
    def handle_interaction(self, frame: np.ndarray):
        """Handle user interaction with the frame"""
        # Convert frame to PIL Image for Streamlit
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if self.point_mode == "Click":
            # Use Streamlit's image click functionality
            clicked = st.image(pil_image, use_column_width=True)
            if clicked:
                self.selected_point = np.array(st.session_state.last_clicked_pos)
                self.tracker.set_reference_point(frame, self.selected_point)
        
        else:  # Brush mode
            # Create canvas for brushing
            canvas = st.image(pil_image, use_column_width=True)
            brush_size = st.sidebar.slider("Brush Size", 1, 50, 10)
            
            if canvas.is_drawing:
                # Get brush strokes and find center
                strokes = canvas.json_data["strokes"]
                if strokes:
                    points = np.array(strokes[-1]["points"])
                    center = points.mean(axis=0)
                    self.selected_point = center
                    self.tracker.set_reference_point(frame, self.selected_point)
    
    def track_and_visualize(self, frame: np.ndarray, frame_idx: int):
        """Track object and create visualization"""
        if self.selected_point is None:
            return None
        
        # Track object
        result = self.tracker.track(frame, frame_idx)
        if result is None:
            return None
        
        position, confidence = result
        
        # Skip if confidence is too low
        if confidence < self.conf_threshold:
            return None
        
        # Create visualization
        vis_frame = frame.copy()
        
        if self.show_mask and self.tracker.last_mask is not None:
            # Overlay segmentation mask
            mask_overlay = np.zeros_like(vis_frame)
            mask_overlay[self.tracker.last_mask] = [0, 255, 0]
            vis_frame = cv2.addWeighted(vis_frame, 0.7, mask_overlay, 0.3, 0)
        
        if self.show_trajectory:
            # Draw trajectory
            points = [(p.x, p.y) for p in self.tracker.tracked_points]
            if len(points) > 1:
                points = np.array(points, np.int32)
                cv2.polylines(vis_frame, [points], False, (0, 255, 255), 2)
        
        # Draw current position
        cv2.circle(vis_frame, 
                  (int(position[0]), int(position[1])), 
                  5, (0, 0, 255), -1)
        
        # Add confidence score
        cv2.putText(vis_frame, 
                   f"Conf: {confidence:.2f}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 255, 0), 
                   2)
        
        return vis_frame, position, confidence

# Usage example:
"""
# Automatic tracking
tracker = SAMPointTracker()
# Initialize with first frame and point
tracker.set_reference_point(first_frame, initial_point)

# Track in subsequent frames
for frame in frames:
    position, confidence = tracker.track(frame, frame_idx)
    
# Interactive tracking
interactive_tracker = InteractiveTracker()
interactive_tracker.initialize_ui()

# In your main loop
frame = get_next_frame()
interactive_tracker.handle_interaction(frame)
result = interactive_tracker.track_and_visualize(frame, frame_idx)
if result:
    vis_frame, position, confidence = result
    # Display or process results
"""