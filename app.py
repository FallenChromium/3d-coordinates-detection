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
from streamlit_image_coordinates import streamlit_image_coordinates

from detectors import BlobDetector, DetectionAlgorithm
from simulation import CameraParams, SimulationEnvironment
from trackers import KalmanTracker, ParticleTracker, TrackingAlgorithm
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
# Evaluation metrics
def calculate_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    error = np.linalg.norm(predicted - ground_truth)
    return {
        "euclidean_error": error,
        "x_error": abs(predicted[0] - ground_truth[0]),
        "y_error": abs(predicted[1] - ground_truth[1]),
        "z_error": abs(predicted[2] - ground_truth[2]),
    }

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))




# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("3D Object Tracking Simulation")
    # streamlit tabs for each Seq of /data/videoset{1..8}/Seq[Number]_settings
    # Load data
    folder_prefix = "./data/step1/videoset0"
    ground_truth_data = pd.read_csv(
        os.path.join(folder_prefix, "Seq0_settings.csv"), sep=",", header=0, index_col=0
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
        # seq1 points
        # st.session_state.update({"viewport_points": [{"x":503,"y":643,"width":1920,"height":1080,"unix_time":1732058105667,"camera":"camera1","frame_idx":183},{"x":491,"y":684,"width":1920,"height":1080,"unix_time":1732058114662,"camera":"camera2","frame_idx":183},{"x":279,"y":743,"width":1920,"height":1080,"unix_time":1732058128106,"camera":"camera3","frame_idx":183},{"x":451,"y":892,"width":1920,"height":1080,"unix_time":1732072682787,"camera":"camera2","frame_idx":16},{"x":238,"y":884,"width":1920,"height":1080,"unix_time":1732072703541,"camera":"camera1","frame_idx":18},{"x":237,"y":883,"width":1920,"height":1080,"unix_time":1732072715264,"camera":"camera1","frame_idx":20},{"x":235,"y":879,"width":1920,"height":1080,"unix_time":1732072727636,"camera":"camera1","frame_idx":22},{"x":499,"y":722,"width":1920,"height":1080,"unix_time":1732072754469,"camera":"camera3","frame_idx":135}]})
        # st.session_state.update({"viewport_points": []})
        st.session_state.update({"viewport_points": [{"x":498,"y":961,"width":1920,"height":1080,"unix_time":1732077535015,"camera":"camera1","frame_idx":"146"},{"x":885,"y":855,"width":1920,"height":1080,"unix_time":1732077588342,"camera":"camera3","frame_idx":"185"},{"x":523,"y":671,"width":1920,"height":1080,"unix_time":1732077610761,"camera":"camera2","frame_idx":"185"},{"x":1061,"y":572,"width":1920,"height":1080,"unix_time":1732077697102,"camera":"camera1","frame_idx":"500"},{"x":1044,"y":484,"width":1920,"height":1080,"unix_time":1732077712486,"camera":"camera2","frame_idx":"500"},{"x":1201,"y":880,"width":1920,"height":1080,"unix_time":1732077732209,"camera":"camera1","frame_idx":"1000"},{"x":1510,"y":656,"width":1920,"height":1080,"unix_time":1732077774707,"camera":"camera2","frame_idx":"1200"},{"x":1533,"y":759,"width":1920,"height":1080,"unix_time":1732077794075,"camera":"camera1","frame_idx":"1182"},{"x":1285,"y":812,"width":1920,"height":1080,"unix_time":1732077813991,"camera":"camera3","frame_idx":"1425"}]})
        settings_data = json.load(
            open(os.path.join(folder_prefix, "Seq0_settings.json"))
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

    frame_idx = st.slider("Frame", 0, 7000, 0)
    frame_idx = st.text_input("Frame", value=frame_idx)
# Simulation controls
    sim: SimulationEnvironment = st.session_state["simulation"]
    detector: DetectionAlgorithm = st.session_state["detector"]
    tracker: TrackingAlgorithm = st.session_state["tracker"]
    # Get camera views
    views = sim.get_camera_views(frame_idx)
    # Perform detection
    # detections = []
    # for view, camera in zip(views, sim.cameras):
    #     detection = detector.detect(view["fused"], camera)
    #     if detection is not None:
    #         detections.append(detection)
    # # Update tracking
    # st.write(detections)
    # track = tracker.update(detections, sim.current_timestamp)  # 0.5s timestep

    # Get ground truth position
    true_pos = ground_truth_data.loc[sim.current_timestamp][
        ["X,m", "Y,m", "Z,m"]
    ].values

    # Calculate metrics
    # metrics = calculate_metrics(track.position, true_pos)

    # Display results
    st.subheader("Camera Views")
    st.caption("Left: Camera 1 View, Middle: Camera 2 View, Right: Camera 3 View")

    point1 = streamlit_image_coordinates(views[0]["fused"])
    st.write(point1)
    if point1 is not None:
        if st.button("Add point from camera1 to query"):
            point1["camera"] = "camera1"
            point1["frame_idx"] = frame_idx
            st.session_state.get("viewport_points").append(point1)

    point2 = streamlit_image_coordinates(views[1]["fused"])
    st.write(point2)
    if point2 is not None:
        if st.button("Add point from camera2 to query"):
            point2["camera"] = "camera2"
            point2["frame_idx"] = frame_idx
            st.session_state.get("viewport_points").append(point2)

    point3 = streamlit_image_coordinates(views[2]["fused"])
    st.write(point3)
    if point3 is not None:
        if st.button("Add point from camera3 to query"):
            point3["camera"] = "camera3"
            point3["frame_idx"] = frame_idx
            st.session_state.get("viewport_points").append(point3)

    st.subheader("CoTracker query points")
    st.write(st.session_state.get("viewport_points"))
    if st.button("Clear query points"):
        st.session_state.update({"viewport_points": []})
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
            # go.Scatter3d(
            #     x=[track.position[0]],
            #     y=[track.position[1]],
            #     z=[track.position[2]],
            #     mode="markers",
            #     name="Predicted",
            #     marker=dict(size=5, color="red"),
            # ),
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

    # st.subheader("Metrics")
    # st.write(f"Euclidean Error: {metrics['euclidean_error']:.2f}m")
    # st.write(f"X Error: {metrics['x_error']:.2f}m")
    # st.write(f"Y Error: {metrics['y_error']:.2f}m")
    # st.write(f"Z Error: {metrics['z_error']:.2f}m")
    # st.write(f"Tracking Uncertainty: {track.uncertainty:.2f}")

    # Advance simulation
    # sim.step()
    if st.button("Track"):

        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16): # only works on modern gayvideo gpus
            query_points = st.session_state.get("viewport_points")
            ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
            for camera_idx in range(1,4):
                st.session_state[f"video_segments{camera_idx}"] = {}
                # fused_filename is Seq0_camera1FL.mp4
                # add another loop to iterate over _part1, _part2, _part3, _part4 .mp4 files
                for part in ["_part1", "_part2", "_part3", "_part4"]:
                    part_multiplier = int(part[-1])
                    # need to get rid of .mp4 extension from fused filename and insert the _part
                    fused_filename = sim.fused_filename(f"camera{camera_idx}")
                    fused_filename = fused_filename[:-4] + part + ".mp4" 
                    predictor = start_predictor()
                    print("predictor init")

                    state = init_predictor_state(predictor, sim.fused_filename(f"camera{camera_idx}"))
                    print("predictor state for camera", camera_idx)
                    points = [[point["x"], point["y"], int(point["frame_idx"])-(part_multiplier-1)*20*30] for point in query_points if point["camera"] == f"camera{camera_idx}"]

                    for frame_i in [p[2] for p in points]:
                        if len(points) > 0:
                            add_keypoint(predictor, state, frame_i, ann_obj_id, [[p[0], p[1]] for p in points if p[2] == frame_i])
                            segments: dict = st.session_state[f"video_segments{camera_idx}"]
                            fragments = propagate_on_video(predictor, state, frame_i, f"camera{camera_idx}")
                            fragments = {str(int(k) + (part_multiplier-1)*20*30): v for k, v in fragments.items()}
                            segments.update(fragments)
                            st.session_state[f"video_segments{camera_idx}"] = segments

            
        
    video_segments = st.session_state.get("video_segments1")
    if video_segments is not None:
        fig, ax = plt.subplots()
        st.write(f"frame {frame_idx} (tracked)")
        ax.imshow(views[0]["rgb"])
        for out_obj_id, out_mask in video_segments[frame_idx].items():
            show_mask(out_mask, ax, obj_id=out_obj_id, random_color=True)
        st.pyplot(fig)

        for camera_idx in range(1,4):
            tracked_frames = {}
            video_segments = st.session_state.get(f"video_segments{camera_idx}")
            for frame, out_masks in video_segments.items():
                out_mask = out_masks[ann_obj_id]
                if np.sum(out_mask) > 0:
                    true_coords = [(y, x) for y in range(out_mask.shape[1]) for x in range(out_mask.shape[2]) if out_mask[0, y, x]]

                    # Step 1: Find the leftmost True value (smallest x value)
                    leftmost = min(true_coords, key=lambda coord: coord[1])[0]

                    # Step 2: Find the rightmost True value (largest x value)
                    rightmost = max(true_coords, key=lambda coord: coord[1])[0]

                    # Step 3: Find the topmost True value (smallest y value)
                    topmost = min(true_coords, key=lambda coord: coord[0])[1]

                    # Step 4: Find the bottommost True value (largest y value)
                    bottommost = max(true_coords, key=lambda coord: coord[0])[1]
                    # mean center
                    center_x = (leftmost + rightmost) / 2
                    center_y = (topmost + bottommost) / 2
                    st.write(f"center: ({center_x}, {center_y})")
                    tracked_frames[frame] = {
                        "center": (center_x, center_y),
                        "leftmost": leftmost,
                        "rightmost": rightmost,
                        "topmost": topmost,
                        "bottommost": bottommost
                    }
            json.dump(st.session_state.viewport_points, open("viewport_points.json", "w"))
            json.dump(tracked_frames, open(f"tracked_frames{camera_idx}.json", "w"))

@st.cache_data(ttl=3600)
def propagate_on_video(_predictor, _state, frame_idx, camera_id):
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in _predictor.propagate_in_video(_state):
        video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
    return video_segments

@st.cache_resource(ttl=3600)
def add_keypoint(_predictor, _state, ann_frame_idx, ann_obj_id, keypoints):
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1 for _ in keypoints], np.int32)
    _, out_obj_ids, out_mask_logits = _predictor.add_new_points_or_box(
                inference_state=_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=keypoints,
                labels=labels,
)

@st.cache_resource(ttl=3600)
def init_predictor_state(_predictor, filename: str):
    state1 = _predictor.init_state(filename)
    return state1

@st.cache_resource(ttl=3600)
def start_predictor():
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-small")
    return predictor
                


if __name__ == "__main__":
    main()
