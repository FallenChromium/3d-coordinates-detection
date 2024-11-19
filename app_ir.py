import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import plotly.graph_objects as go

@dataclass
class TrackingSlice:
    coords: Tuple[int, int, int, int]  # x1, y1, x2, y2
    scale_factor: float
    original_shape: Tuple[int, int]

class CoTrackerWithSAH:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'):
        self.device = device
        # Load CoTracker model from torch hub
        self.model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        self.model.eval()
        
        # Initialize tracking state
        self.tracked_points = []
        self.current_slice: Optional[TrackingSlice] = None
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_slice(self, frame: np.ndarray, center: Tuple[int, int], size: int = 256) -> TrackingSlice:
        """Create a slice around the object (SAHI)"""
        h, w = frame.shape[:2]
        x, y = center
        
        # Calculate slice coordinates
        half_size = size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(w, x + half_size)
        y2 = min(h, y + half_size)
        
        # Calculate scale factor for upsampling
        scale_factor = 4.0  # Can be adjusted based on object size
        
        return TrackingSlice((x1, y1, x2, y2), scale_factor, (h, w))
    
    def process_slice(self, frame: np.ndarray, slice_info: TrackingSlice) -> torch.Tensor:
        """Process image slice for tracking"""
        x1, y1, x2, y2 = slice_info.coords
        slice_img = frame[y1:y2, x1:x2]
        
        # Resize slice for better small object detection
        h, w = slice_img.shape[:2]
        new_h, new_w = int(h * slice_info.scale_factor), int(w * slice_info.scale_factor)
        slice_img = cv2.resize(slice_img, (new_w, new_h))
        
        # Convert to tensor
        return self.transform(Image.fromarray(slice_img)).unsqueeze(0)
    
    def convert_slice_coords(self, points: torch.Tensor, slice_info: TrackingSlice) -> np.ndarray:
        """Convert coordinates from slice space to original frame space"""
        x1, y1, _, _ = slice_info.coords
        scale = slice_info.scale_factor
        
        points = points.cpu().numpy()
        points[:, 0] = points[:, 0] / scale + x1
        points[:, 1] = points[:, 1] / scale + y1
        return points
    
    def track(self, rgb_frames: List[np.ndarray], ir_frames: List[np.ndarray], 
              points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Track points through frame sequence using SAH"""
        with torch.no_grad():
            # Fuse RGB and IR frames
            fused_frames = [self.fuse_rgb_ir(rgb, ir) 
                          for rgb, ir in zip(rgb_frames, ir_frames)]
            
            # Find object center from initial points
            center = points.mean(axis=0).astype(int)
            self.current_slice = self.prepare_slice(fused_frames[0], center)
            
            # Process slices
            sliced_frames = [self.process_slice(frame, self.current_slice) 
                           for frame in fused_frames]
            video_frames = torch.cat(sliced_frames, dim=0).to(self.device)
            
            # Convert points to slice coordinates
            x1, y1, _, _ = self.current_slice.coords
            scale = self.current_slice.scale_factor
            slice_points = torch.tensor(
                [(p[0] - x1) * scale, (p[1] - y1) * scale] 
                for p in points
            ).to(self.device)
            
            # Track points
            tracks, visibility = self.model(
                video_frames, 
                slice_points.unsqueeze(0),
                track_all_frames=True
            )
            
            # Convert back to original coordinate space
            tracks = self.convert_slice_coords(tracks[0], self.current_slice)
            visibility = visibility[0].cpu().numpy()
            
            return tracks, visibility

class MultiCameraTracker:
    def __init__(self):
        self.cotracker = CoTrackerWithSAH()
        self.selected_points = {}  # Store points for each camera
    
    def add_points(self, camera_id: int, frame_idx: int, points: np.ndarray):
        """Add tracking points for a specific camera and frame"""
        if camera_id not in self.selected_points:
            self.selected_points[camera_id] = {}
        self.selected_points[camera_id][frame_idx] = points
    
    def track_all_cameras(self, rgb_sequences: Dict[int, List[np.ndarray]], 
                         ir_sequences: Dict[int, List[np.ndarray]]) -> Dict[int, np.ndarray]:
        """Track objects across all cameras"""
        results = {}
        
        for camera_id in rgb_sequences.keys():
            if camera_id in self.selected_points:
                # Get first frame with points
                first_frame = min(self.selected_points[camera_id].keys())
                points = self.selected_points[camera_id][first_frame]
                
                # Track points
                tracks, visibility = self.cotracker.track(
                    rgb_sequences[camera_id], 
                    ir_sequences[camera_id], 
                    points
                )
                
                results[camera_id] = {
                    'tracks': tracks,
                    'visibility': visibility
                }
        
        return results

@dataclass
class ViewportPoint:
    x: int
    y: int
    camera_id: int
    frame_idx: int

def main():
    # wide with sidebar settings
    st.set_page_config(layout="wide")
    sidebar = st.sidebar.title("Settings")
    with sidebar:
        st.title("Multi-Camera Object Tracking with CoTracker")
        # Video loading
        video_path = st.text_input("Videos Directory", value="./data/step1/videoset1")
        
        if not video_path:
            return
    
    # Load videos
    # get sequence number from the last number of the last folder in the video_path
    if 'rgb_captures' not in st.session_state or 'ir_captures' not in st.session_state:
        sequence_number = video_path[-1]
        rgb_captures = {i: cv2.VideoCapture(f"{video_path}/Seq{sequence_number}_camera{i}.mov") for i in range(1, 4)}
        ir_captures = {i: cv2.VideoCapture(f"{video_path}/Seq{sequence_number}_camera{i}.mov") for i in range(1, 4)}
        st.session_state['rgb_captures'] = rgb_captures
        st.session_state['ir_captures'] = ir_captures

    # Initialize tracker
    tracker = MultiCameraTracker()
    
    # Frame navigation
    frame_idx = st.slider("Frame", 0, 1000, 0)  # Adjust max value based on video length
    
    # Display frames and handle point selection
    st.session_state["viewport_points"] = []
    rgb_captures = st.session_state["rgb_captures"]
    ir_captures = st.session_state["ir_captures"]

    for i in range(1, 4):
        # Read frames
        rgb_captures[i].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ir_captures[i].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret_rgb, rgb_frame = rgb_captures[i].read()
        ret_ir, ir_frame = ir_captures[i].read()
        
        if ret_rgb and ret_ir:
            # Display fused frame
            fused_frame = tracker.cotracker.fuse_rgb_ir(rgb_frame, ir_frame)
            st.image(fused_frame, channels="BGR", use_column_width=True)
            
            # Point selection
            if st.button(f"Select Points (Camera {i})"):
                points = []
                st.write("Click on the image to select points. Press 'Done' when finished.")
                # Note: This is a simplified version. In practice, you'd need to implement
                # a proper point selection mechanism using Streamlit's interactive features
                # or a custom component

    # Tracking button
    if st.button("Track Objects"):
        # Collect frame sequences
        rgb_sequences = {}
        ir_sequences = {}
        
        for i in range(1, 4):
            rgb_sequences[i] = []
            ir_sequences[i] = []
            
            rgb_captures[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ir_captures[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while True:
                ret_rgb, rgb_frame = rgb_captures[i].read()
                ret_ir, ir_frame = ir_captures[i].read()
                
                if not ret_rgb or not ret_ir:
                    break
                
                rgb_sequences[i].append(rgb_frame)
                ir_sequences[i].append(ir_frame)
        
        # Track objects
        results = tracker.track_all_cameras(rgb_sequences, ir_sequences)
        
        # Visualize results
        for camera_id, result in results.items():
            tracks = result['tracks']
            visibility = result['visibility']
            
            # Create trajectory plot
            fig = go.Figure()
            
            # Add tracks
            for track_idx in range(len(tracks)):
                visible_mask = visibility[track_idx] > 0.5
                fig.add_trace(go.Scatter(
                    x=tracks[track_idx, visible_mask, 0],
                    y=tracks[track_idx, visible_mask, 1],
                    mode='lines+markers',
                    name=f'Point {track_idx + 1}'
                ))
            
            fig.update_layout(title=f'Tracking Results - Camera {camera_id}')
            st.plotly_chart(fig)
    
    # Clean up
    for cap in rgb_captures.values():
        cap.release()
    for cap in ir_captures.values():
        cap.release()

if __name__ == "__main__":
    main()