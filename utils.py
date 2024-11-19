import cv2
import numpy as np


def fuse_rgb_ir(rgb_frame: np.ndarray, ir_frame: np.ndarray, 
                    alpha: float = 0.7) -> np.ndarray:
        """Fuse RGB and IR frames with adaptive weighting"""
        # Normalize IR frame to 0-1 range
        ir_norm = cv2.normalize(ir_frame, None, 0, 1, cv2.NORM_MINMAX)
        
        # Convert IR to 3-channel
        # ir_3ch = cv2.cvtColor(ir_norm, cv2.COLOR_GRAY2BGR)
        # actually, this video is gbr as well
        ir_3ch = ir_norm   
        
        # Normalize RGB frame
        rgb_norm = rgb_frame.astype(float) / 255.0
        
        # Calculate local contrast in both frames
        rgb_contrast = cv2.Laplacian(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY), 
                                   cv2.CV_64F).var()
        ir_contrast = cv2.Laplacian(ir_frame, cv2.CV_64F).var()
        
        # Adjust alpha based on contrast ratio
        contrast_ratio = rgb_contrast / (ir_contrast + 1e-6)
        adaptive_alpha = alpha * contrast_ratio / (1 + contrast_ratio)
        
        # Weighted fusion
        fused = cv2.addWeighted(rgb_norm, adaptive_alpha, ir_3ch, 1-adaptive_alpha, 0, dtype=cv2.CV_32F)
        return (fused * 255).astype(np.uint8)