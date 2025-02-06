import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
import time
from lib.pvt import PolypPVT
import warnings
from collections import deque
import skimage.feature
import skimage.color

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PolyDetector:
    def __init__(self, model, history_size=5, confidence_threshold=0.3, 
                 stability_threshold=0.4, min_detection_area=50, 
                 max_detection_area_ratio=0.3, aspect_ratio_range=(0.5, 2.0)):
        """
        Advanced Polyp Detection with Multi-Stage Filtering
        
        Args:
            model (torch.nn.Module): Trained polyp detection model
            history_size (int): Number of frames to consider for moving average
            confidence_threshold (float): Initial confidence threshold
            stability_threshold (float): Threshold for consistent detection
            min_detection_area (int): Minimum area to consider a valid polyp
            max_detection_area_ratio (float): Maximum area ratio relative to frame
            aspect_ratio_range (tuple): Valid aspect ratio range for polyps
        """
        self.model = model
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.min_detection_area = min_detection_area
        self.max_detection_area_ratio = max_detection_area_ratio
        self.aspect_ratio_range = aspect_ratio_range
        
        # Detection history
        self.bbox_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # Debugging flag
        self.debug = True
    
    def color_texture_analysis(self, frame, mask):
        """
        Advanced color and texture analysis for polyp detection
        
        Args:
            frame (numpy.ndarray): Original frame
            mask (numpy.ndarray): Binary mask of potential polyp region
        
        Returns:
            bool: Whether the region is likely a polyp
        """
        # Extract region of interest
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Color-based features
        l, a, b = cv2.split(lab)
        
        # Texture analysis using Local Binary Patterns
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lbp = skimage.feature.local_binary_pattern(gray_roi, P=8, R=1, method='uniform')
        
        # Compute color and texture statistics
        color_std = np.std([a, b])  # Color variation
        texture_entropy = skimage.feature.shannon_entropy(lbp)
        
        # Polyp-specific criteria
        color_criteria = 0.3 < color_std < 0.8  # Moderate color variation
        texture_criteria = 2.0 < texture_entropy < 4.0  # Specific texture complexity
        
        if self.debug:
            print(f"Color Std: {color_std}, Texture Entropy: {texture_entropy}")
            print(f"Color Match: {color_criteria}, Texture Match: {texture_criteria}")
        
        return color_criteria and texture_criteria
    
    def validate_polyp_shape(self, contour, frame_shape, frame):
        """
        Advanced shape validation for potential polyps
        
        Args:
            contour (numpy.ndarray): Detected contour
            frame_shape (tuple): Shape of the input frame
            frame (numpy.ndarray): Original frame
        
        Returns:
            bool: Whether the contour meets polyp shape criteria
        """
        # Calculate contour area and frame area
        contour_area = cv2.contourArea(contour)
        frame_area = frame_shape[0] * frame_shape[1]
        
        # Area-based filtering
        if contour_area < self.min_detection_area:
            if self.debug:
                print(f"Rejected: Area too small ({contour_area})")
            return False
        
        # Prevent oversized detections
        if contour_area > (frame_area * self.max_detection_area_ratio):
            if self.debug:
                print(f"Rejected: Area too large ({contour_area})")
            return False
        
        # Compute bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create mask for the contour
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Color and texture analysis
        if not self.color_texture_analysis(frame, mask):
            if self.debug:
                print("Rejected: Failed color/texture analysis")
            return False
        
        # Aspect ratio check
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio < self.aspect_ratio_range[0] or aspect_ratio > self.aspect_ratio_range[1]:
            if self.debug:
                print(f"Rejected: Irregular aspect ratio ({aspect_ratio})")
            return False
        
        # Convexity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity_defect = 1 - (contour_area / hull_area)
        if convexity_defect > 0.3:  # Allow some irregularity
            if self.debug:
                print(f"Rejected: Irregular shape (convexity defect: {convexity_defect})")
            return False
        
        return True
    
    def find_bounding_box(self, mask, frame):
        """
        Find the bounding box of the largest valid polyp-like component
        
        Args:
            mask (numpy.ndarray): Binary mask of the segmentation
            frame (numpy.ndarray): Original frame
        
        Returns:
            tuple or None: (x, y, w, h) of the bounding box
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and validate contours
        valid_contours = [
            cnt for cnt in contours 
            if self.validate_polyp_shape(cnt, frame.shape, frame)
        ]
        
        if not valid_contours:
            if self.debug:
                print("No valid polyp-like contours found")
            return None
        
        # Find the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if self.debug:
            print(f"Validated bounding box: x={x}, y={y}, w={w}, h={h}")
        
        return (x, y, w, h)
    
    def process_frame(self, frame, input_size=352):
        """
        Process a single frame for polyp detection with temporal consistency
        
        Args:
            frame (numpy.ndarray): Input frame
            input_size (int): Resize input to this size
        
        Returns:
            tuple: Processed frame and detection status
        """
        # Preprocess frame
        frame_resized = cv2.resize(frame, (input_size, input_size))
        input_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        input_tensor = input_tensor.cuda()

        # Inference
        with torch.no_grad():
            P1, P2 = self.model(input_tensor)
            pred = F.interpolate(P1+P2, size=(frame.shape[0], frame.shape[1]), 
                                 mode='bilinear', align_corners=False)
            pred = pred.sigmoid().data.cpu().numpy().squeeze()

        # Postprocess prediction
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_mask = (pred > self.confidence_threshold).astype(np.uint8) * 255

        # Find bounding box
        bbox = self.find_bounding_box(pred_mask, frame)
        
        # Compute detection confidence
        detection_confidence = np.mean(pred) if bbox else 0
        
        if self.debug:
            print(f"Detection confidence: {detection_confidence}")
        
        # Update history
        self.bbox_history.append(bbox)
        self.confidence_history.append(detection_confidence)
        
        # Temporal consistency check
        stable_bbox = self._get_stable_bbox()
        
        # Draw bounding box if stably detected
        output_frame = frame.copy()
        if stable_bbox:
            x, y, w, h = stable_bbox
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output_frame, 'Polyp', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif bbox:  # If no stable bbox but a bbox was found in this frame
            x, y, w, h = bbox
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(output_frame, 'Potential Polyp', (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return output_frame, stable_bbox is not None
    
    def _get_stable_bbox(self):
        """
        Compute stable bounding box based on detection history
        
        Returns:
            tuple or None: Stable bounding box
        """
        # If not enough history, return None
        if len(self.bbox_history) < self.history_size:
            if self.debug:
                print(f"Not enough history: {len(self.bbox_history)}")
            return None
        
        # Check confidence stability
        mean_confidence = np.mean(self.confidence_history)
        if mean_confidence < self.stability_threshold:
            if self.debug:
                print(f"Mean confidence too low: {mean_confidence}")
            return None
        
        # Compute average bounding box
        valid_bboxes = [bbox for bbox in self.bbox_history if bbox is not None]
        if not valid_bboxes:
            if self.debug:
                print("No valid bboxes in history")
            return None
        
        # Compute average bbox
        bboxes_array = np.array(valid_bboxes)
        avg_bbox = np.mean(bboxes_array, axis=0).astype(int)
        
        if self.debug:
            print(f"Stable bbox found: {avg_bbox}")
        
        return tuple(avg_bbox)

def main():
    parser = argparse.ArgumentParser(description='Video Polyp Segmentation')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_path', type=str, default='output_video.avi', help='Path to output video')
    parser.add_argument('--weights', type=str, default='99PolypPVT.pth', help='Path to model weights')
    parser.add_argument('--confidence', type=float, default=0.3, help='Initial confidence threshold')
    parser.add_argument('--history_size', type=int, default=5, help='Number of frames for temporal consistency')
    parser.add_argument('--stability_threshold', type=float, default=0.4, help='Threshold for stable detection')
    parser.add_argument('--min_area', type=int, default=50, help='Minimum detection area')
    parser.add_argument('--max_area_ratio', type=float, default=0.3, help='Maximum area ratio relative to frame')
    args = parser.parse_args()

    # Load model
    model = PolypPVT()
    model.load_state_dict(torch.load(args.weights, map_location=torch.device('cuda')))
    model.cuda()
    model.eval()

    # Initialize polyp detector
    poly_detector = PolyDetector(
        model, 
        history_size=args.history_size, 
        confidence_threshold=args.confidence,
        stability_threshold=args.stability_threshold,
        min_detection_area=args.min_area,
        max_detection_area_ratio=args.max_area_ratio
    )

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    # FPS calculation
    frame_count = 0
    start_time = time.time()

    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, detected = poly_detector.process_frame(frame)
        
        # Calculate and display FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display FPS on frame
        cv2.putText(processed_frame, f'FPS: {current_fps:.2f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame
        out.write(processed_frame)

        # Optional: Display
        cv2.imshow('Polyp Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
