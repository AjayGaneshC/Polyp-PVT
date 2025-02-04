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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PolyDetector:
    def __init__(self, model, history_size=10, confidence_threshold=0.5, 
                 stability_threshold=0.6, min_detection_area=100):
        """
        Advanced Polyp Detection with Temporal Consistency
        
        Args:
            model (torch.nn.Module): Trained polyp detection model
            history_size (int): Number of frames to consider for moving average
            confidence_threshold (float): Initial confidence threshold
            stability_threshold (float): Threshold for consistent detection
            min_detection_area (int): Minimum area to consider a valid polyp
        """
        self.model = model
        self.history_size = history_size
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        self.min_detection_area = min_detection_area
        
        # Detection history
        self.bbox_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
    
    def find_bounding_box(self, mask):
        """
        Find the bounding box of the largest connected component in the mask
        
        Args:
            mask (numpy.ndarray): Binary mask of the segmentation
        
        Returns:
            tuple or None: (x, y, w, h) of the bounding box
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_detection_area]
        
        if not valid_contours:
            return None
        
        # Find the largest contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
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
        bbox = self.find_bounding_box(pred_mask)
        
        # Compute detection confidence
        detection_confidence = np.mean(pred) if bbox else 0
        
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

        return output_frame, stable_bbox is not None
    
    def _get_stable_bbox(self):
        """
        Compute stable bounding box based on detection history
        
        Returns:
            tuple or None: Stable bounding box
        """
        # If not enough history, return None
        if len(self.bbox_history) < self.history_size:
            return None
        
        # Check confidence stability
        mean_confidence = np.mean(self.confidence_history)
        if mean_confidence < self.stability_threshold:
            return None
        
        # Compute average bounding box
        valid_bboxes = [bbox for bbox in self.bbox_history if bbox is not None]
        if not valid_bboxes:
            return None
        
        # Compute average bbox
        bboxes_array = np.array(valid_bboxes)
        avg_bbox = np.mean(bboxes_array, axis=0).astype(int)
        
        return tuple(avg_bbox)

def main():
    parser = argparse.ArgumentParser(description='Video Polyp Segmentation')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_path', type=str, default='output_video.avi', help='Path to output video')
    parser.add_argument('--weights', type=str, default='99PolypPVT.pth', help='Path to model weights')
    parser.add_argument('--confidence', type=float, default=0.5, help='Initial confidence threshold')
    parser.add_argument('--history_size', type=int, default=10, help='Number of frames for temporal consistency')
    parser.add_argument('--stability_threshold', type=float, default=0.6, help='Threshold for stable detection')
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
        stability_threshold=args.stability_threshold
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
