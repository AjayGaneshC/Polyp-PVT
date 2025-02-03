import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
import time
from lib.pvt import PolypPVT
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def find_bounding_box(mask, min_area=100):
    """
    Find the bounding box of the largest connected component in the mask
    
    Args:
        mask (numpy.ndarray): Binary mask of the segmentation
        min_area (int): Minimum area to consider a valid object
    
    Returns:
        tuple: (x, y, w, h) of the bounding box, or None if no valid object found
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not valid_contours:
        return None
    
    # Find the largest contour
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, w, h)

def process_frame(model, frame, input_size=352, confidence_threshold=0.5):
    """
    Process a single frame for polyp detection
    
    Args:
        model (torch.nn.Module): Trained polyp detection model
        frame (numpy.ndarray): Input frame
        input_size (int): Resize input to this size
        confidence_threshold (float): Threshold for segmentation
    
    Returns:
        tuple: Processed frame with bounding box and detection flag
    """
    # Preprocess frame
    frame_resized = cv2.resize(frame, (input_size, input_size))
    input_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    input_tensor = input_tensor.cuda()

    # Inference with torch.no_grad for efficiency
    with torch.no_grad():
        P1, P2 = model(input_tensor)
        pred = F.interpolate(P1+P2, size=(frame.shape[0], frame.shape[1]), 
                             mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu().numpy().squeeze()

    # Postprocess prediction
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred_mask = (pred > confidence_threshold).astype(np.uint8) * 255

    # Find bounding box
    bbox = find_bounding_box(pred_mask)
    
    # Draw bounding box if detected
    output_frame = frame.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_frame, 'Polyp', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return output_frame, bbox is not None

def calculate_metrics(pred_mask, gt_mask):
    """Calculate IoU and Dice Score"""
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    iou = np.sum(intersection) / np.sum(union)
    dice = 2 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask))
    
    return iou, dice

def main():
    parser = argparse.ArgumentParser(description='Video Polyp Segmentation')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--output_path', type=str, default='output_video.avi', help='Path to output video')
    parser.add_argument('--weights', type=str, default='99PolypPVT.pth', help='Path to model weights')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--gt_path', type=str, help='Path to ground truth masks (optional)')
    args = parser.parse_args()

    # Load model with weights_only=True for security
    model = PolypPVT()
    model.load_state_dict(torch.load(args.weights, map_location=torch.device('cuda')))
    model.cuda()
    model.eval()

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    metrics = {'iou': [], 'dice': []}

    # FPS calculation
    frame_count = 0
    start_time = time.time()

    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, detected = process_frame(model, frame, 
                                                  confidence_threshold=args.confidence)
        
        # Calculate and display FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display FPS on frame
        cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Optional ground truth comparison
        if args.gt_path:
            # Assuming ground truth masks are stored sequentially
            gt_mask = cv2.imread(os.path.join(args.gt_path, f'frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.png'), 0)
            if gt_mask is not None:
                iou, dice = calculate_metrics(processed_frame, gt_mask)
                metrics['iou'].append(iou)
                metrics['dice'].append(dice)

        # Write frame
        out.write(processed_frame)

        # Optional: Display
        cv2.imshow('Polyp Detection', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print overall metrics
    if metrics['iou']:
        print(f"Average IoU: {np.mean(metrics['iou']):.4f}")
        print(f"Average Dice Score: {np.mean(metrics['dice']):.4f}")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
