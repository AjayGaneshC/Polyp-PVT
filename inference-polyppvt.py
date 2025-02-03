import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from lib.pvt import PolypPVT

def process_frame(model, frame, input_size=352):
    # Preprocess frame
    frame_resized = cv2.resize(frame, (input_size, input_size))
    input_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    input_tensor = input_tensor.cuda()

    # Inference
    with torch.no_grad():
        P1, P2 = model(input_tensor)
        pred = F.upsample(P1+P2, size=(frame.shape[0], frame.shape[1]), 
                          mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu().numpy().squeeze()

    # Postprocess prediction
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    return pred_mask

def calculate_metrics(pred_mask, gt_mask):
    """Calculate IoU and Dice Score"""
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    iou = np.sum(intersection) / np.sum(union)
    dice = 2 * np.sum(intersection) / (np.sum(pred_mask) + np.sum(gt_mask))
    
    return iou, dice

def main():
    parser = argparse.ArgumentParser(description='Video Polyp Segmentation')
    parser.add_argument('--video_path', type=str, required=True, help='C:\\Users\\HP\\PolypPVT\\Polyp-PVT\\Test-video.avi')
    parser.add_argument('--output_path', type=str, default='output_video.avi', help='Path to output video')
    parser.add_argument('--weights', type=str, default='C:\\Users\\HP\\PolypPVT\\Polyp-PVT\\99PolypPVT.pth', help='Path to model weights')
    parser.add_argument('--gt_path', type=str, help='Path to ground truth masks (optional)')
    args = parser.parse_args()

    # Load model
    model = PolypPVT()
    model.load_state_dict(torch.load(args.weights))
    model.cuda()
    model.eval()

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_path, fourcc, 20.0, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    metrics = {'iou': [], 'dice': []}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        pred_mask = process_frame(model, frame)

        # Optional ground truth comparison
        if args.gt_path:
            # Assuming ground truth masks are stored sequentially
            gt_mask = cv2.imread(os.path.join(args.gt_path, f'frame_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.png'), 0)
            if gt_mask is not None:
                iou, dice = calculate_metrics(pred_mask, gt_mask)
                metrics['iou'].append(iou)
                metrics['dice'].append(dice)

        # Overlay mask on frame
        colored_mask = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
        result = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        out.write(result)
        cv2.imshow('Polyp Segmentation', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Print metrics
    if metrics['iou']:
        print(f"Average IoU: {np.mean(metrics['iou'])}")
        print(f"Average Dice Score: {np.mean(metrics['dice'])}")

if __name__ == '__main__':
    main()
