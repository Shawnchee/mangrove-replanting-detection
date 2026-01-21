"""
RTMP Stream Inference for Mangrove Detection
Reads RTMP stream from DJI Mini 2, applies YOLOv5 inference, and outputs annotated video.
"""

import cv2
import numpy as np
import torch
import time
from datetime import datetime
import argparse
import sys
from pathlib import Path

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.plots import Annotator, colors
from yolov5.models.common import DetectMultiBackend


class RTMPStreamInference:
    def __init__(self, 
                 rtmp_url='rtmp://localhost:1935/drone/dji',
                 model_path='models/best.pt',
                 output_rtmp=None,
                 conf_thres=0.25,
                 iou_thres=0.45,
                 imgsz=640,
                 device='cuda'):
        """
        Initialize RTMP Stream Inference
        
        Args:
            rtmp_url: RTMP URL to read stream from
            model_path: Path to YOLOv5 model (.pt file)
            output_rtmp: Optional RTMP URL to stream annotated video to
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
            imgsz: Input image size
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.rtmp_url = rtmp_url
        self.model_path = model_path
        self.output_rtmp = output_rtmp
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = device
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = DetectMultiBackend(model_path, device=device)
        self.stride = self.model.stride
        self.names = self.model.names
        print(f"Model loaded successfully. Classes: {self.names}")
        
        # Initialize video capture
        print(f"Connecting to RTMP stream: {rtmp_url}")
        self.cap = cv2.VideoCapture(rtmp_url)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open RTMP stream: {rtmp_url}")
        
        # Get stream properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Stream connected: {self.width}x{self.height} @ {self.fps}fps")
        
        # Initialize output stream if specified
        self.out = None
        if output_rtmp:
            self.init_output_stream(output_rtmp)
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
    def init_output_stream(self, output_rtmp):
        """Initialize RTMP output stream using FFmpeg"""
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-f', 'flv',
            output_rtmp
        ]
        
        import subprocess
        self.out = subprocess.Popen(command, stdin=subprocess.PIPE)
        print(f"Output stream initialized: {output_rtmp}")
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        """Resize and pad image while meeting stride-multiple constraints"""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)
    
    def preprocess(self, img):
        """Preprocess image for inference"""
        img_resized, ratio, pad = self.letterbox(img, self.imgsz, auto=True)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_norm, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        img_tensor = torch.from_numpy(img_input).to(self.device)
        return img_tensor, ratio, pad
    
    def postprocess(self, pred, img_shape, img0_shape):
        """Apply NMS and scale boxes to original image"""
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=1000)
        
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img_shape[2:], det[:, :4], img0_shape).round()
                detections.append(det)
        
        return detections
    
    def draw_detections(self, img, detections):
        """Draw bounding boxes and labels on image"""
        annotator = Annotator(img, line_width=2, example=str(self.names))
        
        detection_info = []
        
        if detections and len(detections) > 0:
            det = detections[0]
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{self.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                
                # Store detection info
                detection_info.append({
                    'class': self.names[c],
                    'confidence': float(conf),
                    'bbox': [int(x) for x in xyxy]
                })
        
        return annotator.result(), detection_info
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Preprocess
        img_tensor, ratio, pad = self.preprocess(frame)
        
        # Inference
        pred = self.model(img_tensor)
        
        # Postprocess
        detections = self.postprocess(pred, img_tensor.shape, frame.shape)
        
        # Draw detections
        annotated_frame, detection_info = self.draw_detections(frame.copy(), detections)
        
        return annotated_frame, detection_info
    
    def add_overlay(self, frame, detection_info, fps):
        """Add information overlay to frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Add text
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Mangrove Detection | {timestamp}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Detections: {len(detection_info)}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detection list (bottom)
        if detection_info:
            y_offset = h - 20 * len(detection_info) - 10
            for i, det in enumerate(detection_info):
                text = f"{det['class']}: {det['confidence']:.2%}"
                cv2.putText(frame, text, (10, y_offset + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def run(self, display=True, save_video=None):
        """Run inference on RTMP stream"""
        print("\nStarting inference...")
        print("Press 'q' to quit")
        print("-" * 50)
        
        # Video writer for saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_video, fourcc, self.fps, (self.width, self.height))
            print(f"Saving video to: {save_video}")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame. Stream may have ended.")
                    break
                
                self.frame_count += 1
                
                # Process frame
                start_time = time.time()
                annotated_frame, detection_info = self.process_frame(frame)
                inference_time = time.time() - start_time
                
                # Calculate FPS
                fps = 1.0 / inference_time if inference_time > 0 else 0
                
                # Update detection count
                self.detection_count += len(detection_info)
                
                # Add overlay
                final_frame = self.add_overlay(annotated_frame, detection_info, fps)
                
                # Display
                if display:
                    cv2.imshow('RTMP Stream - Mangrove Detection', final_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save to file
                if video_writer:
                    video_writer.write(final_frame)
                
                # Stream to output RTMP
                if self.out:
                    self.out.stdin.write(final_frame.tobytes())
                
                # Print statistics every 100 frames
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed
                    print(f"Frames: {self.frame_count} | Avg FPS: {avg_fps:.1f} | Total Detections: {self.detection_count}")
        
        except KeyboardInterrupt:
            print("\nStopping inference...")
        
        finally:
            self.cleanup(video_writer)
    
    def cleanup(self, video_writer=None):
        """Cleanup resources"""
        print("\nCleaning up...")
        
        # Release video capture
        if self.cap:
            self.cap.release()
        
        # Release video writer
        if video_writer:
            video_writer.release()
        
        # Close output stream
        if self.out:
            self.out.stdin.close()
            self.out.wait()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\nFinal Statistics:")
        print(f"Total Frames: {self.frame_count}")
        print(f"Total Detections: {self.detection_count}")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total Time: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='RTMP Stream Inference for Mangrove Detection')
    parser.add_argument('--rtmp-url', type=str, default='rtmp://localhost:1935/drone/dji',
                       help='RTMP URL to read stream from')
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Path to YOLOv5 model')
    parser.add_argument('--output-rtmp', type=str, default=None,
                       help='RTMP URL to stream annotated video to')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Inference image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cpu or cuda, default: cuda)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save annotated video to file')
    
    args = parser.parse_args()
    
    # Initialize and run inference
    inference = RTMPStreamInference(
        rtmp_url=args.rtmp_url,
        model_path=args.model,
        output_rtmp=args.output_rtmp,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=args.imgsz,
        device=args.device
    )
    
    inference.run(display=not args.no_display, save_video=args.save_video)


if __name__ == '__main__':
    main()
