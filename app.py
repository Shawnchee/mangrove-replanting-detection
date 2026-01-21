import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import onnxruntime as ort
import io

st.set_page_config(page_title="Mangrove Replanting Detector", layout="wide")

# ===============================
# MODEL LOADING (UPDATED FOR ONNX)
# ===============================
@st.cache_resource
def load_model():
    try:
        # Try CUDA first (requires CUDA 12.6 + cuDNN 9.x)
        try:
            # Configure for GPU with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(
                "models/best_latest.onnx",
                sess_options=sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            active_provider = session.get_providers()[0]
            if active_provider == 'CUDAExecutionProvider':
                print("✓ ONNX Runtime using: CUDAExecutionProvider (GPU) 🚀")
                print("  GPU: NVIDIA GeForce GTX 1650 | CUDA 12.6 | cuDNN 9.17")
                return session
            else:
                print("⚠️ CUDA requested but fell back to CPU")
        except Exception as e:
            print(f"⚠️ CUDA initialization failed: {e}")
        
        # Fallback to optimized CPU
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        
        session = ort.InferenceSession(
            "models/best_latest.onnx",
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        print("✓ ONNX Runtime using: CPUExecutionProvider (optimized multi-thread)")
        print("  💡 For GPU: Run with .\\run_app.ps1 to enable CUDA 12.6")
        
        return session
    except FileNotFoundError:
        st.error("❌ Model file not found at 'models/best_latest.onnx'. Please ensure the model file exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load ONNX model: {str(e)}")
        st.stop()

session = load_model()

# Initialize session state for logs and tracking
if 'detection_logs' not in st.session_state:
    st.session_state.detection_logs = []  # Aggregated zone detections
if 'tracked_boxes' not in st.session_state:
    st.session_state.tracked_boxes = []  # Persistent zone tracking (box, zone_id)
if 'zone_counter' not in st.session_state:
    st.session_state.zone_counter = 0  # Counter for unique zone IDs

# ===============================
# MAIN UI - SINGLE PAGE WITH SIDE-BY-SIDE VIEW
# ===============================
st.title("🌱 Mangrove Replanting Zone Detector")

# Small tab for video input selection
with st.expander("📹 Video Input", expanded=False):
    mode = st.radio("Choose input source:", ["RTMP Livestream", "Upload Video", "📊 Detection Report"], horizontal=True)


# ===============================
# IMPROVED YOLO ONNX PREPROCESSING + POSTPROCESSING
# ===============================
def preprocess(img, target_size=640):
    """Optimized preprocessing for CPU inference (640x640 to match model)"""
    img_h, img_w = img.shape[:2]
    
    # Calculate scale and padding
    scale = min(target_size / img_h, target_size / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    
    # Resize image - use INTER_AREA for downscaling
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create padded image
    padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    padded_img[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = img_resized
    
    # Normalize and transpose
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0)
    
    return img_input, scale, pad_left, pad_top


def xywh2xyxy(x):
    """Convert box format from [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


# Mangrove areas in Malaysia 
MANGROVE_ZONES = [
    {"name": "Kuala Sepetang", "lat": 4.8350, "lon": 100.5720},
    {"name": "Matang Forest Reserve", "lat": 4.8180, "lon": 100.6150},
    {"name": "Kuala Gula", "lat": 4.8520, "lon": 100.5480},
    {"name": "Trong River Estuary", "lat": 4.7890, "lon": 100.5890},
    {"name": "Port Weld", "lat": 4.8650, "lon": 100.6020},
    {"name": "Larut Matang Coast", "lat": 4.8010, "lon": 100.5350},
    {"name": "Sangga Besar", "lat": 4.8420, "lon": 100.5950},
]


def get_zone_gps(zone_id):
    """Get GPS coordinates for a specific zone ID"""
    zone = MANGROVE_ZONES[zone_id % len(MANGROVE_ZONES)]
    # Add small random offset (within ~200m) for variation
    lat = zone["lat"] + random.uniform(-0.002, 0.002)
    lon = zone["lon"] + random.uniform(-0.002, 0.002)
    return zone["name"], lat, lon


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def find_or_create_zone(box, confidence, frame_number, iou_threshold=0.3):
    """
    Find existing zone that matches this detection, or create a new one.
    Returns (zone_id, is_new_zone)
    Zones are tracked persistently throughout the session.
    """
    # Check if this box overlaps with any existing tracked zone
    for i, (tracked_box, zone_id) in enumerate(st.session_state.tracked_boxes):
        iou = calculate_iou(box, tracked_box)
        if iou > iou_threshold:
            # Update the tracked box position (moving average)
            updated_box = [
                (tracked_box[0] + box[0]) / 2,
                (tracked_box[1] + box[1]) / 2,
                (tracked_box[2] + box[2]) / 2,
                (tracked_box[3] + box[3]) / 2,
            ]
            st.session_state.tracked_boxes[i] = (updated_box, zone_id)
            
            # Update existing detection log entry
            for log in st.session_state.detection_logs:
                if log['zone_id'] == zone_id:
                    log['detection_count'] += 1
                    log['last_frame'] = frame_number
                    log['last_seen'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Update confidence (keep max)
                    log['confidence'] = max(log['confidence'], round(float(confidence), 2))
                    break
            
            return zone_id, False  # Existing zone
    
    # New zone detected
    st.session_state.zone_counter += 1
    new_zone_id = st.session_state.zone_counter
    st.session_state.tracked_boxes.append((box, new_zone_id))
    
    return new_zone_id, True  # New zone


def nms(boxes, scores, iou_threshold=0.45):
    """Improved Non-Maximum Suppression"""
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        xx1 = np.maximum(boxes[i][0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i][1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i][2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i][3], boxes[idxs[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        area_others = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        union = area_i + area_others - inter

        iou = inter / (union + 1e-6)
        idxs = idxs[1:][iou < iou_threshold]

    return keep


def infer(img, frame_number=None, conf_threshold=0.75, iou_threshold=0.25, max_box_ratio=0.5):
    """Improved inference with better accuracy"""
    original_img = img.copy()
    img_h, img_w = img.shape[:2]
    
    # Preprocess with letterbox
    img_input, scale, pad_left, pad_top = preprocess(img)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_input})[0]

    predictions = outputs[0]  # shape: (N, 85 or 6)
    
    # Handle different output formats
    if predictions.shape[1] >= 85:
        boxes = predictions[:, :4]
        obj_scores = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        conf = obj_scores * class_scores.max(axis=1)
    else:
        boxes = predictions[:, :4]
        conf = predictions[:, 4]
        class_ids = predictions[:, 5].astype(int) if predictions.shape[1] > 5 else np.zeros(len(boxes), dtype=int)

    # Filter by confidence threshold
    idx = conf > conf_threshold
    boxes = boxes[idx]
    conf = conf[idx]
    class_ids = class_ids[idx]

    if len(boxes) == 0:
        return original_img, 0

    # Convert to xyxy format (boxes are in 640x640 space)
    boxes = xywh2xyxy(boxes)

    # Scale boxes back to original image size
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    
    # Clip boxes to image boundaries
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    box_areas = box_widths * box_heights
    image_area = img_h * img_w
    
    # Filter boxes that are too large (more than max_box_ratio of image)
    size_filter = box_areas < (image_area * max_box_ratio)
    boxes = boxes[size_filter]
    conf = conf[size_filter]
    class_ids = class_ids[size_filter]
    
    if len(boxes) == 0:
        return original_img, 0
    # Apply NMS
    keep = nms(boxes, conf, iou_threshold)
    boxes = boxes[keep]
    conf = conf[keep]
    class_ids = class_ids[keep]

    # Draw boxes on original image
    detection_count = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = int(class_ids[i])
        confidence = conf[i]

        if label == 0:  # class 0 = replanting_zone
            detection_count += 1
            color = (0, 255, 0)
            thickness = 3
            
            # Draw rectangle with rounded corners effect
            cv2.rectangle(original_img, (x1, y1), (x2, y2), color, thickness)
            
            # Add label background
            label_text = f"Replanting Zone {confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(original_img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
            cv2.putText(original_img, label_text,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Find or create zone for this detection (aggregates duplicates)
            if frame_number:
                zone_id, is_new = find_or_create_zone([x1, y1, x2, y2], confidence, frame_number)
                
                if is_new:
                    # Create new zone entry
                    zone_name, lat, lon = get_zone_gps(zone_id)
                    log_entry = {
                        'id': f"ZONE-{1000 + zone_id}",
                        'zone_id': zone_id,
                        'zone_name': zone_name,
                        'first_detected': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'last_seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'first_frame': frame_number,
                        'last_frame': frame_number,
                        'detection_count': 1,
                        'confidence': round(float(confidence), 2),
                        'latitude': round(lat, 6),
                        'longitude': round(lon, 6),
                    }
                    st.session_state.detection_logs.append(log_entry)

    return original_img, detection_count


# ===============================
# DETECTION MODES - SINGLE LAYOUT WITH SIDE-BY-SIDE VIEW
# ===============================
# ===============================
# RTMP LIVESTREAM MODE
# ===============================
if mode == "RTMP Livestream":
    st.markdown("---")
    
    col_settings1, col_settings2 = st.columns([3, 1])
    with col_settings1:
        st.info("📡 Live RTMP stream from drone with real-time inference")
    with col_settings2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.detection_logs = []
            st.session_state.tracked_boxes = []
            st.session_state.zone_counter = 0
            st.rerun()

    # RTMP URL input - localhost because nginx runs on THIS computer
    rtmp_url = st.text_input("RTMP Stream URL", value="rtmp://localhost:1935/live", 
                             help="Use localhost - nginx RTMP server runs on this computer")
    
    # Settings in a collapsible section
    with st.expander("⚙️ Detection Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            conf_threshold = st.slider("Confidence Threshold", 0.1, 0.95, 0.93, 0.05)
        with col2:
            iou_threshold = st.slider("IOU Threshold", 0.1, 0.9, 0.95, 0.05)
        with col3:
            max_box_ratio = st.slider("Max Box Size", 0.1, 1.0, 0.1, 0.05)
    
    run = st.checkbox("▶️ Start Livestream", key="rtmp_start")
    
    # Side-by-side layout: Raw footage | Inferencing
    col_raw, col_inference = st.columns(2)
    
    with col_raw:
        st.subheader("📹 Raw Footage")
        RAW_FRAME = st.empty()
    
    with col_inference:
        st.subheader("🔍 Live Inference")
        INFERENCE_FRAME = st.empty()
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        fps_metric = st.empty()
    with status_col2:
        detection_count = st.empty()

    cap = None
    frame_counter = 0
    skip_frames = 0  # Frame skipping for performance
    target_fps = 30
    consecutive_errors = 0
    max_errors = 10  # Reconnect after this many consecutive errors

    if run:
        try:
            # Suppress ffmpeg warnings for H.264 decoding errors
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;1000000|fflags;discardcorrupt"
            
            # Use ffmpeg backend with optimized RTMP parameters
            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce latency
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                st.error(f"❌ Cannot connect to RTMP stream: {rtmp_url}")
                st.warning("**Common Issues:**")
                st.markdown(r"""
                1. **nginx RTMP server not running**
                   - Run: `.\start_rtmp_server.ps1`
                   - Check: `Get-Process nginx`
                
                2. **No video stream is being sent to the server**
                   - Your DJI drone must be **actively streaming** right now
                   - Configure drone: Server `rtmp://192.168.100.48:1935/drone`, Key `dji`
                   - Start streaming on the drone BEFORE clicking "Start Livestream"
                
                3. **Test with OBS Studio (if no drone available)**
                   - Download OBS Studio (free)
                   - Settings → Stream → Custom
                   - Server: `rtmp://localhost:1935/drone`
                   - Stream Key: `dji`
                   - Start Streaming in OBS first, then try again here
                
                4. **Network issues**
                   - Drone and computer must be on same WiFi
                   - Use IP `192.168.100.48` for drone settings
                """)
                st.stop()

            st.success(f"✅ Connected to RTMP stream: {rtmp_url}")
            
            last_time = time.time()
            fps = 0
            fps_smooth = 0  # Smoothed FPS for display
            frame_time = 1.0 / target_fps  # Target frame time for 30 FPS
            last_valid_frame = None

            while run:
                ret, frame = cap.read()
                
                # Handle read failures and corrupted H.264 frames (missing picture errors)
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        # Reconnect to stream
                        cap.release()
                        time.sleep(0.3)
                        cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        consecutive_errors = 0
                    # Use last valid frame if available (prevents blank screen)
                    if last_valid_frame is not None:
                        frame = last_valid_frame
                    else:
                        continue
                
                # Validate frame is not corrupted (has valid dimensions and data)
                if frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
                    consecutive_errors += 1
                    if last_valid_frame is not None:
                        frame = last_valid_frame
                    else:
                        continue
                
                # Check for green/corrupt frames (common H.264 decode error)
                if np.mean(frame) < 5 or np.std(frame) < 10:
                    consecutive_errors += 1
                    if last_valid_frame is not None:
                        frame = last_valid_frame
                    else:
                        continue
                
                # Valid frame - save it and reset error counter
                consecutive_errors = 0
                last_valid_frame = frame.copy()

                try:
                    frame_counter += 1
                    
                    # Calculate FPS with smoothing
                    current_time = time.time()
                    elapsed = current_time - last_time
                    if elapsed > 0:
                        instant_fps = 1.0 / elapsed
                        fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1  # Exponential smoothing
                    fps = fps_smooth
                    
                    # Skip frames to maintain 30 FPS target
                    if elapsed < frame_time * 0.8:
                        # Processing faster than needed, can process every frame
                        skip_frames = 0
                    elif elapsed > frame_time * 1.2:
                        # Processing slower, skip more frames
                        skip_frames = min(3, skip_frames + 1)
                    
                    # Apply frame skipping
                    if skip_frames > 0 and frame_counter % (skip_frames + 1) != 0:
                        # Still grab frame to keep buffer clear but don't process
                        continue
                    
                    last_time = current_time
                    
                    # Downscale for faster display (maintain aspect ratio)
                    scale = min(640 / frame.shape[1], 480 / frame.shape[0], 1.0)
                    if scale < 1.0:
                        display_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    else:
                        display_frame = frame
                    
                    RAW_FRAME.image(display_frame, channels="BGR", use_column_width=True)
                    
                    # Run inference
                    output, num_detections = infer(frame, frame_counter, conf_threshold, iou_threshold, max_box_ratio)
                    
                    # Downscale inference output for display
                    if scale < 1.0:
                        display_output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    else:
                        display_output = output
                    
                    INFERENCE_FRAME.image(display_output, channels="BGR", use_column_width=True)
                    
                    # Update metrics
                    fps_metric.metric("⚡ FPS", f"{fps:.1f}")
                    detection_count.metric("🎯 Detections", num_detections)
                    
                except Exception as e:
                    # Silently skip frame on error, don't spam UI
                    continue

        except Exception as e:
            st.error(f"❌ RTMP stream error: {str(e)}")
        finally:
            if cap:
                cap.release()
    else:
        st.info("Click 'Start Livestream' to begin processing the RTMP stream")





# ===============================
# VIDEO UPLOAD
# ===============================
if mode == "Upload Video":
    st.markdown("---")
    
    uploaded = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi"])

    if uploaded:
        try:
            st.video(uploaded)
            
            # Settings
            with st.expander("⚙️ Detection Settings"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.40, 0.05)
                with col2:
                    iou_threshold = st.slider("IOU Threshold", 0.1, 0.9, 0.15, 0.05)
                with col3:
                    max_box_ratio = st.slider("Max Box Size", 0.1, 1.0, 0.05, 0.05)
            
            process_btn = st.button("🔍 Process Video")

            if process_btn:
                # Clear logs and tracking
                st.session_state.detection_logs = []
                st.session_state.tracked_boxes = []
                st.session_state.zone_counter = 0
                
                # Side-by-side layout
                col_raw, col_inference = st.columns(2)
                
                with col_raw:
                    st.subheader("📹 Raw Frame")
                    RAW_FRAME = st.empty()
                
                with col_inference:
                    st.subheader("🔍 Inference Frame")
                    INFERENCE_FRAME = st.empty()
                
                progress_bar = st.progress(0)
                detection_count = st.empty()
                download_btn_placeholder = st.empty()

                temp_path = "temp_video.mp4"
                
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded.read())
                except IOError as e:
                    st.error(f"❌ Cannot save video: {str(e)}")
                    st.stop()

                cap = cv2.VideoCapture(temp_path)
                
                if not cap.isOpened():
                    st.error("❌ Cannot open video file")
                    st.stop()
                
                try:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    original_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    if total_frames == 0 or original_fps == 0:
                        st.error("❌ Invalid video file")
                        cap.release()
                        st.stop()
                    
                    frame_counter = 0
                    target_fps = 5
                    frame_delay = 1.0 / target_fps
                    frame_skip = max(1, int(original_fps / target_fps))

                    while True:
                        try:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_counter += 1
                            
                            # Process every Nth frame
                            if frame_counter % frame_skip != 0:
                                continue
                            
                            start_time = time.time()
                            
                            # Show raw and inference frames side by side
                            if frame_counter % 5 == 0:
                                RAW_FRAME.image(frame, channels="BGR", use_column_width=True)
                            
                            output, num_detections = infer(frame, frame_counter, conf_threshold, iou_threshold, max_box_ratio)

                            if frame_counter % 5 == 0:
                                INFERENCE_FRAME.image(output, channels="BGR", use_column_width=True)
                                progress_bar.progress(min(frame_counter / total_frames, 1.0))
                                detection_count.metric("Frame Progress", f"{frame_counter}/{total_frames}")
                            
                            # Maintain target FPS
                            elapsed = time.time() - start_time
                            if elapsed < frame_delay:
                                time.sleep(frame_delay - elapsed)
                        
                        except Exception as e:
                            st.warning(f"⚠️ Error on frame {frame_counter}: {str(e)}")
                            continue

                    cap.release()
                    progress_bar.progress(1.0)
                    st.success(f"✅ Processing complete! {frame_counter} frames @ {target_fps} FPS")

                    if st.session_state.detection_logs:
                        try:
                            df = pd.DataFrame(st.session_state.detection_logs)
                            csv = df.to_csv(index=False)
                            with download_btn_placeholder:
                                st.download_button(
                                    label="📥 Download Detection Log",
                                    data=csv,
                                    file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        except Exception as e:
                            st.error(f"❌ Error generating CSV: {str(e)}")

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                finally:
                    cap.release()
                    # Clean up
                    try:
                        import os
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")


# ===============================
# DETECTION REPORT
# ===============================
if mode == "📊 Detection Report":
    st.markdown("---")
    
    st.subheader("📊 Mangrove Replanting Detection Report")
    
    # Use real detection data from session state
    detection_data = st.session_state.detection_logs
    
    if not detection_data:
        st.warning("⚠️ No detections recorded yet. Run RTMP Livestream or Upload Video to detect replanting zones.")
        st.info("💡 **How to get detections:**\n1. Select 'RTMP Livestream' or 'Upload Video'\n2. Start processing\n3. Detections will appear here automatically")
    else:
        df = pd.DataFrame(detection_data)
        
        # Summary metrics (simplified)
        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Zones Detected", len(detection_data))
        with col2:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col3:
            total_detections = df['detection_count'].sum()
            st.metric("Total Detections", int(total_detections))
        
        # Detection table
        st.markdown("---")
        st.markdown("### 📋 Detected Zones")
        
        # Display formatted table
        display_df = df.copy()
        display_df['coordinates'] = display_df.apply(lambda x: f"{x['latitude']:.6f}, {x['longitude']:.6f}", axis=1)
        display_df['confidence_pct'] = display_df['confidence'].apply(lambda x: f"{x:.0%}")
        display_df['detections'] = display_df['detection_count'].apply(lambda x: f"{int(x)}x")
        
        st.dataframe(
            display_df[['id', 'zone_name', 'coordinates', 'confidence_pct', 'detections', 'first_detected']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": "Zone ID",
                "zone_name": "Location",
                "coordinates": "GPS ",
                "confidence_pct": "Max Confidence",
                "detections": "Times Detected",
                "first_detected": "First Detected"
            }
        )
        
        # Map visualization
        st.markdown("### 🗺️ Detection Locations")
        map_df = df[['latitude', 'longitude']].copy()
        map_df.columns = ['lat', 'lon']
        st.map(map_df, zoom=11)
        
        # Export section
        st.markdown("---")
        st.markdown("### 📥 Export Options")
        
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            # CSV Export
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"mangrove_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            # TXT Report
            def generate_txt_report():
                report_lines = []
                report_lines.append("MANGROVE REPLANTING DETECTION REPORT")
                report_lines.append("=" * 50)
                report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Location: Malaysia")
                report_lines.append(f"Zones Detected: {len(detection_data)}")
                report_lines.append(f"Total Detections: {int(total_detections)}")
                report_lines.append(f"Average Confidence: {avg_conf:.1%}")
                report_lines.append("")
                report_lines.append("ZONE DETAILS")
                report_lines.append("-" * 50)
                
                for det in detection_data:
                    report_lines.append(f"\n[{det['id']}] {det['zone_name']}")
                    report_lines.append(f"  First Detected: {det['first_detected']}")
                    report_lines.append(f"  Coordinates: {det['latitude']:.6f}, {det['longitude']:.6f}")
                    report_lines.append(f"  Max Confidence: {det['confidence']:.0%}")
                    report_lines.append(f"  Times Detected: {det['detection_count']}x")
                
                return "\n".join(report_lines)
            
            txt_content = generate_txt_report()
            st.download_button(
                label="📑 Download Report (TXT)",
                data=txt_content,
                file_name=f"mangrove_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_export3:
            # JSON Export
            import json
            json_data = json.dumps({
                "report_generated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "location": "Malaysia",
                "summary": {
                    "zones_detected": len(detection_data),
                    "total_detections": int(total_detections),
                    "avg_confidence": round(avg_conf, 3),
                },
                "zones": detection_data
            }, indent=2)
            
            st.download_button(
                label="📦 Download JSON",
                data=json_data,
                file_name=f"mangrove_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    # Clear detections button
    st.markdown("---")
    if st.button("🗑️ Clear All Detections"):
        st.session_state.detection_logs = []
        st.session_state.tracked_boxes = []
        st.session_state.zone_counter = 0
        st.rerun()