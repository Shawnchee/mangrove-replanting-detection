import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import random
import onnxruntime as ort

st.set_page_config(page_title="Mangrove Replanting Detector", layout="wide")

# ===============================
# MODEL LOADING (UPDATED FOR ONNX)
# ===============================
@st.cache_resource
def load_model():
    session = ort.InferenceSession(
        "models/best_latest.onnx",
        providers=["CPUExecutionProvider"]
    )
    return session

session = load_model()

# Initialize session state for logs - ALWAYS CLEAR ON STARTUP
st.session_state.detection_logs = []

st.title("🌱 Mangrove Replanting Zone Detector (ONNX YOLO)")
st.write("Real-time detection streamed to the frontend.")

mode = st.radio("Choose input method:", ["Webcam", "Upload Image", "Upload Video"])


# ===============================
# IMPROVED YOLO ONNX PREPROCESSING + POSTPROCESSING
# ===============================
def preprocess(img, target_size=640):
    """Improved preprocessing with letterbox to maintain aspect ratio"""
    img_h, img_w = img.shape[:2]
    
    # Calculate scale and padding
    scale = min(target_size / img_h, target_size / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    
    # Resize image
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
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


def get_mock_gps():
    """Generate mock GPS coordinates (simulating mangrove region in Southeast Asia)"""
    # Base coordinates around a mangrove area (e.g., Philippines/Indonesia region)
    base_lat = 10.3157
    base_lon = 123.8854
    # Add small random offset (within ~1km)
    lat = base_lat + random.uniform(-0.01, 0.01)
    lon = base_lon + random.uniform(-0.01, 0.01)
    return f"{lat:.6f}, {lon:.6f}"


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

            # Log detection
            log_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'frame': frame_number if frame_number else 'N/A',
                'confidence': f"{confidence:.2f}",
                'bbox': f"[{x1}, {y1}, {x2}, {y2}]",
                'size': f"{x2-x1}x{y2-y1}",
                'gps': get_mock_gps()
            }
            st.session_state.detection_logs.append(log_entry)

    return original_img, detection_count


# ===============================
# WEBCAM MODE
# ===============================
if mode == "Webcam":
    st.warning("Streamlit webcam can be 5–10 FPS depending on device.")
    
    # Clear logs when starting webcam
    if st.button("🗑️ Clear Logs"):
        st.session_state.detection_logs = []
        st.rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        run = st.checkbox("Start Webcam")
        conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
        iou_threshold = st.slider("IOU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
        max_box_ratio = st.slider("Max Box Size (% of image)", 0.1, 1.0, 0.5, 0.05, help="Ignore boxes larger than this ratio")
        FRAME_WINDOW = st.image([])
        detection_count = st.empty()

    with col2:
        st.subheader("📊 Detection Log")
        log_container = st.container()

    cap = None
    frame_counter = 0

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to access webcam.")
                break

            frame_counter += 1
            output, num_detections = infer(frame, frame_counter, conf_threshold, iou_threshold, max_box_ratio)

            FRAME_WINDOW.image(output, channels="BGR")
            detection_count.metric("Replanting Zones Detected", num_detections)

            with log_container:
                if st.session_state.detection_logs:
                    df = pd.DataFrame(st.session_state.detection_logs[-10:])
                    st.dataframe(df, use_container_width=True)

        if cap:
            cap.release()
    else:
        st.stop()


# ===============================
# IMAGE UPLOAD
# ===============================
elif mode == "Upload Image":
    # Clear logs for new image
    st.session_state.detection_logs = []
    
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        col1, col2 = st.columns([2, 1])

        with col1:
            conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
            iou_threshold = st.slider("IOU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
            max_box_ratio = st.slider("Max Box Size (% of image)", 0.1, 1.0, 0.5, 0.05, help="Ignore boxes larger than this ratio")
            
            img = np.array(Image.open(uploaded))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output, num_detections = infer(img, conf_threshold=conf_threshold, iou_threshold=iou_threshold, max_box_ratio=max_box_ratio)

            st.image(output, caption="Detection Result", channels="BGR", use_column_width=True)
            st.metric("Replanting Zones Detected", num_detections)

        with col2:
            st.subheader("📊 Detection Log")
            if st.session_state.detection_logs:
                df = pd.DataFrame(st.session_state.detection_logs)
                st.dataframe(df, use_container_width=True)


# ===============================
# VIDEO UPLOAD
# ===============================
elif mode == "Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.video(uploaded)
            conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
            iou_threshold = st.slider("IOU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
            max_box_ratio = st.slider("Max Box Size (% of image)", 0.1, 1.0, 0.5, 0.05, help="Ignore boxes larger than this ratio")
            process_btn = st.button("🔍 Process Video")
            FRAME_WINDOW = st.image([])
            progress_bar = st.progress(0)
            detection_count = st.empty()

        with col2:
            st.subheader("📊 Detection Log")
            log_container = st.container()
            download_btn_placeholder = st.empty()

        if process_btn:
            # Clear logs for new video
            st.session_state.detection_logs = []

            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_counter = 0
            
            # Set target FPS to 5
            target_fps = 5
            frame_delay = 1.0 / target_fps  # 0.2 seconds between frames
            
            # Calculate frame skip to match target FPS
            frame_skip = max(1, int(original_fps / target_fps))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_counter += 1
                
                # Process only every Nth frame based on original FPS
                if frame_counter % frame_skip != 0:
                    continue
                
                start_time = time.time()
                
                output, num_detections = infer(frame, frame_counter, conf_threshold, iou_threshold, max_box_ratio)

                if frame_counter % 5 == 0:
                    FRAME_WINDOW.image(output, channels="BGR")
                    progress_bar.progress(frame_counter / total_frames)
                    detection_count.metric("Current Frame", f"{frame_counter}/{total_frames}")

                    with log_container:
                        if st.session_state.detection_logs:
                            df = pd.DataFrame(st.session_state.detection_logs[-15:])
                            st.dataframe(df, use_container_width=True, height=400)
                
                # Add delay to maintain target FPS
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

            cap.release()
            progress_bar.progress(1.0)
            st.success(f"✅ Processing complete! Total frames: {frame_counter} | Target: {target_fps} FPS")

            if st.session_state.detection_logs:
                df = pd.DataFrame(st.session_state.detection_logs)
                csv = df.to_csv(index=False)
                with download_btn_placeholder:
                    st.download_button(
                        label="📥 Download Detection Log (CSV)",
                        data=csv,
                        file_name=f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )