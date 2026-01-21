# 🌱 Mangrove Replanting Detection System

A real-time computer vision application for detecting and monitoring mangrove replanting zones using YOLOv5 object detection with ONNX Runtime inference. Now with **RTMP live streaming support for DJI Mini 2 drone**!

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [**NEW: DJI Mini 2 RTMP Streaming**](#dji-mini-2-rtmp-streaming)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project provides an automated solution for detecting mangrove replanting zones from various input sources (webcam, images, videos). It uses a custom-trained YOLOv5 model converted to ONNX format for efficient CPU inference, making it suitable for deployment on edge devices and servers without GPU requirements.

### Key Capabilities
- **Real-time Detection**: Process webcam feeds at 5-10 FPS
- **Batch Processing**: Analyze images and videos with progress tracking
- **Smart Filtering**: Configurable confidence thresholds, NMS, and box size filtering
- **Logging System**: Automatic detection logging with CSV export
- **User-Friendly Interface**: Streamlit-based web application

## ✨ Features

### Detection Modes
1. **Webcam Mode**
   - Real-time streaming detection
   - Live bounding box visualization
   - Frame-by-frame logging

2. **Image Upload Mode**
   - Single image analysis
   - Instant detection results
   - Detection metadata display

3. **Video Upload Mode**
   - Batch video processing
   - Progress tracking with frame counter
   - Exportable detection logs (CSV)

### Advanced Features
- **Letterbox Preprocessing**: Maintains aspect ratio for accurate detection
- **Non-Maximum Suppression (NMS)**: Eliminates overlapping detections
- **Size-Based Filtering**: Ignores overly large false positives
- **Configurable Thresholds**:
  - Confidence threshold (0.1-0.9)
  - IOU threshold for NMS (0.1-0.9)
  - Maximum box size ratio (0.1-1.0)
- **Detection Logging**: Timestamp, frame number, confidence score, bounding box coordinates, and size
- **CSV Export**: Download complete detection history
- **🆕 RTMP Live Streaming**: Real-time inference on drone video feeds

## 🚁 DJI Mini 2 RTMP Streaming

### New Feature: Live Drone Detection

Stream live video from your **DJI Mini 2** drone and apply real-time mangrove detection using RTMP protocol!

#### Quick Start
1. **Setup RTMP Server**: See [NGINX_SETUP.md](NGINX_SETUP.md)
2. **Configure Drone Streaming**: See [DJI_STREAMING_GUIDE.md](DJI_STREAMING_GUIDE.md)
3. **Run Inference**: 
   ```powershell
   python rtmp_inference.py --model models/best.pt
   ```

#### Complete Guides
- 📖 **[QUICK_START.md](QUICK_START.md)** - Complete setup walkthrough
- 🔧 **[NGINX_SETUP.md](NGINX_SETUP.md)** - RTMP server installation
- 🚁 **[DJI_STREAMING_GUIDE.md](DJI_STREAMING_GUIDE.md)** - Drone streaming methods
- 💻 **[COMMANDS.md](COMMANDS.md)** - All available commands

#### Features
- ✅ Real-time YOLOv5 inference on live drone feed
- ✅ Annotated video output with bounding boxes
- ✅ FPS monitoring and detection statistics
- ✅ Optional video recording
- ✅ Re-streaming to other RTMP endpoints
- ✅ GPU acceleration support

#### Example Commands
```powershell
# Basic usage
python rtmp_inference.py --model models/best.pt

# Save processed video
python rtmp_inference.py --model models/best.pt --save-video output.mp4

# Use GPU acceleration
python rtmp_inference.py --model models/best.pt --device cuda

# Re-stream to YouTube/Twitch
python rtmp_inference.py --model models/best.pt --output-rtmp rtmp://your-stream-url
```

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.12+**: Primary programming language
- **Streamlit**: Web application framework
- **ONNX Runtime**: Model inference engine
- **OpenCV**: Image processing and video handling
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and logging
- **🆕 FFmpeg**: Video streaming and processing
- **🆕 Nginx-RTMP**: RTMP streaming server

### Model Framework
- **YOLOv5**: Object detection architecture
- **ONNX**: Cross-platform model format for optimized inference

## 📦 Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager
- Webcam (optional, for real-time detection)
- **🆕 Nginx with RTMP module** (for drone streaming)
- **🆕 FFmpeg** (for video processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/mangrove-replanting-detection.git
cd mangrove-replanting-detection
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg (for RTMP streaming)
```powershell
# Windows (using Chocolatey)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### Step 5: Install Nginx-RTMP (for drone streaming)
See detailed instructions in [NGINX_SETUP.md](NGINX_SETUP.md)
```

### Step 4: Download or Place Model
Ensure your trained ONNX model is placed at:
```
models/best_latest.onnx
```

## 🚀 Usage

### Starting the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using Webcam Mode
1. Select "Webcam" from the input method radio buttons
2. Adjust detection parameters using sliders:
   - **Confidence Threshold**: Minimum confidence for valid detections (default: 0.25)
   - **IOU Threshold**: Overlap threshold for NMS (default: 0.45)
   - **Max Box Size**: Maximum detection size as percentage of image (default: 0.5)
3. Check "Start Webcam" to begin real-time detection
4. View detections in the left panel and logs in the right panel
5. Click "Clear Logs" to reset detection history

### Using Image Upload Mode
1. Select "Upload Image" from input methods
2. Upload an image (JPG, JPEG, PNG)
3. Adjust detection parameters
4. View detection results with bounding boxes
5. Check detection log for detailed information

### Using Video Upload Mode
1. Select "Upload Video" from input methods
2. Upload a video file (MP4, MOV, AVI)
3. Preview the uploaded video
4. Adjust detection parameters
5. Click "Process Video" to start analysis
6. Monitor progress with the progress bar
7. Download detection log as CSV when complete

## 🧠 Model Architecture

### YOLOv5 Custom Model
- **Input Size**: 640x640 pixels
- **Architecture**: YOLOv5 (s/m/l/x variant)
- **Classes**: 1 class - `replanting_zone`
- **Output Format**: Bounding boxes with confidence scores

### Preprocessing Pipeline
1. **Letterbox Resizing**: Maintains aspect ratio with padding
2. **Normalization**: Scales pixel values to [0, 1]
3. **Channel Reordering**: BGR → RGB conversion
4. **Tensor Formatting**: HWC → CHW, adds batch dimension

### Postprocessing Pipeline
1. **Confidence Filtering**: Removes low-confidence predictions
2. **Coordinate Transformation**: xywh → xyxy format
3. **Scale Adjustment**: Maps predictions to original image size
4. **Size Filtering**: Removes oversized detections
5. **Non-Maximum Suppression**: Eliminates redundant boxes
6. **Visualization**: Draws bounding boxes with labels

## ⚙️ Configuration

### Default Parameters
```python
# Detection Parameters
conf_threshold = 0.75      # Confidence threshold
iou_threshold = 0.25       # NMS IOU threshold
max_box_ratio = 0.5        # Maximum box size (50% of image)

# Preprocessing
target_size = 640          # Input size for model
padding_value = 114        # Gray padding value

# Visualization
box_color = (0, 255, 0)    # Green bounding boxes
box_thickness = 3          # Line thickness
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 0.6
```

### Adjustable Parameters (UI)
- **Confidence Threshold**: 0.1 - 0.9 (step: 0.05)
- **IOU Threshold**: 0.1 - 0.9 (step: 0.05)
- **Max Box Size**: 0.1 - 1.0 (step: 0.05)

## 📁 Project Structure

```
mangrove-replanting-detection/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── models/
│   └── best_latest.onnx       # Trained ONNX model
│
├── venv/                       # Virtual environment (not in repo)
│
├── temp_video.mp4             # Temporary file for video processing
│
└── detection_logs/            # Exported CSV logs (optional)
```

## 🚀 Performance Optimization

### Inference Speed
- **ONNX Runtime**: 2-3x faster than PyTorch
- **CPU Optimization**: Efficient memory usage
- **Batch Processing**: Processes every 5th frame for videos

### Accuracy Improvements
1. **Letterbox Preprocessing**: Preserves aspect ratio
2. **Smart NMS**: Reduces false positives
3. **Size Filtering**: Eliminates unrealistic large detections
4. **Confidence Tuning**: Adjustable threshold for precision/recall balance

### Memory Management
- **Streamlit Caching**: Model loaded once and cached
- **Frame Skipping**: Reduces memory for video processing
- **Log Limiting**: Shows only recent 10-15 detections in real-time

## 🐛 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'onnxruntime'
```bash
pip install onnxruntime opencv-python streamlit pandas pillow numpy
```

#### 2. Webcam Not Opening
- Check if another application is using the webcam
- Try changing camera index in code: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- Ensure webcam permissions are granted

#### 3. Model Not Loading
- Verify model path: `models/best_latest.onnx`
- Check if file exists and is a valid ONNX model
- Ensure model was exported correctly from YOLOv5

#### 4. Poor Detection Accuracy
- Lower confidence threshold (try 0.15-0.25)
- Adjust IOU threshold for NMS (try 0.35-0.50)
- Reduce max box size ratio (try 0.3-0.4)
- Retrain model with more diverse data

#### 5. Slow Performance
- Close other applications to free CPU
- Reduce webcam resolution if possible
- Increase frame skip rate for video processing
- Consider using GPU-enabled ONNX Runtime

## 📊 Detection Log Format

### CSV Export Columns
| Column      | Description                           | Example               |
|-------------|---------------------------------------|-----------------------|
| timestamp   | Detection time                        | 2025-11-22 14:30:15  |
| frame       | Frame number (N/A for images)         | 150                  |
| confidence  | Detection confidence score            | 0.87                 |
| bbox        | Bounding box [x1, y1, x2, y2]        | [120, 45, 380, 290]  |
| size        | Box dimensions (width x height)       | 260x245              |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Test changes with all three input modes
- Update README if adding new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv5 architecture
- **ONNX**: Cross-platform model format
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library

## 📞 Contact

For questions or support, please open an issue on GitHub or contact:
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

## 🔮 Future Enhancements

- [ ] GPU acceleration support
- [ ] Multi-class detection (different mangrove species)
- [ ] Geographic coordinate tagging
- [ ] Time-series analysis for growth tracking
- [ ] Mobile application deployment
- [ ] REST API for integration
- [ ] Database integration for persistent logging
- [ ] Automated reporting and statistics

---

**Built with ❤️ for mangrove conservation and environmental monitoring**