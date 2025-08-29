# Car Detection and Illegal Parking Detection System

A comprehensive computer vision system for detecting vehicles, tracking their movements, and identifying illegal parking violations using YOLO object detection and color histogram analysis.

## 🚗 System Overview

This system combines YOLO-based vehicle detection with intelligent parking timer management to automatically identify vehicles that exceed parking time limits. It's designed for parking lot surveillance, traffic monitoring, and automated parking enforcement.

## 📁 Core Components

### 1. `car_detector.py` - Main Detection Engine
The primary module that orchestrates the entire system, handling video processing, vehicle detection, and visualization.

**Key Features:**
- **YOLO Integration**: Uses YOLOv9 for real-time vehicle detection
- **Multi-Vehicle Support**: Detects cars, trucks, buses, and motorcycles
- **Smart Tracking**: Built-in YOLO tracking with enhanced stability
- **Overlap Removal**: Intelligent bounding box deduplication
- **Visualization**: Real-time display with parking status indicators

**Main Classes:**
- `CarDetector`: Core detection and processing class
- Handles video input/output, detection processing, and result visualization

### 2. `timer.py` - Parking Timer Management
Specialized module for managing parking states, timing, and illegal parking detection.

**Key Features:**
- **Parking State Management**: Tracks when vehicles enter/exit parking mode
- **Time Calculation**: Precise frame-based timing (25 FPS)
- **Illegal Detection**: Automatic flagging of vehicles exceeding time limits
- **Color Histogram Analysis**: Movement detection using HSV color comparison
- **Occlusion Handling**: Maintains timer state during temporary vehicle occlusion

**Main Classes:**
- `ParkingTimer`: Manages all parking-related timing and state logic

### 3. `config.py` - System Configuration
Centralized configuration file containing all system parameters and settings.

**Key Features:**
- **Parking Settings**: Configurable time limits and thresholds
- **Detection Parameters**: YOLO confidence and IoU thresholds
- **Vehicle Classes**: Configurable vehicle types to detect
- **Color Analysis**: Histogram bins and comparison methods
- **Tracking Parameters**: Optimized ByteTracker settings

## 🎯 Core Functionality

### Vehicle Detection
- **Multi-class Detection**: Supports cars, trucks, buses, and motorcycles
- **Real-time Processing**: Optimized for live video streams
- **Confidence Filtering**: Configurable detection sensitivity
- **Overlap Resolution**: Intelligent handling of overlapping detections

### Parking Detection
- **Color-based Analysis**: Uses HSV histogram comparison for movement detection
- **Position Locking**: Locks vehicle position when parking starts
- **Time Tracking**: Frame-accurate parking duration calculation
- **State Persistence**: Maintains parking state during temporary occlusion

### Illegal Parking Detection
- **Time-based Monitoring**: Tracks vehicles exceeding 30-second limit
- **IOU Verification**: Confirms vehicle position using intersection-over-union
- **Two-stage Checking**: Time threshold (80%) then position verification
- **Violation Recording**: Maintains detailed violation history

### ID Continuity
- **Stable Tracking**: Prevents ID jumping during detection
- **Position Memory**: Tracks vehicle positions across frames
- **Recovery Logic**: Recovers vehicle IDs after temporary occlusion
- **Duplicate Prevention**: Eliminates multiple IDs for same vehicle

## 🚀 Usage

### Basic Video Processing
```bash
# Process video with default settings
python car_detector.py --input videos/parking_lot.mp4 --output processed_video.mp4

# Generate visualization without video output
python car_detector.py --input videos/parking_lot.mp4 --vis

# Use custom model
python car_detector.py --input videos/parking_lot.mp4 --model yolov9c.pt --vis
```

### Output Files
- **Processed Video**: Enhanced video with detection overlays and parking status
- **Visualization Image**: Summary image showing illegal parking violations
- **Console Output**: Real-time detection statistics and violation alerts

## ⚙️ Configuration

### Parking Settings
```python
# config.py
ALLOWED_PARKING_TIME = 30     # seconds - parking time limit
PARKING_THRESHOLD = 25         # frames - time to confirm parking
OCCLUSION_TOLERANCE_FRAMES = 375  # frames - occlusion tolerance
```

### Detection Settings
```python
# config.py
CONFIDENCE_THRESHOLD = 0.35    # YOLO detection confidence
IOU_THRESHOLD = 0.4           # Intersection over Union threshold
HISTOGRAM_BINS = 32           # Color analysis precision
```

### Vehicle Classes
```python
# config.py
VEHICLE_CLASSES = {
    2: 'car',          # Passenger vehicles
    3: 'motorcycle',   # Two-wheelers
    5: 'bus',          # Public transport
    7: 'truck'         # Commercial vehicles
}
```

## 🔧 System Architecture

### Data Flow
1. **Video Input** → Frame extraction
2. **YOLO Detection** → Vehicle detection and tracking
3. **Position Analysis** → Movement detection and parking state
4. **Timer Management** → Parking duration calculation
5. **Violation Detection** → Illegal parking identification
6. **Output Generation** → Processed video and visualization

### Key Algorithms
- **HSV Histogram Comparison**: Movement detection using color similarity
- **IOU Calculation**: Position verification for illegal parking
- **Frame-based Timing**: Precise time calculation independent of processing speed
- **Overlap Resolution**: Intelligent bounding box deduplication

## 📊 Performance Features

### Real-time Processing
- **Frame Rate**: Optimized for 25 FPS video streams
- **Memory Management**: Efficient position memory and cleanup
- **Tracking Stability**: Enhanced ByteTracker integration
- **Overlap Handling**: Fast duplicate detection and removal

### Accuracy Improvements
- **Color Smoothing**: Reduces false movement detection
- **Position Locking**: Maintains accurate parking positions
- **Occlusion Tolerance**: Handles temporary vehicle disappearance
- **ID Recovery**: Maintains vehicle identity across frames

## 🎨 Visualization Features

### Real-time Display
- **Color-coded Status**: Green (moving), Yellow (parked), Red (illegal)
- **Timer Information**: Real-time parking duration display
- **Confidence Scores**: Detection confidence indicators
- **Status Labels**: Clear parking state information

### Summary Visualization
- **Violation Summary**: Complete illegal parking report
- **Vehicle Images**: Actual detected vehicle snapshots
- **Duration Display**: Parking time for each violation
- **Timestamp Information**: When violations occurred

## 🔍 Troubleshooting

### Common Issues
1. **Low Detection Rate**: Lower confidence threshold in config.py
2. **False Movement Detection**: Adjust color change threshold
3. **ID Jumping**: Increase track buffer and confirmation frames
4. **Memory Issues**: Reduce position memory size

### Performance Optimization
- **Model Selection**: Use lighter YOLO models for faster processing
- **Frame Skipping**: Process every nth frame for speed
- **Resolution Reduction**: Lower input video resolution
- **Batch Processing**: Process multiple frames simultaneously

## 📈 Future Enhancements

### Planned Features
- **Multi-camera Support**: Synchronized multi-view processing
- **License Plate Recognition**: OCR integration for vehicle identification
- **Database Integration**: Persistent violation storage
- **API Interface**: REST API for external system integration
- **Mobile App**: Real-time monitoring and alerts

### Extensibility
- **Plugin System**: Modular architecture for custom features
- **Custom Models**: Support for specialized detection models
- **Export Formats**: Multiple output format support
- **Integration APIs**: Easy integration with existing systems

## 📝 License and Support

This system is designed for educational and research purposes. For commercial use, please ensure compliance with local regulations and obtain necessary licenses.

---

**Note**: This system is optimized for parking lot surveillance and traffic monitoring applications. Adjust parameters based on your specific use case and environmental conditions.
