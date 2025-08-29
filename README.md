# Car Detection and Illegal Parking Detection System

A comprehensive computer vision system for detecting vehicles, tracking their movements, and identifying illegal parking violations using YOLO object detection and color histogram analysis.

## 🚗 System Overview

This system combines YOLO-based vehicle detection with intelligent parking timer management to automatically identify vehicles that exceed parking time limits. It's designed for parking lot surveillance, traffic monitoring, and automated parking enforcement.

### 🔧 Architecture

#### Data Flow
1. **Video Input** → Frame extraction
2. **YOLO Detection** → Vehicle detection and tracking
3. **Position Analysis** → Movement detection and parking state
4. **Timer Management** → Parking duration calculation
5. **Violation Detection** → Illegal parking identification
6. **Output Generation** → Processed video and visualization

#### Key Algorithms
- **HSV Histogram Comparison**: Movement detection using color similarity
- **IOU Calculation**: Position verification for illegal parking
- **Frame-based Timing**: Precise time calculation independent of processing speed
- **Overlap Resolution**: Intelligent bounding box deduplication


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

## 🔄 System Workflow

### 📋 **Complete Processing Pipeline**

#### **Phase 1: Video Input & Frame Processing**
1. **Video Loading**: System loads input video file
2. **Frame Extraction**: Extracts individual frames at 25 FPS
3. **Preprocessing**: Resizes frames for optimal YOLO processing

#### **Phase 2: Vehicle Detection & Tracking**
1. **YOLO Detection**: 
   - Processes each frame through YOLOv9 model
   - Detects vehicles with confidence scores
   - Identifies vehicle classes (car, truck, bus, motorcycle)
2. **Tracking Assignment**: 
   - Assigns unique track IDs to detected vehicles
   - Maintains ID consistency across frames
   - Handles vehicle occlusion and reappearance

#### **Phase 3: Parking State Analysis**
1. **Movement Detection**:
   - Execute per frame  
   - Calculates HSV color histogram for each vehicle
   - Compares current histogram with previous frame
   - Determines if vehicle is moving or stationary
2. **Parking Confirmation**:
   - Requires 25 consecutive frames of stationary behavior
   - Locks vehicle position when parking is confirmed
   - Starts parking timer for the vehicle

#### **Phase 4: Timer Management**
1. **Parking Duration Tracking**:
   - Counts frames since parking confirmation
   - Converts frame count to real-time seconds (25 FPS)
   - Updates timer for all parked vehicles
2. **Illegal Parking Detection**:
   - Monitors vehicles approaching 30-second limit
   - Activates two-stage verification system

#### **Phase 5: Illegal Parking Verification**
1. **Time Threshold Check** (80% of limit):
   - Checks if vehicle is within 24 seconds of limit
   - Only proceeds if time condition is met
2. **Position Verification** (IOU > 0.5):
   - Calculates intersection-over-union with locked position
   - Confirms vehicle hasn't moved significantly
   - Triggers illegal parking mode if both conditions met

#### **Phase 6: Result Generation**
1. **Video Processing**:
   - Adds detection overlays and status indicators
   - Shows real-time parking timers
   - Highlights illegal parking violations
2. **Visualization**:
   - Generates summary image of violations
   - Creates violation report with timestamps
   - Saves processed video with annotations


### 📊 **Data Flow Architecture**

```
Input Video ──▶ Frame Buffer ──▶ Detection Queue ──▶ Processing Pipeline
     │              │                   │                   │
     ▼              ▼                   ▼                   ▼
  Metadata     Frame Data         Detection Results    Timer Updates
     │              │                   │                   │
     ▼              ▼                   ▼                   ▼
  Output Path   Processed Frames   Tracked Vehicles   Violation List
```

### 🎯 **Key Decision Points**

1. **Parking Confirmation**: 25 consecutive stationary frames required
2. **Illegal Threshold**: 80% of 30-second limit (24 seconds)
3. **Position Tolerance**: IOU > 0.5 for position verification
4. **Occlusion Handling**: 375 frames tolerance for temporary disappearance
5. **Overlap Resolution**: Automatic removal of duplicate detections

### 🔧 **Error Handling & Recovery**

- **Detection Failures**: Graceful fallback to previous frame data
- **Tracking Loss**: ID recovery through position and color similarity
- **Timer Corruption**: Automatic state restoration from backup data
- **Memory Issues**: Automatic cleanup of old tracking data

## 🌈 **HSV Color Analysis in Our System**

### 📊 **How We Use HSV**

#### **Movement Detection Process**
1. **Extract** vehicle region from bounding box
2. **Convert** BGR image to HSV color space
3. **Calculate** color histogram (32 bins per channel)
4. **Compare** current frame with previous frame
5. **Determine** if vehicle is moving based on similarity threshold


### 🔧 **HSV Advantages**
- **Lighting Invariant**: Works in shadows and bright sunlight
- **Movement Sensitive**: Detects even subtle position changes
- **Noise Resistant**: Handles video compression and camera artifacts

### 🚗 **Real-World Usage**
- **Parking Confirmation**: Requires 25 consecutive stationary frames
- **Movement Detection**: Identifies when vehicles change position
- **Illegal Parking**: Confirms vehicles haven't moved from parked position

## 🚀 Usage

### Basic Video Processing
```bash
# Process video with default settings
python car_detector.py --input videos/parking_lot.mp4 --output processed_video.mp4

# Generate visualization without video output
python car_detector.py --input videos/parking_lot.mp4 --vis
```
### Output Files
- **Processed Video**: Enhanced video with detection overlays and parking status
- **Visualization Image**: Summary image showing illegal parking violations
- **Console Output**: Real-time detection statistics and violation alerts




## 📝 License and Support

This system is designed for educational and research purposes. For commercial use, please ensure compliance with local regulations and obtain necessary licenses.

---

**Note**: This system is optimized for parking lot surveillance and traffic monitoring applications. Adjust parameters based on your specific use case and environmental conditions.
