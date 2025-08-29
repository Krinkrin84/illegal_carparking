import cv2
import argparse

# ===== ESSENTIAL CONFIGURATION =====

# Core parking settings - Essential for parking detection
ALLOWED_PARKING_TIME = 30     # seconds (5 minutes) - realistic parking time limit
PARKING_THRESHOLD = 25         # frames (1 second at 25fps) - adjusted for 25fps
MOVEMENT_DETECTION_THRESHOLD = 20  # pixels - center point movement threshold for parking prevention

# Detection settings - Essential for car detection
CONFIDENCE_THRESHOLD = 0.35    # Lower confidence for more sensitive detection
IOU_THRESHOLD = 0.4           # Slightly lower IoU for better detection overlap
DEFAULT_MODEL = "yolov9c.pt"   # YOLO model file (using YOLOv8 for better compatibility)

# Vehicle class detection settings - Configurable vehicle types to detect
VEHICLE_CLASSES = {
    2: 'car',          # 汽车
    3: 'motorcycle',   # 摩托车
    5: 'bus',          # 公交车
    7: 'truck'         # 卡车
}

# List of class IDs to detect (easy to modify)
DETECTABLE_VEHICLE_IDS = list(VEHICLE_CLASSES.keys())  # [1, 2, 3, 5, 6, 7]

# Color histogram settings - Essential for movement detection
COLOR_CHANGE_THRESHOLD = 0.8   # Lower threshold for more sensitive movement detection
HISTOGRAM_BINS = 32            # More bins for better color analysis
HISTOGRAM_METHOD = cv2.HISTCMP_CORREL  # Histogram comparison method

# Occlusion handling - Essential for tracking continuity
OCCLUSION_TOLERANCE_FRAMES = 375  # 15 seconds at 25fps - adjusted for 25fps

# Tracking settings - Essential for stable tracking
TRACK_BUFFER = 30              # 1.2 seconds at 25fps - updated for better performance
TRACK_THRESH = 0.6             # Higher threshold for more stable tracking
MATCH_THRESH = 0.8             # Higher match threshold for better association

# ================================
# Additional tracking parameters - Required by the system
# ================================
# Additional tracking parameters - Required by the system
TRACK_STABILITY_BUFFER = 38    # 1.5 seconds at 25fps - adjusted for 25fps
TRACK_CONFIRMATION_FRAMES = 8  # More frames for confirmation
TRACK_REAPPEAR_THRESHOLD = 0.3 # Lower threshold for reappearance
TRACK_SIZE_SMOOTHING = 0.7     # Less smoothing for more responsive tracking

# Position memory parameters - Required for ID recovery
POSITION_MEMORY_SIZE = 1000      # 2.5 seconds at 25fps - adjusted for 25fps
POSITION_RECOVERY_THRESHOLD = 0.7  # Higher threshold for position recovery
POSITION_SEARCH_RADIUS = 150   # Smaller search radius for precision

# Occlusion parameters - Required for occlusion handling
OCCLUSION_RECOVERY_THRESHOLD = 0.4  # Higher threshold for occlusion recovery
LONG_OCCLUSION_THRESHOLD = 167  # 6.7 seconds at 25fps - adjusted for 25fps

# Color histogram parameters - Required for movement detection
COLOR_CHANGE_SMOOTHING = 0.3   # Less smoothing for more responsive detection

# ByteTracker parameters - Single optimized configuration
TRACK_HIGH_THRESH = 0.6       # High confidence threshold for track confirmation
TRACK_LOW_THRESH = 0.1        # Low confidence threshold for track maintenance
NEW_TRACK_THRESH = 0.7        # Threshold for creating new tracks
TRACK_BUFFER_BYTETRACKER = 30 # Track buffer for ByteTracker
MATCH_THRESH_BYTETRACKER = 0.8 # Match threshold for ByteTracker


class TrackerArgs:
    """ByteTracker configuration class with optimized parameters"""
    def __init__(self, track_thresh=None, track_buffer=None, 
                 match_thresh=None, mot20=False):
        # Core ByteTracker parameters
        self.track_thresh = track_thresh or TRACK_HIGH_THRESH
        self.track_buffer = track_buffer or TRACK_BUFFER_BYTETRACKER
        self.match_thresh = match_thresh or MATCH_THRESH_BYTETRACKER
        self.mot20 = mot20
        
        # Additional required parameters
        self.track_low_thresh = TRACK_LOW_THRESH
        self.new_track_thresh = NEW_TRACK_THRESH
        self.fuse_score = False
        
        # Required by ByteTracker demo code
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10


def parse_arguments():
    """Parse command line arguments for car detection"""
    parser = argparse.ArgumentParser(description="Car Detection and Parking Timer System")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="YOLO model file")
    parser.add_argument("--input", type=str, required=True, help="Input video file path")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--vis", action="store_true", help="Generate illegal parking visualization images (no window display)")

    
    return parser.parse_args()


def print_configuration():
    """Print current configuration settings"""
    print(f"================Configuration===================")
    print(f"Core Parking Settings (25 FPS):")
    print(f"  Allowed parking time: {ALLOWED_PARKING_TIME} seconds")
    print(f"  Parking threshold: {PARKING_THRESHOLD} frames (1 second at 25fps)")
    print(f"  Movement detection threshold: {MOVEMENT_DETECTION_THRESHOLD} pixels")
    print(f"Detection Settings (SENSITIVE MODE):")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD} (Lower for more detection)")
    print(f"  IoU threshold: {IOU_THRESHOLD} (Optimized for overlap)")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"Vehicle Detection Classes:")
    print(f"  Detectable vehicle types: {len(VEHICLE_CLASSES)} types")
    for class_id, class_name in VEHICLE_CLASSES.items():
        print(f"    {class_id}: {class_name}")
    print(f"  Class IDs to detect: {DETECTABLE_VEHICLE_IDS}")
    print(f"Movement Detection:")
    print(f"  Color change threshold: {COLOR_CHANGE_THRESHOLD}")
    print(f"  Color smoothing: {COLOR_CHANGE_SMOOTHING}")
    print(f"  Histogram bins: {HISTOGRAM_BINS}")
    print(f"Tracking Settings (OPTIMIZED MODE):")
    print(f"  Track buffer: {TRACK_BUFFER} frames")
    print(f"  Track threshold: {TRACK_THRESH} (Higher for more stable tracking)")
    print(f"  Match threshold: {MATCH_THRESH} (Higher for better association)")
    print(f"  Occlusion tolerance: {OCCLUSION_TOLERANCE_FRAMES} frames")
    print(f"ByteTracker Implementation (OPTIMIZED MODE):")
    print(f"  Using: Enhanced ByteTracker with optimized parameters")
    print(f"  Integration: YOLO built-in + Enhanced ByteTracker")
    print(f"  Persistence: Enabled (persist=True)")
    print(f"  Track high threshold: {TRACK_HIGH_THRESH} (Higher for stability)")
    print(f"  Track low threshold: {TRACK_LOW_THRESH} (Lower for sensitivity)")
    print(f"  New track threshold: {NEW_TRACK_THRESH} (Higher for quality)")
    print(f"  Track buffer: {TRACK_BUFFER_BYTETRACKER} frames (Optimized)")
    print(f"  Match threshold: {MATCH_THRESH_BYTETRACKER} (Higher for accuracy)")
    print(f"================================================") 