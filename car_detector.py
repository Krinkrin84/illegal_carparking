"""
Car Detection using YOLO with ByteTracker

Example usage:
    python car_detector.py --input videos/carpark2.mp4 --output ./output/videos/2.mp4

To generate illegal parking visualization images:
    python car_detector.py --input video.mp4 --vis

Note: Tracking and parking timer are always enabled and required for car detection.
Note: --vis flag generates images without displaying windows.
Note: Only single video file processing is supported.
"""

from config import *
import cv2
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from tqdm import tqdm
from timer import ParkingTimer

# Use YOLO's built-in tracking
print("✓ Using YOLO built-in tracking")

class CarDetector:
    def __init__(self, model_path, conf_threshold=None, iou_threshold=None, 
                 parking_threshold=None, track_buffer=None):
        """Initialize car detector with YOLO model and built-in ByteTracker"""
        self.model_path = model_path
        
        # Core parking system configuration (from global variables)
        self.color_threshold = COLOR_CHANGE_THRESHOLD
        self.color_smoothing = COLOR_CHANGE_SMOOTHING
        self.allowed_parking_time = ALLOWED_PARKING_TIME
        self.parking_threshold = parking_threshold or PARKING_THRESHOLD
        self.occlusion_tolerance_frames = OCCLUSION_TOLERANCE_FRAMES
        self.movement_detection_threshold = MOVEMENT_DETECTION_THRESHOLD
        
        # Other configuration (from global variables)
        self.conf_threshold = conf_threshold or CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or IOU_THRESHOLD
        
        # Tracking stability parameters
        self.track_reappear_threshold = TRACK_REAPPEAR_THRESHOLD
        
        # Color histogram parameters (from global variables)
        self.histogram_bins = HISTOGRAM_BINS
        self.histogram_method = HISTOGRAM_METHOD
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # YOLO model with built-in tracking (no separate tracker needed)
        self.track_buffer = track_buffer or TRACK_BUFFER
        
        # Parking timer system
        self.parking_timer = ParkingTimer(
            allowed_parking_time=ALLOWED_PARKING_TIME,
            parking_threshold=parking_threshold or PARKING_THRESHOLD,
            occlusion_tolerance_frames=OCCLUSION_TOLERANCE_FRAMES,
            detector=self
        )
        self.frame_count = 0
        self.last_detections = []
        
        # Position memory system removed - simplified system
        
        # ID continuity protection system removed - simplified system
        
        # Bounding box overlap removal system
        self.overlap_removal_enabled = True
        self.overlap_iou_threshold = 0.3
        self.overlap_confidence_priority = True
        
        # FPS counter for inference speed measurement
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.fps_measurement_active = False
        
        # Property accessors for backward compatibility
        @property
        def car_timers(self):
            return self.parking_timer.car_timers
        
        @property
        def illegal_parking_violations(self):
            return self.parking_timer.illegal_parking_violations
        
        self.load_model()
    
    def load_model(self):
        """Load YOLO model from file"""
        print(f"Loading model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Tracking: YOLO built-in tracking")
        print(f"Parking timer: On")
        print(f"Color histogram: Enabled")
        print(f"Fixed parking histogram: Enabled (uses parking start position)")
        print(f"Track reappear threshold: {self.track_reappear_threshold:.2f}")
        print(f"Occlusion tolerance: {self.occlusion_tolerance_frames} frames (2x for illegal cars)")
        print(f"================================================")
        try:
            # Try to load model with PyTorch compatibility fix
            import torch
            import os
            
            # Set environment variable to allow unsafe loading for trusted models
            os.environ['TORCH_WEIGHTS_ONLY'] = '0'
            
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try alternative loading method
            try:
                print("Trying alternative loading method...")
                import torch
                # Use weights_only=False for older model formats
                torch.hub._load_local = lambda *args, **kwargs: torch.load(*args, weights_only=False, **kwargs)
                self.model = YOLO(self.model_path)
                print("Model loaded successfully with weights_only=False!")
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                raise
    
    def calculate_color_histogram(self, image, bbox):
        """Calculate color histogram for a bounding box region"""
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Extract the region of interest
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # Convert to HSV color space for better color representation
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for H and S channels (ignore V for lighting invariance)
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [HISTOGRAM_BINS, HISTOGRAM_BINS], [0, 180, 0, 256])
        
        # Normalize histogram
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        return hist
    
    def compare_histograms(self, hist1, hist2):
        """Compare two histograms and return similarity score"""
        if hist1 is None or hist2 is None:
            return 0.0
        
        # Compare histograms using the specified method
        similarity = cv2.compareHist(hist1, hist2, HISTOGRAM_METHOD)
        
        # For correlation method, values range from -1 to 1, where 1 is perfect match
        # For other methods, higher values indicate more similarity
        if HISTOGRAM_METHOD == cv2.HISTCMP_CORREL:
            return max(0, similarity)  # Convert to 0-1 range
        else:
            return similarity
    

    
    def get_color_similarity_score(self, track_id, image, bbox):
        """Get color similarity score for a track"""
        return self.parking_timer.get_color_similarity_score(track_id, image, bbox)
    
    def update_car_timer(self, track_id, bbox, timestamp, image=None):
        """Update car's parking timer with illegal parking detection"""
        self.parking_timer.update_car_timer(track_id, bbox, timestamp, image, self.frame_count)
    
    def update_all_timers_continuously(self):
        """On every frame, update the timers for all parked cars."""
        self.parking_timer.update_all_timers_continuously()
    
    def get_parking_time_str(self, track_id):
        """Get formatted parking time string for a car"""
        return self.parking_timer.get_parking_time_str(track_id)
    
    def get_parking_status_str(self, track_id):
        """Get parking status string including illegal parking"""
        return self.parking_timer.get_parking_status_str(track_id)
    

    
    def cleanup_old_timers(self, current_track_ids):
        """Remove timers for cars that are no longer visible, with occlusion tolerance"""
        self.parking_timer.cleanup_old_timers(current_track_ids)
        
        # Clean up position memory
        # Position memory cleanup removed
    
    def detect_cars(self, image):
        """Detect all vehicles (cars, trucks, buses, motorcycles, etc.) in image with enhanced tracking and parking timer integration"""
        detections = []
        current_track_ids = []
        
        try:
            # Run YOLO detection with tracking
            results = self.model.track(image, persist=True, conf=self.conf_threshold, 
                                     iou=self.iou_threshold, verbose=False)
            
            # Validate tracking alignment
            self._validate_tracking_alignment(results)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                car_boxes = results[0].boxes.xyxy.cpu().numpy()
                car_scores = results[0].boxes.conf.cpu().numpy()
                car_classes = results[0].boxes.cls.cpu().numpy()
                car_track_ids = results[0].boxes.id.cpu().numpy()
                
                # Process YOLO detections
                for i, (bbox, score, cls, track_id) in enumerate(zip(car_boxes, car_scores, car_classes, car_track_ids)):
                    # Use configuration vehicle class list
                    if cls in DETECTABLE_VEHICLE_IDS:
                        bbox_int = [int(coord) for coord in bbox]
                        original_track_id = int(track_id)
                        
                        # ID continuity protection
                        protected_track_id = self._protect_id_continuity(original_track_id, bbox_int, image)
                        
                        # Check if vehicle already exists in parking system
                        existing_car_id = self.find_existing_car_by_position(bbox_int, image)
                        
                        if existing_car_id is not None:
                            # Use existing ID from parking system
                            final_track_id = existing_car_id
                        else:
                            # Simplified ID handling - no recovery needed
                            final_track_id = protected_track_id
                        
                        # Update position memory for the final track ID
                        # Position memory update removed
                        
                        # Update car timer with the final track ID
                        timestamp = datetime.now()
                        self.update_car_timer(final_track_id, bbox_int, timestamp, image)
                        
                        # Add to detections
                        detections.append({
                            'bbox': bbox_int,
                            'confidence': float(score),
                            'track_id': final_track_id
                        })
                        
                        current_track_ids.append(final_track_id)
            
            # Remove overlapping bounding boxes
            if self.overlap_removal_enabled:
                original_count = len(detections)
                detections = self.remove_overlapping_bboxes(detections)
                if len(detections) < original_count:
                    print(f"🗑️  Overlap removal: {original_count} → {len(detections)} detections")
            
            # Check for ID duplicates
            if len(detections) > 1:
                original_count = len(detections)
                detections = self.check_frame_id_duplicates(detections, image)
                if len(detections) < original_count:
                    print(f"🔍 Duplicate removal: {original_count} → {len(detections)} detections")
            
            # Check if any parked cars are about to become illegal (only when they're close to time limit)
            if hasattr(self, 'parking_timer') and self.parking_timer is not None:
                for track_id, car_data in self.parking_timer.car_timers.items():
                    if car_data.get('is_parked', False) and not car_data.get('is_illegal', False):
                        # Only check IOU when car is close to becoming illegal (80% of allowed time)
                        if self.parking_timer.is_car_close_to_illegal_limit(track_id):
                            # Now check IOU overlap to confirm car is still in position
                            if self.is_car_about_to_become_illegal(track_id, results):
                                print(f"🚨 Car #{track_id}: Conditions met for illegal parking mode activation")
            
            self.frame_count += 1
            self.last_detections = detections
            return detections
            
        except Exception as e:
            print(f"❌ Error in detect_cars: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def find_existing_car_by_position(self, new_bbox, image):
        """
        Find existing car in car_data by comparing position and HSV color
        
        Args:
            new_bbox: New bounding box from YOLO
            image: Current image
            
        Returns:
            track_id if match found, None otherwise
        """
        if not hasattr(self, 'parking_timer') or self.parking_timer is None:
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        for track_id, car_data in self.parking_timer.car_timers.items():
            # Match ALL cars in car_data, not just parked ones
            # This ensures we don't lose any vehicles
            
            # Get last known position
            last_position = car_data.get('last_position')
            if last_position is None:
                continue
            
            # Calculate distance between new bbox and last position
            distance = self.calculate_bbox_distance(new_bbox, last_position)
            
            # If distance is too large, skip this car
            if distance > 200:  # 200px threshold for position matching
                continue
            
            # Calculate color similarity using HSV histogram
            color_similarity = 1.0
            if image is not None:
                # Use parking histogram if available, otherwise calculate new one
                if car_data.get('parking_histogram') is not None:
                    # For parked cars, use the locked parking position for HSV comparison
                    if car_data.get('is_parked', False) and car_data.get('parking_histogram_bbox') is not None:
                        # Use the locked parking position
                        parking_bbox = car_data['parking_histogram_bbox']
                        new_hist = self.calculate_color_histogram(image, parking_bbox)
                        if new_hist is not None:
                            color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                    else:
                        # For non-parked cars, use current bbox
                        new_hist = self.calculate_color_histogram(image, new_bbox)
                        if new_hist is not None:
                            color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                else:
                    # Fallback: compare with last position histogram
                    last_hist = self.calculate_color_histogram(image, last_position)
                    new_hist = self.calculate_color_histogram(image, new_bbox)
                    if last_hist is not None and new_hist is not None:
                        color_similarity = self.compare_histograms(last_hist, new_hist)
            
            # Calculate overall similarity (position + color)
            position_similarity = max(0, 1.0 - distance / 200.0)  # Normalize distance to 0-1
            overall_similarity = (position_similarity * 0.6) + (color_similarity * 0.4)  # Weighted combination
            
            # Update best match if this is better
            if overall_similarity > best_similarity and overall_similarity > 0.5:  # Minimum threshold
                best_similarity = overall_similarity
                best_match_id = track_id
        
        return best_match_id
    
    def _validate_tracking_alignment(self, yolo_results):
        """Validate alignment between YOLO detection and tracking results"""
        if len(yolo_results) == 0:
            return
        
        result = yolo_results[0]
        if result.boxes is None or result.boxes.id is None:
            return
        
        # Validate array lengths match
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()
        
        # Check if arrays have the same length
        if len(boxes) != len(scores) or len(scores) != len(classes) or len(classes) != len(track_ids):
            print(f"❌ Array length mismatch: boxes={len(boxes)}, scores={len(scores)}, classes={len(classes)}, track_ids={len(track_ids)}")
            return
        
        # Count car detections
        car_count = np.sum(classes == 2)
        unique_track_ids = len(np.unique(track_ids))
        
        # Check for potential tracking issues
        duplicate_tracks = 0
        invalid_tracks = 0
        
        if car_count > 0:
            # Check for potential tracking issues (duplicate/invalid IDs)
            duplicate_tracks = len(track_ids) - len(np.unique(track_ids))
            invalid_tracks = np.sum(track_ids <= 0)
        
        if duplicate_tracks > 0:
            print(f"⚠️  Found {duplicate_tracks} duplicate track IDs")
        
        if invalid_tracks > 0:
            print(f"⚠️  Found {invalid_tracks} invalid track IDs (≤0)")
    
    def is_car_about_to_become_illegal(self, track_id, current_yolo_results):
        """
        Check if a car is about to change to illegal mode in the next frame
        by checking if any YOLO detection bounding box has > 0.5 IOU with the car's locked position
        
        This function works in conjunction with timer.py's check_car_about_to_become_illegal method.
        First, this function checks IOU overlap, then if HSV comparison passes, the timer method
        determines if the car should actually become illegal.
        
        Args:
            track_id: The track ID of the car to check
            current_yolo_results: Current YOLO detection results
            
        Returns:
            bool: True if car should become illegal parking mode, False otherwise
        """
        if not hasattr(self, 'parking_timer') or self.parking_timer is None:
            return False
        
        if track_id not in self.parking_timer.car_timers:
            return False
        
        car_data = self.parking_timer.car_timers[track_id]
        
        # Check if car has a locked parking position
        if not car_data.get('is_parked', False) or car_data.get('parking_histogram_bbox') is None:
            return False
        
        # Get the locked parking position bbox
        locked_bbox = car_data['parking_histogram_bbox']
        
        # Check if current YOLO results have any detection overlapping with locked position
        if len(current_yolo_results) == 0:
            return False
        
        result = current_yolo_results[0]
        if result.boxes is None:
            return False
        
        # Get all current YOLO detections
        yolo_boxes = result.boxes.xyxy.cpu().numpy()
        yolo_classes = result.boxes.cls.cpu().numpy()
        
        # Check each YOLO detection for overlap with locked position
        for i, (bbox, cls) in enumerate(zip(yolo_boxes, yolo_classes)):
            # Only consider vehicle detections
            if cls in DETECTABLE_VEHICLE_IDS:
                bbox_int = [int(coord) for coord in bbox]
                
                # Calculate IOU between YOLO detection and locked position
                iou = self.calculate_iou(bbox_int, locked_bbox)
                
                # If IOU > 0.5, check with timer if car should become illegal
                if iou > 0.5:
                    print(f"🚨 Car #{track_id}: High IOU ({iou:.3f}) with locked position - checking timer conditions")
                    
                    # Now check with timer if car should actually become illegal
                    if self.parking_timer.check_car_about_to_become_illegal(track_id, current_yolo_results):
                        print(f"✅ Car #{track_id}: Timer conditions met - ready for illegal parking mode")
                        return True
                    else:
                        print(f"⚠️  Car #{track_id}: IOU threshold met but timer conditions not satisfied")
        
        return False
    
    def draw_detections(self, image, detections):
        """Draw detections on image with enhanced visualization"""
        result_image = image.copy()
        
        # Add rule text
        rule_text = "Parking Detection System - 5 min limit"
        cv2.putText(result_image, rule_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw YOLO bounding boxes for all detections
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            track_id = detection['track_id']
            
            x1, y1, x2, y2 = bbox
            
            # ===== HSV HISTOGRAM DISPLAY (COMMENT OUT WHEN NOT NEEDED) =====
            # Uncomment the following lines to show HSV histograms for each detection
            # if track_id >= 0:  # Only for tracked objects
            #     self.show_hsv_histogram(result_image, bbox, track_id)
            #     # Uncomment the next line to also show comparison with previous histogram
            #     # self.show_comparison_histograms(result_image, bbox, track_id)
            # ===== END HSV HISTOGRAM DISPLAY =====
            
            # Choose color based on parking status
            if isinstance(track_id, (int, str)) and track_id in self.parking_timer.car_timers:
                car_data = self.parking_timer.car_timers[track_id]
                if car_data['is_parked']:
                    if car_data['is_illegal']:
                        color = (0, 0, 255)  # Red for illegal parking
                    else:
                        color = (0, 255, 255)  # Yellow for legal parking
                else:
                    color = (0, 255, 0)  # Green for moving cars
            else:
                color = (0, 255, 0)  # Default green
            
            # Draw bounding box and labels (always show since parking spaces are removed)
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Create integrated label with timer
            if isinstance(track_id, (int, str)):
                label = f"Car #{track_id}: {confidence:.2f}"
            else:
                label = f"Car: {confidence:.2f}"
            
            # Add timer information to the same label
            if isinstance(track_id, (int, str)) and track_id in self.parking_timer.car_timers:
                car_data = self.parking_timer.car_timers[track_id]
                
                if car_data['is_parked']:
                    # Show parking status and timer
                    parking_status = self.get_parking_status_str(track_id)
                    label += f" | {parking_status}"
                else:
                    # Show detection method info
                    label += " | Detected"
            
            # Draw integrated label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image
    

    

    

    

    

    

    

    

    
    def process_video(self, video_path, output_path=None, show_progress=True):
        """Process video for car detection with progress tracking"""
        print(f"Processing video: {video_path}")
        
        # Start FPS measurement
        self.start_fps_measurement()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        
        if show_progress:
            pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection and processing
            self.last_detections = self.detect_cars(frame)
            self.update_all_timers_continuously()
            result_frame = self.draw_detections(frame, self.last_detections)
            
            if writer:
                writer.write(result_frame)
            
            self.last_processed_frame = frame.copy()
            frame_count += 1
            self.update_fps_counter()
            
            if show_progress:
                pbar.update(1)
        
        # Cleanup
        self.stop_fps_measurement()
        cap.release()
        if writer:
            writer.release()
        if show_progress:
            pbar.close()
        
        # Final statistics
        total_cars = len(self.last_detections) if self.last_detections else 0
        unique_tracks = len(self.parking_timer.car_timers) if hasattr(self, 'parking_timer') else 0
        
        # Display final illegal parking violations summary
        if hasattr(self, 'parking_timer') and self.parking_timer is not None:
            # Use the new method to get unique violations
            illegal_violations = self.parking_timer.get_unique_illegal_parking_violations()
            if illegal_violations:
                print("\n" + "="*50)
                print("🚨 FINAL ILLEGAL PARKING SUMMARY")
                print("="*50)
                print(f"{'Car ID':<8} {'Illegal Time (s)':<15}")
                print("-" * 50)
                for violation in illegal_violations:
                    car_id = violation['track_id']
                    duration = violation['parking_duration']
                    print(f"{car_id:<8} {duration:<15.1f}")
                print("="*50)
        
        print(f"Complete: {frame_count} frames, {total_cars} cars, {unique_tracks} tracks")
        if output_path:
            print(f"Saved to: {output_path}")
        
        return total_cars, unique_tracks

    

    

    

    

    

    

    

    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    

    


    def start_fps_measurement(self):
        """Start FPS measurement"""
        import time
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.fps_measurement_active = True
    
    def update_fps_counter(self):
        """Update FPS counter - call this for each processed frame"""
        if self.fps_measurement_active:
            self.fps_frame_count += 1
    
    def get_fps_stats(self):
        """Get current FPS statistics"""
        if self.fps_start_time is None:
            return None
        
        import time
        if self.fps_measurement_active:
            current_time = time.time()
        else:
            # If measurement stopped, use the stored end time
            current_time = getattr(self, 'fps_end_time', time.time())
        
        elapsed_time = current_time - self.fps_start_time
        
        if elapsed_time > 0 and self.fps_frame_count > 0:
            fps = self.fps_frame_count / elapsed_time
            return {
                'fps': fps,
                'total_frames': self.fps_frame_count,
                'elapsed_time': elapsed_time,
                'avg_frame_time': elapsed_time / self.fps_frame_count
            }
        return None
    
    def stop_fps_measurement(self):
        """Stop FPS measurement"""
        import time
        self.fps_end_time = time.time()
        self.fps_measurement_active = False



    # update_position_memory function removed - simplified system
    
    def find_track_by_position(self, bbox, search_radius=None):
        """Find a track by its position in memory"""
        if search_radius is None:
            search_radius = self.position_search_radius
        
        current_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        current_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        best_match = None
        best_similarity = 0
        
        for track_id, position_history in self.position_memory.items():
            if not position_history:
                continue
            
            # Get the most recent position
            latest_position = position_history[-1]
            latest_center = latest_position['center']
            latest_area = latest_position['area']
            
            # Calculate distance
            distance = ((current_center[0] - latest_center[0]) ** 2 + 
                       (current_center[1] - latest_center[1]) ** 2) ** 0.5
            
            # Calculate area similarity
            area_similarity = min(current_area, latest_area) / max(current_area, latest_area) if max(current_area, latest_area) > 0 else 0
            
            # Check if within search radius
            if distance <= search_radius:
                # Calculate overall similarity
                position_similarity = max(0, 1 - distance / search_radius)
                total_similarity = position_similarity * 0.7 + area_similarity * 0.3
                
                if total_similarity > best_similarity and total_similarity >= self.position_recovery_threshold:
                    best_similarity = total_similarity
                    best_match = {
                        'track_id': track_id,
                        'similarity': total_similarity,
                        'distance': distance,
                        'area_similarity': area_similarity,
                        'position_similarity': position_similarity,
                        'last_seen': latest_position['frame']
                    }
        
        return best_match
    
    # recover_id_by_position function removed - simplified system
    
    # clean_position_memory function removed - simplified system
    

    

    

    

    

    
    def calculate_bbox_distance(self, bbox1, bbox2):
        """Calculate the distance between two bounding box centers"""
        if bbox1 is None or bbox2 is None:
            return float('inf')
        
        # Calculate centers
        center1 = [(bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2]
        center2 = [(bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2]
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance
    

    

    

    
    def reset_for_new_video(self):
        """Reset all tracking and ID data for a new video"""
        print("🔄 Resetting tracking data for new video...")
        
        # Reset frame counter
        self.frame_count = 0
        
        # Position memory and ID continuity protection removed - simplified system
        
        # Reset parking timer (this will reset car_timers and illegal_parking_violations)
        self.parking_timer.reset_for_new_video()
        
        # Reset FPS measurement
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.fps_measurement_active = False
        
        print("✅ Tracking data reset complete")
    

    

    

    

    def remove_overlapping_bboxes(self, detections):
        """
        Remove overlapping bounding boxes based on IOU threshold
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'confidence', 'track_id'
            
        Returns:
            List of detections with overlapping bboxes removed
        """
        if not self.overlap_removal_enabled or len(detections) <= 1:
            return detections
        
        # Sort detections by confidence (highest first) if confidence priority is enabled
        if self.overlap_confidence_priority:
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Keep track of which detections to remove
        to_remove = set()
        
        # Compare each pair of detections
        for i in range(len(detections)):
            if i in to_remove:
                continue
                
            bbox1 = detections[i]['bbox']
            
            for j in range(i + 1, len(detections)):
                if j in to_remove:
                    continue
                    
                bbox2 = detections[j]['bbox']
                
                # Calculate IOU between the two bounding boxes
                iou = self.calculate_iou(bbox1, bbox2)
                
                # If IOU exceeds threshold, mark the lower priority detection for removal
                if iou > self.overlap_iou_threshold:
                    if self.overlap_confidence_priority:
                        # Keep higher confidence, remove lower confidence
                        if detections[i]['confidence'] >= detections[j]['confidence']:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
                            break  # Current i is removed, move to next i
        
        # Remove marked detections
        filtered_detections = [det for idx, det in enumerate(detections) if idx not in to_remove]
        
        if len(to_remove) > 0:
            print(f"🗑️  Removed {len(to_remove)} overlapping detections")
        
        return filtered_detections
    
    def calculate_bbox_area(self, bbox):
        """Calculate area of a bounding box"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    


    def create_illegal_parking_visualization(self, width=1920, height=1080):
        """Create a simple visualization image for illegal parking"""
        # Create white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add title
        title = "Illegal Parking Visualization"
        cv2.putText(image, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        return image

    def draw_car_on_visualization(self, image, car_id, bbox, status, parking_time=None):
        """Draw a simple car representation on the visualization image"""
        x1, y1, x2, y2 = bbox
        
        # Choose color based on status
        if status == "illegal":
            color = (0, 0, 255)  # Red
        elif status == "legal":
            color = (0, 255, 255)  # Yellow
        elif status == "moving":
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue
        
        # Draw simple rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Add car ID
        cv2.putText(image, f"#{car_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image

    def visualize_illegal_parking(self, save_path="illegal_parking_visualization.jpg", show_window=False):
        """Generate simple illegal parking visualization"""
        if hasattr(self, 'parking_timer') and self.parking_timer is not None:
            # Use the new method to get unique violations
            illegal_violations = self.parking_timer.get_unique_illegal_parking_violations()
            car_timers = self.parking_timer.car_timers
            
            print(f"🔍 Found {len(illegal_violations)} unique illegal parking violations")
            print(f"🔍 Available car timers: {len(car_timers)}")
            
            # Display illegal parking violations in table format
            if illegal_violations:
                print("\n" + "="*50)
                print("🚨 ILLEGAL PARKING VIOLATIONS")
                print("="*50)
                print(f"{'Car ID':<8} {'Illegal Time (s)':<15}")
                print("-" * 50)
                for violation in illegal_violations:
                    car_id = violation['track_id']
                    duration = violation['parking_duration']
                    print(f"{car_id:<8} {duration:<15.1f}")
                print("="*50 + "\n")
                
            if not illegal_violations:
                print("✅ No illegal parking violations found")
                return None
            
            # Check if we have last processed frame
            if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
                print(f"📸 Last processed frame available: {self.last_processed_frame.shape}")
            else:
                print("⚠️  No last processed frame available")
            
            # Create simple visualization
            vis_image = self.create_illegal_cars_visualization(illegal_violations, car_timers)
            
            if vis_image is not None:
                cv2.imwrite(save_path, vis_image)
                print(f"💾 Visualization saved to: {save_path}")
            else:
                print("❌ Failed to create visualization")
            
            return vis_image
        else:
            print("❌ No parking timer data available")
            return None

    def create_illegal_cars_visualization(self, illegal_violations, car_timers):
        """Create enhanced visualization showing illegal cars with actual parking images"""
        if not illegal_violations:
            return None
        
        # Create enhanced layout with actual parking images
        num_cars = len(illegal_violations)
        cols = min(3, num_cars)  # Max 3 columns for better layout
        rows = (num_cars + cols - 1) // cols
        
        # Enhanced dimensions for better image display
        car_width = 350  # Width for car images
        car_height = 250  # Height for car images
        margin = 30
        title_height = 80
        
        # Calculate total size
        total_width = cols * car_width + (cols + 1) * margin
        total_height = title_height + rows * car_height + (rows + 1) * margin
        
        # Create background
        vis_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Add title
        title = f"Illegal Parking Violations: {num_cars} cars"
        cv2.putText(vis_image, title, (margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Add subtitle with timestamp
        subtitle = f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(vis_image, subtitle, (margin, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Add enhanced car representations with actual parking images
        for i, violation in enumerate(illegal_violations):
            car_id = violation['track_id']
            duration = violation['parking_duration']
            
            # Calculate position
            row = i // cols
            col = i % cols
            x_start = margin + col * (car_width + margin)
            y_start = title_height + row * (car_height + margin)
            
            # Try to get the actual car image from the last processed frame
            car_image = None
            if hasattr(self, 'last_processed_frame') and self.last_processed_frame is not None:
                # Get car bbox from car_timers if available
                if car_id in car_timers and 'last_position' in car_timers[car_id]:
                    bbox = car_timers[car_id]['last_position']
                    if bbox is not None:
                        try:
                            # Crop the car from the last processed frame
                            x1, y1, x2, y2 = map(int, bbox)
                            h, w = self.last_processed_frame.shape[:2]
                            
                            print(f"🔍 Car #{car_id}: bbox={bbox}, frame_size={w}x{h}")
                            
                            # Ensure coordinates are within bounds
                            x1 = max(0, min(x1, w-1))
                            y1 = max(0, min(y1, h-1))
                            x2 = max(0, min(x2, w-1))
                            y2 = max(0, min(y2, h-1))
                            
                            if x2 > x1 and y2 > y1:
                                car_image = self.last_processed_frame[y1:y2, x1:x2].copy()
                            else:
                                pass  # Invalid bbox after bounds check
                        except Exception as e:
                            pass  # Error cropping car image
                    else:
                        pass  # No bbox available
                else:
                    pass  # Not found in car_timers or no last_position
            else:
                pass  # No last_processed_frame available
            
            # Use the actual car image if available
            if car_image is not None:
                try:
                    # Resize the car image to fit the allocated space
                    resized_image = cv2.resize(car_image, (car_width, car_height))
                    
                    # Place the resized image
                    vis_image[y_start:y_start + car_height, x_start:x_start + car_width] = resized_image
                    
                    # Add red border to indicate illegal parking
                    cv2.rectangle(vis_image, (x_start, y_start), 
                                 (x_start + car_width, y_start + car_height), (0, 0, 255), 2)
                    
                    # Add information overlay
                    info_bg = np.ones((50, car_width, 3), dtype=np.uint8) * 255
                    vis_image[y_start + car_height - 50:y_start + car_height, x_start:x_start + car_width] = info_bg
                    
                    # Add car ID and duration
                    cv2.putText(vis_image, f"Car #{car_id}", (x_start + 10, y_start + car_height - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(vis_image, f"Duration: {duration:.1f}s", (x_start + 10, y_start + car_height - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except Exception as e:
                    # Fallback to simple box
                    self._draw_simple_car_box(vis_image, x_start, y_start, car_width, car_height, car_id, duration)
            else:
                # Fallback: Draw simple car box
                self._draw_simple_car_box(vis_image, x_start, y_start, car_width, car_height, car_id, duration)
        
        return vis_image
    
    def _draw_simple_car_box(self, vis_image, x_start, y_start, car_width, car_height, car_id, duration):
        """Helper method to draw simple car representation when image is not available"""
        # Draw simple car box representation
        cv2.rectangle(vis_image, (x_start, y_start), 
                     (x_start + car_width, y_start + car_height), (0, 0, 255), 2)
        
        # Add car ID
        cv2.putText(vis_image, f"Car #{car_id}", (x_start + 10, y_start + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Add duration
        cv2.putText(vis_image, f"Duration: {duration:.1f}s", (x_start + 10, y_start + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def check_frame_id_duplicates(self, current_detections, image):
        """
        Check for ID duplicates in the same position using HSV comparison and position locking
        
        This function prevents multiple IDs from occupying the same position by:
        1. Using HSV histogram comparison to identify duplicate vehicles
        2. Leveraging position locking after parking mode
        3. Removing the ID with greater number when duplicates are detected
        
        Args:
            current_detections: List of current detections with bbox, confidence, track_id
            image: Current frame image for HSV calculation
            
        Returns:
            List of cleaned detections with duplicates removed
        """
        if not current_detections or len(current_detections) < 2:
            return current_detections
        
        # Enhanced ID duplicate detection
        track_id_counts = {}
        for detection in current_detections:
            track_id = detection['track_id']
            if track_id not in track_id_counts:
                track_id_counts[track_id] = []
            track_id_counts[track_id].append(detection)
        
        # Process duplicate detections with same track_id
        duplicates_to_remove = set()
        for track_id, detections_with_same_id in track_id_counts.items():
            if len(detections_with_same_id) > 1:
                # Sort by confidence, keep highest
                sorted_detections = sorted(detections_with_same_id, key=lambda x: x['confidence'], reverse=True)
                
                # Keep first (highest confidence), mark others as duplicate
                for i in range(1, len(sorted_detections)):
                    duplicates_to_remove.add(id(sorted_detections[i]))
        
        # Check position overlaps
        position_duplicates = self._check_position_overlaps(current_detections, image)
        duplicates_to_remove.update(position_duplicates)
        
        # Filter out duplicate detections
        final_cleaned_detections = []
        removed_count = 0
        
        for detection in current_detections:
            if id(detection) not in duplicates_to_remove:
                final_cleaned_detections.append(detection)
            else:
                removed_count += 1
        
        if removed_count > 0:
            print(f"🔍 Removed {removed_count} duplicate detections")
        
        return final_cleaned_detections
    
    def _check_position_overlaps(self, detections, image):
        """Check position overlaps and return IDs to remove"""
        duplicates_to_remove = set()
        
        # Check each pair of detections
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                detection1 = detections[i]
                detection2 = detections[j]
                
                # Calculate IoU
                iou = self._calculate_iou(detection1['bbox'], detection2['bbox'])
                
                # If IoU > 0.3, consider as overlap
                if iou > 0.3:
                    # Keep higher confidence, remove lower confidence
                    if detection1['confidence'] >= detection2['confidence']:
                        duplicates_to_remove.add(id(detection2))
                    else:
                        duplicates_to_remove.add(id(detection1))
        
        return duplicates_to_remove
    
    def _calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的IoU"""
        # 计算交集
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # 计算并集
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    

    


    # _recover_occluded_vehicle_id function removed - simplified system
            
    def _recover_by_position_memory(self, bbox, image):
        """Recover vehicle ID by checking position memory"""
        if not hasattr(self, 'position_memory'):
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        for track_id, position_history in self.position_memory.items():
            if not position_history or track_id not in self.parking_timer.car_timers:
                continue
            
            # Get the most recent position
            last_position_data = position_history[-1] if position_history else None
            if last_position_data is None:
                continue
            
            # Extract bbox from position data (it's stored as a dictionary)
            last_position = last_position_data.get('bbox', None)
            if last_position is None:
                continue
            
            # Calculate position similarity
            distance = self.calculate_bbox_distance(bbox, last_position)
            if distance <= 200:  # 200px threshold for position matching
                # Check if this car was recently detected (within last 30 frames)
                car_data = self.parking_timer.car_timers[track_id]
                last_seen = car_data.get('last_seen_frame', 0)
                frames_since_last_seen = self.frame_count - last_seen
                
                if frames_since_last_seen <= 30:  # Recently seen
                    position_similarity = max(0, 1.0 - distance / 200.0)
                    
                    # Check color similarity if available
                    color_similarity = 1.0
                    if image is not None and car_data.get('parking_histogram') is not None:
                        # For parked cars, use the locked parking position for HSV comparison
                        if car_data.get('is_parked', False) and car_data.get('parking_histogram_bbox') is not None:
                            # Use the locked parking position
                            parking_bbox = car_data['parking_histogram_bbox']
                            new_hist = self.calculate_color_histogram(image, parking_bbox)
                            if new_hist is not None:
                                color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                        else:
                            # For non-parked cars, use current bbox
                            new_hist = self.calculate_color_histogram(image, bbox)
                            if new_hist is not None:
                                color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                    
                    # Combined similarity score
                    overall_similarity = (position_similarity * 0.6) + (color_similarity * 0.4)
                    
                    if overall_similarity > best_similarity and overall_similarity > 0.7:
                        best_similarity = overall_similarity
                        best_match_id = track_id
        
        return best_match_id
    
    def _recover_by_color_similarity(self, bbox, image):
        """Recover vehicle ID by checking color similarity with recently occluded vehicles"""
        if not hasattr(self, 'parking_timer') or image is None:
            return None
        
        best_match_id = None
        best_similarity = 0.0
        
        for track_id, car_data in self.parking_timer.car_timers.items():
            # Check if this car was recently detected but is now missing
            last_seen = car_data.get('last_seen_frame', 0)
            frames_since_last_seen = self.frame_count - last_seen
            
            if frames_since_last_seen <= 50:  # Recently seen (within 2 seconds at 25fps)
                # Check color similarity
                if car_data.get('parking_histogram') is not None:
                    # For parked cars, use the locked parking position for HSV comparison
                    if car_data.get('is_parked', False) and car_data.get('parking_histogram_bbox') is not None:
                        # Use the locked parking position
                        parking_bbox = car_data['parking_histogram_bbox']
                        new_hist = self.calculate_color_histogram(image, parking_bbox)
                        if new_hist is not None:
                            color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                            
                            if color_similarity > best_similarity and color_similarity > 0.8:
                                best_similarity = color_similarity
                                best_match_id = track_id
                    else:
                        # For non-parked cars, use current bbox
                        new_hist = self.calculate_color_histogram(image, bbox)
                        if new_hist is not None:
                            color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                            
                            if color_similarity > best_similarity and color_similarity > 0.8:
                                best_similarity = color_similarity
                                best_match_id = track_id
        
        return best_match_id
    
    def _recover_by_temporal_proximity(self, bbox, image):
        """Recover vehicle ID by checking temporal proximity and movement patterns"""
        if not hasattr(self, 'parking_timer'):
            return None
        
        best_match_id = None
        best_score = 0.0
        
        for track_id, car_data in self.parking_timer.car_timers.items():
            # Check if this car was recently detected
            last_seen = car_data.get('last_seen_frame', 0)
            frames_since_last_seen = self.frame_count - last_seen
            
            if frames_since_last_seen <= 75:  # Within 3 seconds at 25fps
                # Calculate recovery score based on multiple factors
                score = 0.0
                
                # Factor 1: Temporal proximity (more recent = higher score)
                temporal_score = max(0, 1.0 - frames_since_last_seen / 75.0)
                score += temporal_score * 0.4
                
                # Factor 2: Position similarity if available
                if hasattr(self, 'position_memory') and track_id in self.position_memory:
                    position_history = self.position_memory[track_id]
                    if position_history:
                        last_position_data = position_history[-1]
                        last_position = last_position_data.get('bbox', None)
                        if last_position is not None:
                            distance = self.calculate_bbox_distance(bbox, last_position)
                            if distance <= 300:  # 300px threshold
                                position_score = max(0, 1.0 - distance / 300.0)
                                score += position_score * 0.3
                
                # Factor 3: Color similarity if available
                if image is not None and car_data.get('parking_histogram') is not None:
                    # For parked cars, use the locked parking position for HSV comparison
                    if car_data.get('is_parked', False) and car_data.get('parking_histogram_bbox') is not None:
                        # Use the locked parking position
                        parking_bbox = car_data['parking_histogram_bbox']
                        new_hist = self.calculate_color_histogram(image, parking_bbox)
                        if new_hist is not None:
                            color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                            score += color_similarity * 0.3
                    else:
                        # For non-parked cars, use current bbox
                        new_hist = self.calculate_color_histogram(image, bbox)
                        if new_hist is not None:
                            color_similarity = self.compare_histograms(car_data['parking_histogram'], new_hist)
                            score += color_similarity * 0.3
                
                if score > best_score and score > 0.6:
                    best_score = score
                    best_match_id = track_id
        
        return best_match_id
    
    def _ensure_timer_state_preserved(self, recovered_id, bbox, image):
        """
        Ensure recovered ID exists in timer, maintaining original timer state
        
        Args:
            recovered_id: Recovered vehicle ID
            bbox: Current detection box
            image: Current image
        """
        if not hasattr(self, 'parking_timer') or self.parking_timer is None:
            return
        
        # Check if recovered ID exists in timer
        if recovered_id not in self.parking_timer.car_timers:
            print(f"⚠️  Recovered ID {recovered_id} not in timer, attempting to restore timer state")
            
            # Improved timer recovery logic
            recovery_success = False
            
            # Strategy 1: Restore from position memory
            if hasattr(self, 'position_memory') and recovered_id in self.position_memory:
                position_history = self.position_memory[recovered_id]
                if position_history:
                    recovery_success = self._restore_timer_from_position_memory(recovered_id, bbox, image, position_history)
            
            # Strategy 2: Restore from parking timer history (if position memory fails)
            if not recovery_success and hasattr(self, 'parking_timer'):
                recovery_success = self._restore_timer_from_parking_history(recovered_id, bbox, image)
            
            # Strategy 3: Create new timer entry (last resort)
            if not recovery_success:
                print(f"⚠️  Unable to restore timer state for ID {recovered_id}, creating new entry")
                timestamp = datetime.now()
                self.parking_timer.update_car_timer(recovered_id, bbox, timestamp, image, self.frame_count)
        else:
            print(f"✅ Recovered ID {recovered_id} timer state is normal")
    
    def _restore_timer_from_position_memory(self, recovered_id, bbox, image, position_history):
        """Restore timer state from position memory"""
        try:
            # Get last position data
            last_position_data = position_history[-1]
            last_bbox = last_position_data.get('bbox', None)
            last_timestamp = last_position_data.get('timestamp', None)
            
            if last_bbox is not None and last_timestamp is not None:
                # Calculate frames since last detection
                frames_since_last_seen = self.frame_count - last_position_data.get('frame', self.frame_count)
                
                # If time interval is not too long, try to maintain original state
                if frames_since_last_seen <= self.occlusion_tolerance_frames:
                    print(f"🔄 Restoring timer state for ID {recovered_id} from position memory")
                    
                    # Reinitialize timer
                    self.parking_timer.update_car_timer(recovered_id, bbox, last_timestamp, image, self.frame_count)
                    
                    # Try to restore parking state
                    if 'parking_start_time' in last_position_data:
                        print(f"✅ Successfully restored parking state for ID {recovered_id}")
                        return True
                    else:
                        print(f"⚠️  Parking state for ID {recovered_id} cannot be restored")
                else:
                    print(f"⚠️  Time interval too long ({frames_since_last_seen} frames), cannot restore state for ID {recovered_id}")
            else:
                print(f"⚠️  Position memory data incomplete, cannot restore state for ID {recovered_id}")
        except Exception as e:
            print(f"❌ Error restoring timer from position memory: {e}")
        
        return False
    
    def _restore_timer_from_parking_history(self, recovered_id, bbox, image):
        """Restore timer state from parking history"""
        try:
            # Check if there are other vehicle history data that can serve as reference
            if hasattr(self, 'parking_timer') and self.parking_timer is not None:
                # Find recently cleaned timer data
                for track_id, car_data in self.parking_timer.car_timers.items():
                    if track_id != recovered_id:
                        # Check if there are similar positions or features
                        if self._is_similar_to_existing_car(recovered_id, bbox, track_id, car_data):
                            print(f"🔄 Restoring timer state for ID {recovered_id} from similar vehicle {track_id}")
                            
                            # Create new timer entry but maintain some historical information
                            timestamp = datetime.now()
                            self.parking_timer.update_car_timer(recovered_id, bbox, timestamp, image, self.frame_count)
                            
                            return True
            
            print(f"⚠️  Unable to find similar vehicle in parking history to restore ID {recovered_id}")
        except Exception as e:
            print(f"❌ Error restoring timer from parking history: {e}")
        
        return False
    
    def _is_similar_to_existing_car(self, recovered_id, bbox, existing_id, existing_car_data):
        """Check if recovered vehicle is similar to existing vehicle"""
        try:
            # Check position similarity
            if 'last_position' in existing_car_data:
                existing_bbox = existing_car_data['last_position']
                distance = self.calculate_bbox_distance(bbox, existing_bbox)
                
                # If positions are close, consider them similar
                if distance <= 100:  # 100 pixel threshold
                    return True
            
            # Check temporal similarity (recently detected vehicles)
            last_seen = existing_car_data.get('last_detected_frame', 0)
            frames_since_last_seen = self.frame_count - last_seen
            
            # If recently detected, may be related vehicle
            if frames_since_last_seen <= 50:  # Within 2 seconds
                return True
                
        except Exception as e:
            print(f"❌ Error checking vehicle similarity: {e}")
        
        return False

    def _protect_id_continuity(self, original_track_id, bbox, image):
        """
        Simplified ID continuity protection - returns original track ID
        """
        return original_track_id
    
    # _allocate_continuous_id method removed - simplified system
    

    

    

    



def create_output_dirs():
    """Create output directories for results"""
    os.makedirs("./output/videos", exist_ok=True)

def process_single_file(detector, input_path, output_path, args):
    """Process a single file (video only)"""
    try:
        detector.process_video(str(input_path), str(output_path))
        

            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    """Main function to run car detection"""
    from config import parse_arguments
    args = parse_arguments()
    
    # Create output directories
    create_output_dirs()
    
    # Initialize detector
    detector = CarDetector(args.model)
    
    # Check input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path {args.input} not found!")
        return
    
    # Generate timestamp for both output and visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate output path
    if args.output is None:
        output_path = f"./output/videos/cars_{timestamp}_{input_path.name}"
    else:
        output_path = args.output
    
    # Process single file
    process_single_file(detector, input_path, output_path, args)
    
    # Generate visualization if requested (creates image files, no window display)
    if args.vis:
        # Save visualization in videos folder
        vis_path = f"./output/videos/visualization_{timestamp}_{input_path.stem}.jpg"
        
        # Generate enhanced visualization with actual parking images
        vis_image = detector.visualize_illegal_parking(vis_path)
        
        print(f"✅ Processing completed!")
        print(f"📁 Results saved in: {output_path}")

if __name__ == "__main__":
    main() 