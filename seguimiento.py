import numpy as np
from collections import deque
import logging
from typing import Dict, List, Tuple, Any, Set, Deque

# Import config and utility functions
from utiles import (
    CONFIG, calculate_iou, calculate_center, calculate_size, calculate_distance,
    calculate_overlap_area, is_box_contained, get_most_common_class, is_in_roi
)

logger = logging.getLogger(__name__)

# Load tracking parameters from config
TRACKING_CONFIG = CONFIG['tracking']
IOU_THRESHOLD = TRACKING_CONFIG['iou_threshold']
OVERLAP_THRESHOLD = TRACKING_CONFIG['overlap_threshold'] # Note: OVERLAP_THRESHOLD is defined but not used directly, overlap logic is in calculate_match_score and proximity checks
DISAPPEAR_THRESHOLD = TRACKING_CONFIG['disappear_threshold']
MIN_CONSECUTIVE_DETECTIONS = TRACKING_CONFIG['min_consecutive_detections']
MAX_PREDICTION_FRAMES = TRACKING_CONFIG['max_prediction_frames']
CLASS_HISTORY_SIZE = TRACKING_CONFIG['class_history_size']
COOLDOWN_FRAMES = TRACKING_CONFIG['cooldown_frames']

# Load tunable parameters from config
PARAMS = TRACKING_CONFIG['parameters']
SCORE_WEIGHTS = PARAMS['score_weights']
VELOCITY_ALPHA = PARAMS['velocity_alpha']
PREDICTION_MATCH_THRESHOLD = PARAMS['prediction_match_threshold']
PROXIMITY_THRESHOLDS = PARAMS['proximity_thresholds']
COUNTING_STABILITY_THRESHOLD = PARAMS['counting_stability_threshold']
MIN_CONSECUTIVE_ROI_FRAMES = PARAMS.get('min_consecutive_roi_frames', 1) # Load new param, default to 1
# Read the new parameter
CLASS_CONFIRMATION_FRAMES = PARAMS.get('class_confirmation_frames', 3) # Default to 3 if not in config

# Load model confidence settings
MODEL_CONFIG = CONFIG['model']
CONF_SETTINGS = MODEL_CONFIG['confidence']
DEFAULT_CONF = CONF_SETTINGS.get('default', 0.25) # Default confidence if not specified

def group_overlapping_detections(current_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Groups highly overlapping detections, keeping the one with highest confidence."""
    sorted_detections = sorted(enumerate(current_detections), key=lambda x: x[1]['confidence'], reverse=True)
    grouped_detections = []
    used_detection_indices = set()
    
    for i, detection1 in sorted_detections:
        if i in used_detection_indices:
            continue
            
        group = [detection1]
        used_detection_indices.add(i)
        
        for j, detection2 in enumerate(current_detections):
            if j in used_detection_indices or i == j:
                continue
                
            overlap = calculate_overlap_area(detection1['box'], detection2['box'])
            iou = calculate_iou(detection1['box'], detection2['box'])
            
            if (overlap > 0.9 or iou > 0.85 or 
                is_box_contained(detection1['box'], detection2['box'], threshold=0.95)):
                group.append(detection2)
                used_detection_indices.add(j)
                
        best_detection = max(group, key=lambda x: x['confidence'])
        grouped_detections.append(best_detection)
    
    return grouped_detections

def calculate_match_score(
    detection: Dict[str, Any],
    vehicle_data: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    processed_frames: int
) -> float:
    """Calculates a score indicating how well a detection matches a tracked vehicle."""
    det_center = calculate_center(detection['box'])
    det_size = calculate_size(detection['box'])
    veh_center = calculate_center(vehicle_data['box'])
    veh_size = calculate_size(vehicle_data['box'])
    
    iou = calculate_iou(vehicle_data['box'], detection['box'])
    area_overlap = calculate_overlap_area(vehicle_data['box'], detection['box'])
    distance = calculate_distance(det_center, veh_center)
    norm_distance = distance / max(frame_width, frame_height)
    size_ratio = min(det_size, veh_size) / max(det_size, veh_size)
    
    direction_score = 1.0
    if 'velocity' in vehicle_data and np.linalg.norm(vehicle_data['velocity']) > 1.0:
        dx = det_center[0] - veh_center[0]
        dy = det_center[1] - veh_center[1]
        current_direction = np.array([dx, dy])
        
        prev_direction = np.array(vehicle_data['velocity'])
        if np.linalg.norm(current_direction) > 0 and np.linalg.norm(prev_direction) > 0:
            current_direction = current_direction / np.linalg.norm(current_direction)
            prev_direction = prev_direction / np.linalg.norm(prev_direction)
            cos_sim = np.dot(current_direction, prev_direction)
            direction_score = (cos_sim + 1) / 2
    
    frames_since_seen = processed_frames - vehicle_data['last_seen']
    recency_factor = max(0, 1.0 - (frames_since_seen / DISAPPEAR_THRESHOLD))
    
    class_consistency = 1.0
    if 'class_history' in vehicle_data and len(vehicle_data['class_history']) > 1:
        unique_classes = len(set(vehicle_data['class_history']))
        class_consistency = 1.0 / unique_classes # Simple consistency measure

    # Use weights from config
    score = (iou * SCORE_WEIGHTS['iou']) + \
            (area_overlap * SCORE_WEIGHTS['area_overlap']) + \
            (size_ratio * SCORE_WEIGHTS['size_ratio']) + \
            ((1.0 - norm_distance) * SCORE_WEIGHTS['norm_distance']) + \
            (direction_score * SCORE_WEIGHTS['direction_score']) + \
            (recency_factor * SCORE_WEIGHTS['recency_factor'])

    if is_box_contained(vehicle_data['box'], detection['box']):
        score += SCORE_WEIGHTS['contained_bonus']

    return score

def update_tracked_vehicle(
    vehicle_data: Dict[str, Any],
    detection: Dict[str, Any],
    processed_frames: int,
    frame_width: int, # Add frame dimensions
    frame_height: int # Add frame dimensions
) -> Dict[str, Any]:
    """Updates the state of a tracked vehicle with a new matching detection."""
    old_center = calculate_center(vehicle_data['box'])
    new_center = calculate_center(detection['box'])
    
    if 'velocity' not in vehicle_data:
        vehicle_data['velocity'] = (0, 0)

    # Use alpha from config
    alpha = VELOCITY_ALPHA
    dx = new_center[0] - old_center[0]
    dy = new_center[1] - old_center[1]
    vehicle_data['velocity'] = (
        alpha * dx + (1-alpha) * vehicle_data['velocity'][0],
        alpha * dy + (1-alpha) * vehicle_data['velocity'][1]
    )
    
    if 'class_history' not in vehicle_data:
        vehicle_data['class_history'] = deque(maxlen=CLASS_HISTORY_SIZE)
    vehicle_data['class_history'].append(detection['class_id'])

    # Determine the majority class from history *before* applying locks
    current_majority_class = get_most_common_class(vehicle_data['class_history'])

    # --- Class Confirmation Logic ---
    previous_majority_class = vehicle_data.get('previous_majority_class', None)
    consecutive_class_frames = vehicle_data.get('consecutive_class_frames', 0)

    if current_majority_class == previous_majority_class:
        consecutive_class_frames += 1
    else:
        # Reset confirmation if majority class changes
        consecutive_class_frames = 1
        vehicle_data['confirmed_class_id'] = None # Reset confirmed class

    vehicle_data['previous_majority_class'] = current_majority_class
    vehicle_data['consecutive_class_frames'] = consecutive_class_frames

    # Confirm the class if it has been stable long enough
    if vehicle_data['confirmed_class_id'] is None and consecutive_class_frames >= CLASS_CONFIRMATION_FRAMES:
        vehicle_data['confirmed_class_id'] = current_majority_class
        logger.debug(f"Confirmed class for vehicle {vehicle_data['id']} as {current_majority_class} after {consecutive_class_frames} frames.")
    # --- End Class Confirmation Logic ---


    # --- Update Main Class ID (Respecting Confirmation and Count Lock) ---
    if not vehicle_data.get('is_counted', False):
        confirmed_class = vehicle_data.get('confirmed_class_id')
        initial_class_id = vehicle_data.get('initial_class_id')
        vehicle_classes_to_lock = {2, 7, 3} # Lock car, truck, motorcycle

        if confirmed_class is not None:
             # Prioritize confirmed class if available
             vehicle_data['class_id'] = confirmed_class
        elif initial_class_id is not None and initial_class_id in vehicle_classes_to_lock:
            # Fallback to initial lock for specific classes if not yet confirmed
            vehicle_data['class_id'] = initial_class_id
        else:
            # Fallback to current majority if no confirmation or initial lock applies
            vehicle_data['class_id'] = current_majority_class
    # --- End Update Main Class ID ---

    vehicle_data['box'] = detection['box']
    vehicle_data['last_seen'] = processed_frames
    vehicle_data['confidence'] = detection['confidence']
    vehicle_data['consecutive_matches'] = vehicle_data.get('consecutive_matches', 0) + 1
    vehicle_data['predicted'] = False
    vehicle_data['detection_stability'] = vehicle_data.get('detection_stability', 0) + 1

    # --- Update Consecutive ROI Frames --- 
    if is_in_roi(detection['box'], frame_width, frame_height):
        vehicle_data['consecutive_roi_frames'] = vehicle_data.get('consecutive_roi_frames', 0) + 1
    else:
        vehicle_data['consecutive_roi_frames'] = 0 # Reset if outside ROI
    # --- End Update --- 

    return vehicle_data

def match_with_predicted_positions(
    remaining_detections: List[Dict[str, Any]],
    tracked_vehicles: Dict[int, Dict[str, Any]],
    matched_vehicles: Set[int],
    processed_frames: int,
    frame_width: int,
    frame_height: int
) -> Tuple[List[Dict[str, Any]], Set[int]]:
    """Attempts to match remaining detections with predicted positions of lost tracks."""
    still_unmatched = []
    
    for detection in remaining_detections:
        det_center = calculate_center(detection['box'])
        potential_match = False
        
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            if vehicle_id in matched_vehicles:
                continue
                
            frames_not_seen = processed_frames - vehicle_data['last_seen']
            if frames_not_seen < MAX_PREDICTION_FRAMES and 'velocity' in vehicle_data:
                old_center = calculate_center(vehicle_data['box'])
                predicted_x = old_center[0] + vehicle_data['velocity'][0] * frames_not_seen
                predicted_y = old_center[1] + vehicle_data['velocity'][1] * frames_not_seen
                
                distance = calculate_distance(det_center, (predicted_x, predicted_y))
                normalized_dist = distance / max(frame_width, frame_height)
                
                det_size = calculate_size(detection['box'])
                veh_size = calculate_size(vehicle_data['box'])
                size_ratio = min(det_size, veh_size) / max(det_size, veh_size)

                # Use threshold from config
                match_quality = (1 - normalized_dist) * 0.7 + size_ratio * 0.3 # Keep this simple heuristic or make it configurable too?
                if match_quality > PREDICTION_MATCH_THRESHOLD:
                    if 'class_history' not in vehicle_data:
                        vehicle_data['class_history'] = deque(maxlen=CLASS_HISTORY_SIZE)
                    vehicle_data['class_history'].append(detection['class_id'])
                    
                    vehicle_data['box'] = detection['box']
                    vehicle_data['last_seen'] = processed_frames
                    vehicle_data['confidence'] = detection['confidence']
                    vehicle_data['predicted'] = False
                    vehicle_data['consecutive_matches'] = vehicle_data.get('consecutive_matches', 0) + 1
                    matched_vehicles.add(vehicle_id)
                    potential_match = True
                    break
        
        if not potential_match:
            still_unmatched.append(detection)
    
    return still_unmatched, matched_vehicles

def process_detections(
    current_detections: List[Dict[str, Any]],
    tracked_vehicles: Dict[int, Dict[str, Any]],
    processed_frames: int,
    unique_vehicle_counts: Dict[int, int],
    frame_width: int,
    frame_height: int
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, int]]:
    """Processes current frame detections to update tracked vehicles and count new ones."""
    matched_vehicles = set()
    next_vehicle_id = max(tracked_vehicles.keys()) + 1 if tracked_vehicles else 0

    # --- Filter detections based on class-specific confidence BEFORE grouping/matching ---
    filtered_detections = []
    for det in current_detections:
        class_id = det['class_id']
        confidence = det['confidence']
        # Determine the confidence threshold for this specific class
        class_conf_threshold = CONF_SETTINGS.get(class_id, DEFAULT_CONF)
        if confidence >= class_conf_threshold:
            filtered_detections.append(det)
        # else: # Optional: Log discarded detections
        #     logger.debug(f"Discarding detection (Class: {class_id}, Conf: {confidence:.2f}) below threshold {class_conf_threshold:.2f}")
    # ------------------------------------------------------------------------------------

    # Use the filtered list for grouping and matching
    grouped_detections = group_overlapping_detections(filtered_detections) # Use filtered list

    unmatched_detections = []

    # --- Calculate ROI boundaries once ---
    roi_enabled = CONFIG['roi']['enabled']
    roi_coords = CONFIG['roi']['coords']
    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, frame_width, frame_height # Default to full frame if ROI disabled
    if roi_enabled:
        roi_x1 = int(roi_coords[0] * frame_width)
        roi_y1 = int(roi_coords[1] * frame_height)
        roi_x2 = int(roi_coords[2] * frame_width)
        roi_y2 = int(roi_coords[3] * frame_height)
    # -------------------------------------

    for detection in grouped_detections:
        best_match = None
        best_score = IOU_THRESHOLD # Use IOU_THRESHOLD as minimum score for matching existing tracks

        for vehicle_id, vehicle_data in tracked_vehicles.items():
            # Only consider matching if the vehicle hasn't been matched already in this frame
            if vehicle_id in matched_vehicles:
                continue

            score = calculate_match_score(detection, vehicle_data, frame_width, frame_height, processed_frames)
            if score > best_score:
                best_score = score
                best_match = vehicle_id

        if best_match is not None:
            # Check if this detection is already assigned to another track (can happen with overlapping high scores)
            if best_match not in matched_vehicles:
                # --- Check for ROI entry from right or bottom --- 
                vehicle_data = tracked_vehicles[best_match]
                assign_new_id = False
                if roi_enabled:
                    prev_center = calculate_center(vehicle_data['box'])
                    new_center = calculate_center(detection['box'])

                    prev_in_roi = roi_x1 <= prev_center[0] <= roi_x2 and roi_y1 <= prev_center[1] <= roi_y2
                    new_in_roi = roi_x1 <= new_center[0] <= roi_x2 and roi_y1 <= new_center[1] <= roi_y2

                    if not prev_in_roi and new_in_roi:
                        entered_from_right = prev_center[0] >= roi_x2 # Check if previous X was beyond right edge
                        entered_from_bottom = prev_center[1] >= roi_y2 # Check if previous Y was beyond bottom edge

                        if entered_from_right or entered_from_bottom:
                            logger.info(f"Vehicle ID {best_match} entered ROI from invalid direction (right/bottom). Assigning new ID.")
                            assign_new_id = True
                # ------------------------------------------------

                if assign_new_id:
                    tracked_vehicles.pop(best_match, None) # Remove the old track
                    unmatched_detections.append(detection) # Treat as a new detection
                    # Do not add best_match to matched_vehicles
                else:
                    # Normal update
                    tracked_vehicles[best_match] = update_tracked_vehicle(vehicle_data, detection, processed_frames, frame_width, frame_height)
                    matched_vehicles.add(best_match)
            else:
                 # This detection matched strongly with a vehicle that was already matched by another detection.
                 # Add it to unmatched for potential prediction matching or new track creation if confidence is high enough.
                 unmatched_detections.append(detection)
        else:
            # No existing track matched well enough, add to unmatched list
            unmatched_detections.append(detection)

    # Attempt to match remaining detections with predicted positions
    unmatched_detections, matched_vehicles = match_with_predicted_positions(
        unmatched_detections, tracked_vehicles, matched_vehicles, processed_frames, frame_width, frame_height
    )

    # Create new tracks for remaining unmatched detections (already passed confidence check)
    for detection in unmatched_detections:
        # The confidence check is already done at the beginning of the function.
        # We just need to check proximity before creating a new track.
        det_center = calculate_center(detection['box'])
        too_close_to_existing = False
        detection_class = detection['class_id'] # Get detection class once

        for vehicle_id, vehicle_data in tracked_vehicles.items():
            # Check proximity only against vehicles that were *actually seen* recently or are predicted
            if vehicle_id not in matched_vehicles and not vehicle_data.get('predicted', False):
                 continue

            veh_center = calculate_center(vehicle_data['box'])
            distance = calculate_distance(det_center, veh_center)
            frames_since_seen = processed_frames - vehicle_data['last_seen']
            overlap = calculate_overlap_area(vehicle_data['box'], detection['box'])
            track_class = vehicle_data.get('class_id') # Get track class

            # Define conditions for being too close
            close_by_distance = False
            close_by_overlap = False

            # Check distance thresholds
            if vehicle_id in matched_vehicles and distance < PROXIMITY_THRESHOLDS['distance_close'] and frames_since_seen < COOLDOWN_FRAMES:
                close_by_distance = True
            elif vehicle_id not in matched_vehicles and vehicle_data.get('predicted', False) and distance < PROXIMITY_THRESHOLDS['distance_disappeared']:
                close_by_distance = True

            # Check overlap threshold
            if vehicle_id in matched_vehicles and overlap > PROXIMITY_THRESHOLDS['overlap_close'] and frames_since_seen < PROXIMITY_THRESHOLDS['frames_close']:
                close_by_overlap = True

            # Check if thresholds are met
            if close_by_distance or close_by_overlap:
                # --- Add Class Check --- 
                problem_classes = {2, 7} # Car and Truck
                # If one is car and the other is truck, DO NOT consider them too close based on proximity alone
                if detection_class in problem_classes and track_class in problem_classes and detection_class != track_class:
                    # logger.debug(f"Skipping proximity merge for detection (Class: {detection_class}) near track {vehicle_id} (Class: {track_class}) due to class difference.")
                    continue # Skip setting too_close_to_existing = True for this specific track
                # --- End Class Check ---
                
                # Otherwise (same class, or other classes involved), they are too close
                too_close_to_existing = True
                # logger.debug(f"New track too close (Dist: {distance:.1f}, Overlap: {overlap:.2f}) to track {vehicle_id}. Type: {'matched' if vehicle_id in matched_vehicles else 'predicted'}")
                break # Exit the inner loop, no need to check other tracks

        if not too_close_to_existing:
            logger.debug(f"Creating new track {next_vehicle_id} at frame {processed_frames} (Class: {detection['class_id']}, Conf: {detection['confidence']:.2f})")
            initial_in_roi = is_in_roi(detection['box'], frame_width, frame_height)
            initial_roi_frames = 1 if initial_in_roi else 0
            initial_class_id = detection['class_id']

            tracked_vehicles[next_vehicle_id] = {
                'id': next_vehicle_id,
                'box': detection['box'],
                'class_id': initial_class_id, # Initial class based on first detection
                'initial_class_id': initial_class_id,
                'class_history': deque([initial_class_id], maxlen=CLASS_HISTORY_SIZE),
                'last_seen': processed_frames,
                'confidence': detection['confidence'],
                'velocity': (0, 0),
                'consecutive_matches': 1,
                'is_counted': False,
                'predicted': False,
                'detection_stability': 1,
                'consecutive_roi_frames': initial_roi_frames,
                'confirmed_class_id': None, # Initialize confirmed class as None
                'consecutive_class_frames': 1, # Initialize consecutive frames
                'previous_majority_class': initial_class_id # Initialize previous majority
            }
            matched_vehicles.add(next_vehicle_id)
            next_vehicle_id += 1
        # else: # Optional logging
            # logger.debug(f"Skipping new track creation for detection (Class: {detection['class_id']}, Conf: {detection['confidence']:.2f}) due to proximity.")


    # --- Update state for vehicles NOT matched in this frame ---
    vehicles_to_remove = []
    for vehicle_id, vehicle_data in tracked_vehicles.items():
        if vehicle_id not in matched_vehicles:
            frames_since_seen = processed_frames - vehicle_data['last_seen']

            # Predict position if recently lost
            if frames_since_seen < MAX_PREDICTION_FRAMES and 'velocity' in vehicle_data:
                old_box = vehicle_data['box']
                old_center = calculate_center(old_box)

                # Predict based on last known velocity
                new_center_x = old_center[0] + vehicle_data['velocity'][0]
                new_center_y = old_center[1] + vehicle_data['velocity'][1]

                # Keep box size the same during prediction
                width = old_box[2] - old_box[0]
                height = old_box[3] - old_box[1]

                new_box = [
                    new_center_x - width / 2,
                    new_center_y - height / 2,
                    new_center_x + width / 2,
                    new_center_y + height / 2
                ]

                # Basic boundary check (optional, adjust as needed)
                new_box[0] = max(0, new_box[0])
                new_box[1] = max(0, new_box[1])
                new_box[2] = min(frame_width, new_box[2])
                new_box[3] = min(frame_height, new_box[3])

                vehicle_data['box'] = new_box
                vehicle_data['predicted'] = True
                vehicle_data['consecutive_matches'] = 0 # Reset consecutive matches on prediction
                # Decay stability when not detected
                if 'detection_stability' in vehicle_data:
                    vehicle_data['detection_stability'] = max(0, vehicle_data['detection_stability'] - 0.5) # Decay faster?

            # Mark for removal if lost for too long
            elif frames_since_seen > DISAPPEAR_THRESHOLD:
                vehicles_to_remove.append(vehicle_id)
            else:
                 # Vehicle is lost but not yet disappeared or predicted (e.g., velocity not established)
                 vehicle_data['predicted'] = False # Ensure predicted flag is false
                 vehicle_data['consecutive_matches'] = 0
                 if 'detection_stability' in vehicle_data:
                     vehicle_data['detection_stability'] = max(0, vehicle_data['detection_stability'] - 0.1) # Slow decay

    # --- Counting Logic ---
    # Use configured vehicle classes for counting check
    vehicle_classes_for_counting = set(CONFIG['model']['vehicle_classes'])

    for vehicle_id, vehicle_data in tracked_vehicles.items():
         # Check if vehicle exists after potential removal list generation
         if vehicle_id in vehicles_to_remove:
             continue

         consecutive_matches = vehicle_data.get('consecutive_matches', 0)
         stability = vehicle_data.get('detection_stability', 0)
         is_counted = vehicle_data.get('is_counted', False)
         # current_class_id = vehicle_data.get('class_id') # No longer primary for counting check
         confirmed_class_id = vehicle_data.get('confirmed_class_id') # Get the confirmed class
         is_predicted = vehicle_data.get('predicted', False) # Don't count predicted states
         consecutive_roi = vehicle_data.get('consecutive_roi_frames', 0) # Get consecutive ROI frames

         # --- Updated Counting Condition ---
         # Count only if: not already counted, not predicted, class is confirmed,
         # meets stability/match thresholds, and has been in ROI long enough.
         if not is_counted and not is_predicted and confirmed_class_id is not None and \
            confirmed_class_id in vehicle_classes_for_counting and \
            consecutive_matches >= MIN_CONSECUTIVE_DETECTIONS and \
            stability >= COUNTING_STABILITY_THRESHOLD and \
            consecutive_roi >= MIN_CONSECUTIVE_ROI_FRAMES:
         # --- End Updated Counting Condition ---

             # Additional check: Ensure the vehicle center is within the ROI if enabled
             should_count = True
             if CONFIG['roi']['enabled']:
                 # Use pre-calculated ROI boundaries
                 center_x, center_y = calculate_center(vehicle_data['box'])
                 if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
                     should_count = False
                     # logger.debug(f"Vehicle {vehicle_id} met stability but is outside ROI, not counting yet.")

             if should_count:
                 vehicle_data['is_counted'] = True
                 # Use the confirmed_class_id for counting
                 count_class = confirmed_class_id
                 if count_class not in unique_vehicle_counts:
                     unique_vehicle_counts[count_class] = 0
                 unique_vehicle_counts[count_class] += 1
                 logger.info(f"Counted vehicle ID {vehicle_id} (Confirmed Class: {count_class}, Stable, ROI Frames: {consecutive_roi}). Total for class {count_class}: {unique_vehicle_counts[count_class]}")

    # --- Remove disappeared tracks ---
    for vehicle_id in vehicles_to_remove:
        logger.debug(f"Removing disappeared track {vehicle_id} at frame {processed_frames}")
        tracked_vehicles.pop(vehicle_id, None) # Use pop with default None

    return tracked_vehicles, unique_vehicle_counts