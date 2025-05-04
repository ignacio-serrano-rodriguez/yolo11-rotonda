import numpy as np
from collections import deque
from configuracion import (
    IOU_THRESHOLD, OVERLAP_THRESHOLD, DISAPPEAR_THRESHOLD, 
    MIN_CONSECUTIVE_DETECTIONS, MAX_PREDICTION_FRAMES, CLASS_HISTORY_SIZE,
    COOLDOWN_FRAMES
)
from utiles import (
    calculate_iou, calculate_center, calculate_size, calculate_distance, 
    calculate_overlap_area, is_box_contained, get_most_common_class
)

def group_overlapping_detections(current_detections):
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

def calculate_match_score(detection, vehicle_data, frame_width, frame_height, processed_frames):
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
        class_consistency = 1.0 / unique_classes
        
    score = (iou * 0.4) + (area_overlap * 0.2) + (size_ratio * 0.1) + \
           ((1.0 - norm_distance) * 0.15) + (direction_score * 0.1) + \
           (recency_factor * 0.05)
    
    if is_box_contained(vehicle_data['box'], detection['box']):
        score += 0.15
        
    return score

def update_tracked_vehicle(vehicle_data, detection, processed_frames):
    old_center = calculate_center(vehicle_data['box'])
    new_center = calculate_center(detection['box'])
    
    if 'velocity' not in vehicle_data:
        vehicle_data['velocity'] = (0, 0)
    
    alpha = 0.7
    dx = new_center[0] - old_center[0]
    dy = new_center[1] - old_center[1]
    vehicle_data['velocity'] = (
        alpha * dx + (1-alpha) * vehicle_data['velocity'][0],
        alpha * dy + (1-alpha) * vehicle_data['velocity'][1]
    )
    
    if 'class_history' not in vehicle_data:
        vehicle_data['class_history'] = deque(maxlen=CLASS_HISTORY_SIZE)
    vehicle_data['class_history'].append(detection['class_id'])
    
    vehicle_data['class_id'] = get_most_common_class(vehicle_data['class_history'])
    vehicle_data['box'] = detection['box']
    vehicle_data['last_seen'] = processed_frames
    vehicle_data['confidence'] = detection['confidence']
    vehicle_data['consecutive_matches'] = vehicle_data.get('consecutive_matches', 0) + 1
    vehicle_data['predicted'] = False
    vehicle_data['detection_stability'] = vehicle_data.get('detection_stability', 0) + 1
    
    return vehicle_data

def match_with_predicted_positions(remaining_detections, tracked_vehicles, matched_vehicles, processed_frames, frame_width, frame_height):
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
                
                match_quality = (1 - normalized_dist) * 0.7 + size_ratio * 0.3
                if match_quality > 0.6:
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

def process_detections(current_detections, tracked_vehicles, processed_frames, unique_vehicle_counts, frame_width, frame_height):
    matched_vehicles = set()
    next_vehicle_id = max(tracked_vehicles.keys()) + 1 if tracked_vehicles else 0
    
    grouped_detections = group_overlapping_detections(current_detections)
    
    unmatched_detections = []
    
    for detection in grouped_detections:
        best_match = None
        best_score = IOU_THRESHOLD
        
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            score = calculate_match_score(detection, vehicle_data, frame_width, frame_height, processed_frames)
            if score > best_score:
                best_score = score
                best_match = vehicle_id
        
        if best_match is not None:
            tracked_vehicles[best_match] = update_tracked_vehicle(tracked_vehicles[best_match], detection, processed_frames)
            matched_vehicles.add(best_match)
        else:
            unmatched_detections.append(detection)
    
    unmatched_detections, matched_vehicles = match_with_predicted_positions(
        unmatched_detections, tracked_vehicles, matched_vehicles, processed_frames, frame_width, frame_height
    )
    
    for detection in unmatched_detections:
        det_center = calculate_center(detection['box'])
        too_close_to_existing = False
        
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            veh_center = calculate_center(vehicle_data['box'])
            distance = calculate_distance(det_center, veh_center)
            frames_since_seen = processed_frames - vehicle_data['last_seen']
            
            if distance < 20 and frames_since_seen < COOLDOWN_FRAMES:
                too_close_to_existing = True
                break
                
            if frames_since_seen >= DISAPPEAR_THRESHOLD and distance < 15:
                too_close_to_existing = True
                break
                
            overlap = calculate_overlap_area(vehicle_data['box'], detection['box'])
            if overlap > 0.7 and frames_since_seen < 5:
                too_close_to_existing = True
                break
                
        if not too_close_to_existing:
            tracked_vehicles[next_vehicle_id] = {
                'box': detection['box'],
                'class_id': detection['class_id'],
                'class_history': deque([detection['class_id']], maxlen=CLASS_HISTORY_SIZE),
                'last_seen': processed_frames,
                'confidence': detection['confidence'],
                'velocity': (0, 0),
                'consecutive_matches': 1,
                'is_counted': False,
                'predicted': False,
                'detection_stability': 1
            }
            matched_vehicles.add(next_vehicle_id)
            next_vehicle_id += 1
    
    for vehicle_id, vehicle_data in tracked_vehicles.items():
        if vehicle_id not in matched_vehicles and 'velocity' in vehicle_data:
            frames_since_seen = processed_frames - vehicle_data['last_seen']
            
            if frames_since_seen < MAX_PREDICTION_FRAMES:
                old_box = vehicle_data['box']
                old_center = calculate_center(old_box)
                
                new_center_x = old_center[0] + vehicle_data['velocity'][0]
                new_center_y = old_center[1] + vehicle_data['velocity'][1]
                
                dx = new_center_x - old_center[0]
                dy = new_center_y - old_center[1]
                
                new_box = [
                    old_box[0] + dx,
                    old_box[1] + dy,
                    old_box[2] + dx,
                    old_box[3] + dy
                ]
                
                vehicle_data['box'] = new_box
                vehicle_data['predicted'] = True
                
                if 'detection_stability' in vehicle_data:
                    vehicle_data['detection_stability'] = max(0, vehicle_data['detection_stability'] - 0.5)
    
    for vehicle_id, vehicle_data in tracked_vehicles.items():
        consecutive_matches = vehicle_data.get('consecutive_matches', 0)
        stability = vehicle_data.get('detection_stability', 0)
        is_counted = vehicle_data.get('is_counted', False)
        
        if not is_counted and consecutive_matches >= MIN_CONSECUTIVE_DETECTIONS and stability >= 2:
            vehicle_data['is_counted'] = True
            
            if 2 not in unique_vehicle_counts:
                unique_vehicle_counts[2] = 0
                
            unique_vehicle_counts[2] += 1
    
    vehicles_to_remove = []
    for vehicle_id, vehicle_data in tracked_vehicles.items():
        if processed_frames - vehicle_data['last_seen'] > DISAPPEAR_THRESHOLD:
            vehicles_to_remove.append(vehicle_id)
    
    for vehicle_id in vehicles_to_remove:
        tracked_vehicles.pop(vehicle_id)
    
    return tracked_vehicles, unique_vehicle_counts