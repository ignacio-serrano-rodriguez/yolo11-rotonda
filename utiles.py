import json
import os
import sys
import io
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple, Any, Deque, Optional, Union

# Configure logging
log_config = yaml.safe_load(open('config.yaml', 'r')).get('logging', {})
log_level = log_config.get('level', 'INFO').upper()
log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format=log_format)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Convert class names keys to integers
        if 'model' in config and 'class_names' in config['model']:
            config['model']['class_names'] = {int(k): v for k, v in config['model']['class_names'].items()}
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config: {e}")
        sys.exit(1)

# Load config globally or pass it around. Loading globally for simplicity here.
# Consider dependency injection for larger projects.
CONFIG: Dict[str, Any] = load_config()

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_size(box: List[float]) -> float:
    """Calculates the area of a bounding box."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def calculate_center(box: List[float]) -> Tuple[float, float]:
    """Calculates the center coordinates of a bounding box."""
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return (center_x, center_y)

def calculate_distance(center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_overlap_area(box1: List[float], box2: List[float]) -> float:
    """Calculates the overlap area as a fraction of the smaller box area."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    smaller_area = min(area_box1, area_box2)
    return intersection / smaller_area if smaller_area > 0 else 0.0

def is_box_contained(box1: List[float], box2: List[float], threshold: float = 0.8) -> bool:
    """Checks if box1 is largely contained within box2 or vice-versa."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    smaller_area = min(area_box1, area_box2)
    return intersection / smaller_area >= threshold

def get_most_common_class(class_history: Deque[int]) -> int:
    """Determines the most frequent class ID in a history deque."""
    if not class_history:
        return 2
    
    class_counts = {}
    for cls in class_history:
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1
    
    return max(class_counts, key=class_counts.get)

def is_in_roi(box: List[float], frame_width: int, frame_height: int) -> bool:
    """Checks if the center of a bounding box is within the defined ROI."""
    roi_config = CONFIG['roi']
    if not roi_config['enabled']:
        return True

    roi_coords = roi_config['coords']
    roi_pixels = [
        int(roi_coords[0] * frame_width),
        int(roi_coords[1] * frame_height),
        int(roi_coords[2] * frame_width),
        int(roi_coords[3] * frame_height)
    ]
    
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    
    intersection_x1 = max(box[0], roi_pixels[0])
    intersection_y1 = max(box[1], roi_pixels[1])
    intersection_x2 = min(box[2], roi_pixels[2])
    intersection_y2 = min(box[3], roi_pixels[3])
    
    if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        
        if intersection_area / box_area > 0.3: # Threshold for partial overlap
            return True
    
    return (roi_pixels[0] <= center_x <= roi_pixels[2] and 
            roi_pixels[1] <= center_y <= roi_pixels[3])

def save_vehicle_counts_to_json(counts: Dict[int, int]):
    """Saves the total vehicle count to a JSON file, updating the existing count.

    Only saves the total number of vehicles detected across all classes.
    """
    output_file = CONFIG['output']['count_file']
    # Use the configured key for the total count, default to "total_vehiculos"
    total_key = CONFIG['model']['spanish_names'].get("vehicle_total", "total_vehiculos")

    # Calculate total vehicles detected in this session
    total_current_session = sum(counts.values())

    # Load existing data if file exists
    existing_data = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if not isinstance(existing_data, dict):
                logger.warning(f"Existing data in {output_file} is not a dictionary. Overwriting with total count.")
                existing_data = {}
            logger.info(f"Loaded existing counts from: {output_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading or parsing existing count file {output_file}. Starting fresh. Error: {e}")
            existing_data = {} # Reset if file is corrupted

    # Get the previous total count, default to 0 if key doesn't exist or data is invalid
    previous_total = existing_data.get(total_key, 0)
    if not isinstance(previous_total, int):
        logger.warning(f"Existing value for '{total_key}' in {output_file} is not an integer. Resetting total.")
        previous_total = 0

    # Calculate the new total count
    new_total = previous_total + total_current_session

    # Prepare data to save (only the total count)
    data_to_save = {total_key: new_total}

    logger.info(f"Total vehicles this session: {total_current_session}. New total count to save: {new_total}")

    # Save the updated data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False) # ensure_ascii=False for Spanish names
        logger.info(f"Total vehicle count updated successfully in {output_file}")
    except IOError as e:
        logger.error(f"Error writing to count file {output_file}. Error: {e}")