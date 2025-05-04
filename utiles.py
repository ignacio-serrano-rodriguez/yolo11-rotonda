import json
import os
import sys
import io
import numpy as np
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
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
CONFIG = load_config()

def calculate_iou(box1, box2):
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

def calculate_size(box):
    """Calculates the area of a bounding box."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def calculate_center(box):
    """Calculates the center coordinates of a bounding box."""
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return (center_x, center_y)

def calculate_distance(center1, center2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_overlap_area(box1, box2):
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

def is_box_contained(box1, box2, threshold=0.8):
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

def get_most_common_class(class_history):
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

def is_in_roi(box, frame_width, frame_height):
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

def save_vehicle_counts_to_json(counts):
    """Saves the counted vehicle numbers to a JSON file, updating existing counts."""
    output_file = CONFIG['output']['count_file']
    current_data = {}

    # Use the spanish name from config if available, otherwise default to "vehiculo"
    spanish_name = CONFIG['model']['spanish_names'].get("vehicle", "vehiculo")
    total_vehicles = sum(counts.values()) # Assuming counts keys are class IDs, sum all detected vehicles
    current_data[spanish_name] = total_vehicles

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            if spanish_name in existing_data:
                existing_data[spanish_name] += total_vehicles
            else:
                existing_data[spanish_name] = total_vehicles

            data_to_save = existing_data
            logger.info(f"Updating existing count file: {output_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading count file {output_file}. Creating a new one. Error: {e}")
            data_to_save = current_data
    else:
        data_to_save = current_data
        logger.info(f"Creating new count file: {output_file}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        logger.info(f"Vehicle counts saved successfully to {output_file}")
    except IOError as e:
        logger.error(f"Error writing to count file {output_file}. Error: {e}")