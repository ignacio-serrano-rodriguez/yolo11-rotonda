import json
import os
import sys
import io
import numpy as np
from configuracion import SPANISH_NAMES, ROI_ENABLED

class FilteredStderr(io.StringIO):
    def __init__(self, filtered_message):
        super().__init__()
        self.stderr = sys.stderr
        self.filtered_message = filtered_message
        
    def write(self, message):
        if self.filtered_message not in message:
            self.stderr.write(message)
            
    def flush(self):
        self.stderr.flush()

def calculate_iou(box1, box2):
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
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

def calculate_center(box):
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return (center_x, center_y)

def calculate_distance(center1, center2):
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calculate_overlap_area(box1, box2):
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
    if not class_history:
        return 2
    
    class_counts = {}
    for cls in class_history:
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1
    
    return max(class_counts, key=class_counts.get)

def is_in_roi(box, roi, frame_width, frame_height):
    if not ROI_ENABLED:
        return True
    
    roi_pixels = [
        int(roi[0] * frame_width),
        int(roi[1] * frame_height),
        int(roi[2] * frame_width),
        int(roi[3] * frame_height)
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
        
        if intersection_area / box_area > 0.3:
            return True
    
    return (roi_pixels[0] <= center_x <= roi_pixels[2] and 
            roi_pixels[1] <= center_y <= roi_pixels[3])

def save_vehicle_counts_to_json(counts, output_file='conteo.json'):
    current_data = {}
    
    total_vehicles = sum(counts.values())
    current_data["vehiculo"] = total_vehicles
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
            if "vehiculo" in existing_data:
                existing_data["vehiculo"] += total_vehicles
            else:
                existing_data["vehiculo"] = total_vehicles
                    
            data_to_save = existing_data
            print(f"Actualizando conteo existente.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error al leer el fichero de conteo. Creando uno nuevo.")
            data_to_save = current_data
    else:
        data_to_save = current_data
        print(f"Creando nuevo fichero de conteo.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)