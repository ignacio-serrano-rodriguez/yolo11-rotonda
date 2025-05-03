import json
import os
import sys
import io
import numpy as np
from configuracion import SPANISH_NAMES, ROI_ENABLED

# Clase personalizada para filtrar mensajes de error específicos
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

# Función para calcular IOU (Intersection over Union) de dos bounding boxes
def calculate_iou(box1, box2):
    """
    Calcula el IOU entre dos bounding boxes
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Área de intersección
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Área de la unión
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    
    # IOU
    iou = intersection / union if union > 0 else 0
    return iou

# Función para calcular el tamaño de un bounding box
def calculate_size(box):
    """Calcula el tamaño (área) de un bounding box"""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width * height

# Función para calcular el centro de un bounding box
def calculate_center(box):
    """Calcula el centro de un bounding box"""
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return (center_x, center_y)

# Función para calcular la distancia euclidiana entre dos puntos
def calculate_distance(center1, center2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# Función para guardar los conteos de vehículos en un archivo JSON
def save_vehicle_counts_to_json(counts, output_file='conteo.json'):
    """
    Guarda los conteos de vehículos en un archivo JSON.
    Si el archivo ya existe, incrementa los valores en lugar de reemplazarlos.
    
    Args:
        counts (dict): Diccionario con los conteos de vehículos por clase
        output_file (str): Nombre del archivo de salida
    """
    # Preparar datos para guardar - solo conteo total
    current_data = {}
    
    # Añadir total de vehículos
    total_vehicles = sum(counts.values())
    current_data["vehiculo"] = total_vehicles
    
    # Comprobar si el archivo existe y leer los datos actuales
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
            # Actualizar solo el conteo total de vehículos
            if "vehiculo" in existing_data:
                existing_data["vehiculo"] += total_vehicles
            else:
                existing_data["vehiculo"] = total_vehicles
                    
            # Usar los datos actualizados
            data_to_save = existing_data
            print(f"Actualizando conteo existente.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error al leer el fichero de conteo. Creando uno nuevo.")
            data_to_save = current_data
    else:
        data_to_save = current_data
        print(f"Creando nuevo fichero de conteo.")
    
    # Guardar el archivo JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"Conteo guardado en {output_file}: {data_to_save}")

# Función para verificar si una detección está dentro del ROI
def is_in_roi(box, roi, frame_width, frame_height):
    """
    Comprueba si un bounding box está dentro del ROI
    box format: [x1, y1, x2, y2] en pixeles
    roi format: [x1, y1, x2, y2] en porcentaje (0-1)
    """
    if not ROI_ENABLED:
        return True  # Si el ROI está desactivado, todas las detecciones están "dentro"
    
    # Convertir ROI de porcentajes a píxeles
    roi_pixels = [
        int(roi[0] * frame_width),
        int(roi[1] * frame_height),
        int(roi[2] * frame_width),
        int(roi[3] * frame_height)
    ]
    
    # Calcular el centro del box de detección
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    
    # Comprobar si el centro de la detección está dentro del ROI
    return (roi_pixels[0] <= center_x <= roi_pixels[2] and 
            roi_pixels[1] <= center_y <= roi_pixels[3])

# Función para calcular el área de solapamiento relativa entre dos bounding boxes
def calculate_overlap_area(box1, box2):
    """
    Calcula el área de solapamiento relativa entre dos bounding boxes
    Retorna: (área de intersección / área del box más pequeño)
    """
    # Calcular intersección
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Si no hay intersección, retornar 0
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Área de intersección
    intersection = (x2 - x1) * (y2 - y1)
    
    # Áreas de cada box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Retornar intersección relativa al box más pequeño
    smaller_area = min(area_box1, area_box2)
    return intersection / smaller_area if smaller_area > 0 else 0.0

# Función para determinar la clase más frecuente en un historial
def get_most_common_class(class_history):
    """
    Determina la clase más frecuente en el historial
    """
    if not class_history:
        return 2  # Valor por defecto si no hay historial
    
    class_counts = {}
    for cls in class_history:
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1
    
    return max(class_counts, key=class_counts.get)

# Función para verificar si dos vehículos tienen una relación de inclusión
def is_box_contained(box1, box2, threshold=0.8):
    """
    Verifica si un box está contenido en otro (con un umbral)
    """
    # Calcular intersección
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Si no hay intersección, retornar False
    if x2 < x1 or y2 < y1:
        return False
    
    # Área de intersección
    intersection = (x2 - x1) * (y2 - y1)
    
    # Áreas de cada box
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Verificar si el área más pequeña está contenida en la más grande
    smaller_area = min(area_box1, area_box2)
    return intersection / smaller_area >= threshold
