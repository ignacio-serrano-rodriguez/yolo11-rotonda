import cv2
import yt_dlp as youtube_dl
from ultralytics import YOLO
import torch
from datetime import datetime
import os
from configuracion import CLASS_NAMES, ROI_ENABLED, ROI_COORDS, ROI_COLOR, ROI_THICKNESS
import yt_dlp
from utiles import calculate_center, is_in_roi

def get_youtube_stream(video_url):
    """Obtiene una URL de streaming de vídeo de YouTube con calidad moderada y sin cortes"""
    ydl_opts = {
        # Primero intentar calidad media (480p-720p), con fallback a menor calidad
        'format': 'best[height>=480][height<=720]/best[height>=360][height<=480]/best',
        'quiet': True,
        # Aumentamos el buffer para mejorar la estabilidad
        'buffersize': 16384,
        # Ignoramos cualquier error SSL/certificados
        'no-check-certificate': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            # Si el extractor devuelve directamente una URL
            if 'url' in info:
                return info['url']
            
            # Si tenemos formatos disponibles
            elif 'formats' in info and info['formats']:
                # Primero, intenta encontrar un formato adecuado dentro del rango preferido (480p-720p)
                suitable_formats = [f for f in info['formats'] 
                                   if f.get('height', 0) >= 480 and f.get('height', 0) <= 720 
                                   and f.get('url')]
                
                # Si no hay formatos en el rango preferido, acepta rangos más bajos (360p-480p)
                if not suitable_formats:
                    suitable_formats = [f for f in info['formats'] 
                                       if f.get('height', 0) >= 360 and f.get('height', 0) <= 480
                                       and f.get('url')]
                
                # Si aún no tenemos formatos, acepta cualquiera disponible
                if not suitable_formats:
                    suitable_formats = [f for f in info['formats'] if f.get('url')]
                
                if suitable_formats:
                    # Preferir formatos más estables (mp4 > webm)
                    for f in suitable_formats:
                        if f.get('ext') == 'mp4':
                            return f['url']
                    
                    # Si no hay mp4, usar cualquier formato disponible
                    return suitable_formats[0]['url']
            
            raise Exception("No se encontró URL de stream")
    except Exception as e:
        raise Exception(f"Error al obtener stream: {str(e)}")

def initialize_model(device):
    """Inicializa y devuelve el modelo YOLO"""
    model = YOLO('yolo11m.pt')
    model.to(device)
    return model

def create_video_writer(cap, target_fps):
    """Crea un objeto VideoWriter para la salida"""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Crear directorio de videos si no existe
    videos_dir = 'videos'
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
        print(f"Directorio creado: {videos_dir}")
    
    # Generar un nombre de archivo con la fecha y hora actuales
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = os.path.join(videos_dir, f'video_{current_datetime}.avi')
    
    return cv2.VideoWriter(
        video_filename, 
        cv2.VideoWriter_fourcc(*'MJPG'), 
        target_fps, 
        (frame_width, frame_height)
    ), frame_width, frame_height, video_filename

def annotate_frame(frame, results, unique_vehicle_counts, frame_width, frame_height, tracked_vehicles=None):
    """Anota el frame con los resultados de detección y el contador de vehículos"""
    annotated_frame = results[0].plot()
    
    # Dibujar el ROI en el frame si está habilitado
    if ROI_ENABLED:
        roi_x1 = int(ROI_COORDS[0] * frame_width)
        roi_y1 = int(ROI_COORDS[1] * frame_height)
        roi_x2 = int(ROI_COORDS[2] * frame_width)
        roi_y2 = int(ROI_COORDS[3] * frame_height)
        
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, ROI_THICKNESS)
        cv2.putText(annotated_frame, "ROI", (roi_x1 + 10, roi_y1 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR, 2)
    
    # Visualizar vehículos rastreados
    if tracked_vehicles:
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            box = vehicle_data['box']
            is_counted = vehicle_data.get('is_counted', False)
            is_predicted = vehicle_data.get('predicted', False)
            stability = vehicle_data.get('detection_stability', 0)
            class_id = vehicle_data.get('class_id', 2)
            
            # Mapear ID de clase a nombre
            class_name = CLASS_NAMES.get(class_id, "vehicle")
            
            # Color basado en si el vehículo ha sido contado y su estabilidad
            if is_counted:
                color = (0, 255, 0)  # Verde si está contado
            elif stability >= 3:
                color = (0, 255, 255)  # Amarillo si es estable pero no contado
            else:
                color = (0, 165, 255)  # Naranja si no es estable
            
            # Si es una posición predicha, usar línea punteada
            thickness = 1 if is_predicted else 2
            line_type = cv2.LINE_AA
            
            if is_predicted:
                # Dibujar caja con líneas discontinuas
                import numpy as np
                pts = np.array([[box[0], box[1]], [box[2], box[1]], 
                                [box[2], box[3]], [box[0], box[3]]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, color, thickness, lineType=line_type)
            else:
                # Dibujar caja normal
                cv2.rectangle(annotated_frame, 
                              (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), 
                              color, thickness, lineType=line_type)
            
            # Mostrar ID del vehículo, clase y estabilidad
            label_text = f"ID:{vehicle_id} ({class_name[:3]}) S:{int(stability)}"
            cv2.putText(annotated_frame, label_text, 
                        (int(box[0]), int(box[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # Mostrar el conteo de vehículos en el frame (esquina superior derecha en naranja)
    color = (0, 128, 255)
    
    # Calcular el tamaño del texto para posicionarlo correctamente en la parte superior derecha
    title_text = "Conteo"
    title_size, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
    # Posición del título (alineado a la derecha)
    title_x = frame_width - title_size[0] - 10
    y_pos = 40
    
    cv2.putText(annotated_frame, title_text, (title_x, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y_pos += 30
    
    # Mostrar el total de vehículos en lugar de conteos por clase
    total_vehicles = sum(unique_vehicle_counts.values())
    text = f"Vehiculos: {total_vehicles}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = frame_width - text_size[0] - 10
    
    cv2.putText(annotated_frame, text, (text_x, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
    return annotated_frame
