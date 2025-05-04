import cv2
import os
import yt_dlp
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from configuracion import CLASS_NAMES, ROI_ENABLED, ROI_COORDS, ROI_COLOR, ROI_THICKNESS
from utiles import calculate_center, is_in_roi

def get_youtube_stream(video_url):
    ydl_opts = {
        'format': 'best[height>=360][height<=480]/best[height<=720]/best',
        'quiet': True,
        'buffersize': 32768,
        'no-check-certificate': True,
        'socket_timeout': 30,
        'retries': 10
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            if 'url' in info:
                return info['url']
            
            elif 'formats' in info and info['formats']:
                suitable_formats = [f for f in info['formats'] 
                                   if f.get('height', 0) >= 360 and f.get('height', 0) <= 480 
                                   and f.get('url')]
                
                if not suitable_formats:
                    suitable_formats = [f for f in info['formats'] if f.get('url')]
                
                if suitable_formats:
                    mp4_formats = [f for f in suitable_formats if f.get('ext') == 'mp4']
                    if mp4_formats:
                        mp4_formats.sort(key=lambda x: x.get('tbr', 0))
                        if mp4_formats:
                            return mp4_formats[0]['url']
                    
                    suitable_formats.sort(key=lambda x: x.get('tbr', 0))
                    return suitable_formats[0]['url']
            
            raise Exception("No se encontró URL de stream")
    except Exception as e:
        raise Exception(f"Error al obtener stream: {str(e)}")

def initialize_model(device):
    model = YOLO('yolo11m.pt')
    model.to(device)
    return model

def create_video_writer(cap, target_fps):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    videos_dir = 'videos'
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
        print(f"Directorio creado: {videos_dir}")
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = os.path.join(videos_dir, f'video_{current_datetime}.avi')
    
    return cv2.VideoWriter(
        video_filename, 
        cv2.VideoWriter_fourcc(*'MJPG'), 
        target_fps, 
        (frame_width, frame_height)
    ), frame_width, frame_height, video_filename

def annotate_frame(frame, results, unique_vehicle_counts, frame_width, frame_height, tracked_vehicles=None):
    annotated_frame = results[0].plot()
    
    if ROI_ENABLED:
        roi_x1 = int(ROI_COORDS[0] * frame_width)
        roi_y1 = int(ROI_COORDS[1] * frame_height)
        roi_x2 = int(ROI_COORDS[2] * frame_width)
        roi_y2 = int(ROI_COORDS[3] * frame_height)
        
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, -1)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), ROI_COLOR, ROI_THICKNESS)
        cv2.putText(annotated_frame, "ROI", (roi_x1 + 10, roi_y1 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_COLOR, 2)
    
    if tracked_vehicles:
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            box = vehicle_data['box']
            is_counted = vehicle_data.get('is_counted', False)
            is_predicted = vehicle_data.get('predicted', False)
            stability = vehicle_data.get('detection_stability', 0)
            class_id = vehicle_data.get('class_id', 2)
            
            class_name = CLASS_NAMES.get(class_id, "vehicle")
            
            if is_counted:
                color = (0, 255, 0)
            elif stability >= 3:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            thickness = 1 if is_predicted else 2
            line_type = cv2.LINE_AA
            
            if is_predicted:
                pts = np.array([[box[0], box[1]], [box[2], box[1]], 
                                [box[2], box[3]], [box[0], box[3]]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [pts], True, color, thickness, lineType=line_type)
            else:
                cv2.rectangle(annotated_frame, 
                              (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), 
                              color, thickness, lineType=line_type)
            
            label_text = f"ID:{vehicle_id} ({class_name[:3]}) S:{int(stability)}"
            cv2.putText(annotated_frame, label_text, 
                        (int(box[0]), int(box[1] - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    color = (0, 128, 255)
    title_text = "Conteo"
    title_size, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    title_x = frame_width - title_size[0] - 10
    y_pos = 40

    cv2.putText(annotated_frame, title_text, (title_x, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y_pos += 30

    total_vehicles = sum(unique_vehicle_counts.values())
    text = f"Total: {total_vehicles}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = frame_width - text_size[0] - 10

    cv2.putText(annotated_frame, text, (text_x, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
    return annotated_frame