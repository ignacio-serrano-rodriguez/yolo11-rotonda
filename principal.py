import time
import torch
import cv2
import numpy as np
from collections import deque
import os
import logging
import sys
from typing import Dict, Any, Optional, Tuple, List
from ultralytics import YOLO # Assuming YOLO type can be imported

# Import config and utility functions
from utiles import CONFIG, save_vehicle_counts_to_json, is_in_roi, calculate_center
from seguimiento import process_detections
from video import get_youtube_stream, create_video_writer, annotate_frame, initialize_model

def setup_gpu() -> str:
    # Use GPU config
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.get('gpu', {}).get('cuda_visible_devices', "0")

    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Usando GPU: {gpu_name}")
        print(f"Versión de CUDA: {torch.version.cuda}")
        torch.set_default_device('cuda')
        return device
    else:
        device = 'cpu'
        print("No se ha detectado ninguna GPU. Usando CPU.")
        print("CUDA disponible:", torch.cuda.is_available())
        print("Versión de PyTorch:", torch.__version__)
        return device

def process_video(
    model: YOLO,
    device: str,
    video_stream: str,
    duration: int,
    target_fps: int
) -> Optional[Dict[int, int]]:
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_stream)
    
    if not cap.isOpened():
        print("Error al abrir el stream de YouTube.")
        return None
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    
    output_video, frame_width, frame_height, video_filename = create_video_writer(cap, target_fps)
    print(f"Guardando video de salida: {video_filename}")
    
    frame_count = int(duration * target_fps)
    # Use CONFIG for vehicle classes
    unique_vehicle_counts: Dict[int, int] = {class_id: 0 for class_id in CONFIG['model']['vehicle_classes']}
    tracked_vehicles: Dict[int, Dict[str, Any]] = {}
    
    frame_buffer: deque = deque(maxlen=60)
    
    processed_frames = 0
    real_frame = 0
    
    # Use CONFIG for parameters
    print(f"Resolución del vídeo: {CONFIG['video']['input_resolution']}x{CONFIG['video']['input_resolution']}\nSalto de frame: {CONFIG['video']['frame_skip']}\nConfianza: {CONFIG['video']['confidence']}")
    print(f"Parámetros de tracking: IOU={CONFIG['tracking']['iou_threshold']}, Desaparición={CONFIG['tracking']['disappear_threshold']}")
    
    current_results: Optional[Any] = None
    stream_failed = False
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    
    print("Prellenando buffer de frames...")
    while len(frame_buffer) < frame_buffer.maxlen * 0.5:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame.copy())
        
    while processed_frames < frame_count:
        if not stream_failed:
            ret, frame = cap.read()
            if ret:
                frame_buffer.append(frame.copy())
                reconnect_attempts = 0
            else:
                stream_failed = True
                print("Error al leer frame. Usando buffer...")
        
        if stream_failed:
            if reconnect_attempts < max_reconnect_attempts:
                try:
                    print(f"Intento de reconexión {reconnect_attempts+1}/{max_reconnect_attempts}...")
                    cap.release()
                    time.sleep(1)
                    video_stream = get_youtube_stream(CONFIG['video']['youtube_url'])
                    cap = cv2.VideoCapture(video_stream)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
                    
                    if cap.isOpened():
                        success_reads = 0
                        for _ in range(5):
                            ret, test_frame = cap.read()
                            if ret:
                                success_reads += 1
                                frame_buffer.append(test_frame.copy())
                        
                        if success_reads >= 3:
                            stream_failed = False
                            print("Reconexión exitosa.")
                        else:
                            reconnect_attempts += 1
                    else:
                        reconnect_attempts += 1
                except Exception as e:
                    print(f"Error al reconectar: {e}")
                    reconnect_attempts += 1
        
        if len(frame_buffer) > 0:
            current_frame: np.ndarray = frame_buffer.popleft()
        elif 'annotated_frame' in locals() and isinstance(annotated_frame, np.ndarray):
            current_frame = annotated_frame.copy()
            cv2.putText(current_frame, "BUFFER VACÍO - FRAME DUPLICADO", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 0, 255), 2)
        else:
            print("No hay frames disponibles para procesar.")
            break
        
        real_frame += 1
        
        # Use CONFIG for frame skip
        skip_factor = CONFIG['video']['frame_skip']
        if len(frame_buffer) < frame_buffer.maxlen * 0.3:
            skip_factor = max(2, CONFIG['video']['frame_skip'] * 2)  # Double the skip or at least 2
        
        process_this_frame = (real_frame - 1) % skip_factor == 0
        
        if process_this_frame:
            results = model.predict(
                current_frame, 
                # Use CONFIG for input resolution, confidence, vehicle classes
                imgsz=CONFIG['video']['input_resolution'],
                conf=CONFIG['video']['confidence'], 
                verbose=False, 
                classes=CONFIG['model']['vehicle_classes'],
                device=device
            )
            current_results = results
            
            current_detections: List[Dict[str, Any]] = []
            
            for result in results:
                for detection in result.boxes:
                    class_id = int(detection.cls)
                    # Use CONFIG for vehicle classes
                    if class_id in CONFIG['model']['vehicle_classes']:
                        confidence = float(detection.conf)
                        box = detection.xyxy[0].cpu().numpy()
                        
                        # Use CONFIG for ROI coords
                        in_roi = is_in_roi(box, frame_width, frame_height) # is_in_roi now uses CONFIG internally
                        if in_roi:
                            current_detections.append({'box': box, 'class_id': class_id, 'confidence': confidence})
                            if processed_frames % 30 == 0:
                                center: Tuple[float, float] = calculate_center(box)
            
            tracked_vehicles, unique_vehicle_counts = process_detections(
                current_detections, 
                tracked_vehicles, 
                processed_frames, 
                unique_vehicle_counts,
                frame_width,
                frame_height
            )
        
        if current_results is not None:
            annotated_frame: np.ndarray = annotate_frame(
                current_frame, 
                current_results, 
                unique_vehicle_counts, 
                frame_width, 
                frame_height,
                tracked_vehicles
            )
        else:
            annotated_frame = current_frame.copy()
        
        buffer_fill = len(frame_buffer) / frame_buffer.maxlen
        buffer_color = (0, 255, 0) if buffer_fill > 0.5 else (0, 165, 255) if buffer_fill > 0.2 else (0, 0, 255)
        
        output_video.write(annotated_frame)
        processed_frames += 1
        
        if processed_frames % 30 == 0:
            elapsed = time.time() - start_time
            fps_current = processed_frames / elapsed
            remaining = (frame_count - processed_frames) / fps_current
            print(f"- ({processed_frames/frame_count*100:.1f}%) Procesado - {processed_frames}/{frame_count} Frames procesados - {remaining:.1f}s restantes aprox")
    
    cap.release()
    output_video.release()
    
    return unique_vehicle_counts

def main() -> None:
    device = setup_gpu()
    
    print("Cargando modelo YOLO...")
    model: YOLO = initialize_model(device)
    
    print("Conexión al stream de YouTube...")
    try:
        # Use CONFIG for YouTube URL
        video_stream: str = get_youtube_stream(CONFIG['video']['youtube_url'])
    except Exception as e:
        print(f"Error al conectar con YouTube: {e}")
        return
    
    print("Procesando vídeo...")
    # Use CONFIG for duration and target FPS
    unique_vehicle_counts: Optional[Dict[int, int]] = process_video(
        model, 
        device, 
        video_stream, 
        CONFIG['video']['duration'], 
        CONFIG['video']['target_fps']
    )
    
    if unique_vehicle_counts:
        save_vehicle_counts_to_json(unique_vehicle_counts)
        print("Procesamiento completado con éxito.")

if __name__ == "__main__":
    main()