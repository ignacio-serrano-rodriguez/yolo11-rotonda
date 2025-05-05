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
    
    frame_count = int(duration * target_fps)
    # Use CONFIG for vehicle classes
    unique_vehicle_counts: Dict[int, int] = {class_id: 0 for class_id in CONFIG['model']['vehicle_classes']}
    tracked_vehicles: Dict[int, Dict[str, Any]] = {}
    
    frame_buffer: deque = deque(maxlen=60)
    
    processed_frames = 0
    real_frame = 0
    
    current_results: Optional[Any] = None
    stream_failed = False
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    
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
        
        if stream_failed:
            if reconnect_attempts < max_reconnect_attempts:
                try:
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
                        else:
                            reconnect_attempts += 1
                    else:
                        reconnect_attempts += 1
                except Exception as e:
                    reconnect_attempts += 1
        
        if len(frame_buffer) > 0:
            current_frame: np.ndarray = frame_buffer.popleft()
        elif 'annotated_frame' in locals() and isinstance(annotated_frame, np.ndarray):
            current_frame = annotated_frame.copy()
        else:
            break
        
        real_frame += 1
        
        # Use CONFIG for frame skip
        skip_factor = CONFIG['video']['frame_skip']
        if len(frame_buffer) < frame_buffer.maxlen * 0.3:
            skip_factor = max(2, CONFIG['video']['frame_skip'] * 2)  # Double the skip or at least 2
        
        process_this_frame = (real_frame - 1) % skip_factor == 0
        
        if process_this_frame:
            # Get the default confidence and per-class overrides from config
            conf_settings = CONFIG['model']['confidence']
            # --- Find the minimum confidence threshold across all classes --- 
            all_conf_values = [v for v in conf_settings.values() if isinstance(v, (int, float))]
            min_conf_for_predict = min(all_conf_values) if all_conf_values else 0.01 # Use lowest defined or a small default
            # ----------------------------------------------------------------

            # Pass the minimum confidence to model.predict to ensure low-confidence classes are not filtered out early
            results = model.predict(
                current_frame,
                imgsz=CONFIG['video']['input_resolution'],
                conf=min_conf_for_predict, # Use the calculated minimum confidence value here
                verbose=False,
                classes=CONFIG['model']['vehicle_classes'],
                device=device
            )
            current_results = results # Keep results for annotation

            # --- Extract ALL detections passing the minimum threshold --- 
            raw_detections: List[Dict[str, Any]] = []
            for result in results:
                for detection in result.boxes:
                    raw_detections.append({
                        'box': detection.xyxy[0].cpu().numpy(),
                        'class_id': int(detection.cls),
                        'confidence': float(detection.conf)
                    })
            # ----------------------------------------------------------
            
            # Pass raw detections to the tracking function (filtering will happen there)
            tracked_vehicles, unique_vehicle_counts = process_detections(
                raw_detections, # Pass the unfiltered (by class conf) list
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
        
        output_video.write(annotated_frame)
        processed_frames += 1
        
        if processed_frames % 30 == 0 or processed_frames == frame_count: # Print every 30 frames or on the last frame
            elapsed = time.time() - start_time
            fps_current = processed_frames / elapsed if elapsed > 0 else 0
            remaining_frames = frame_count - processed_frames
            remaining_time = remaining_frames / fps_current if fps_current > 0 else 0
            print(f"Procesado: {processed_frames/frame_count*100:.1f}% - Tiempo restante aproximado: {remaining_time:.1f}s - Tiempo transcurrido: {elapsed:.1f}s")
    
    cap.release()
    output_video.release()
    
    return unique_vehicle_counts

def main() -> None:
    device = setup_gpu()
    
    model: YOLO = initialize_model(device)
    
    try:
        video_stream: str = get_youtube_stream(CONFIG['video']['youtube_url'])
    except Exception as e:
        print(f"Error al conectar con YouTube: {e}") # Keep error message
        return
    
    unique_vehicle_counts: Optional[Dict[int, int]] = process_video(
        model, 
        device, 
        video_stream, 
        CONFIG['video']['duration'], 
        CONFIG['video']['target_fps']
    )
    
    if unique_vehicle_counts:
        save_vehicle_counts_to_json(unique_vehicle_counts)
        print("Procesamiento completado con éxito.") # Keep completion message

if __name__ == "__main__":
    main()