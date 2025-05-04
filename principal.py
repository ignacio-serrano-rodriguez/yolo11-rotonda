import time
import torch
import contextlib
import cv2
import numpy as np
from collections import deque

from configuracion import *
from utiles import FilteredStderr, save_vehicle_counts_to_json, is_in_roi, calculate_center
from seguimiento import process_detections
from video import get_youtube_stream, create_video_writer, annotate_frame, initialize_model

def setup_gpu():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
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

def process_video(model, device, video_stream, duration, target_fps):
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_stream)
    
    if not cap.isOpened():
        print("Error al abrir el stream de YouTube.")
        return None
    
    output_video, frame_width, frame_height, video_filename = create_video_writer(cap, target_fps)
    print(f"Guardando video de salida: {video_filename}")
    
    frame_count = int(duration * target_fps)
    unique_vehicle_counts = {class_id: 0 for class_id in VEHICLE_CLASSES}
    tracked_vehicles = {}
    frame_buffer = deque(maxlen=30)
    
    processed_frames = 0
    real_frame = 0
    
    print(f"Resolución del vídeo: {INPUT_RESOLUTION}x{INPUT_RESOLUTION}\nSalto de frame: {FRAME_SKIP}\nConfianza: {CONFIDENCE}")
    print(f"Parámetros de tracking: IOU={IOU_THRESHOLD}, Desaparición={DISAPPEAR_THRESHOLD}, Min_Detecciones={MIN_CONSECUTIVE_DETECTIONS}")
    
    current_results = None
    stream_failed = False
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    
    print("Prellenando buffer de frames...")
    while len(frame_buffer) < frame_buffer.maxlen and len(frame_buffer) < 15:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame.copy())
    
    filtered_stderr = FilteredStderr("Cannot reuse HTTP connection for different host")
    
    with contextlib.redirect_stderr(filtered_stderr):
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
                        video_stream = get_youtube_stream(YOUTUBE_URL)
                        cap = cv2.VideoCapture(video_stream)
                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret:
                                stream_failed = False
                                print("Reconexión exitosa.")
                                frame_buffer.append(test_frame.copy())
                            else:
                                reconnect_attempts += 1
                        else:
                            reconnect_attempts += 1
                    except Exception as e:
                        print(f"Error al reconectar: {e}")
                        reconnect_attempts += 1
            
            if len(frame_buffer) > 0:
                current_frame = frame_buffer.popleft()
            elif 'annotated_frame' in locals():
                current_frame = annotated_frame.copy()
            else:
                print("No hay frames disponibles para procesar.")
                break
            
            real_frame += 1
            process_this_frame = (real_frame - 1) % FRAME_SKIP == 0
            
            if process_this_frame:
                results = model.predict(
                    current_frame, 
                    imgsz=INPUT_RESOLUTION,
                    conf=CONFIDENCE, 
                    verbose=False, 
                    classes=VEHICLE_CLASSES,
                    device=device
                )
                current_results = results
                
                current_detections = []
                
                for result in results:
                    for detection in result.boxes:
                        class_id = int(detection.cls)
                        if class_id in VEHICLE_CLASSES:
                            confidence = float(detection.conf)
                            box = detection.xyxy[0].cpu().numpy()
                            
                            in_roi = is_in_roi(box, ROI_COORDS, frame_width, frame_height)
                            if in_roi:
                                current_detections.append({'box': box, 'class_id': class_id, 'confidence': confidence})
                                if processed_frames % 10 == 0:
                                    center = calculate_center(box)
                                    print(f"Detectado vehículo en ROI: centro={center}, confianza={confidence:.2f}")
                
                tracked_vehicles, unique_vehicle_counts = process_detections(
                    current_detections, 
                    tracked_vehicles, 
                    processed_frames, 
                    unique_vehicle_counts,
                    frame_width,
                    frame_height
                )
            
            if current_results is not None:
                annotated_frame = annotate_frame(
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
            
            if processed_frames % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = processed_frames / elapsed
                remaining = (frame_count - processed_frames) / fps_current
                print(f"- ({processed_frames/frame_count*100:.1f}%) Procesado - {processed_frames}/{frame_count} Frames procesados - {remaining:.1f}s restantes aprox")
    
    cap.release()
    output_video.release()
    
    print("Procesamiento completado.")
    return unique_vehicle_counts

def main():
    device = setup_gpu()
    
    print("Cargando modelo YOLO...")
    model = initialize_model(device)
    
    print("Conexión al stream de YouTube...")
    try:
        video_stream = get_youtube_stream(YOUTUBE_URL)
    except Exception as e:
        print(f"Error al conectar con YouTube: {e}")
        return
    
    print("Procesando vídeo...")
    unique_vehicle_counts = process_video(
        model, 
        device, 
        video_stream, 
        DURATION, 
        TARGET_FPS
    )
    
    if unique_vehicle_counts:
        save_vehicle_counts_to_json(unique_vehicle_counts)
        print("Procesamiento completado con éxito.")

if __name__ == "__main__":
    main()