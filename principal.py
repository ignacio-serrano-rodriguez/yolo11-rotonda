import time
import torch
import contextlib

from configuracion import *
from utiles import FilteredStderr, save_vehicle_counts_to_json, is_in_roi
from seguimiento import process_detections
from video import (
    get_youtube_stream, 
    create_video_writer, 
    annotate_frame, 
    initialize_model
)

def setup_gpu():
    """Configura y devuelve el dispositivo apropiado (GPU/CPU)"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Utilizar primero la GPU
    
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
    """Procesa el video, detecta y cuenta vehículos"""
    import cv2
    import numpy as np
    
    start_time = time.time()
    
    # Abrir el stream de video
    cap = cv2.VideoCapture(video_stream)
    
    # Verificar si el stream de video se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir el stream de YouTube.")
        return None
    
    # Crear un VideoWriter para guardar el video de salida
    output_video, frame_width, frame_height, video_filename = create_video_writer(cap, target_fps)
    print(f"Guardando video de salida como: {video_filename}")
    
    # Calcular la cantidad total de cuadros
    frame_count = int(duration * target_fps)
    
    # Diccionario para contar las detecciones únicas de cada clase de vehículo
    unique_vehicle_counts = {2: 0}  # Usar ID 2 como contador genérico para todos los vehículos
    
    # Diccionario para rastrear vehículos únicos
    tracked_vehicles = {}
    
    # Leer y procesar los cuadros del video
    processed_frames = 0
    real_frame = 0
    
    print(f"Resolución del vídeo: {INPUT_RESOLUTION}x{INPUT_RESOLUTION}\nSalto de frame: {FRAME_SKIP}\nConfianza: {CONFIDENCE}")
    print(f"Parámetros de tracking: IOU={IOU_THRESHOLD}, Desaparición={DISAPPEAR_THRESHOLD}, Min_Detecciones={MIN_CONSECUTIVE_DETECTIONS}")
    print(f"Historial de clases: {CLASS_HISTORY_SIZE} frames")
    
    # Crear filtro para mensajes de error específicos
    filtered_stderr = FilteredStderr("Cannot reuse HTTP connection for different host")
    
    # Usar el filtro para suprimir los mensajes de error de conexión HTTP
    with contextlib.redirect_stderr(filtered_stderr):
        while processed_frames < frame_count:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video o error al leer.")
                break
            
            real_frame += 1
            
            # Procesar sólo cada enésimo fotograma
            if (real_frame - 1) % FRAME_SKIP != 0:
                continue
    
            # Realizar la predicción usando YOLO
            results = model.predict(
                frame, 
                imgsz=INPUT_RESOLUTION,
                conf=CONFIDENCE, 
                verbose=False, 
                classes=VEHICLE_CLASSES,
                device=device
            )
            
            # Lista de detecciones actuales
            current_detections = []
            
            # Extraer las detecciones del frame actual
            for result in results:
                for detection in result.boxes:
                    class_id = int(detection.cls)
                    if class_id in VEHICLE_CLASSES:
                        confidence = float(detection.conf)
                        box = detection.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        
                        # Solo procesar detecciones dentro del ROI
                        if is_in_roi(box, ROI_COORDS, frame_width, frame_height):
                            current_detections.append({'box': box, 'class_id': class_id, 'confidence': confidence})
            
            # Actualizar vehículos rastreados
            tracked_vehicles, unique_vehicle_counts = process_detections(
                current_detections, 
                tracked_vehicles, 
                processed_frames, 
                unique_vehicle_counts,
                frame_width,
                frame_height
            )
            
            # Dibujar los resultados en el cuadro
            annotated_frame = annotate_frame(
                frame, 
                results, 
                unique_vehicle_counts, 
                frame_width, 
                frame_height,
                tracked_vehicles
            )
    
            # Escribir el cuadro procesado en el video de salida
            output_video.write(annotated_frame)
    
            # Incrementar el contador de cuadros procesados
            processed_frames += 1
            
            # Imprimir el progreso cada 30 fotogramas
            if processed_frames % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = processed_frames / elapsed
                remaining = (frame_count - processed_frames) / fps_current
                print(f"- ({processed_frames/frame_count*100:.1f}%) Procesado - {processed_frames}/{frame_count} Frames procesados - {remaining:.1f}s restantes aprox")
    
    # Liberar recursos
    cap.release()
    output_video.release()
    
    end_time = time.time()
    total_time = end_time - start_time
    average_fps = processed_frames / total_time
    
    print(f"Tiempo de procesamiento total: {total_time:.2f} segundos")
    print(f"Vehículos contados: {unique_vehicle_counts[2]}")
    
    return unique_vehicle_counts

def main():
    """Función principal que orquesta todo el proceso"""
    # Configurar GPU y obtener dispositivo
    device = setup_gpu()
    
    # Inicializar modelo
    print("Cargando modelo YOLO...")
    model = initialize_model(device)
    
    # Obtener el stream del video
    print("Conexión al stream de YouTube...")
    try:
        video_stream = get_youtube_stream(YOUTUBE_URL)
    except Exception as e:
        print(f"Error al conectar con YouTube: {e}")
        return
    
    # Procesar el video
    print("Procesando vídeo...")
    unique_vehicle_counts = process_video(
        model, 
        device, 
        video_stream, 
        DURATION, 
        TARGET_FPS
    )
    
    if unique_vehicle_counts:
        # Guardar conteo de vehículos en JSON
        save_vehicle_counts_to_json(unique_vehicle_counts)
        print("Procesamiento completado con éxito.")

if __name__ == "__main__":
    main()