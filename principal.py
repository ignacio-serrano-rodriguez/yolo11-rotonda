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
    """Procesa el video, detecta y cuenta vehículos con buffer para evitar cortes"""
    import cv2
    import numpy as np
    from collections import deque
    
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
    
    # Crear buffer de frames para manejar interrupciones
    frame_buffer = deque(maxlen=30)  # Buffer de 30 frames (~1 segundo a 30fps)
    
    # Leer y procesar los cuadros del video
    processed_frames = 0
    real_frame = 0
    
    print(f"Resolución del vídeo: {INPUT_RESOLUTION}x{INPUT_RESOLUTION}\nSalto de frame: {FRAME_SKIP}\nConfianza: {CONFIDENCE}")
    print(f"Parámetros de tracking: IOU={IOU_THRESHOLD}, Desaparición={DISAPPEAR_THRESHOLD}, Min_Detecciones={MIN_CONSECUTIVE_DETECTIONS}")
    print(f"Historial de clases: {CLASS_HISTORY_SIZE} frames")
    
    # Variables para manejar el estado del procesamiento
    current_results = None
    stream_failed = False
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    
    # Prellenar el buffer con algunos frames
    print("Prellenando buffer de frames...")
    while len(frame_buffer) < frame_buffer.maxlen and len(frame_buffer) < 15:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame.copy())
    
    # Crear filtro para mensajes de error específicos
    filtered_stderr = FilteredStderr("Cannot reuse HTTP connection for different host")
    
    # Usar el filtro para suprimir los mensajes de error de conexión HTTP
    with contextlib.redirect_stderr(filtered_stderr):
        while processed_frames < frame_count:
            # Intentar obtener un nuevo frame solo si no estamos en modo de fallo
            if not stream_failed:
                ret, frame = cap.read()
                if ret:
                    # Añadir frame al buffer
                    frame_buffer.append(frame.copy())
                    reconnect_attempts = 0  # Resetear contador de reconexión
                else:
                    stream_failed = True
                    print("Error al leer frame. Usando buffer...")
            
            # Si el stream falló, intentar reconectar
            if stream_failed:
                if reconnect_attempts < max_reconnect_attempts:
                    try:
                        print(f"Intento de reconexión {reconnect_attempts+1}/{max_reconnect_attempts}...")
                        cap.release()
                        time.sleep(1)  # Esperar brevemente antes de reintentar
                        video_stream = get_youtube_stream(YOUTUBE_URL)
                        cap = cv2.VideoCapture(video_stream)
                        if cap.isOpened():
                            ret, test_frame = cap.read()
                            if ret:
                                # Reconexión exitosa
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
            
            # Obtener un frame para procesar - del buffer si está disponible o el último procesado
            if len(frame_buffer) > 0:
                current_frame = frame_buffer.popleft()
            elif 'annotated_frame' in locals():
                # Si el buffer está vacío, duplicar el último frame anotado
                current_frame = annotated_frame.copy()
            else:
                # Si no hay frames en el buffer ni frames procesados, no podemos continuar
                print("No hay frames disponibles para procesar.")
                break
            
            real_frame += 1
            
            # Decidir si procesamos este frame para detección
            process_this_frame = (real_frame - 1) % FRAME_SKIP == 0
            
            if process_this_frame:
                # Realizar la predicción usando YOLO
                results = model.predict(
                    current_frame, 
                    imgsz=INPUT_RESOLUTION,
                    conf=CONFIDENCE, 
                    verbose=False, 
                    classes=VEHICLE_CLASSES,
                    device=device
                )
                current_results = results
                
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
            
            # Anotar cada frame (procesado o no)
            if current_results is not None:
                # Dibujar los resultados en el cuadro
                annotated_frame = annotate_frame(
                    current_frame, 
                    current_results, 
                    unique_vehicle_counts, 
                    frame_width, 
                    frame_height,
                    tracked_vehicles
                )
            else:
                # Si aún no tenemos resultados (primeros frames)
                annotated_frame = current_frame.copy()
            
            # Escribir cada frame al video de salida
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
    
    print("Procesamiento completado. Vídeo guardado en:", video_filename)
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