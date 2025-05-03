import numpy as np
from collections import deque
from configuracion import (
    IOU_THRESHOLD, OVERLAP_THRESHOLD, DISAPPEAR_THRESHOLD, 
    MIN_CONSECUTIVE_DETECTIONS, MAX_PREDICTION_FRAMES, CLASS_HISTORY_SIZE,
    COOLDOWN_FRAMES
)
from utiles import (
    calculate_iou, calculate_center, calculate_size, calculate_distance, 
    calculate_overlap_area, is_box_contained, get_most_common_class
)

def process_detections(current_detections, tracked_vehicles, processed_frames, unique_vehicle_counts, frame_width, frame_height):
    """Procesa las detecciones actuales y actualiza el seguimiento de vehículos"""
    matched_vehicles = set()
    next_vehicle_id = max(tracked_vehicles.keys()) + 1 if tracked_vehicles else 0
    
    # Detecciones de grupo que probablemente se refieran al mismo vehículo físico
    grouped_detections = []
    used_detection_indices = set()
    
    # Ordenar las detecciones por confianza (primero la más alta)
    sorted_detections = sorted(enumerate(current_detections), key=lambda x: x[1]['confidence'], reverse=True)
    
    # Agrupar detecciones superpuestas - Se encarga de múltiples detecciones del mismo vehículo.
    for i, detection1 in sorted_detections:
        if i in used_detection_indices:
            continue
            
        group = [detection1]
        used_detection_indices.add(i)
        
        # Buscar todas las detecciones solapadas
        for j, detection2 in enumerate(current_detections):
            if j in used_detection_indices or i == j:
                continue
                
            # Compruebe si es probable que se trate del mismo vehículo con diferente clase
            overlap = calculate_overlap_area(detection1['box'], detection2['box'])
            iou = calculate_iou(detection1['box'], detection2['box'])
            
            # MODIFICADO: Aumentar umbral para considerar detecciones separadas
            # Esto evita agrupar vehículos diferentes que estén cercanos
            if (overlap > 0.9 or iou > 0.85 or 
                is_box_contained(detection1['box'], detection2['box'], threshold=0.95)):
                group.append(detection2)
                used_detection_indices.add(j)
                
        # Utilizar la detección de mayor confianza del grupo
        best_detection = max(group, key=lambda x: x['confidence'])
        grouped_detections.append(best_detection)
    
    # Actualizar las pistas activas con nuevas detecciones
    unmatched_detections = []
    
    # Primera pasada de concordancia: concordancia con los vehículos existentes utilizando nuestra puntuación compuesta.
    for detection in grouped_detections:
        best_match = None
        best_score = IOU_THRESHOLD  # Umbral mínimo para considerar una coincidencia
        det_center = calculate_center(detection['box'])
        det_size = calculate_size(detection['box'])
        
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            # Calcular IOU entre detección y vehículo rastreado
            iou = calculate_iou(vehicle_data['box'], detection['box'])
            
            # Calcular superposición de áreas (mejor para manejar cajas contenidas)
            area_overlap = calculate_overlap_area(vehicle_data['box'], detection['box'])
            
            # Calcular la distancia entre ejes normalizada por el tamaño del fotograma
            veh_center = calculate_center(vehicle_data['box'])
            distance = calculate_distance(det_center, veh_center)
            norm_distance = distance / max(frame_width, frame_height)
            
            # Calcular la similitud de tamaño (1,0 = mismo tamaño)
            veh_size = calculate_size(vehicle_data['box'])
            size_ratio = min(det_size, veh_size) / max(det_size, veh_size)
            
            # Comprobar la coherencia de la dirección si se dispone de la velocidad
            direction_score = 1.0
            if 'velocity' in vehicle_data and np.linalg.norm(vehicle_data['velocity']) > 1.0:
                # Calcular la dirección del movimiento
                dx = det_center[0] - veh_center[0]
                dy = det_center[1] - veh_center[1]
                current_direction = np.array([dx, dy])
                
                # Normalizar direcciones
                prev_direction = np.array(vehicle_data['velocity'])
                if np.linalg.norm(current_direction) > 0 and np.linalg.norm(prev_direction) > 0:
                    current_direction = current_direction / np.linalg.norm(current_direction)
                    prev_direction = prev_direction / np.linalg.norm(prev_direction)
                    
                    # Coseno de similitud de direcciones (-1 a 1, donde 1 es la misma dirección)
                    cos_sim = np.dot(current_direction, prev_direction)
                    # Convertir a una puntuación entre 0 y 1
                    direction_score = (cos_sim + 1) / 2
            
            # Factor del tiempo transcurrido desde la última vez que se vio (favorece a los vehículos vistos recientemente)
            frames_since_seen = processed_frames - vehicle_data['last_seen']
            recency_factor = max(0, 1.0 - (frames_since_seen / DISAPPEAR_THRESHOLD))
            
            # Penalización por cambio de clase: menor si el vehículo cambia a menudo de clase.
            class_consistency = 1.0
            if 'class_history' in vehicle_data and len(vehicle_data['class_history']) > 1:
                # Contar clases únicas en la historia
                unique_classes = len(set(vehicle_data['class_history']))
                # Más clases únicas = más cambios = menor puntuación
                class_consistency = 1.0 / unique_classes
                
            # Puntuación de concordancia combinada (componentes ponderados)
            score = (iou * 0.4) + (area_overlap * 0.2) + (size_ratio * 0.1) + \
                   ((1.0 - norm_distance) * 0.15) + (direction_score * 0.1) + \
                   (recency_factor * 0.05)
            
            # Si las casillas tienen una relación de contención significativa, aumenta la puntuación
            if is_box_contained(vehicle_data['box'], detection['box']):
                score += 0.15
                
            if score > best_score:
                best_score = score
                best_match = vehicle_id
        
        if best_match is not None:
            # Actualizar vehículo oruga
            vehicle_data = tracked_vehicles[best_match]
            
            # Calculate movement vector for future prediction
            old_center = calculate_center(vehicle_data['box'])
            new_center = calculate_center(detection['box'])
            if 'velocity' not in vehicle_data:
                vehicle_data['velocity'] = (0, 0)
            
            # Actualizar la velocidad como media móvil
            alpha = 0.7  # Peso para la nueva medición
            dx = new_center[0] - old_center[0]
            dy = new_center[1] - old_center[1]
            vehicle_data['velocity'] = (
                alpha * dx + (1-alpha) * vehicle_data['velocity'][0],
                alpha * dy + (1-alpha) * vehicle_data['velocity'][1]
            )
            
            # Actualizar el historial de clases de este vehículo
            if 'class_history' not in vehicle_data:
                vehicle_data['class_history'] = deque(maxlen=CLASS_HISTORY_SIZE)
            vehicle_data['class_history'].append(detection['class_id'])
            
            # Establecer la clase actual a la más común en la historia
            vehicle_data['class_id'] = get_most_common_class(vehicle_data['class_history'])
            
            vehicle_data['box'] = detection['box']
            vehicle_data['last_seen'] = processed_frames
            vehicle_data['confidence'] = detection['confidence']
            vehicle_data['consecutive_matches'] = vehicle_data.get('consecutive_matches', 0) + 1
            vehicle_data['predicted'] = False
            vehicle_data['detection_stability'] = vehicle_data.get('detection_stability', 0) + 1
            matched_vehicles.add(best_match)
        else:
            # No se han encontrado coincidencias, manéjese como posible vehículo nuevo
            unmatched_detections.append(detection)
    
    # Segunda pasada de coincidencia: intenta coincidir con los vehículos perdidos recientemente utilizando la posición prevista.
    remaining_unmatched = []
    for detection in unmatched_detections:
        det_center = calculate_center(detection['box'])
        potential_match = False
        
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            if vehicle_id in matched_vehicles:
                continue
                
            # Para vehículos vistos recientemente pero no en este marco
            frames_not_seen = processed_frames - vehicle_data['last_seen']
            if frames_not_seen < MAX_PREDICTION_FRAMES:
                # Predecir dónde debe estar el vehículo en función de la última velocidad
                if 'velocity' in vehicle_data:
                    # Predecir la posición en función de la velocidad
                    old_center = calculate_center(vehicle_data['box'])
                    predicted_x = old_center[0] + vehicle_data['velocity'][0] * frames_not_seen
                    predicted_y = old_center[1] + vehicle_data['velocity'][1] * frames_not_seen
                    
                    # Calcular la distancia entre la detección y la posición prevista
                    distance = calculate_distance(det_center, (predicted_x, predicted_y))
                    # Normalizar por tamaño de fotograma
                    normalized_dist = distance / max(frame_width, frame_height)
                    
                    # Calcular la similitud de tamaño
                    det_size = calculate_size(detection['box'])
                    veh_size = calculate_size(vehicle_data['box'])
                    size_ratio = min(det_size, veh_size) / max(det_size, veh_size)
                    
                    # Si la posición prevista es lo suficientemente cercana y el tamaño es similar
                    match_quality = (1 - normalized_dist) * 0.7 + size_ratio * 0.3
                    if match_quality > 0.6:  # Umbral combinado de posición y tamaño
                        # Actualizar el historial de clases
                        if 'class_history' not in vehicle_data:
                            vehicle_data['class_history'] = deque(maxlen=CLASS_HISTORY_SIZE)
                        vehicle_data['class_history'].append(detection['class_id'])
                        
                        vehicle_data['box'] = detection['box']
                        vehicle_data['last_seen'] = processed_frames
                        vehicle_data['confidence'] = detection['confidence']
                        vehicle_data['predicted'] = False
                        vehicle_data['consecutive_matches'] = vehicle_data.get('consecutive_matches', 0) + 1
                        matched_vehicles.add(vehicle_id)
                        potential_match = True
                        break
        
        if not potential_match:
            remaining_unmatched.append(detection)
    
    # Procesar detecciones realmente inigualables: nuevos vehículos potenciales
    for detection in remaining_unmatched:
        det_center = calculate_center(detection['box'])
        
        # Compruebe si esta detección se encuentra en una región en la que hemos contabilizado recientemente un vehículo
        # Esto ayuda a evitar el doble recuento cuando los vehículos desaparecen y reaparecen temporalmente.
        too_close_to_existing = False
        potential_duplicate = False
        duplicate_vehicle_id = None
        
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            # Compruebe todos los vehículos, incluidos los emparejados
            veh_center = calculate_center(vehicle_data['box'])
            distance = calculate_distance(det_center, veh_center)
            overlap = calculate_overlap_area(vehicle_data['box'], detection['box'])
            
            # MODIFICACIÓN: Reducir la distancia mínima para considerar vehículos diferentes
            # Esto permitirá contar vehículos cercanos entre sí
            frames_since_seen = processed_frames - vehicle_data['last_seen']
            if distance < 35 and frames_since_seen < COOLDOWN_FRAMES:  # Reducido de 60 a 35 píxeles
                too_close_to_existing = True
                break
                
            # MODIFICACIÓN: Reducir la distancia para verificar vehículos que se han ido
            if frames_since_seen >= DISAPPEAR_THRESHOLD and distance < 25:  # Reducido de 40 a 25 píxeles
                too_close_to_existing = True
                break
                
            # MODIFICACIÓN: Aumentar el umbral de solapamiento para detectar duplicados
            # Esto permitirá que detecciones parciales cercanas se consideren vehículos diferentes
            if overlap > 0.8 and frames_since_seen < 5:  # Aumentado de 0.6 a 0.8
                potential_duplicate = True
                duplicate_vehicle_id = vehicle_id
                break
                
        # Si la detección no está demasiado cerca de un vehículo existente
        if not too_close_to_existing and not potential_duplicate:
            # Crear una nueva pista de vehículos
            tracked_vehicles[next_vehicle_id] = {
                'box': detection['box'],
                'class_id': detection['class_id'],
                'class_history': deque([detection['class_id']], maxlen=CLASS_HISTORY_SIZE),
                'last_seen': processed_frames,
                'confidence': detection['confidence'],
                'velocity': (0, 0),
                'consecutive_matches': 1,
                'is_counted': False,
                'predicted': False,
                'detection_stability': 1
            }
            
            matched_vehicles.add(next_vehicle_id)
            next_vehicle_id += 1
        elif potential_duplicate and duplicate_vehicle_id is not None:
            # Esto es probablemente una detección parcial de un vehículo existente
            # Actualizar el vehículo existente con la nueva detección
            vehicle_data = tracked_vehicles[duplicate_vehicle_id]
            
            # Añadir esta clase al historial
            if 'class_history' not in vehicle_data:
                vehicle_data['class_history'] = deque(maxlen=CLASS_HISTORY_SIZE)
            vehicle_data['class_history'].append(detection['class_id'])
            
            # Actualizar otros campos si aún no coinciden
            if duplicate_vehicle_id not in matched_vehicles:
                vehicle_data['last_seen'] = processed_frames
                vehicle_data['predicted'] = False
                matched_vehicles.add(duplicate_vehicle_id)
    
    # Modificar la sección donde se verifica si las detecciones están demasiado cerca

    # En la función process_detections, ubicar la sección que determina new_vehicles
    for detection in remaining_unmatched:
        det_center = calculate_center(detection['box'])
        
        # Verificar si esta detección está en una región donde recientemente contamos un vehículo
        too_close_to_existing = False
        potential_duplicate = False
        duplicate_vehicle_id = None
        
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            # Verificar todos los vehículos, incluyendo los ya emparejados
            veh_center = calculate_center(vehicle_data['box'])
            distance = calculate_distance(det_center, veh_center)
            overlap = calculate_overlap_area(vehicle_data['box'], detection['box'])
            
            # MODIFICACIÓN: Reducir aún más el umbral de distancia para permitir
            # vehículos muy cercanos entre sí
            frames_since_seen = processed_frames - vehicle_data['last_seen']
            if distance < 20 and frames_since_seen < COOLDOWN_FRAMES:  # Reducido de 35 a 20 píxeles
                too_close_to_existing = True
                break
                
            # Reducir la distancia para verificar vehículos que ya pasaron
            if frames_since_seen >= DISAPPEAR_THRESHOLD and distance < 15:  # Reducido de 25 a 15 píxeles
                too_close_to_existing = True
                break
                
            # Reducir umbral de superposición para distinguir mejor vehículos cercanos
            if overlap > 0.7 and frames_since_seen < 5:  # Reducido de 0.8 a 0.7
                potential_duplicate = True
                duplicate_vehicle_id = vehicle_id
                break
                
        # Si la detección no está demasiado cerca de un vehículo existente
        if not too_close_to_existing and not potential_duplicate:
            # Crear un nuevo seguimiento de vehículo
            tracked_vehicles[next_vehicle_id] = {
                'box': detection['box'],
                'class_id': detection['class_id'],
                'class_history': deque([detection['class_id']], maxlen=CLASS_HISTORY_SIZE),
                'last_seen': processed_frames,
                'confidence': detection['confidence'],
                'velocity': (0, 0),
                'consecutive_matches': 1,
                'is_counted': False,
                'predicted': False,
                'detection_stability': 1
            }
            
            matched_vehicles.add(next_vehicle_id)
            next_vehicle_id += 1
    
    # Actualizar las predicciones de posición de los vehículos no emparejados
    for vehicle_id, vehicle_data in tracked_vehicles.items():
        if vehicle_id not in matched_vehicles and 'velocity' in vehicle_data:
            frames_since_seen = processed_frames - vehicle_data['last_seen']
            
            # Predecir sólo para vehículos vistos recientemente
            if frames_since_seen < MAX_PREDICTION_FRAMES:
                # Obtener la caja y el centro actuales
                old_box = vehicle_data['box']
                old_center = calculate_center(old_box)
                
                # Predecir el nuevo centro en función de la velocidad
                new_center_x = old_center[0] + vehicle_data['velocity'][0] * 1  # Predict just 1 frame ahead
                new_center_y = old_center[1] + vehicle_data['velocity'][1] * 1
                
                # Calcular el movimiento de la caja
                dx = new_center_x - old_center[0]
                dy = new_center_y - old_center[1]
                
                # Actualizar la posición de la caja
                new_box = [
                    old_box[0] + dx,
                    old_box[1] + dy,
                    old_box[2] + dx,
                    old_box[3] + dy
                ]
                
                vehicle_data['box'] = new_box
                vehicle_data['predicted'] = True
                # Disminución de la estabilidad de detección para las posiciones previstas
                if 'detection_stability' in vehicle_data:
                    vehicle_data['detection_stability'] = max(0, vehicle_data['detection_stability'] - 0.5)
    
    # Actualizar los contadores de los vehículos que han sido rastreados correctamente durante varias tramas.
    # y tienen suficiente estabilidad
    for vehicle_id, vehicle_data in tracked_vehicles.items():
        # Sólo se contabilizan los vehículos que han sido objeto de un seguimiento coherente
        consecutive_matches = vehicle_data.get('consecutive_matches', 0)
        stability = vehicle_data.get('detection_stability', 0)
        is_counted = vehicle_data.get('is_counted', False)
        class_id = vehicle_data.get('class_id', 2)  # Clase predeterminada si no existe
        
        # MODIFICACIÓN: Reducir el umbral de estabilidad para acelerar el conteo
        # de vehículos en el ROI
        # MODIFICACIÓN: Reducir requisitos para contar un vehículo en el ROI
        if not is_counted and consecutive_matches >= MIN_CONSECUTIVE_DETECTIONS and stability >= 2:  # Cambio de 3 a 2
            vehicle_data['is_counted'] = True
            
            # Usar siempre ID 2 como contador genérico para todos los vehículos
            if 2 not in unique_vehicle_counts:
                unique_vehicle_counts[2] = 0
                
            # Incrementar el contador para todos los vehículos independientemente de su clase
            unique_vehicle_counts[2] += 1
    
    # Retire los vehículos que no se han visto durante un tiempo
    vehicles_to_remove = []
    for vehicle_id, vehicle_data in tracked_vehicles.items():
        if processed_frames - vehicle_data['last_seen'] > DISAPPEAR_THRESHOLD:
            vehicles_to_remove.append(vehicle_id)
    
    for vehicle_id in vehicles_to_remove:
        tracked_vehicles.pop(vehicle_id)
    
    return tracked_vehicles, unique_vehicle_counts