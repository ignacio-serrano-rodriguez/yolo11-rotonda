import cv2
import os
import yt_dlp
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging

# Import config from utiles
from utiles import CONFIG, calculate_center, is_in_roi

logger = logging.getLogger(__name__)

def get_youtube_stream(video_url):
    """Fetches the best available stream URL for a given YouTube video URL."""
    ydl_opts = {
        'format': 'best[height>=360][height<=480]/best[height<=720]/best',
        'quiet': True,
        'buffersize': 32768,
        'no-check-certificate': True,
        'socket_timeout': 30,
        'retries': 10
    }

    logger.info(f"Attempting to fetch stream URL for: {video_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

            if 'url' in info:
                logger.info("Direct stream URL found.")
                return info['url']

            elif 'formats' in info and info['formats']:
                # Prioritize formats between 360p and 480p
                suitable_formats = [f for f in info['formats']
                                   if f.get('height', 0) >= 360 and f.get('height', 0) <= 480
                                   and f.get('url')]

                if not suitable_formats:
                    # Fallback to any format with a URL if no suitable resolution found
                    logger.warning("No stream found in 360p-480p range, looking for any available format.")
                    suitable_formats = [f for f in info['formats'] if f.get('url')]

                if suitable_formats:
                    # Prefer mp4 if available, otherwise sort by bitrate
                    mp4_formats = [f for f in suitable_formats if f.get('ext') == 'mp4']
                    if mp4_formats:
                        mp4_formats.sort(key=lambda x: x.get('tbr', 0)) # Sort by bitrate (lower might be more stable)
                        logger.info(f"Found MP4 stream: {mp4_formats[0]['url']}")
                        return mp4_formats[0]['url']

                    # If no mp4, sort all suitable formats by bitrate
                    suitable_formats.sort(key=lambda x: x.get('tbr', 0))
                    logger.info(f"Found non-MP4 stream: {suitable_formats[0]['url']}")
                    return suitable_formats[0]['url']

            logger.error("No suitable stream URL found in video info.")
            raise ValueError("No se encontró URL de stream") # Keep Spanish for consistency?
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error: {e}")
        raise ConnectionError(f"Error al obtener stream (yt-dlp): {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error getting YouTube stream: {e}")
        raise ConnectionError(f"Error inesperado al obtener stream: {str(e)}")

def initialize_model(device):
    """Initializes and loads the YOLO model onto the specified device."""
    model_path = CONFIG['model']['model_path']
    logger.info(f"Loading model from: {model_path} onto device: {device}")
    try:
        model = YOLO(model_path)
        model.to(device)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def create_video_writer(cap, target_fps):
    """Creates a VideoWriter object for saving the annotated video."""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videos_dir = CONFIG['output']['video_dir']
    if not os.path.exists(videos_dir):
        try:
            os.makedirs(videos_dir)
            logger.info(f"Created output directory: {videos_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {videos_dir}: {e}")
            # Fallback or raise error?
            videos_dir = "." # Fallback to current directory

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = os.path.join(videos_dir, f'video_{current_datetime}.avi')

    # Consider allowing codec choice via config
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Or 'XVID', 'mp4v'
    logger.info(f"Creating video writer. Filename: {video_filename}, FPS: {target_fps}, Resolution: {frame_width}x{frame_height}")

    try:
        writer = cv2.VideoWriter(
            video_filename,
            fourcc,
            target_fps,
            (frame_width, frame_height)
        )
        if not writer.isOpened():
             raise IOError("VideoWriter failed to open.")
        return writer, frame_width, frame_height, video_filename
    except Exception as e:
        logger.error(f"Failed to create VideoWriter: {e}")
        raise

def annotate_frame(frame, results, unique_vehicle_counts, frame_width, frame_height, tracked_vehicles=None):
    """Annotates a single frame with detection boxes, ROI, tracking info, and counts."""
    annotated_frame = results[0].plot() # Use YOLO's plotting

    roi_config = CONFIG['roi']
    if roi_config['enabled']:
        roi_coords = roi_config['coords']
        roi_color = tuple(roi_config['color']) # Ensure color is a tuple
        roi_thickness = roi_config['thickness']

        roi_x1 = int(roi_coords[0] * frame_width)
        roi_y1 = int(roi_coords[1] * frame_height)
        roi_x2 = int(roi_coords[2] * frame_width)
        roi_y2 = int(roi_coords[3] * frame_height)

        # Draw semi-transparent overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, -1)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

        # Draw ROI border and text
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, roi_thickness)
        cv2.putText(annotated_frame, "ROI", (roi_x1 + 10, roi_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

    # Draw tracking information if available
    if tracked_vehicles:
        class_names = CONFIG['model']['class_names']
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            box = vehicle_data['box']
            is_counted = vehicle_data.get('is_counted', False)
            is_predicted = vehicle_data.get('predicted', False)
            stability = vehicle_data.get('detection_stability', 0)
            class_id = vehicle_data.get('class_id', 2) # Default to a vehicle class if missing

            class_name = class_names.get(class_id, "vehicle") # Use class name from config

            # Determine color based on state
            if is_counted:
                color = (0, 255, 0) # Green
            elif stability >= 3:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 165, 255) # Orange

            thickness = 1 if is_predicted else 2
            line_type = cv2.LINE_AA

            # Draw box (dashed if predicted)
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

            # Draw label
            label_text = f"ID:{vehicle_id} ({class_name[:3]}) S:{int(stability)}"
            cv2.putText(annotated_frame, label_text,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Draw count display
    color = (0, 128, 255) # Orange-Red for count display
    title_text = "Conteo"
    title_size, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    title_x = frame_width - title_size[0] - 10
    y_pos = 40

    cv2.putText(annotated_frame, title_text, (title_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y_pos += 30

    # Use the spanish name from config
    spanish_name = CONFIG['model']['spanish_names'].get("vehicle", "vehiculo")
    total_vehicles = sum(unique_vehicle_counts.values()) # Sum all counted vehicles
    text = f"{spanish_name.capitalize()}: {total_vehicles}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_x = frame_width - text_size[0] - 10

    cv2.putText(annotated_frame, text, (text_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated_frame