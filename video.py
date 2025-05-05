import cv2
import os
import yt_dlp
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional

# Import config from utiles
from utiles import CONFIG, calculate_center, is_in_roi

logger = logging.getLogger(__name__)

def get_youtube_stream(video_url: str) -> str:
    """Fetches the best available stream URL for a given YouTube video URL."""
    ydl_opts = {
        'format': f'best[height>={CONFIG["youtube"]["min_height"]}][height<={CONFIG["youtube"]["max_height"]}]/best[height<={CONFIG["youtube"]["fallback_max_height"]}]/best',
        'quiet': True,
        'buffersize': CONFIG["youtube"]["buffer_size"],
        'no-check-certificate': True,
        'socket_timeout': CONFIG["youtube"]["socket_timeout"],
        'retries': CONFIG["youtube"]["retries"]
    }

    logger.info(f"Attempting to fetch stream URL for: {video_url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

            if 'url' in info:
                logger.info("Direct stream URL found.")
                return info['url']

            elif 'formats' in info and info['formats']:
                # Use height parameters from config
                suitable_formats = [f for f in info['formats']
                                   if f.get('height', 0) >= CONFIG["youtube"]["min_height"] and f.get('height', 0) <= CONFIG["youtube"]["max_height"]
                                   and f.get('url')]

                if not suitable_formats:
                    # Fallback to any format with a URL if no suitable resolution found
                    logger.warning(f"No stream found in {CONFIG['youtube']['min_height']}p-{CONFIG['youtube']['max_height']}p range, looking for any available format.")
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

def initialize_model(device: str) -> YOLO:
    """Initializes and loads the YOLO model onto the specified device."""
    model_path = CONFIG['model']['model_path']
    # Use print for the required output
    print(f"Cargando modelo ({model_path}) en el dispositivo ({device})")
    try:
        model = YOLO(model_path)
        model.to(device)
        # logger.info("Model loaded successfully.") # Removed success message
        return model
    except Exception as e:
        # Use logger.error for actual errors
        logging.error(f"Failed to load model: {e}")
        raise

def create_video_writer(cap: cv2.VideoCapture, target_fps: int) -> Tuple[cv2.VideoWriter, int, int, str]:
    """Creates a VideoWriter object for saving the annotated video."""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videos_dir = CONFIG['output']['video_dir']
    if not os.path.exists(videos_dir):
        try:
            os.makedirs(videos_dir)
            # logger.info(f"Created output directory: {videos_dir}") # Suppressed
        except OSError as e:
            logging.error(f"Failed to create output directory {videos_dir}: {e}")
            videos_dir = "."

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_filename = os.path.join(videos_dir, f'video_{current_datetime}.avi')

    fourcc = cv2.VideoWriter_fourcc(*CONFIG['output']['video']['codec'])
    # Use print for the required output
    print(f"Guardando video de salida en: {video_filename}, FPS: {target_fps}, Resolution: {frame_width}x{frame_height}, Codec: {CONFIG['output']['video']['codec']}")

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
        logging.error(f"Failed to create VideoWriter: {e}")
        raise

# Define a list of distinct colors (BGR format)
# Add more colors if you have more classes
CLASS_COLORS = [
    (255, 0, 0),    # Blue    (Index 0)
    (0, 255, 0),    # Lime    (Index 1)
    (0, 0, 255),    # Red     (Index 2)
    (255, 255, 0),  # Cyan    (Index 3)
    (255, 0, 255),  # Magenta (Index 4)
    (255, 165, 0),  # Orange (Index 5 - Assuming Bus ID is 5, changed from Yellow)
    (128, 0, 0),    # Navy    (Index 6)
    (0, 128, 0),    # Green   (Index 7)
    (0, 0, 128),    # Maroon  (Index 8)
    (128, 128, 0),  # Teal    (Index 9)
    (128, 0, 128),  # Purple  (Index 10)
    (0, 128, 128),  # Olive   (Index 11)
    (192, 192, 192),# Silver  (Index 12)
    (128, 128, 128),# Gray    (Index 13)
    (0, 255, 255),  # Yellow (Index 14 - Changed from Orange to avoid duplicate)
    (255, 192, 203) # Pink    (Index 15)
]
DEFAULT_COLOR = (100, 100, 100) # Default color for unknown classes

def annotate_frame(
    frame: np.ndarray,
    results: List[Any], # Type hint for results might need refinement based on ultralytics output
    unique_vehicle_counts: Dict[int, int],
    frame_width: int,
    frame_height: int,
    tracked_vehicles: Optional[Dict[int, Dict[str, Any]]] = None
) -> np.ndarray:
    """Annotates a single frame with detection boxes, ROI, tracking info, and counts."""
    annotated_frame = frame.copy() # Start with a fresh copy of the frame
    # Define a darker green color (BGR)
    green_color = (0, 128, 0) # Changed from (0, 255, 0)

    roi_config = CONFIG['roi']
    if roi_config['enabled']:
        # ... (ROI drawing code remains the same)
        roi_coords = roi_config['coords']
        # Use the defined darker green color for ROI
        roi_color = green_color # Use darker green for ROI
        roi_thickness = roi_config['thickness']

        roi_x1 = int(roi_coords[0] * frame_width)
        roi_y1 = int(roi_coords[1] * frame_height)
        roi_x2 = int(roi_coords[2] * frame_width)
        roi_y2 = int(roi_coords[3] * frame_height)

        # --- Remove semi-transparent overlay --- 
        # overlay_color = tuple(roi_config['color'])
        # overlay = annotated_frame.copy()
        # cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), overlay_color, -1)
        # cv2.addWeighted(overlay, CONFIG['annotation']['roi_overlay_alpha'], annotated_frame, 1.0 - CONFIG['annotation']['roi_overlay_alpha'], 0, annotated_frame)
        # --- End removal --- 

        # Draw ROI border and text using darker green color
        cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, roi_thickness)
        cv2.putText(annotated_frame, "ROI", (roi_x1 + 10, roi_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)

    # Draw tracking information if available
    if tracked_vehicles:
        class_names = CONFIG['model']['class_names']
        for vehicle_id, vehicle_data in tracked_vehicles.items():
            box = vehicle_data['box']
            is_predicted = vehicle_data.get('predicted', False)
            stability = vehicle_data.get('detection_stability', 0)
            class_id = vehicle_data.get('class_id', 2) # Default to a vehicle class if missing

            class_name = class_names.get(class_id, f"cls_{class_id}") # Use class name or ID

            # --- Determine color based on class ID --- 
            # Use modulo operator to cycle through colors if more classes than defined colors
            color_index = class_id % len(CLASS_COLORS)
            color = CLASS_COLORS[color_index]
            # --- End color determination --- 

            thickness = 1 if is_predicted else 2
            line_type = cv2.LINE_AA

            # Draw box (dashed if predicted)
            if is_predicted:
                # Draw dashed line for predicted box
                dash_length = 10
                num_dashes_x = int((box[2] - box[0]) / dash_length)
                num_dashes_y = int((box[3] - box[1]) / dash_length)
                # Draw top and bottom dashed lines
                for i in range(0, num_dashes_x, 2):
                    start_x = int(box[0] + i * dash_length)
                    end_x = int(min(box[0] + (i + 1) * dash_length, box[2]))
                    cv2.line(annotated_frame, (start_x, int(box[1])), (end_x, int(box[1])), color, thickness, lineType=line_type)
                    cv2.line(annotated_frame, (start_x, int(box[3])), (end_x, int(box[3])), color, thickness, lineType=line_type)
                # Draw left and right dashed lines
                for i in range(0, num_dashes_y, 2):
                    start_y = int(box[1] + i * dash_length)
                    end_y = int(min(box[1] + (i + 1) * dash_length, box[3]))
                    cv2.line(annotated_frame, (int(box[0]), start_y), (int(box[0]), end_y), color, thickness, lineType=line_type)
                    cv2.line(annotated_frame, (int(box[2]), start_y), (int(box[2]), end_y), color, thickness, lineType=line_type)
            else:
                cv2.rectangle(annotated_frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color, thickness, lineType=line_type)

            # Draw label
            label_text = f"ID:{vehicle_id} ({class_name[:3]}) S:{int(stability)}"
            # Use font scale from config
            cv2.putText(annotated_frame, label_text,
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, CONFIG['annotation']['tracking_label_font_scale'], color, 1, cv2.LINE_AA)

    # Draw count display
    # Use the defined darker green color for count display
    count_display_color = green_color # Use darker green for count text
    title_text = "Conteo"
    title_size, _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, CONFIG['annotation']['count_display_title_scale'], 2)
    title_x = frame_width - title_size[0] - 10 # Align title to the right
    y_pos = 40

    cv2.putText(annotated_frame, title_text, (title_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, CONFIG['annotation']['count_display_title_scale'], count_display_color, 2)
    y_pos += 30

    # --- Remove display counts per class --- 
    # class_names = CONFIG['model']['class_names']
    # spanish_names = CONFIG['model']['spanish_names']
    # text_scale = CONFIG['annotation']['count_display_text_scale']
    # text_thickness = 2

    # # Sort class IDs for consistent display order (optional)
    # sorted_class_ids = sorted(unique_vehicle_counts.keys())

    # for class_id in sorted_class_ids:
    #     count = unique_vehicle_counts[class_id]
    #     # Get English name, then Spanish name
    #     english_name = class_names.get(class_id, f"Clase {class_id}")
    #     display_name = spanish_names.get(english_name, english_name).capitalize()
    #     text = f"{display_name}: {count}"
    #     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    #     text_x = frame_width - text_size[0] - 10 # Align text to the right
    #     cv2.putText(annotated_frame, text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, text_scale, count_display_color, text_thickness)
    #     y_pos += 20 # Increment y position for the next line
    # --- End removal --- 

    # --- Display Total Count --- 
    # y_pos += 10 # Adjust position if per-class counts are removed
    spanish_names = CONFIG['model']['spanish_names'] # Define spanish_names here
    text_scale = CONFIG['annotation']['count_display_text_scale'] # Ensure text_scale is defined
    text_thickness = 2 # Ensure text_thickness is defined
    total_spanish_name = spanish_names.get("vehicle_total", "Total").capitalize()
    total_vehicles = sum(unique_vehicle_counts.values()) # Sum all counted vehicles
    text = f"{total_spanish_name}: {total_vehicles}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
    text_x = frame_width - text_size[0] - 10 # Align text to the right
    cv2.putText(annotated_frame, text, (text_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, text_scale, count_display_color, text_thickness)
    # --- End Total Count --- 

    return annotated_frame