# Parámetros de rendimiento
INPUT_RESOLUTION = 640  # Ajuste a una resolución más baja para un procesamiento más rápido (original: 640)
FRAME_SKIP = 1          # Procesar sólo cada enésima imagen (1 = cada imagen, 2 = cada dos imágenes, etc.)
CONFIDENCE = 0.31       # Fijar umbral de confianza (más alto = menos detecciones pero más rápido)

# CONFIGURACIÓN DEL VÍDEO
# Vehículos rotonda Cádiz
YOUTUBE_URL = "https://www.youtube.com/watch?v=0dKNLFFcHFU" 
DURATION = 60           # Duración del vídeo de salida en segundos
TARGET_FPS = 30         # FPS objetivo para el vídeo de salida

# Configuración de clases y detección
CLASS_NAMES = {
    2: "vehicle",
    3: "vehicle",
    5: "vehicle",
    7: "vehicle"
}

# Traducción de nombres de clase
SPANISH_NAMES = {
    "vehicle": "vehiculo"
}

# Incluir todos los tipos de vehículos en COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7]  # coche, moto, autobús, camión

# Parámetros para la detección de vehículos únicos
IOU_THRESHOLD = 0.15        # Umbral de IOU más bajo para hacer más fácil el matching
OVERLAP_THRESHOLD = 0.4    # Umbral para agrupar detecciones superpuestas (más bajo)
DISAPPEAR_THRESHOLD = 40   # Número de frames para considerar que un vehículo ha abandonado la escena
COOLDOWN_FRAMES = 8        # Frames a esperar antes de contar un nuevo vehículo (reducido)
MIN_CONSECUTIVE_DETECTIONS = 1  # Solo necesita ser detectado una vez para ser contado
MAX_PREDICTION_FRAMES = 15  # Predecir posición por más frames cuando no es detectado
CLASS_HISTORY_SIZE = 5      # Tamaño del historial de clases para estabilizar la identificación

# =====================================
# CONFIGURACIÓN DE ROI (REGION OF INTEREST)
# =====================================
# Define el rectángulo donde se realizará el conteo (x1, y1, x2, y2)
# Los valores son porcentajes del ancho/alto del video (0-1)
ROI_ENABLED = True  # Activar/desactivar el ROI
ROI_COORDS = [0.8, 0.63, 1.0, 0.85]  # [x1, y1, x2, y2]
ROI_COLOR = (0, 255, 0)  # Color del rectángulo (BGR)
ROI_THICKNESS = 2  # Grosor de la línea del rectángulo
