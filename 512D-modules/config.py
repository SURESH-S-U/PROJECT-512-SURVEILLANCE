import os
import warnings
import faiss
import numpy as np
import insightface

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

# Model configuration
MODEL_NAME = 'buffalo_l'
MODEL_PROVIDERS = ['CUDAExecutionProvider']
DET_SIZE = (640, 640)
DET_THRESHOLD = 0.75

# Initialize the model
model = insightface.app.FaceAnalysis(name=MODEL_NAME, providers=MODEL_PROVIDERS)
model.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=DET_THRESHOLD)
print("\n✅ Model loaded successfully!")

# Directory paths
FACES_DIR = "Data/Faces"
KNOWN_FACES_DIR = os.path.join(FACES_DIR, "known")
UNKNOWN_FACES_DIR = os.path.join(FACES_DIR, "unknown")

# Time intervals
FACE_UPDATE_INTERVAL = 5  # seconds
SAVE_INTERVAL = 5  # seconds
GRACE_PERIOD = 15  # seconds

# Thresholds
UNKNOWN_THRESHOLD = 10
SIMILARITY_THRESHOLD = 0.7
IOU_THRESHOLD = 0.5

# Database settings
EMBEDDING_DIM = 512

# File paths
DB_FAISS_FILE = "face_db.faiss"
NAME_MAPPING_FILE = "name_mapping.json"
KNOWN_LOG_FILE = "detect_known.json"
UNKNOWN_LOG_FILE = "detect_unknown.json"

# Global variables (initialized with default values, will be updated by other modules)
cleanup_timer = None
face_last_saved = {}
face_trackers = {}
known_detections = {}
unknown_detections = {}
last_save_time = 0
unknown_counter = 1