import os
import json
import time

# Import from config
from config import (
    KNOWN_LOG_FILE, UNKNOWN_LOG_FILE,
    known_detections, unknown_detections
)

def load_detection_logs():
    """Load existing detection logs if they exist."""
    global known_detections, unknown_detections
    
    try:
        if os.path.exists(KNOWN_LOG_FILE):
            with open(KNOWN_LOG_FILE, 'r') as f:
                known_detections.update(json.load(f))
            print(f"✅ Loaded {len(known_detections)} known face logs")
        else:
            known_detections.clear()
            
        if os.path.exists(UNKNOWN_LOG_FILE):
            with open(UNKNOWN_LOG_FILE, 'r') as f:
                unknown_detections.update(json.load(f))
            print(f"✅ Loaded {len(unknown_detections)} unknown face logs")
        else:
            unknown_detections.clear()
    except Exception as e:
        print(f"⚠️ Error loading detection logs: {e}")
        known_detections.clear()
        unknown_detections.clear()

def save_detection_logs():
    """Save detection logs to JSON files."""
    try:
        with open(KNOWN_LOG_FILE, 'w') as f:
            json.dump(known_detections, f, indent=2)
        
        with open(UNKNOWN_LOG_FILE, 'w') as f:
            json.dump(unknown_detections, f, indent=2)
            
        print(f"💾 Detection logs saved: {len(known_detections)} known, {len(unknown_detections)} unknown")
    except Exception as e:
        print(f"⚠️ Error saving detection logs: {e}")

def log_known_detection(face_id, camera_id, timestamp):
    """Log a known face detection."""
    known_detections[face_id] = {
        "id": face_id,
        "camera_id": camera_id,
        "timestamp": timestamp
    }

def log_unknown_detection(face_id, camera_id, embedding, timestamp):
    """Log an unknown face detection."""
    unknown_detections[face_id] = {
        "id": face_id,
        "camera_id": camera_id,
        "embedding": embedding.tolist() if embedding is not None else None,
        "timestamp": timestamp
    }

def get_known_detections():
    """Get all known face detections."""
    return known_detections

def get_unknown_detections():
    """Get all unknown face detections."""
    return unknown_detections