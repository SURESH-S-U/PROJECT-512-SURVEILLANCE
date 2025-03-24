import os
import json
import time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Import from config
from config import (
    KNOWN_LOG_FILE, UNKNOWN_LOG_FILE,
    known_detections, unknown_detections
)

# Initialize Flask app (but don't start it yet)
app = None

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

def log_known_detection(face_id, camera_id, timestamp, name=None, face_image=None):
    """Log a known face detection."""
    known_detections[face_id] = {
        "_id": face_id,  # Match frontend expectation
        "id": face_id,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "name": name if name else "Unknown",  # Add name field
        "face_image": face_image if face_image else None  # Add face_image field
    }
    save_detection_logs()

def log_unknown_detection(face_id, camera_id, embedding, timestamp, face_image=None):
    """Log an unknown face detection."""
    unknown_detections[face_id] = {
        "_id": face_id,  # Match frontend expectation
        "id": face_id,
        "camera_id": camera_id,
        "embedding": embedding.tolist() if embedding is not None else None,
        "timestamp": timestamp,
        "name": "Unknown",  # Add name field
        "face_image": face_image if face_image else None  # Add face_image field
    }
    save_detection_logs()

def get_known_detections():
    """Get all known face detections."""
    return known_detections

def get_unknown_detections():
    """Get all unknown face detections."""
    return unknown_detections

# Load detection logs when the module is imported
load_detection_logs()

# Flask-related functionality
def create_flask_app():
    """Create and configure the Flask app."""
    global app
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    @app.route('/detection_data', methods=['GET'])
    def get_detection_data():
        """Route to fetch detection data."""
        try:
            # Combine known and unknown detections into a single list
            detections = list(known_detections.values()) + list(unknown_detections.values())
            return jsonify(detections)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/video_feed', methods=['GET'])
    def video_feed():
        """Route to serve the video feed."""
        # Assuming you have a function or a way to get the video feed
        # For now, this is a placeholder
        return send_from_directory('path_to_video_feed', 'video_feed.mjpeg')

    @app.route('/stop', methods=['POST'])
    def stop_backend():
        """Route to stop the backend process."""
        try:
            # Add your logic to stop the backend process here
            return jsonify({"status": "stopped"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app

# Ensure the Flask app does not run automatically when imported
if __name__ == '__main__':
    # Create and run the Flask app if this file is executed directly
    flask_app = create_flask_app()
    flask_app.run(host='0.0.0.0', port=5000, debug=True)