"""
Test script to verify that the modular structure works correctly.
This script imports from all modules and runs a basic test.
"""

import os
import cv2
import numpy as np

# Import from all modules
from config import model, KNOWN_FACES_DIR, UNKNOWN_FACES_DIR
from face_detection import detect_and_embed, normalize, bbox_iou
from database import load_db, save_db, add_face, id_to_name, index
from file_utils import save_face_image, generate_face_hash, start_cleanup_thread
from detection_logging import load_detection_logs, save_detection_logs
from tracking import search_face
from video import VideoStream
from api import (
    initialize_system, get_all_faces, get_face_by_id,
    add_face_from_image, search_face_in_image
)

def test_imports():
    """Test that all imports work correctly."""
    print("✅ All imports successful!")

def test_directories():
    """Test that required directories exist."""
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
    print(f"✅ Directories created/verified: {KNOWN_FACES_DIR}, {UNKNOWN_FACES_DIR}")

def test_model():
    """Test that the face recognition model is loaded."""
    if model is not None:
        print("✅ Face recognition model loaded!")
    else:
        print("❌ Face recognition model not loaded!")

def test_database():
    """Test database operations."""
    # Load the database
    load_db()
    print(f"✅ Database loaded with {len(id_to_name)} faces")

def test_detection():
    """Test face detection on a sample image if available."""
    # Check if there are any images in the known faces directory
    if os.path.exists(KNOWN_FACES_DIR):
        for person_id in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    image_path = os.path.join(person_dir, images[0])
                    img = cv2.imread(image_path)
                    if img is not None:
                        faces, embeddings = detect_and_embed(img)
                        if faces:
                            print(f"✅ Face detection successful! Found {len(faces)} faces in {image_path}")
                            return
    
    print("⚠️ No test images found or face detection failed")

def main():
    """Run all tests."""
    print("\n🧪 Testing modular structure...\n")
    
    test_imports()
    test_directories()
    test_model()
    test_database()
    test_detection()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()