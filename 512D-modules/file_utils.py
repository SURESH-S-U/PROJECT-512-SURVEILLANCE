import os
import cv2
import hashlib
import time
from datetime import datetime
from threading import Timer

# Import configuration
from config import (
    KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, FACE_UPDATE_INTERVAL,
    face_last_saved, cleanup_timer
)

def generate_face_hash(embedding):
    """Generate a unique hash for a face embedding to avoid duplicates."""
    if embedding is None:
        return None
    hash_obj = hashlib.md5(embedding.tobytes())
    return hash_obj.hexdigest()[:10]


def save_face_image(frame, bbox, name, embedding):
    """Save a detected face image with timestamp and hash, deleting previous images."""
    global face_last_saved
    
    current_time = time.time()
    face_id = name.replace(" (tracking)", "")
    
    # Check if this face was recently saved
    if face_id in face_last_saved and current_time - face_last_saved[face_id] < FACE_UPDATE_INTERVAL:
        return
    
    # Create directory structure
    if "Unknown" in face_id:
        face_dir = os.path.join(UNKNOWN_FACES_DIR, face_id)
    else:
        face_dir = os.path.join(KNOWN_FACES_DIR, face_id)
    
    os.makedirs(face_dir, exist_ok=True)
    
    # Extract face from frame
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    # Add padding to the face crop
    pad = 20
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    face_img = frame[y1:y2, x1:x2]
    
    if face_img.size == 0:
        return  # Skip if face extraction failed
    
    # Generate timestamp and hash for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    face_hash = generate_face_hash(embedding)
    if face_hash is None:
        face_hash = str(hash(str(bbox)))[:10]  # Fallback if embedding is None
    
    filename = f"{timestamp}_{face_hash}.jpg"
    file_path = os.path.join(face_dir, filename)
    
    # Delete previous images for this person (if not an Unknown)
    if "Unknown" not in face_id:
        for old_file in os.listdir(face_dir):
            if old_file.endswith(('.jpg', '.jpeg', '.png')):
                old_path = os.path.join(face_dir, old_file)
                try:
                    os.remove(old_path)
                    print(f"🗑️ Removed previous image for {face_id}: {old_file}")
                except Exception as e:
                    print(f"⚠️ Error removing old image {old_path}: {e}")
    
    # Save the face image
    cv2.imwrite(file_path, face_img)
    face_last_saved[face_id] = current_time
    
    print(f"💾 Saved face image for {face_id} to {file_path}")


def delete_old_images(interval=60):
    """Periodically clean up old face images for known faces."""
    global cleanup_timer
    
    current_time = time.time()
    
    try:
        # Process known faces - keep only the most recent images
        if os.path.exists(KNOWN_FACES_DIR):
            for name_dir in os.listdir(KNOWN_FACES_DIR):
                person_dir = os.path.join(KNOWN_FACES_DIR, name_dir)
                if os.path.isdir(person_dir):
                    images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                    if len(images) > 5:  # Keep only 5 most recent images per person
                        # Sort by timestamp (which is at the beginning of filename)
                        images.sort(reverse=True)
                        for old_img in images[5:]:
                            os.remove(os.path.join(person_dir, old_img))
                            print(f"🗑️ Removed old image for {name_dir}: {old_img}")
    
        # Do not delete unknown faces automatically as per requirements
    except Exception as e:
        print(f"⚠️ Error in cleanup: {e}")
    
    # Schedule next cleanup
    cleanup_timer = Timer(interval, delete_old_images, [interval])
    cleanup_timer.daemon = True
    cleanup_timer.start()


def start_cleanup_thread():
    """Start the background cleanup thread."""
    # Create required directories if they don't exist
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
    
    # Start the cleanup thread
    delete_old_images(60)


def capture_and_save(name, src=0):
    """Captures all profile images using the webcam and saves them."""
    angles = ["front", "left", "right", "up"]
    os.makedirs(f"Data/Images/{name}", exist_ok=True)

    for angle in angles:
        input(f"📸 Look {angle} and press ENTER to capture...")
        
        cam = cv2.VideoCapture(src)
        if not cam.isOpened():
            print("❌ Error: Could not open camera.")
            return

        ret, frame = cam.read()
        cam.release()
        if not ret:
            print(f"❌ Error: Could not capture {angle} image.")
            return
        
        img_path = f"Data/Images/{name}/{angle}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"✅ {angle} image saved.")