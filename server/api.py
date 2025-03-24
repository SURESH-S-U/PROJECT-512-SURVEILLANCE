import os
import cv2
import numpy as np
import json
from datetime import datetime
import faiss

# Import from other modules
from config import (
    KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, 
    DB_FAISS_FILE, NAME_MAPPING_FILE
)
from database import (
    load_db, save_db, add_face, add_unknown_face, 
    id_to_name, index
)
from face_detection import detect_and_embed, normalize
from file_utils import save_face_image, generate_face_hash
from detection_logging import (
    load_detection_logs, save_detection_logs,
    get_known_detections, get_unknown_detections
)
from tracking import search_face

# API Functions for Flask Routes

def initialize_system():
    """Initialize the face recognition system."""
    load_db()
    load_detection_logs()
    return {"status": "success", "message": "System initialized"}

def get_all_faces():
    """Get all known and unknown faces."""
    known_faces = []
    unknown_faces = []
    
    # Get known faces
    if os.path.exists(KNOWN_FACES_DIR):
        for person_id in os.listdir(KNOWN_FACES_DIR):
            person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    latest_image = sorted(images)[-1]
                    image_path = os.path.join(person_dir, latest_image)
                    known_faces.append({
                        "id": person_id,
                        "name": person_id,
                        "image_path": image_path,
                        "timestamp": latest_image.split('_')[0]
                    })
    
    # Get unknown faces
    if os.path.exists(UNKNOWN_FACES_DIR):
        for person_id in os.listdir(UNKNOWN_FACES_DIR):
            person_dir = os.path.join(UNKNOWN_FACES_DIR, person_id)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    latest_image = sorted(images)[-1]
                    image_path = os.path.join(person_dir, latest_image)
                    unknown_faces.append({
                        "id": person_id,
                        "name": person_id,
                        "image_path": image_path,
                        "timestamp": latest_image.split('_')[0]
                    })
    
    return {
        "known_faces": known_faces,
        "unknown_faces": unknown_faces,
        "total_known": len(known_faces),
        "total_unknown": len(unknown_faces)
    }

def get_face_by_id(face_id):
    """Get a specific face by ID."""
    # Check in known faces
    person_dir = os.path.join(KNOWN_FACES_DIR, face_id)
    if os.path.exists(person_dir) and os.path.isdir(person_dir):
        images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            latest_image = sorted(images)[-1]
            image_path = os.path.join(person_dir, latest_image)
            return {
                "id": face_id,
                "name": face_id,
                "image_path": image_path,
                "timestamp": latest_image.split('_')[0],
                "type": "known"
            }
    
    # Check in unknown faces
    person_dir = os.path.join(UNKNOWN_FACES_DIR, face_id)
    if os.path.exists(person_dir) and os.path.isdir(person_dir):
        images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            latest_image = sorted(images)[-1]
            image_path = os.path.join(person_dir, latest_image)
            return {
                "id": face_id,
                "name": face_id,
                "image_path": image_path,
                "timestamp": latest_image.split('_')[0],
                "type": "unknown"
            }
    
    return {"status": "error", "message": f"Face with ID {face_id} not found"}

def add_face_from_image(name, image_data):
    """Add a face from an image (bytes or file path)."""
    try:
        # Handle different input types
        if isinstance(image_data, str):
            # Assume it's a file path
            if not os.path.exists(image_data):
                return {"status": "error", "message": "Image file not found"}
            img = cv2.imread(image_data)
        else:
            # Assume it's bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Failed to decode image"}
        
        # Detect face and add to database
        faces, embeddings = detect_and_embed(img)
        if embeddings is None or len(embeddings) == 0:
            return {"status": "error", "message": "No face detected in image"}
        
        # Add face to database
        add_face(name, None)  # We'll handle the embedding directly
        
        # Save the face image
        bbox = faces[0].bbox
        embedding = embeddings[0]
        save_face_image(img, bbox, name, embedding)
        
        return {
            "status": "success", 
            "message": f"Face added for {name}",
            "face_id": name
        }
    except Exception as e:
        return {"status": "error", "message": f"Error adding face: {str(e)}"}

def rename_face(old_id, new_id):
    """Rename a face in the database."""
    try:
        # Check if the face exists
        old_is_known = False
        old_dir = os.path.join(KNOWN_FACES_DIR, old_id)
        if os.path.exists(old_dir) and os.path.isdir(old_dir):
            old_is_known = True
        else:
            old_dir = os.path.join(UNKNOWN_FACES_DIR, old_id)
            if not (os.path.exists(old_dir) and os.path.isdir(old_dir)):
                return {"status": "error", "message": f"Face with ID {old_id} not found"}
        
        # Create new directory
        if old_is_known or not new_id.startswith("Unknown"):
            new_dir = os.path.join(KNOWN_FACES_DIR, new_id)
        else:
            new_dir = os.path.join(UNKNOWN_FACES_DIR, new_id)
        
        os.makedirs(new_dir, exist_ok=True)
        
        # Move images
        for img_file in os.listdir(old_dir):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                old_path = os.path.join(old_dir, img_file)
                new_path = os.path.join(new_dir, img_file)
                os.rename(old_path, new_path)
        
        # Update database
        if old_id in id_to_name:
            idx = id_to_name.index(old_id)
            id_to_name[idx] = new_id
            save_db()
        
        # Remove old directory
        os.rmdir(old_dir)
        
        return {
            "status": "success",
            "message": f"Face renamed from {old_id} to {new_id}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error renaming face: {str(e)}"}

def delete_face(face_id):
    """Delete a face from the database."""
    global index
    try:
        # Check if the face exists in known faces
        known_dir = os.path.join(KNOWN_FACES_DIR, face_id)
        unknown_dir = os.path.join(UNKNOWN_FACES_DIR, face_id)
        
        face_found = False
        
        if os.path.exists(known_dir) and os.path.isdir(known_dir):
            import shutil
            shutil.rmtree(known_dir)
            face_found = True
        
        if os.path.exists(unknown_dir) and os.path.isdir(unknown_dir):
            import shutil
            shutil.rmtree(unknown_dir)
            face_found = True
        
        if not face_found:
            return {"status": "error", "message": f"Face with ID {face_id} not found"}
        
        # Update database
        if face_id in id_to_name:
            idx = id_to_name.index(face_id)
            
            # Create a new index without this face
            all_embeddings = faiss.vector_to_array(index.xb).reshape(index.ntotal, index.d)
            keep_indices = [i for i in range(len(id_to_name)) if i != idx]
            
            new_index = faiss.IndexFlatIP(index.d)
            if keep_indices:  # Only add if there are embeddings to keep
                new_index.add(all_embeddings[keep_indices])
            
            # Update global variables
            index = new_index
            id_to_name.pop(idx)
            save_db()
        
        return {
            "status": "success",
            "message": f"Face {face_id} deleted successfully"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error deleting face: {str(e)}"}

def get_detection_logs():
    """Get all detection logs."""
    known = get_known_detections()
    unknown = get_unknown_detections()
    
    return {
        "known_detections": known,
        "unknown_detections": unknown,
        "total_known": len(known),
        "total_unknown": len(unknown)
    }

def search_face_in_image(image_data):
    """Search for faces in an image."""
    try:
        # Handle different input types
        if isinstance(image_data, str):
            # Assume it's a file path
            if not os.path.exists(image_data):
                return {"status": "error", "message": "Image file not found"}
            img = cv2.imread(image_data)
        else:
            # Assume it's bytes
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Failed to decode image"}
        
        # Search for faces
        results = search_face(img)
        
        # Format results
        faces = []
        for bbox, name, score in results:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            faces.append({
                "name": name.replace(" (tracking)", ""),
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2]
            })
        
        return {
            "status": "success",
            "faces": faces,
            "count": len(faces)
        }
    except Exception as e:
        return {"status": "error", "message": f"Error searching face: {str(e)}"}

def get_database_stats():
    """Get statistics about the face database."""
    try:
        # Count known and unknown faces in database
        known_count = sum(1 for name in id_to_name if not name.startswith("Unknown"))
        unknown_count = sum(1 for name in id_to_name if name.startswith("Unknown"))
        
        # Count face images
        known_images = 0
        if os.path.exists(KNOWN_FACES_DIR):
            for person_id in os.listdir(KNOWN_FACES_DIR):
                person_dir = os.path.join(KNOWN_FACES_DIR, person_id)
                if os.path.isdir(person_dir):
                    known_images += len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        unknown_images = 0
        if os.path.exists(UNKNOWN_FACES_DIR):
            for person_id in os.listdir(UNKNOWN_FACES_DIR):
                person_dir = os.path.join(UNKNOWN_FACES_DIR, person_id)
                if os.path.isdir(person_dir):
                    unknown_images += len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Get database file sizes
        db_size = os.path.getsize(DB_FAISS_FILE) if os.path.exists(DB_FAISS_FILE) else 0
        mapping_size = os.path.getsize(NAME_MAPPING_FILE) if os.path.exists(NAME_MAPPING_FILE) else 0
        
        return {
            "status": "success",
            "total_faces": len(id_to_name),
            "known_faces": known_count,
            "unknown_faces": unknown_count,
            "known_images": known_images,
            "unknown_images": unknown_images,
            "database_size_bytes": db_size,
            "mapping_size_bytes": mapping_size,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": f"Error getting database stats: {str(e)}"}