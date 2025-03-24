import time
from datetime import datetime

# Import from other modules
from config import (
    face_trackers, GRACE_PERIOD, UNKNOWN_THRESHOLD, 
    IOU_THRESHOLD
)
from face_detection import detect_and_embed, bbox_iou
import database  # Import the whole module instead of specific variables
from file_utils import save_face_image
from detection_logging import log_known_detection, log_unknown_detection

def search_face(img, camera_id="default"):
    """Search for faces in an image using FAISS, with improved tracking for transient faces."""
    global face_trackers
    
    faces, test_embeddings = detect_and_embed(img)
    if test_embeddings is None:
        # No faces detected in this frame, but keep track of previously known faces
        current_time = time.time()
        results = []
        
        # Check for recently seen faces and include them with "tracking" label
        for track_id, info in list(face_trackers.items()):
            if current_time - info["last_seen"] < GRACE_PERIOD:
                # Still within grace period, return the last known identity
                results.append((info["bbox"], info["name"] + " (tracking)", info["score"]))
            else:
                # Remove old tracks
                face_trackers.pop(track_id, None)
                
        return results
    
    # Faces were detected, run recognition
    results = []
    current_time = time.time()
    timestamp = datetime.now().isoformat()
    
    # Process detected faces
    for idx, face in enumerate(faces):
        bbox = face.bbox
        embedding = test_embeddings[idx]
        
        # Search for face in database
        best_match_idx, best_score = database.search_face_db(embedding)
        
        matched_track_id = None
        
        # Try to match with existing tracks using IOU
        for track_id, info in face_trackers.items():
            iou = bbox_iou(bbox, info["bbox"])
            if iou > IOU_THRESHOLD:
                matched_track_id = track_id
                break
        
        # If this face matches a known track
        if matched_track_id is not None:
            track_info = face_trackers[matched_track_id]
            if "Identifying" not in track_info["name"]:
                save_face_image(img, bbox, track_info["name"], track_info["embedding"])
            # If face is recognized confidently
            if best_match_idx != -1 and best_score > 0.5:
                # Ensure index is valid before accessing database.id_to_name
                if 0 <= best_match_idx < len(database.id_to_name):
                    matched_name = database.id_to_name[best_match_idx]
                    
                    # Update tracker with new recognition
                    track_info["name"] = matched_name
                else:
                    # Index is out of range, keep current name
                    print(f"⚠️ Warning: Index {best_match_idx} out of range for id_to_name (len={len(database.id_to_name)})")
                track_info["last_seen"] = current_time
                track_info["bbox"] = bbox
                track_info["embedding"] = embedding
                track_info["unrecognized_count"] = 0
                track_info["score"] = best_score
                
                # Log face detection based on whether it's known or unknown
                if 0 <= best_match_idx < len(database.id_to_name):
                    matched_name = database.id_to_name[best_match_idx]
                    if matched_name.startswith("Unknown"):
                        log_unknown_detection(matched_name, camera_id, embedding, timestamp)
                    else:
                        log_known_detection(matched_name, camera_id, timestamp)
            else:
                # Face not recognized confidently, but we have a tracking history
                track_info["last_seen"] = current_time
                track_info["bbox"] = bbox
                track_info["embedding"] = embedding
                
                # Only increment counter if score is really bad
                if best_score < 0.3:
                    track_info["unrecognized_count"] += 1
                
                # Only register as Unknown after many consecutive failures
                # and if we've already tried recognizing confidently before
                if track_info["unrecognized_count"] >= UNKNOWN_THRESHOLD and "Unknown" not in track_info["name"]:
                    unknown_name = database.add_unknown_face(embedding)
                    track_info["name"] = unknown_name
                    track_info["score"] = 0.0
                    
                    # Log unknown face detection
                    log_unknown_detection(unknown_name, camera_id, embedding, timestamp)
            
            results.append((bbox, track_info["name"], track_info["score"]))
            
        else:
            # New face detected that doesn't match any existing tracks
            new_track_id = str(len(face_trackers) + 1)
            
            if best_match_idx != -1 and best_score > 0.4:
                # Ensure index is valid before accessing database.id_to_name
                if 0 <= best_match_idx < len(database.id_to_name):
                    # Confidently recognized new face
                    matched_name = database.id_to_name[best_match_idx]
                    face_trackers[new_track_id] = {
                        "name": matched_name,
                        "last_seen": current_time,
                        "bbox": bbox,
                        "embedding": embedding,
                        "unrecognized_count": 0,
                        "score": best_score
                    }
                    results.append((bbox, matched_name, best_score))
                    
                    # Log face detection based on whether it's known or unknown
                    if matched_name.startswith("Unknown"):
                        log_unknown_detection(matched_name, camera_id, embedding, timestamp)
                    else:
                        log_known_detection(matched_name, camera_id, timestamp)
                    
                    # Save face image
                    save_face_image(img, bbox, matched_name, embedding)
                else:
                    # Index is out of range, treat as unrecognized
                    print(f"⚠️ Warning: Index {best_match_idx} out of range for id_to_name (len={len(database.id_to_name)})")
                    face_trackers[new_track_id] = {
                        "name": "Identifying...",
                        "last_seen": current_time,
                        "bbox": bbox,
                        "embedding": embedding,
                        "unrecognized_count": 1,
                        "score": 0.0
                    }
                    results.append((bbox, "Identifying...", 0.0))
                
                # Note: log_known_detection and save_face_image are already called inside the if block
            else:
                # Unrecognized new face - don't immediately create Unknown entry
                # Instead, start tracking and wait for more confident recognition
                face_trackers[new_track_id] = {
                    "name": "Identifying...",
                    "last_seen": current_time,
                    "bbox": bbox,
                    "embedding": embedding,
                    "unrecognized_count": 1,
                    "score": 0.0
                }
                results.append((bbox, "Identifying...", 0.0))
    
    # Clean up old tracks
    for track_id in list(face_trackers.keys()):
        if current_time - face_trackers[track_id]["last_seen"] > GRACE_PERIOD:
            face_trackers.pop(track_id, None)
    
    return results