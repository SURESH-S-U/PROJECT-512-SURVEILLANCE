import insightface
import cv2
import numpy as np
import faiss
import os
import json
import time
import threading
from queue import Queue
import warnings
import json
from datetime import datetime
import hashlib
import shutil
from threading import Timer
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.75)  # Higher threshold for better quality
print("\n✅ Model loaded successfully!")


# Add these global variables
FACES_DIR = "Data/Faces"
KNOWN_FACES_DIR = os.path.join(FACES_DIR, "known")
UNKNOWN_FACES_DIR = os.path.join(FACES_DIR, "unknown")
FACE_UPDATE_INTERVAL = 10  # seconds
cleanup_timer = None
face_last_saved = {}  # Track when each face was last saved


face_db = {}
id_to_name = []
unknown_counter = 1
embedding_dim = 512  
index = faiss.IndexFlatIP(embedding_dim) 
db_file = "face_db.json"
known_faces = set()

face_trackers = {}  
GRACE_PERIOD = 15  
UNKNOWN_THRESHOLD = 15  
SIMILARITY_THRESHOLD = 0.7  
IOU_THRESHOLD = 0.5 




# Add after other global variables
known_log_file = "detect_known.json"
unknown_log_file = "detect_unknown.json"
known_detections = {}
unknown_detections = {}
last_save_time = time.time()
SAVE_INTERVAL = 5 

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



def load_detection_logs():
    """Load existing detection logs if they exist."""
    global known_detections, unknown_detections
    
    try:
        if os.path.exists(known_log_file):
            with open(known_log_file, 'r') as f:
                known_detections = json.load(f)
            print(f"✅ Loaded {len(known_detections)} known face logs")
        else:
            known_detections = {}
            
        if os.path.exists(unknown_log_file):
            with open(unknown_log_file, 'r') as f:
                unknown_detections = json.load(f)
            print(f"✅ Loaded {len(unknown_detections)} unknown face logs")
        else:
            unknown_detections = {}
    except Exception as e:
        print(f"⚠️ Error loading detection logs: {e}")
        known_detections = {}
        unknown_detections = {}

def save_detection_logs():
    """Save detection logs to JSON files."""
    try:
        with open(known_log_file, 'w') as f:
            json.dump(known_detections, f, indent=2)
        
        with open(unknown_log_file, 'w') as f:
            json.dump(unknown_detections, f, indent=2)
            
        print(f"💾 Detection logs saved: {len(known_detections)} known, {len(unknown_detections)} unknown")
    except Exception as e:
        print(f"⚠️ Error saving detection logs: {e}")



def bbox_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    box1 = [float(x) for x in box1]
    box2 = [float(x) for x in box2]
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def normalize(emb):
    """Normalize embeddings for cosine similarity."""
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

def detect_and_embed(img):
    """Detects faces in an image and extracts embeddings."""
    faces = model.get(img)
    if not faces:
        return [], None

    # Validate face bounding boxes
    valid_faces = []
    for face in faces:
        w = face.bbox[2] - face.bbox[0]
        h = face.bbox[3] - face.bbox[1]
        if w > 10 and h > 10:  # Minimum reasonable face size
            valid_faces.append(face)
    
    if not valid_faces:
        return [], None

    # Sort valid faces by size
    valid_faces = sorted(valid_faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)
    
    embeddings = np.array([face.embedding for face in valid_faces], dtype=np.float32)
    return valid_faces, embeddings

def add_unknown_face(embedding):
    global unknown_counter, id_to_name

    unknown_name = f"Unknown{unknown_counter}"
    unknown_counter += 1
    normalized_embedding = normalize(embedding.reshape(1, -1))
    
    index.add(normalized_embedding)
    id_to_name.append(unknown_name)
    save_db()
    print(f"⚠️ New unknown face added as {unknown_name}")
    return unknown_name

def check_similar_unknown_faces(embeddings, name):
    global index, id_to_name

    if index.ntotal == 0:
        return 0

    all_embeddings = faiss.vector_to_array(index.xb).reshape(index.ntotal, embedding_dim)
    all_names = id_to_name
    similar_unknowns = set()

    for i, current_name in enumerate(all_names):
        if "Unknown" in current_name:
            current_emb = all_embeddings[i]
            similarity = np.dot(normalize(embeddings), current_emb).max()
            if similarity > SIMILARITY_THRESHOLD:
                similar_unknowns.add(current_name)
                print(f"🔄 Found similar unknown: {current_name} ({similarity:.2f})")

    if similar_unknowns:
        keep_indices = [i for i, n in enumerate(all_names) if n not in similar_unknowns]
        new_index = faiss.IndexFlatIP(embedding_dim)
        new_index.add(all_embeddings[keep_indices])
        index = new_index
        id_to_name = [all_names[i] for i in keep_indices]
        save_db()
        print(f"🗑️ Removed {len(similar_unknowns)} unknowns")
        return len(similar_unknowns)
    return 0

def add_face(name, img_path=None):
    global index, id_to_name


    person_dir = f"Data/Images/{name}"
    
    # If a specific image is provided, use it
    if img_path:
        img_paths = [img_path]
    else:
        if not os.path.exists(person_dir):
            print(f"❌ No images found for {name}!")
            return
        img_paths = [os.path.join(person_dir, img) for img in os.listdir(person_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]

    all_embeddings = []
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Error reading {img_path}")
            continue
        
        faces, embeddings = detect_and_embed(img)
        if embeddings is None or len(embeddings) == 0:
            print(f"⚠️ No face detected in {img_path}")
            continue

        embeddings = normalize(embeddings)  # Normalize for FAISS
        all_embeddings.append(embeddings[0])  # Only take the largest face

    if not all_embeddings:
        print(f"❌ No valid faces added for {name}")
        return

    all_embeddings = np.vstack(all_embeddings)  # Stack all embeddings together
    
    # Check for similar unknown faces and replace them
    removed_count = check_similar_unknown_faces(all_embeddings, name)
    
    index.add(all_embeddings)
    if name not in id_to_name:
        id_to_name.append(name)
    save_db()
    print(f"✅ {name} added with {len(all_embeddings)} images!" + 
          (f" Replaced {removed_count} unknowns." if removed_count else ""))

def search_face(img):
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
    norm_test_embeddings = normalize(test_embeddings)
    D, I = index.search(norm_test_embeddings, 1)  # Search for the closest match
    
    results = []
    current_time = time.time()
    
    # Process detected faces
    for idx, (face, embedding, best_match_idx, best_score) in enumerate(zip(faces, norm_test_embeddings, I[:, 0], D[:, 0])):
        bbox = face.bbox
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
                matched_name = id_to_name[best_match_idx]
                
                # Update tracker with new recognition
                track_info["name"] = matched_name
                track_info["last_seen"] = current_time
                track_info["bbox"] = bbox
                track_info["embedding"] = embedding
                track_info["unrecognized_count"] = 0
                track_info["score"] = best_score
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
                    unknown_name = add_unknown_face(embedding)
                    track_info["name"] = unknown_name
                    track_info["score"] = 0.0
            
            
            results.append((bbox, track_info["name"], track_info["score"]))
            
        else:
            # New face detected that doesn't match any existing tracks
            new_track_id = str(len(face_trackers) + 1)
            
            if best_match_idx != -1 and best_score > 0.4:
                # Confidently recognized new face
                matched_name = id_to_name[best_match_idx]
                face_trackers[new_track_id] = {
                    "name": matched_name,
                    "last_seen": current_time,
                    "bbox": bbox,
                    "embedding": embedding,
                    "unrecognized_count": 0,
                    "score": best_score
                }
                results.append((bbox, matched_name, best_score))
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
            if best_match_idx != -1 and best_score > 0.4:
                save_face_image(img, bbox, matched_name, embedding)
    
    # Clean up old tracks
    for track_id in list(face_trackers.keys()):
        if current_time - face_trackers[track_id]["last_seen"] > GRACE_PERIOD:
            face_trackers.pop(track_id, None)
    
    return results

def capture_and_save(name,src=0):
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
        
    add_face(name)  # Load all images after capturing



class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.retry_count = 0
        self.thread = None

    def update(self):
        while not self.stopped:
            grabbed = self.stream.grab()
            if not grabbed:
                self.retry_count += 1
                if self.retry_count > 5:
                    self.stop()
                time.sleep(0.1)
                continue
            self.retry_count = 0
            self.grabbed, self.frame = self.stream.retrieve()

    def start(self):
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
        self.stream.release()

    def read(self):
        return self.frame


def get_box_style(face_type, is_overlapping=False):
    """
    Get visual style for a bounding box based on face type and overlap status.
    
    Args:
        face_type: Type of face ("known", "unknown", or "identifying")
        is_overlapping: Whether this box overlaps with others
        
    Returns:
        Tuple of (color, thickness, line_type)
    """
    # Simple color scheme as per user request (BGR format)
    if face_type == "unknown":
        color = (0, 0, 255)    # Red for unknown faces
    elif face_type == "identifying":
        color = (0, 255, 255)  # Yellow for identifying faces
    else:
        color = (0, 255, 0)    # Green for known faces
    
    # Increase thickness for overlapping boxes
    thickness = 3 if is_overlapping else 2
    
    # Use different line types for better distinction of overlapping boxes
    line_type = cv2.LINE_AA if is_overlapping else cv2.LINE_8
    
    return color, thickness, line_type

def real_time_recognition(src):
    """Real-time face recognition with dynamic camera source and detection logging."""
    global known_detections, unknown_detections, last_save_time
    
    # Determine if the source is a webcam index or URL
    try:
        src_int = src
        vs = VideoStream(src=src_int).start()
        window_name = f"Webcam {src_int} Face Recognition"
        camera_id = f"webcam_{src_int}"
    except ValueError:
        vs = VideoStream(src=src).start()
        window_name = f"Camera Stream: {src}"
        # Extract a simpler camera_id from the source URL
        camera_id = src.split("@")[-1].replace(".", "_") if "@" in src else "camera_stream"
    
    time.sleep(2.0)
    print(f"📷 Starting {window_name}. Press 'q' to quit.")
    
    frame_counter = 0
    PROCESS_EVERY = 2
    label_positions = {}  # Track label positions to prevent overlap

    while True:
        frame = vs.read()
        if frame is None:
            print("❌ Error: Could not read frame.")
            break

        frame_counter += 1
        if frame_counter % PROCESS_EVERY != 0:
            continue

        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        results = search_face(small_frame)
        
        current_time = time.time()
        timestamp = datetime.now().isoformat()
        
        scale_factor = 2
        
        # Scale all bounding boxes
        scaled_results = []
        for bbox, name, score in results:
            x1, y1, x2, y2 = [int(coord * scale_factor) for coord in bbox]
            scaled_results.append(((x1, y1, x2, y2), name, score))
        
        # Check for overlapping boxes
        overlapping_boxes = set()
        for i, (bbox1, _, _) in enumerate(scaled_results):
            for j, (bbox2, _, _) in enumerate(scaled_results):
                if i != j:
                    # Calculate overlap
                    x1 = max(bbox1[0], bbox2[0])
                    y1 = max(bbox1[1], bbox2[1])
                    x2 = min(bbox1[2], bbox2[2])
                    y2 = min(bbox1[3], bbox2[3])
                    
                    if x1 < x2 and y1 < y2:
                        # Boxes overlap
                        overlapping_boxes.add(i)
                        overlapping_boxes.add(j)
        
        # Reset label positions for this frame
        label_positions = {}
        
        # Draw the boxes and labels with different styles
        for i, (bbox, name, score) in enumerate(scaled_results):
            x1, y1, x2, y2 = bbox
            
            # Determine face type for styling
            face_type = "known"
            if "Identifying" in name:
                face_type = "identifying"
            elif "Unknown" in name:
                face_type = "unknown"
                
            # Get style based on face type and overlap status
            color, thickness, line_type = get_box_style(face_type, i in overlapping_boxes)
            
            if "Unknown" in name and name.replace(" (tracking)", "") in id_to_name:
                # Log unknown face detection
                face_id = name.replace(" (tracking)", "")
                
                # Find the corresponding tracker and get embedding
                embedding = None
                for track_id, info in face_trackers.items():
                    if info["name"] == face_id:
                        embedding = info["embedding"]  # Keep as numpy array for image saving
                        embedding_list = info["embedding"].tolist()  # Convert to list for JSON
                        
                        # Save face image for unknown face
                        if "Identifying" not in face_id:
                            full_size_bbox = [x1, y1, x2, y2]
                            save_face_image(frame, full_size_bbox, face_id, embedding)
                        break
                
                if embedding is not None:
                    unknown_detections[face_id] = {
                        "id": face_id,
                        "camera_id": camera_id,
                        "embedding": embedding_list,
                        "timestamp": timestamp
                    }
            
            elif not "Identifying" in name:
                # Log known face detection (excluding "Identifying...")
                face_id = name.replace(" (tracking)", "")
                if face_id in id_to_name:
                    known_detections[face_id] = {
                        "id": face_id,
                        "camera_id": camera_id,
                        "timestamp": timestamp
                    }
                    
                    # Find the corresponding tracker to get embedding for saving
                    embedding = None
                    for track_id, info in face_trackers.items():
                        if info["name"] == face_id:
                            embedding = info["embedding"]
                            break
                    
                    # Save face image for known face
                    full_size_bbox = [x1, y1, x2, y2]
                    save_face_image(frame, full_size_bbox, face_id, embedding)
            
            # Draw the rectangle with the determined style
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_type)
            
            # Prepare label text
            label = name.replace(" (tracking)", "")
            score_text = f" ({score:.2f})" if score > 0 else ""
            label_text = f"{label}{score_text}"
            
            # Find a good position for the label to avoid overlap
            label_y = y1 - 10  # Default position
            
            # Check if this position overlaps with existing labels
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_width, text_height = text_size
            
            # Create a rectangle representing the text area
            text_x1, text_y1 = x1, label_y - text_height
            text_x2, text_y2 = x1 + text_width, label_y + 5
            
            # Check for overlaps with existing label positions
            overlap = False
            for pos_id, (tx1, ty1, tx2, ty2) in label_positions.items():
                if (text_x1 < tx2 and text_x2 > tx1 and 
                    text_y1 < ty2 and text_y2 > ty1):
                    overlap = True
                    break
            
            # If overlap, try placing the label inside the box at the bottom
            if overlap:
                label_y = y2 - 5
                text_y1, text_y2 = label_y - text_height, label_y + 5
            
            # Store this label position
            label_id = hash(f"{x1}{y1}{x2}{y2}")
            label_positions[label_id] = (text_x1, text_y1, text_x2, text_y2)
            
            # Draw the label
            cv2.putText(frame, label_text, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        # Save logs every SAVE_INTERVAL seconds
        if current_time - last_save_time >= SAVE_INTERVAL:
            save_detection_logs()
            last_save_time = current_time

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save logs before exiting
            save_detection_logs()
            break

    vs.stop()
    cv2.destroyAllWindows()

def save_db():
    """Save FAISS index and name mappings."""
    faiss.write_index(index, "face_db.faiss")
    with open("name_mapping.json", "w") as f:
        json.dump(id_to_name, f)
    print("💾 Database saved!")

def load_db():
    global index, id_to_name, unknown_counter

    if not (os.path.exists("face_db.faiss") and os.path.exists("name_mapping.json")):
        print("⚠️ No database found. Starting fresh!")
        index = faiss.IndexFlatIP(embedding_dim)
        id_to_name = []
        unknown_counter = 1
        return

    index = faiss.read_index("face_db.faiss")
    with open("name_mapping.json", "r") as f:
        id_to_name = json.load(f)

    # Initialize unknown counter
    unknown_counter = 1
    for name in id_to_name:
        if name.startswith("Unknown"):
            try:
                num = int(name[7:])
                unknown_counter = max(unknown_counter, num + 1)
            except ValueError:
                pass
    print(f"✅ Loaded {len(id_to_name)} faces from database!")

def delete_files_and_folders(json_files, folder_name = 'Data'):
    for file in json_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted file: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"File not found: {file}")
    
    # Delete folder
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        try:
            shutil.rmtree(folder_name)
            print(f"Deleted folder: {folder_name}")
        except Exception as e:
            print(f"Error deleting folder {folder_name}: {e}")
    else:
        print(f"Folder not found: {folder_name}")

def main():
    load_db()  # Load saved embeddings at startup
    load_detection_logs()   # Load saved embeddings at startup
    start_cleanup_thread()
    src = 0  #"rtsp://admin:bitsathY@192.168.1.1"
    while True:
        print("\nOptions:")
        print("1️⃣ Add a face from an image file")
        print("2️⃣ Capture and add a face using webcam (multiple angles)")
        print("3️⃣ Search for a face in an image")
        print("4️⃣ Start real-time face recognition")
        print("5️⃣ Exit")
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            name = input("Enter name: ").strip()
            img_path = input("Enter image path: ").strip()
            if os.path.exists(img_path):
                add_face(name, img_path)
            else:
                print("❌ File not found!")
        elif choice == '2':
            name = input("Enter name: ").strip()
            capture_and_save(name,src=src)
        elif choice == '3':
            img_path = input("Enter test image path: ").strip()
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                results = search_face(img)
                for bbox, name, score in results:
                    print(f"✅ Matched: {name} (Similarity: {score:.2f})")
            else:
                print("❌ File not found!")
        elif choice == '4':
            real_time_recognition(src)
        elif choice == '5':
            # Example usage
            json_files_list = [known_log_file, unknown_log_file]
            delete_files_and_folders(json_files_list)
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid choice, try again.")

if __name__ == "__main__":
    try:
        main()

    finally:
        if cleanup_timer is not None:
            cleanup_timer.cancel()