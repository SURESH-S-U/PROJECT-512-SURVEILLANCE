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

warnings.filterwarnings("ignore", category=RuntimeWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")

model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)  # Higher threshold for better quality
print("\n✅ Model loaded successfully!")

face_db = {}
id_to_name = []
unknown_counter = 1
embedding_dim = 512  
index = faiss.IndexFlatIP(embedding_dim)  
db_file = "face_db.json"

face_trackers = {}  
GRACE_PERIOD = 15  
UNKNOWN_THRESHOLD = 15  
SIMILARITY_THRESHOLD = 0.7  
IOU_THRESHOLD = 0.5 

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
    """Add a new unknown face to the database and FAISS index."""
    global unknown_counter, face_db, id_to_name

    unknown_name = f"Unknown{unknown_counter}"
    unknown_counter += 1

    embedding = embedding.reshape(1, -1)
    normalized_embedding = normalize(embedding)

    # Store unknown face embedding
    face_db[unknown_name] = normalized_embedding.tolist()
    index.add(normalized_embedding)
    id_to_name.append(unknown_name)

    save_db()
    print(f"⚠️ New unknown face added as {unknown_name}")

    return unknown_name

def check_similar_unknown_faces(embeddings, name):
    """Check if there are similar unknown faces and replace them with the new named face."""
    global face_db, index, id_to_name
    
    similar_unknowns = []
    embeddings = normalize(embeddings)
    
    # Look for similar unknown faces
    for face_name, face_embeddings in list(face_db.items()):
        if "Unknown" in face_name:
            face_embeddings = np.array(face_embeddings, dtype=np.float32)
            # Calculate similarity with each unknown face
            for i, unknown_emb in enumerate(face_embeddings):
                unknown_emb = unknown_emb.reshape(1, -1)
                similarity = np.dot(embeddings, unknown_emb.T).max()
                
                if similarity > SIMILARITY_THRESHOLD:
                    similar_unknowns.append(face_name)
                    print(f"🔄 Found similar unknown face: {face_name} (similarity: {similarity:.2f})")
                    break
    
    # Remove similar unknown faces
    if similar_unknowns:
        # Rebuild the index
        index = faiss.IndexFlatIP(embedding_dim)
        new_id_to_name = []
        
        # Remove the similar unknown faces
        for unknown_name in similar_unknowns:
            print(f"🗑️ Removing {unknown_name} as it's similar to the newly added {name}")
            face_db.pop(unknown_name, None)
        
        # Rebuild index with remaining faces
        for face_name, face_embeddings in face_db.items():
            face_embeddings = np.array(face_embeddings, dtype=np.float32)
            index.add(face_embeddings)
            new_id_to_name.extend([face_name] * face_embeddings.shape[0])
        
        id_to_name = new_id_to_name
        save_db()
        
        return len(similar_unknowns)
    
    return 0

def add_face(name, img_path=None):
    """Adds all face images of a person to the database."""
    global face_db, index, id_to_name

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
    
    # Add embeddings to FAISS and store in database
    face_db[name] = all_embeddings.tolist()
    index.add(all_embeddings)  # Add all images to FAISS
    id_to_name.extend([name] * all_embeddings.shape[0])  # Map each embedding to the name
    
    save_db()
    print(f"✅ {name} added with {len(all_embeddings)} images!" + 
          (f" Replaced {removed_count} similar unknown face(s)." if removed_count > 0 else ""))

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
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


def real_time_recognition(src):
    """Real-time face recognition with dynamic camera source."""
    # Determine if the source is a webcam index or URL
    try:
        src_int = src
        vs = VideoStream(src=src_int).start()
        window_name = f"Webcam {src_int} Face Recognition"
    except ValueError:
        vs = VideoStream(src=src).start()
        window_name = f"Camera Stream: {src}"
    
    time.sleep(2.0)
    print(f"📷 Starting {window_name}. Press 'q' to quit.")
    
    frame_counter = 0
    PROCESS_EVERY = 2

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

        scale_factor = 2
        for bbox, name, score in results:
            x1, y1, x2, y2 = [int(coord * scale_factor) for coord in bbox]
            
            if "Unknown" in name:
                color = (0, 0, 255)
            elif "tracking" in name:
                color = (255, 165, 0)
            elif "Identifying" in name:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = name.replace(" (tracking)", "")
            score_text = f" ({score:.2f})" if score > 0 else ""
            cv2.putText(frame, f"{label}{score_text}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

def save_db():
    """Save face embeddings to a JSON file for persistence."""
    with open(db_file, 'w') as f:
        json.dump(face_db, f)
    print("💾 Database saved!")

def load_db():
    """Load face embeddings from JSON file and restore FAISS index."""
    global face_db, index, id_to_name, unknown_counter

    if not os.path.exists(db_file):
        print("⚠️ No database found. Starting fresh!")
        return
    
    with open(db_file, 'r') as f:
        face_db = json.load(f)

    all_embeddings = []
    id_to_name.clear()
    
    for name, embeddings in face_db.items():
        embeddings = np.array(embeddings, dtype=np.float32)
        all_embeddings.append(embeddings)
        id_to_name.extend([name] * embeddings.shape[0])

        # Track the last unknown ID
        if "Unknown" in name:
            try:
                unknown_num = int(name.replace("Unknown", ""))
                unknown_counter = max(unknown_counter, unknown_num + 1)
            except ValueError:
                pass

    if all_embeddings:
        index.add(np.vstack(all_embeddings))
    
    print(f"✅ Loaded {len(face_db)} faces from database!")

def main():
    load_db()  # Load saved embeddings at startup
    src = "rtsp://admin:bitsathY@192.168.1.11" #"rtsp://admin:bitsathY@192.168.1.1"
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
            print("👋 Exiting...")
            break
        else:
            print("❌ Invalid choice, try again.")

if __name__ == "__main__":
    main()