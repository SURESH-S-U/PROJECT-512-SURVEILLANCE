import os
import json
import faiss
import numpy as np

# Import from other modules
from config import (
    EMBEDDING_DIM, DB_FAISS_FILE, NAME_MAPPING_FILE,
    SIMILARITY_THRESHOLD, unknown_counter
)
from face_detection import normalize, detect_and_embed

# Initialize global variables
index = faiss.IndexFlatIP(EMBEDDING_DIM)
id_to_name = []

def save_db():
    """Save FAISS index and name mappings."""
    faiss.write_index(index, DB_FAISS_FILE)
    with open(NAME_MAPPING_FILE, "w") as f:
        json.dump(id_to_name, f)
    print("💾 Database saved!")

def load_db():
    """Load FAISS index and name mappings."""
    global index, id_to_name, unknown_counter

    if not (os.path.exists(DB_FAISS_FILE) and os.path.exists(NAME_MAPPING_FILE)):
        print("⚠️ No database found. Starting fresh!")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        id_to_name = []
        unknown_counter = 1
        return

    index = faiss.read_index(DB_FAISS_FILE)
    with open(NAME_MAPPING_FILE, "r") as f:
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

def add_unknown_face(embedding):
    """Add an unknown face to the database."""
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
    """Check for similar unknown faces and remove them."""
    global index, id_to_name

    if index.ntotal == 0:
        return 0

    all_embeddings = faiss.vector_to_array(index.xb).reshape(index.ntotal, EMBEDDING_DIM)
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
        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        new_index.add(all_embeddings[keep_indices])
        index = new_index
        id_to_name = [all_names[i] for i in keep_indices]
        save_db()
        print(f"🗑️ Removed {len(similar_unknowns)} unknowns")
        return len(similar_unknowns)
    return 0

def add_face(name, img_path=None):
    """Add a face to the database."""
    global index, id_to_name
    import cv2  # Import here to avoid circular imports

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

def search_face_db(embedding):
    """Search for a face in the database."""
    if index.ntotal == 0:
        return -1, 0.0
    
    norm_embedding = normalize(embedding.reshape(1, -1))
    D, I = index.search(norm_embedding, 1)
    
    return I[0][0], D[0][0]