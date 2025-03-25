import os
import cv2
import numpy as np
import json
import insightface
import faiss

# Initialize face model (same as in main program)
model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# Initialize FAISS index and name mapping
embedding_dim = 512
index = faiss.IndexFlatIP(embedding_dim)
id_to_name = []

def process_dataset(dataset_path):
    global index, id_to_name
    
    for student_dir in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student_dir)
        if not os.path.isdir(student_path):
            continue
            
        student_name = student_dir.replace("student-", "").replace("_", " ").title()
        print(f"Processing: {student_name}")
        
        valid_embeddings = []
        
        # Process all images in student directory
        for img_file in os.listdir(student_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(student_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Detect faces
            faces = model.get(img)
            if not faces:
                continue
                
            # Get largest face's embedding
            face_sizes = [(face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1]) for face in faces]
            main_face = faces[np.argmax(face_sizes)]
            embedding = main_face.embedding
            
            # Normalize and store
            normalized_embedding = embedding / np.linalg.norm(embedding)
            valid_embeddings.append(normalized_embedding)
        
        if valid_embeddings:
            # Add all embeddings to index
            embeddings_array = np.array(valid_embeddings, dtype=np.float32)
            index.add(embeddings_array)
            
            # Add name for each embedding
            id_to_name.extend([student_name] * len(valid_embeddings))
            
    # Save the generated files
    faiss.write_index(index, "face_db.faiss")
    with open("name_mapping.json", "w") as f:
        json.dump(id_to_name, f)
    print(f"Created index with {len(id_to_name)} entries")

if __name__ == "__main__":
    dataset_path = "dataset"  # Path to your dataset root
    process_dataset(dataset_path)