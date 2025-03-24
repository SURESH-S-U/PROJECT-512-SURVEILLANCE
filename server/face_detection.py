import numpy as np

# Import configuration
from config import model

def normalize(emb):
    """Normalize embeddings for cosine similarity."""
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)

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