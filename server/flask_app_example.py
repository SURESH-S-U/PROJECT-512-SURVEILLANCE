from flask import Flask, request, jsonify, send_file
import os
import json

# Import functions from api module
from api import (
    initialize_system,
    get_all_faces,
    get_face_by_id,
    add_face_from_image,
    rename_face,
    delete_face,
    get_detection_logs,
    search_face_in_image,
    get_database_stats
)

app = Flask(__name__)

# Initialize the system when the app starts
@app.before_first_request
def setup():
    initialize_system()
    print("Face recognition system initialized")

# Routes for face management
@app.route('/api/faces', methods=['GET'])
def api_get_all_faces():
    """Get all known and unknown faces."""
    return jsonify(get_all_faces())

@app.route('/api/faces/<face_id>', methods=['GET'])
def api_get_face(face_id):
    """Get a specific face by ID."""
    result = get_face_by_id(face_id)
    
    # If the result includes an image path, send the image file
    if 'image_path' in result and os.path.exists(result['image_path']):
        return send_file(result['image_path'])
    
    return jsonify(result)

@app.route('/api/faces', methods=['POST'])
def api_add_face():
    """Add a new face."""
    if 'name' not in request.form:
        return jsonify({"status": "error", "message": "Name is required"})
    
    name = request.form['name']
    
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "Image file is required"})
    
    image_file = request.files['image']
    image_data = image_file.read()
    
    result = add_face_from_image(name, image_data)
    return jsonify(result)

@app.route('/api/faces/<face_id>', methods=['PUT'])
def api_update_face(face_id):
    """Rename a face."""
    data = request.json
    if not data or 'new_id' not in data:
        return jsonify({"status": "error", "message": "New ID is required"})
    
    new_id = data['new_id']
    result = rename_face(face_id, new_id)
    return jsonify(result)

@app.route('/api/faces/<face_id>', methods=['DELETE'])
def api_delete_face(face_id):
    """Delete a face."""
    result = delete_face(face_id)
    return jsonify(result)

# Routes for face recognition
@app.route('/api/recognize', methods=['POST'])
def api_recognize_face():
    """Recognize faces in an uploaded image."""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "Image file is required"})
    
    image_file = request.files['image']
    image_data = image_file.read()
    
    result = search_face_in_image(image_data)
    return jsonify(result)

# Routes for detection logs
@app.route('/api/logs', methods=['GET'])
def api_get_logs():
    """Get detection logs."""
    return jsonify(get_detection_logs())

# Routes for database stats
@app.route('/api/stats', methods=['GET'])
def api_get_stats():
    """Get database statistics."""
    return jsonify(get_database_stats())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)