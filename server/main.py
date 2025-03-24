import os
from database import load_db, add_face
from detection_logging import load_detection_logs, create_flask_app
from file_utils import start_cleanup_thread, capture_and_save
from video import real_time_recognition
import cv2
from threading import Thread
from waitress import serve  # Import waitress for production-ready WSGI server

def start_flask_app():
    """Start the Flask app using Waitress."""
    flask_app = create_flask_app()
    print("✅ Flask API started at http://localhost:5000")
    serve(flask_app, host='0.0.0.0', port=5000)  # Use Waitress to serve the app

def main():
    """Main application entry point with command-line interface."""
    # Initialize the system
    load_db()  # Load saved embeddings at startup
    load_detection_logs()  # Load detection logs
    start_cleanup_thread()  # Start the cleanup thread

    # Start the Flask API in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.daemon = True  # Daemonize thread to exit when the main program exits
    flask_thread.start()

    # Default camera source
    src = 0  # Default camera source
    
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
            capture_and_save(name, src=src)
        elif choice == '3':
            img_path = input("Enter test image path: ").strip()
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                from tracking import search_face
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully...")
    finally:
        from config import cleanup_timer
        if cleanup_timer is not None:
            cleanup_timer.cancel()