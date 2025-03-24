import os

# Import from other modules
from database import load_db, add_face
from detection_logging import load_detection_logs
from file_utils import start_cleanup_thread, capture_and_save
from video import real_time_recognition
import cv2

def main():
    """Main application entry point with command-line interface."""
    # Initialize the system
    load_db()  # Load saved embeddings at startup
    load_detection_logs()  # Load detection logs
    start_cleanup_thread()  # Start the cleanup thread
    
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
    finally:
        from config import cleanup_timer
        if cleanup_timer is not None:
            cleanup_timer.cancel()