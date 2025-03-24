import cv2
import time
import threading
from datetime import datetime

# Import from other modules
from config import SAVE_INTERVAL, last_save_time
from tracking import search_face
from detection_logging import save_detection_logs

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


def real_time_recognition(src):
    """Real-time face recognition with dynamic camera source and detection logging."""
    global last_save_time
    
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

    while True:
        frame = vs.read()
        if frame is None:
            print("❌ Error: Could not read frame.")
            break

        frame_counter += 1
        if frame_counter % PROCESS_EVERY != 0:
            continue

        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        results = search_face(small_frame, camera_id)
        
        current_time = time.time()
        
        scale_factor = 2
        for bbox, name, score in results:
            x1, y1, x2, y2 = [int(coord * scale_factor) for coord in bbox]
            
            # Skip faces that are still being identified
            if "Identifying" in name:
                color = (0, 165, 255)
            elif "Unknown" in name:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = name.replace(" (tracking)", "")
            score_text = f" ({score:.2f})" if score > 0 else ""
            cv2.putText(frame, f"{label}{score_text}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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