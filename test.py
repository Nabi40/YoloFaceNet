import cv2
import os
import numpy as np
import torch
import mediapipe as mp
import psycopg2
import threading  # ✅ Allows Flask & OpenCV to run together
import signal  # ✅ Captures CTRL+C to stop Flask & OpenCV
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import cosine_similarity
from flask import Flask, render_template
from database import fetch_attendance, insert_attendance 
from flask import Flask, render_template, jsonify
from database import fetch_attendance # ✅ Import insert_attendance

app = Flask(__name__)  # Initialize Flask
flask_thread = None  # Global variable for the Flask thread

class FaceRecognitionSystem:
    def __init__(self, input_folder):
        """Initialize the face recognition system with models and reference images."""
        self.input_folder = input_folder
        self.running = True  # ✅ Ensures graceful shutdown

        # ✅ Define transformations before calling load_reference_images
        self.resize_transform = Resize((160, 160))
        self.to_tensor = ToTensor()

        self.yolo_model, self.facenet = self.load_models()
        self.reference_features = self.load_reference_images()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        self.last_detected_person = None
        self.person_in_frame = False  # ✅ Tracks if a person is in the frame

    def load_models(self):
        """Load YOLO and FaceNet models."""
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        yolo_model = YOLO(model_path)
        facenet = InceptionResnetV1(pretrained='vggface2').eval()
        return yolo_model, facenet

    def load_reference_images(self):
        """Load reference images and compute embeddings."""
        reference_features = {}
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                person_name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.input_folder, filename)

                image = Image.open(image_path).convert("RGB")
                image_tensor = self.to_tensor(self.resize_transform(image)).unsqueeze(0)

                # Normalize for FaceNet
                image_tensor = (image_tensor - 0.5) / 0.5

                with torch.no_grad():
                    embedding = self.facenet(image_tensor)

                reference_features[person_name] = embedding

        print(f"✅ Loaded {len(reference_features)} reference images.")
        return reference_features

    def detect_face(self, frame):
        """Detect faces using YOLO model."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        results = self.yolo_model(pil_frame)
        return results, frame_rgb

    def recognize_face(self, detected_face):
        """Compare detected face with stored embeddings."""
        detected_tensor = self.to_tensor(self.resize_transform(Image.fromarray(detected_face))).unsqueeze(0)
        detected_tensor = (detected_tensor - 0.5) / 0.5

        with torch.no_grad():
            detected_embedding = self.facenet(detected_tensor)

        best_match = "Unknown"
        best_score = 0.0

        for person_name, ref_embedding in self.reference_features.items():
            similarity = cosine_similarity(ref_embedding, detected_embedding).item()
            similarity = max(0.0, similarity)

            if similarity > best_score and similarity > 0.5:
                best_score = similarity
                best_match = person_name

        # ✅ Store Matched Data
        self.dataMatch = {
            "best_match": best_match,
            "name": best_match if best_match != "Unknown" else "No Match",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accuracy": round(best_score * 100, 2),
        }

        print(f"🔍 Best Match: {best_match}, Accuracy: {best_score:.2f}%, Time: {self.dataMatch['time']}")

        return best_match, best_score

    def run(self):
        """Main function to run real-time face recognition system."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam. Check your camera settings.")
            return

        while self.running:  # ✅ Loop will exit when `CTRL+C` is pressed
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not capture frame from webcam.")
                continue

            results, frame_rgb = self.detect_face(frame)
            face_found = False

            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    detected_face = frame_rgb[y1:y2, x1:x2]

                    # Face recognition
                    best_match, best_score = self.recognize_face(detected_face)
                    face_found = True

                    # ✅ Post attendance if a new person enters
                    if best_match != "Unknown" and not self.person_in_frame:
                        print(f"✅ Recording Attendance: {best_match} at {self.dataMatch['time']}")
                        insert_attendance(best_match, self.dataMatch["accuracy"])  # ✅ Insert Data into Database
                        self.person_in_frame = True  # ✅ Mark as "in frame"
                        self.last_detected_person = best_match  # ✅ Track last person

                    # ✅ Draw rectangle around face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if not face_found:
                if self.person_in_frame:
                    print("👀 Person left the frame. Resetting detection.")
                self.person_in_frame = False  # ✅ Reset when no face is found

            # ✅ Keep Webcam Running and Show Frame
            cv2.imshow("Face Recognition", frame)

            key = cv2.waitKey(1)
            if key == ord("q"):
                print("🚀 Exiting Face Recognition...")
                break

        cap.release()
        cv2.destroyAllWindows()


@app.route("/")
def index():
    """Render the attendance table."""
    records = fetch_attendance()  # ✅ Fetch attendance records
    return render_template("index2.html", records=records)  # ✅ Render template with data


def signal_handler(sig, frame):
    """Handle CTRL+C (SIGINT) to terminate Flask and OpenCV properly."""
    print("\n🛑 Stopping Face Recognition & Flask Server...")
    face_recognition.running = False  # ✅ Stop OpenCV loop
    os._exit(0)  # ✅ Force terminate all threads


@app.route("/get_attendance")
def get_attendance():
    """Fetch latest attendance records and return them for auto-refresh."""
    records = fetch_attendance()  # ✅ Fetch latest records
    return jsonify({"attendance": records})  # ✅ Return records in JSON format



if __name__ == "__main__":
    input_folder = r"C:\Warning\Projects\YoloFaceNet\data"
    face_recognition = FaceRecognitionSystem(input_folder)

    # ✅ Capture CTRL+C and stop everything
    signal.signal(signal.SIGINT, signal_handler)

    # ✅ Run Flask in a separate thread (Daemon mode)
    flask_thread = threading.Thread(target=lambda: app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False), daemon=True)
    flask_thread.start()

    # ✅ Run Face Recognition
    face_recognition.run()
