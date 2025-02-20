import cv2
import os
import numpy as np
import torch
import mediapipe as mp
import base64
import threading  
import signal  
from datetime import datetime
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import cosine_similarity
from flask import Flask, render_template, jsonify
from database import fetch_attendance, insert_attendance  
import time

app = Flask(__name__)  
flask_thread = None  

# ✅ Ensure 'captured_images' directory exists
IMAGE_SAVE_PATH = "captured_images"
os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

class FaceRecognitionSystem:
    def __init__(self, input_folder):
        """Initialize the face recognition system with models and reference images."""
        self.input_folder = input_folder
        self.running = True  
        self.captured_images = set()  

        self.resize_transform = Resize((160, 160))
        self.to_tensor = ToTensor()

        self.yolo_model, self.facenet = self.load_models()
        self.reference_features = self.load_reference_images()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        self.last_detected_person = None
        self.person_in_frame = False  

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

    def save_image(self, face_img, person_name):

        if person_name in self.captured_images:
            return None  
        time.sleep(1) 
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = f"{IMAGE_SAVE_PATH}/{timestamp}_{person_name}.jpg"

        # Convert RGB to BGR for OpenCV and save image
        cv2.imwrite(image_filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        print(f"📸 Image saved: {image_filename}")
        
        self.captured_images.add(person_name)
        
        return image_filename

    def recognize_face(self, detected_face):
        """Compare detected face with stored embeddings and save image."""
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

        image_path = self.save_image(detected_face, best_match)  

        if image_path:
            try:
                with open(image_path, "rb") as img_file:
                    image_binary = img_file.read()  
                insert_attendance(best_match, round(best_score * 100, 2), image_binary)
            except Exception as e:
                print(f"❌ Error saving attendance: {e}")

        print(f"🔍 Best Match: {best_match}, Accuracy: {best_score:.2f}%, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return best_match, best_score, image_path

    def run(self):
        """Main function to run real-time face recognition system."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ Error: Could not open webcam. Check your camera settings.")
            return

        while self.running:  
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

                    best_match, best_score, image_path = self.recognize_face(detected_face)
                    face_found = True

                    if best_match != "Unknown" and not self.person_in_frame:
                        print(f"✅ Recording Attendance: {best_match} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        self.person_in_frame = True  
                        self.last_detected_person = best_match  

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, best_match, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if not face_found:
                if self.person_in_frame:
                    print("👀 Person left the frame. Resetting detection.")
                self.person_in_frame = False  

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
    records = fetch_attendance()  
    return render_template("index2.html", records=records)  

@app.route("/get_attendance")
def get_attendance():
    """Fetch latest attendance records and return them as JSON with Base64 images."""
    try:
        records = fetch_attendance()
        
        for record in records:
            if record["image"]:  
                record["image"] = base64.b64encode(record["image"]).decode("utf-8")  # ✅ Convert BYTEA to Base64
            else:
                record["image"] = None  # ✅ Handle cases where no image exists

        return jsonify({"attendance": records})
    except Exception as e:
        print(f"❌ Error fetching attendance: {e}")
        return jsonify({"error": str(e)})


def signal_handler(sig, frame):
    """Handle CTRL+C (SIGINT) to terminate Flask and OpenCV properly."""
    print("\n🛑 Stopping Face Recognition & Flask Server...")
    face_recognition.running = False  
    os._exit(0)



if __name__ == "__main__":
    input_folder = r"C:\Warning\Projects\YoloFaceNet\data"
    face_recognition = FaceRecognitionSystem(input_folder)

    signal.signal(signal.SIGINT, signal_handler)

    flask_thread = threading.Thread(target=lambda: app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False), daemon=True)
    flask_thread.start()

    face_recognition.run()
