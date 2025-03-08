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
from huggingface_hub import hf_hub_download, snapshot_download
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import cosine_similarity
from flask import Flask, render_template, jsonify
from database import fetch_attendance, insert_attendance  
import time

app = Flask(__name__)  
flask_thread = None  

# ✅ Load StyleGAN3 (`balgot/text-2-stylegan3`)
try:
    from diffusers import StyleGANPipeline  # Hugging Face pipeline
    stylegan_path = snapshot_download(repo_id="balgot/text-2-stylegan3")
    stylegan_pipe = StyleGANPipeline.from_pretrained(stylegan_path).to("cuda")
    print("🚀 StyleGAN3 (balgot/text-2-stylegan3) loaded successfully!")
except ImportError:
    print("⚠️ StyleGAN3 not installed. Skipping augmentation.")
    stylegan_pipe = None

# ✅ Load Real-ESRGAN (`ai-forever/Real-ESRGAN`)
try:
    from realesrgan import RealESRGAN
    realesrgan_path = snapshot_download(repo_id="ai-forever/Real-ESRGAN")
    realesrgan_model = RealESRGAN.from_pretrained(realesrgan_path).to("cuda")
    print("🚀 Real-ESRGAN (ai-forever/Real-ESRGAN) loaded successfully!")
except ImportError:
    print("⚠️ Real-ESRGAN not installed. Skipping enhancement.")
    realesrgan_model = None

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
        """Process captured face image, apply StyleGAN3 & Real-ESRGAN, and return binary data."""
        
        if person_name in self.captured_images:
            return None  

        time.sleep(1)  # ✅ Delay to prevent rapid duplicate processing

        # ✅ Convert RGB to BGR for OpenCV
        face_img_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        face_pil = Image.fromarray(face_img_bgr)

        # ✅ Apply StyleGAN3 Augmentation (if available)
        if stylegan_pipe:
            face_pil = stylegan_pipe(face_pil)  # Generate synthetic variations

        # ✅ Apply Real-ESRGAN Super-Resolution (if available)
        if realesrgan_model:
            face_pil = realesrgan_model.enhance(face_pil)

        # ✅ Encode image as JPEG and convert to binary
        _, img_encoded = cv2.imencode('.jpg', np.array(face_pil))
        image_binary = img_encoded.tobytes()

        print(f"📸 Image processed and enhanced for {person_name}, ready to be stored in database.")

        self.captured_images.add(person_name)
        return image_binary  

    def recognize_face(self, detected_face):
        """Compare detected face with stored embeddings and insert image into the database."""
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

        image_binary = self.save_image(detected_face, best_match)

        if image_binary:
            try:
                insert_attendance(best_match, round(best_score * 100, 2), image_binary)
                print(f"✅ Attendance Recorded: {best_match}, Accuracy: {best_score:.2f}%")
            except Exception as e:
                print(f"❌ Database Error: {e}")

        return best_match, best_score

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

                    best_match, best_score = self.recognize_face(detected_face)
                    face_found = True

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

if __name__ == "__main__":
    input_folder = r"C:\Warning\Projects\YoloFaceNet\data"
    face_recognition = FaceRecognitionSystem(input_folder)

    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

    flask_thread = threading.Thread(target=lambda: app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False), daemon=True)
    flask_thread.start()

    face_recognition.run()
