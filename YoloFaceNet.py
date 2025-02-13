import cv2
import os
import time
import numpy as np
import torch
import mediapipe as mp
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import cosine_similarity

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo_model = YOLO(model_path)

facenet = InceptionResnetV1(pretrained='vggface2').eval()

resize_transform = Resize((160, 160))
to_tensor = ToTensor()

# Mediapipe for Blink & Movement Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ✅ Folder Path for Reference Images
input_folder = r"C:\Warning\Projects\YoloFaceNet\data"
reference_features = {}

# Reference Images
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        person_name = os.path.splitext(filename)[0]
        image_path = os.path.join(input_folder, filename)

        image = Image.open(image_path).convert("RGB")
        image_tensor = to_tensor(resize_transform(image)).unsqueeze(0)

        # ✅ FaceNet Normalization
        image_tensor = (image_tensor - 0.5) / 0.5

        with torch.no_grad():
            embedding = facenet(image_tensor)  # ✅ Extract Deep Features

        reference_features[person_name] = embedding

print(f"✅ Loaded {len(reference_features)} reference images.")

# Start Webcam for Real-Time Face Recognition
cap = cv2.VideoCapture(0)


blink_counter = 0
movement_counter = 0
fake_face_threshold = 8
real_face_threshold = 5  
face_detected_count = 0  
spoof_attack_detected = False  

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not capture frame from webcam.")
        continue  # Keeps Webcam Active

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)

    # ✅ Run YOLOv8 for Face Detection
    results = yolo_model(pil_frame)

    face_found = False
    fake_face_detected = False

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])


            detected_face = frame_rgb[y1:y2, x1:x2]

            # ✅ Anti-Spoofing: Check for 2D Image on a Phone Screen
            gray = cv2.cvtColor(detected_face, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)  # Edge Detection

            # ✅ Count white pixels in edge image (more edges = likely a screen)
            edge_density = np.sum(edges == 255) / edges.size

            if edge_density > 0.1:  # Adjust threshold if needed
                print("⚠️ Possible Screen Detected! Fake Face?")
                spoof_attack_detected = True
            else:
                spoof_attack_detected = False

            # ✅ Detect Facial Landmarks for Blink & Movement Detection
            face_landmarks = face_mesh.process(detected_face)

            if face_landmarks.multi_face_landmarks:
                face_detected_count += 1  # ✅ Increment real face count

                if face_detected_count > real_face_threshold:
                    fake_face_detected = False  # ✅ Enough real detections, consider real
            else:
                blink_counter += 1  # ✅ Track missing landmark detections
                movement_counter += 1  # ✅ Track missing movement

                if blink_counter > fake_face_threshold or movement_counter > fake_face_threshold:
                    fake_face_detected = True  # ✅ Consider as fake only after multiple failures
                    print("⚠️ Warning: No Blink or Movement Detected! Fake Face?")
                else:
                    fake_face_detected = False  # ✅ Wait before deciding



            detected_tensor = to_tensor(resize_transform(Image.fromarray(detected_face))).unsqueeze(0)
            detected_tensor = (detected_tensor - 0.5) / 0.5  # ✅ FaceNet Normalization

            with torch.no_grad():
                detected_embedding = facenet(detected_tensor)

            best_match = "Unknown"
            best_score = 0.0

            # Compare with Stored Encodings
            for person_name, ref_embedding in reference_features.items():
                similarity = cosine_similarity(ref_embedding, detected_embedding).item()
                similarity = max(0.0, similarity)

                print(f"Comparing with {person_name}: Accuracy = {similarity:.2f}%")

                if similarity > best_score and similarity > 0.5:
                    best_score = similarity
                    best_match = person_name

            print(f"🔍 Best Match: {best_match}, Accuracy: {best_score:.2f}%")

            face_found = True

            #isplay Results with Accuracy & Anti-Spoofing Alerts
            if spoof_attack_detected:
                text = "SPOOF ATTACK DETECTED!"
                color = (0, 0, 255)  # Red for Fake Face
            elif fake_face_detected:
                text = "FAKE FACE DETECTED!"
                color = (0, 0, 255)  # Red for Fake Face
            else:
                text = f"{best_match} ({best_score:.2f}%)" if best_match != "Unknown" else "NO MATCH"
                color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if not face_found:
        print("👀 No face detected, but webcam stays ON!")
        cv2.putText(frame, "NO FACE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if spoof_attack_detected:
        cv2.putText(frame, "SPOOF ATTACK DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

 
    cv2.imshow("YOLO + FaceNet Face Recognition (Liveness & Anti-Spoofing)", frame)


    key = cv2.waitKey(1)
    if key == ord("q"):
        print("🚀 Exiting Face Recognition...")
        break

cap.release()
cv2.destroyAllWindows()
