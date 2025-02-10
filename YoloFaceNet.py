import torch
import cv2
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torch.nn.functional import cosine_similarity

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo_model = YOLO(model_path)

facenet = InceptionResnetV1(pretrained='vggface2').eval()

resize_transform = Resize((160, 160))  # ✅ FaceNet requires exactly 160x160
to_tensor = ToTensor()


input_folder = r"C:\Warning\code\YoloFaceNet\data"
reference_features = {}

for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):  
        person_name = os.path.splitext(filename)[0]  
        image_path = os.path.join(input_folder, filename)

        image = Image.open(image_path).convert("RGB")
        image_tensor = to_tensor(resize_transform(image)).unsqueeze(0)

        # ✅ FaceNet normalization
        image_tensor = (image_tensor - 0.5) / 0.5  

        with torch.no_grad():
            embedding = facenet(image_tensor)  # ✅ Extract deep features

        reference_features[person_name] = embedding

print(f"✅ Loaded {len(reference_features)} reference images.")

# ✅ Start Webcam for Real-Time Face Recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not capture frame from webcam.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)

    results = yolo_model(pil_frame)

    face_found = False
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])

            # ✅ Ensure bounding box does not go outside the image
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))

            detected_face = pil_frame.crop((x1, y1, x2, y2))

            detected_tensor = to_tensor(resize_transform(detected_face)).unsqueeze(0)  
            detected_tensor = (detected_tensor - 0.5) / 0.5  # ✅ FaceNet normalization

            with torch.no_grad():
                detected_embedding = facenet(detected_tensor)

            best_match = "Unknown"
            best_score = 0.0

            for person_name, ref_embedding in reference_features.items():
                similarity = cosine_similarity(ref_embedding, detected_embedding).item()
                similarity = max(0.0, similarity)

                print(f"Comparing with {person_name}: Similarity Score = {similarity}")

                if similarity > best_score and similarity > 0.5:  # ✅ Adjusted threshold
                    best_score = similarity
                    best_match = person_name  

            print(f"🔍 Best Match: {best_match}, Similarity Score: {best_score}")

            face_found = True

            if best_match != "Unknown":
                text = f"{best_match} (MATCH)"
                color = (0, 255, 0)  
            else:
                text = "NO MATCH"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if not face_found:
        cv2.putText(frame, "NO FACE DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("YOLO + FaceNet Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
