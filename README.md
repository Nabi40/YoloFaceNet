# YoloFaceNet
YOLOv8 + FaceNet + Django + PostgreSQL    

This project implements a real-time face recognition system that detects faces using YOLOv8, identifies them using FaceNet embeddings, and logs attendance into a PostgreSQL database. It also provides a Flask-based web interface to view attendance records.    



----

## How It Works  
Reference Images:  
 - Load all .jpg/.png images from data/ folder.  
 - Convert them to 160x160 FaceNet embeddings and store them.  

YOLO Face Detection:  
 - Frame-by-frame webcam capture using OpenCV.  
 - YOLO detects faces and crops the face region.  

Face Recognition:  
  - Each face is passed to FaceNet for embedding.   
  - Compare embedding with reference images using cosine similarity.    
  - If similarity > 0.5, the person is recognized.  

    
Attendance Logging:    
  - Recognized person’s name and timestamp are saved to PostgreSQL.    
  - Only logs attendance when a new person enters the frame.    
 
Web Interface:  
  - Flask app shows attendance logs at http://localhost:5000/  
  - Auto-refresh supported via /get_attendance route.  


---


## Features  
- Real-time face detection & recognition using webcam  
- Attendance auto-logging with timestamp & accuracy  
- Live attendance dashboard via Flask  
- Modular & extensible code structure  
- Graceful shutdown with CTRL+C (thanks to signal module)  


---


## Limitations❌   
- YOLO model is pretrained, not fine-tuned on your data  
- Lighting, camera angle, and image quality affect accuracy  
- No spoof detection (e.g., photo or video replay attacks)  
- FaceNet threshold is hardcoded (0.5) — may need tuning  
- Attendance duplicates may occur if a person leaves and re-enters frequently  


