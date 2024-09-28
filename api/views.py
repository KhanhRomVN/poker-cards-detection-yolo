import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from datetime import datetime

# Load the trained model
MODEL_PATH = os.path.join(settings.BASE_DIR, "model", "poker_card_detection", "weights", "best.pt")
model = YOLO(MODEL_PATH)

# Create result directories
RESULT_DIR = os.path.join(settings.BASE_DIR, "result", "poker-card-detection-result")
IMAGE_RESULT_DIR = os.path.join(RESULT_DIR, "image")
VIDEO_RESULT_DIR = os.path.join(RESULT_DIR, "video")
os.makedirs(IMAGE_RESULT_DIR, exist_ok=True)
os.makedirs(VIDEO_RESULT_DIR, exist_ok=True)

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            label = f"{result.names[int(cls)]} {conf:.2f}"
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, label, color='white', fontweight='bold', bbox=dict(facecolor='red', alpha=0.5))
            
            detections.append({
                "class": result.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(x) for x in [x1, y1, x2, y2]]
            })
    
    plt.axis('off')
    
    plot_filename = os.path.join(IMAGE_RESULT_DIR, f"detection_{os.path.basename(image_path)}")
    plt.savefig(plot_filename)
    plt.close()
    
    return detections, plot_filename

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_path = os.path.join(VIDEO_RESULT_DIR, f"detection_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = f"{result.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    
    print(f"Video processing completed. Output saved to {output_path}")
    return output_path

@require_http_methods(["GET", "POST"])
def poker_card_detection(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get('file')
        if not uploaded_file:
            return JsonResponse({"error": "No file uploaded"}, status=400)
        
        file_name = default_storage.save(uploaded_file.name, ContentFile(uploaded_file.read()))
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            detections, plot_path = process_image(file_path)
            # Tạo URL tương đối cho ảnh kết quả
            relative_path = os.path.relpath(plot_path, settings.MEDIA_ROOT)
            image_url = settings.MEDIA_URL + relative_path
            result = {
                "type": "image",
                "detections": detections,
                "image_url": image_url
            }
        elif file_name.lower().endswith(('.mp4', '.avi', '.mov')):
            output_path = process_video(file_path)
            # Tạo URL tương đối cho video kết quả
            relative_path = os.path.relpath(output_path, settings.MEDIA_ROOT)
            video_url = settings.MEDIA_URL + relative_path
            result = {
                "type": "video",
                "video_url": video_url
            }
        else:
            return JsonResponse({"error": "Unsupported file type"}, status=400)
        
        # Clean up the uploaded file
        default_storage.delete(file_name)
        
        return JsonResponse(result)
    
    return render(request, 'poker_card_detection.html')