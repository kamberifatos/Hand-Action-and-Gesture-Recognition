import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

def main():
    # 1. Load YOLOv8 for spatial object tracking
    yolo_model = YOLO("yolov8n.pt")
    
    # 2. Load Pre-trained VideoMAE Transformer from Hugging Face
    # This model natively recognizes 400 different human-object interactions
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    video_model = VideoMAEForVideoClassification.from_pretrained(model_name)
    
    # Put model in evaluation mode for faster laptop CPU inference
    video_model.eval()

    # 3. Read your already available gesture/action video file
    video_path = "path_to_your_available_video.mp4" # <-- Put your video file name here
    cap = cv2.VideoCapture(video_path)
    
    # Configuration for the Video Transformer
    # VideoMAE expects a sequence of exactly 16 frames scaled to 224x224 pixels
    FRAME_WINDOW_SIZE = 16
    frame_buffer = []
    
    print(f"Processing video: {video_path}...")
    current_action = "Analyzing sequence..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        # --- STAGE 1: YOLO SPATIAL CHECK ---
        # Look for target objects in the current frame
        yolo_results = yolo_model(frame, stream=True, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                c_id = int(box.cls[0])
                c_name = yolo_model.names[c_id]
                # If it's an object of interest, draw it
                if c_name in ["cell phone", "bottle", "cup"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, c_name.upper(), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- STAGE 2: TEMPORAL SEQUENCE BUFFER ---
        # Convert frame to RGB and resize for the Video Transformer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        frame_buffer.append(resized_frame)

        # Once we have 16 consecutive frames, trigger the Video Transformer
        if len(frame_buffer) == FRAME_WINDOW_SIZE:
            # Convert list of 16 frames to a single numpy array batch shape: (16, 224, 224, 3)
            video_sequence = np.array(frame_buffer)
            
            # Pre-process the sequence format for PyTorch
            inputs = processor(list(video_sequence), return_tensors="pt")
            
            with torch.no_grad():
                outputs = video_model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                
            # Decode the numerical ID to the text action label
            current_action = video_model.config.id2label[predicted_class_idx]
            
            # Slide the time window forward by removing the oldest frame
            frame_buffer.pop(0)

        # --- STAGE 3: UI OVERLAY ---
        # Display the classification result in real-time on top of the video
        cv2.rectangle(annotated_frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"ACTION: {current_action.upper()}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Two-Stage Hybrid AI Video Pipeline", annotated_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing finished successfully.")

if __name__ == "__main__":
    main()