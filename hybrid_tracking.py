import cv2
import math
from ultralytics import YOLO

# Explicit imports to stop VS Code / Pyright from throwing red lines
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

def main():
    # 1. Initialize YOLOv8 Nano Model
    yolo_model = YOLO("yolov8n.pt")
    
    # 2. Initialize Optimized MediaPipe Hands
    hands = mp_hands.Hands(  # type: ignore
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("--- PHASE 3: SPATIAL FUSION RUNNING ---")
    print("Move your hand close to a tracked object to trigger interaction.")
    print("Press 'q' to exit.")

    frame_count = 0
    yolo_boxes_cache = []
    
    # Interaction threshold in pixels (adjust based on your camera distance)
    INTERACTION_THRESHOLD = 120

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = frame.copy()
        
        # Coordinates to store for math fusion
        wrist_coords = None
        detected_objects = []

        # --- 1. YOLOv8 OBJECT DETECTION (THROTTLED) ---
        if frame_count % 3 == 0 or not yolo_boxes_cache:
            yolo_results = yolo_model(frame, stream=True, verbose=False)
            yolo_boxes_cache = []
            
            for r in yolo_results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate center point of the object
                    obj_cx = int((x1 + x2) / 2)
                    obj_cy = int((y1 + y2) / 2)
                    
                    yolo_boxes_cache.append((class_name, conf, (x1, y1, x2, y2), (obj_cx, obj_cy)))

        # --- 2. MEDIAPIPE HAND TRACKING ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        mp_results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if mp_results.multi_hand_landmarks:  # type: ignore
            for hand_landmarks in mp_results.multi_hand_landmarks:  # type: ignore
                mp_draw.draw_landmarks(  # type: ignore
                    annotated_frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS  # type: ignore
                )
                
                # Extract Wrist (Landmark 0)
                wrist = hand_landmarks.landmark[0]
                h, w, _ = frame.shape
                wrist_coords = (int(wrist.x * w), int(wrist.y * h))
                
                # Draw the red anchor dot on wrist
                cv2.circle(annotated_frame, wrist_coords, 8, (0, 0, 255), -1)

# --- 3. SPATIAL FUSION LOGIC ---
        for class_name, conf, (x1, y1, x2, y2), (obj_cx, obj_cy) in yolo_boxes_cache:
            is_interacting = False
            distance = 9999.0
            
            # If a hand is visible, calculate distance to the current object
            if wrist_coords is not None:
                distance = math.sqrt((obj_cx - wrist_coords[0])**2 + (obj_cy - wrist_coords[1])**2)
                
                if distance < INTERACTION_THRESHOLD:
                    is_interacting = True
            
            # FIXED: Added explicit check 'and wrist_coords is not None' to satisfy Pyright type narrowing
            if is_interacting and wrist_coords is not None:
                box_color = (255, 0, 0) # Blue for active interaction
                
                # Draw a line from wrist to object center
                cv2.line(annotated_frame, wrist_coords, (obj_cx, obj_cy), (255, 255, 0), 2)
                
                # Put distance text on the line
                cv2.putText(annotated_frame, f"{int(distance)}px", (int((wrist_coords[0]+obj_cx)/2), int((wrist_coords[1]+obj_cy)/2) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                box_color = (0, 255, 0) # Green for passive observation
            
            # Render the Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Render Label
            status = " [HOLDING]" if is_interacting else ""
            label = f"{class_name.upper()}{status} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # 4. Display the fusion feed
        cv2.imshow("Hybrid AI - Phase 3: Spatial Fusion", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()