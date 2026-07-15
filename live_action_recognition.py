import time
import cv2
import torch
import numpy as np
from threading import Lock, Thread
from collections import deque
from ultralytics import YOLO
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    yolo_model = YOLO("yolov8n.pt")

    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    video_model: VideoMAEForVideoClassification = VideoMAEForVideoClassification.from_pretrained(model_name)
    video_model.to(device)
    video_model.eval()
    if device.type == "cuda":
        video_model.half()

    object_action_hint = {
        "cell phone": "using phone",
        "bottle": "drinking",
        "cup": "drinking",
        "book": "reading",
        "laptop": "using laptop",
        "keyboard": "typing",
    }

    FRAME_WINDOW_SIZE = 16
    frame_buffer = deque(maxlen=FRAME_WINDOW_SIZE)
    buffer_lock = Lock()
    prediction_lock = Lock()
    latest_action = "waiting for buffer..."
    latest_hint = ""
    stop_flag = False
    current_action_history = deque(maxlen=6)

    def inference_worker():
        nonlocal latest_action, latest_hint, stop_flag
        while not stop_flag:
            with buffer_lock:
                frames = list(frame_buffer) if len(frame_buffer) == FRAME_WINDOW_SIZE else None

            if frames is None:
                time.sleep(0.02)
                continue

            try:
                inputs = processor(frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = video_model(**inputs)
                    predicted_class_idx = outputs.logits.argmax(-1).item()

                raw_action = video_model.config.id2label[predicted_class_idx].replace("_", " ").lower()
                with prediction_lock:
                    if latest_hint in ["using phone", "drinking"]:
                        latest_action = latest_hint
                    else:
                        latest_action = raw_action
                    current_action_history.append(latest_action)
            except Exception as exc:
                print(f"Inference error: {exc}")
                time.sleep(0.1)

            time.sleep(0.08)

    ai_thread = Thread(target=inference_worker, daemon=True)
    ai_thread.start()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: camera cannot be opened")
        stop_flag = True
        ai_thread.join(timeout=1.0)
        return

    print("--- LIVE TWO-STAGE HYBRID AI RUNNING ---")
    print("Press 'q' to exit.")

    frame_count = 0
    yolo_skip = 5
    object_labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        if frame_count % yolo_skip == 0:
            object_labels = []
            yolo_results = yolo_model(frame, stream=True, verbose=False)
            for result in yolo_results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]
                    if class_name in object_action_hint:
                        object_labels.append(class_name)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            annotated_frame,
                            class_name.upper(),
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (224, 224))
        with buffer_lock:
            frame_buffer.append(resized_frame)

        hint_action = ""
        if object_labels:
            hint_action = object_action_hint.get(object_labels[-1], "")

        with prediction_lock:
            latest_hint = hint_action
            if current_action_history:
                current_action = max(set(current_action_history), key=current_action_history.count)
            else:
                current_action = latest_action

        if hint_action and hint_action != current_action:
            display_action = f"{hint_action.upper()} (hint)"
        else:
            display_action = current_action.upper()

        cv2.rectangle(annotated_frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            f"LIVE ACTION: {display_action}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        buffer_pct = int((len(frame_buffer) / FRAME_WINDOW_SIZE) * 100)
        cv2.putText(
            annotated_frame,
            f"Buffer: {buffer_pct}%",
            (w - 130, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Spatio-Temporal Hybrid AI", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_flag = True
    ai_thread.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()