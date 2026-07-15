import cv2
import numpy as np
import mediapipe as mp
import time
try:
    from tensorflow.keras.models import load_model
except Exception:
    from keras.models import load_model

# --- AYARLAR ---
actions = np.array(['alma', 'birakma', 'cevirme'])
sequence_length = 45  # Model eğitilirken kullandığın kare sayısı 45

# Eğittiğin modeli yükle
try:
    model = load_model('aksiyon.h5')
except Exception as e:
    print(f"Model yüklenemedi: {e}")
    exit()

# MediaPipe Kurulumu (Task API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

# --- DEĞİŞKENLER ---
sequence = []  # Son 30 kareyi tutacak liste
threshold = 0.8  # Tahmin güven aralığı (%80'den eminse yazdır)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Koordinatları Çıkar
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            keypoints = np.array([[lm.x, lm.y, lm.z]
                                 for lm in landmarks]).flatten()
        else:
            keypoints = np.zeros(21*3)

        # Yeni koordinatı kuyruğa ekle
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]  # Sadece son 30 kareyi tut

        # Tahmin Yap (Elimizde yeterli kare biriktiyse)
        if len(sequence) == sequence_length and model is not None:
            # Modeli tahmin için hazırla (1, 30, 63) formatında olmalı
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

            # En yüksek olasılıklı hareketi bul
            if res[np.argmax(res)] > threshold:
                action_text = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]

                # Ekrana Yazdır
                cv2.rectangle(frame, (0, 0), (250, 40), (245, 117, 16), -1)
                cv2.putText(frame, f'{action_text.upper()} ({confidence:.2f})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Bitirme Projesi - Canlı Tahmin', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
