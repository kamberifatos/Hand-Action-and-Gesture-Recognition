import os
import numpy as np
import cv2
import mediapipe as mp
import time

# --- AYARLAR ---
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['alma', 'birakma', 'cevirme'])  # Hareketlerin isimleri
no_sequences = 30  # Her hareket için 30 farklı video kaydı
sequence_length = 45  # Her video 45 kareden oluşur

# Klasörleri oluştur
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action,
                    str(sequence)), exist_ok=True)

# MediaPipe Ayarları (Senin çalışan kodunla aynı)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,  # Veri toplarken tek el daha sağlıklıdır
    min_hand_detection_confidence=0.7
)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                success, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Ekrana Bilgilendirme Yazıları
                cv2.putText(frame, f'HAREKET: {action.upper()} | KAYIT NO: {sequence}', (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if frame_num == 0:
                    cv2.putText(frame, 'HAZIRLAN VE BASLA!', (150, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                    cv2.imshow('Veri Toplama Merkezi', frame)
                    cv2.waitKey(2000)  # Hareketler arası 2 saniye mola
                else:
                    cv2.imshow('Veri Toplama Merkezi', frame)

                # --- KOORDİNATLARI ÇIKAR VE KAYDET ---
                if result.hand_landmarks:
                    # 21 nokta x 3 (x,y,z) = 63 sayısal değer
                    landmarks = result.hand_landmarks[0]
                    keypoints = np.array([[lm.x, lm.y, lm.z]
                                         for lm in landmarks]).flatten()
                else:
                    # El tespit edilemezse sıfır dizisi kaydet (Hata almamak için)
                    keypoints = np.zeros(21*3)

                # .npy olarak kaydet
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
print("Veri toplama tamamlandı!")
