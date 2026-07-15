import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard


# --- AYARLAR ---
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['alma', 'birakma', 'cevirme'])
no_sequences = 30
# Her video 45 kareden oluşacak şekilde ayarlandı
sequence_length = 45

# Etiketleme sözlüğü
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action,
                          str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = np.array(to_categorical(labels)).astype(int)

# Veriyi Eğitim ve Test olarak ayır (%5 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- LSTM MODEL MİMARİSİ ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu',
          input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# --- EĞİTİM ---
print("Model eğitimi başlıyor...")
model.fit(X_train, y_train, epochs=200)  # 200 tur eğitim yapacak

model.summary()

# Modeli kaydet
model.save('aksiyon.h5')
print("Model başarıyla 'aksiyon.h5' olarak kaydedildi!")
