import os, cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

DATA_DIR = "Dataset"
IMG_SIZE = 64

X, y = [], []

for label, folder in enumerate(["cats", "dogs"]):
    path = os.path.join(DATA_DIR, folder)
    print("Reading from:", path)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        X.append(image)
        y.append(label)


X = np.array(X) / 255.0
y = np.array(y)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5)

model.save("models/cnn.h5")
print("CNN model saved")
