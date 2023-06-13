import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from keras.models import load_model
from keras.utils.image_utils import img_to_array


def pic():
    # Создаем окно Tkinter
    root = tk.Tk()
    root.withdraw()

    # Запускаем диалоговое окно для выбора файла
    file_path = filedialog.askopenfilename()


# Загружаем модель из файла
    model = load_model('model.h5')

    # Список эмоций, которые может распознавать модель
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Загружаем картинку
# img = image.load_img(file_path, target_size=(48, 48), grayscale=True)

    face_classifier = cv2.CascadeClassifier(
        r'haarcascade_frontalface_default.xml')
    classifier = load_model(r'model.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    img = cv2.imread(file_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 10)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 10)
        else:
            cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 10)
    dim = (400, 300)
    imgRes = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    while True:
        cv2.imshow('Emotion Detector', imgRes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
