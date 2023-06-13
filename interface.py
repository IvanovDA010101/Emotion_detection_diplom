from tkinter import *
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tkinter import filedialog

root = Tk()
root.title("Hello METANIT.COM")
root.geometry("500x450+660+320")
image = cv2.imread("C:\\Users\\timar\\Desktop\\diplom\Emotion_Detection_CNN\\test\\happy\\PrivateTest_218533.jpg")
# # imageFrame = ttk.Frame(root, width=300, height=300)
label_root = Image.fromarray(image)
# imgtk = ImageTk.PhotoImage(image=img)


def pic():
    # Создаем окно Tkinter
    # root1 = Tk()
    # root1.withdraw()

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
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    dim = (400, 350)
    imgRes = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("test",imgRes)
    fr = Image.fromarray(imgRes)
    frame_tk = ImageTk.PhotoImage(image=fr)
    label_root.image = frame_tk
    ttk.Label(root, image=frame_tk).place(x=10, y=40, width=350, height=380)
    # ttk.Label(root, image=frame_tk).place(x=10, y=40, width=350, height=380)


pass

btn_foto = ttk.Button(text="Фото", command=pic)
btn_foto.place(relx=0, rely=0)


def web_camera():
    face_classifier = cv2.CascadeClassifier(
        r'C:\Users\timar\Desktop\diplom\Emotion_Detection_CNN\haarcascade_frontalface_default.xml')
    classifier = load_model(r'C:\Users\timar\Desktop\diplom\Emotion_Detection_CNN\model.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fr = Image.fromarray(frame)
        frame_tk = ImageTk.PhotoImage(image=fr)
        label_root.image = frame_tk
        ttk.Label(root, image=frame_tk).place(x=10, y=40, width=350, height=380)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    # cv2.destroyAllWindows()


pass

btn_camera = ttk.Button(text="Камера", command=web_camera)
btn_camera.place(relx=0.2, rely=0)

btn_edu = ttk.Button(text="Обучение")
btn_edu.place(relx=0.4, rely=0)

btn_gr = ttk.Button(text="Графики")
btn_gr.place(relx=0.6, rely=0)

btn_start = ttk.Button(text="Старт")
btn_start.place(relx=0.83, rely=0.80)

btn_stop = ttk.Button(text="Стоп")
btn_stop.place(relx=0.83, rely=0.87)

# ttk.Label(root, image=imgtk).place(x=10, y=40, width=350, height=380)

root.mainloop()
