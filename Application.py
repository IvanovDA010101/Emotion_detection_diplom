import tkinter as tk
from tkinter import filedialog

import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array

face_classifier = cv2.CascadeClassifier(
    r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Нейтральный', 'Грусть', 'Удивление']


class Form:
    def __init__(self, window):
        self.window = window
        self.video_button = None
        self.photo_button = None
        self.graphics_button = None
        self.training_button = None

        self.create_buttons()

    def create_buttons(self):
        self.video_button = tk.Button(self.window, text="Видео", command=self.open_video_window)
        self.video_button.pack()

        self.photo_button = tk.Button(self.window, text="Фото", command=self.open_photo)
        self.photo_button.pack()

        self.graphics_button = tk.Button(self.window, text="Графики", command=self.show_graphics)
        self.graphics_button.pack()

        self.training_button = tk.Button(self.window, text="Обучение", command=self.training)
        self.training_button.pack()

    def open_video_window(self):
        video_window = tk.Toplevel(self.window)
        video_window.title("Видео")
        video_stream = VideoStream(video_window)

    def open_photo(self):
        photo_window = tk.Toplevel(self.window)
        photo_window.title("Фото")
        photo_viewer = PhotoViewer(photo_window)

    def show_graphics(self):
        graphics_window = tk.Toplevel(self.window)
        graphics_window.title("Графики")
        graphics_window = GraphicsViewer(graphics_window)

    def training(self):
        edu_window = tk.Toplevel(self.window)
        edu_window.title("Обучение")
        edu_viewer = ModelTrainer(edu_window)


class GraphicsViewer:
    def __init__(self, window):
        window.geometry('1000x500')
        self.window = window
        self.canvas = None
        self.graphics = None
        self.display_graphics()

    def display_graphics(self):
        self.canvas = tk.Canvas(self.window,width=1000, height=500)
        self.canvas.pack()
        self.display_info()

    def display_info(self):
        picture = Image.open('graph.jpg').resize((1000,500))
        self.graphics = ImageTk.PhotoImage(picture)
        self.canvas.create_image(0, 0, image=self.graphics, anchor=tk.NW)


class ModelTrainer:
    def __init__(self, window):
        self.window = window
        self.canvas = None
        self.info_button = None
        self.start_button = None
        self.info = None

        self.model = None
        self.train_set = None
        self.test_set = None
        self.display_info()

    def display_info(self):
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack()

        self.info_button = tk.Button(self.window, text="Модель", command=self.load_model)
        self.info_button.pack()

        start_button = tk.Button(self.window, text="Старт", command=self.train_model)
        start_button.pack()

    def update(self, path):
        picture = Image.open(path).resize((400, 250))
        self.info = ImageTk.PhotoImage(picture)
        self.canvas.create_image(0, 0, image=self.info, anchor=tk.NW)

    def load_model(self):
        picture_size = 48
        folder_path = "data for model\\"
        batch_size = 128

        datagen_train = ImageDataGenerator()
        datagen_val = ImageDataGenerator()

        train_set = datagen_train.flow_from_directory(folder_path + "train",
                                                      target_size=(picture_size, picture_size),
                                                      color_mode="grayscale",
                                                      batch_size=batch_size,
                                                      class_mode='categorical',
                                                      shuffle=True)

        test_set = datagen_val.flow_from_directory(folder_path + "test",
                                                   target_size=(picture_size, picture_size),
                                                   color_mode="grayscale",
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   shuffle=False)

        no_of_classes = 7

        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(no_of_classes, activation='softmax'))

        opt = Adam(lr=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        # plot_model(model,to_file="result.png", show_shapes=True)
        # model.summary()
        summary_buffer = io.StringIO()
        model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
        summary_output = summary_buffer.getvalue()
        summary_buffer.close()

        plt.rc('figure', figsize=(8, 5))
        plt.text(0.01, 0.05, str(summary_output), {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.tight_layout()
        path_to_info = 'results.jpg'
        plt.savefig(path_to_info)
        self.update(path_to_info)
        self.model = model
        self.test_set = test_set
        self.train_set = train_set

    def train_model(self):
        model = self.model
        train_set = self.train_set
        test_set = self.test_set
        checkpoint = ModelCheckpoint("./model.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=3,
                                       verbose=1,
                                       restore_best_weights=True
                                       )

        reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.2,
                                                patience=3,
                                                verbose=1,
                                                min_delta=0.0001)

        callbacks_list = [early_stopping, checkpoint, reduce_learningrate]

        epochs = 48

        history = model.fit_generator(generator=train_set,
                                      steps_per_epoch=train_set.n // train_set.batch_size,
                                      epochs=epochs,
                                      validation_data=test_set,
                                      validation_steps=test_set.n // test_set.batch_size,
                                      callbacks=callbacks_list
                                      )

        plt.style.use('dark_background')
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.suptitle('Optimizer : Adam', fontsize=10)
        plt.ylabel('Loss', fontsize=16)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.ylabel('Accuracy', fontsize=16)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.savefig("graph.jpg")
        self.update("graph.jpg")


class VideoStream:
    def __init__(self, window, video_source=0):
        self.window = window
        self.video_source = video_source
        self.vid = None
        self.canvas = None
        self.is_streaming = False
        self.start_button = None
        self.stop_button = None
        self.video_stream()

    def video_stream(self):
        self.vid = cv2.VideoCapture(self.video_source)
        width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = tk.Canvas(self.window, width=width, height=height)
        self.canvas.pack()

        self.start_button = tk.Button(self.window, text="Старт", command=self.start_stream)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(self.window, text="Стоп", command=self.stop_stream)
        self.stop_button.pack(side=tk.LEFT)

        self.update()

    def start_stream(self):
        self.is_streaming = True

    def stop_stream(self):
        self.is_streaming = False

    def update(self):
        if self.is_streaming:
            ret, frame = self.vid.read()
            if ret:
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
                        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'Нет лиц', (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                imgColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(imgColor))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update)


class PhotoViewer:
    def __init__(self, window):
        self.window = window
        self.canvas = None
        self.photo_button = None
        self.photo = None
        self.display_photo()

    def display_photo(self):
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack()

        self.photo_button = tk.Button(self.window, text="Фото", command=self.show_photo)
        self.photo_button.pack()

    def show_photo(self):
        file_path = tk.filedialog.askopenfilename()
        if file_path:
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
                    cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 10, (0, 255, 0), 10)
                else:
                    cv2.putText(img, 'Нет лиц', (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 10)
            dim = (400, 350)
            imgRes = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            imgColor = cv2.cvtColor(imgRes, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(imgColor)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)


def start_form():
    root = tk.Tk()
    root.title("Распознавание эмоций")
    root.geometry('200x110+300+200')
    form = Form(root)
    root.mainloop()


if __name__ == "__main__":
    start_form()
