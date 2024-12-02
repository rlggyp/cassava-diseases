import sys
import cv2
import os
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QRadioButton, QStackedWidget, QHBoxLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import tensorflow.lite as tflite
from ultralytics import YOLO

saved_path = "output_images"

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

class ImageInferencePage(QWidget):
    def __init__(self, tflite_model, yolo_model, labels, min_conf_threshold=0.1):
        super().__init__()
        self.tflite_model = tflite_model
        self.yolo_model = yolo_model
        self.labels = labels
        self.min_conf_threshold = min_conf_threshold

        input_details = self.tflite_model.get_input_details()
        self.tflite_height = input_details[0]['shape'][1]
        self.tflite_width = input_details[0]['shape'][2]

        self.init_ui()
        self.current_model = "SSD"  # Default model is SSD MobileNet

    def init_ui(self):
        layout = QVBoxLayout()

        # self.label = QLabel("Select an image to predict")
        # layout.addWidget(self.label)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["SSD MobileNet", "YOLOv8"])
        self.model_selector.currentTextChanged.connect(self.change_model)
        layout.addWidget(self.model_selector)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        layout.addWidget(self.image_label)

        select_button = QPushButton("Select Image")
        select_button.clicked.connect(self.load_image)

        self.capture_button = QPushButton("Start Camera")
        self.capture_button.clicked.connect(self.toggle_camera)

        self.from_camera = False
        self.camera_running = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        h_layout = QHBoxLayout()
        h_layout.addWidget(select_button)
        h_layout.addWidget(self.capture_button)
        layout.addLayout(h_layout)

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.predict_image)
        layout.addWidget(predict_button)

        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        layout.addWidget(save_button)

        self.setLayout(layout)

        self.image_path = None
        self.capture = None
        self.image_result = None

    def change_model(self, model_name):
        self.current_model = "YOLO" if model_name == "YOLOv8" else "SSD"

    def save_image(self):
        if self.image_result is None:
            return

        datetime_now = datetime.now().strftime("_%Y-%m-%d-%H%M%S") 
        save_path = os.path.join(saved_path, f'image{datetime_now}.jpg')
        cv2.imwrite(save_path, self.image_result)

    def toggle_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Failed to open camera.")
                return

            self.camera_running = True
            self.from_camera = True
            self.capture_button.setText('Stop Camera')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer.start(30)  # Update frame every 30ms
        else:
            if self.cap:
                self.cap.release()
            self.camera_running = False
            self.capture_button.setText('Start Camera')
            self.timer.stop()

    def update_frame(self):
        ret, self.capture = self.cap.read()
        self.display_image(self.capture)

    def load_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg)", options=options)
        if self.image_path:
            self.from_camera = False
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height()))

    def predict_image(self):
        if self.image_path or self.capture is not None:
            if self.current_model == "YOLO":
                self.predict_with_yolo()
            else:
                self.predict_with_ssd()
        else:
            self.image_label.setText("No image selected or captured!")

    def predict_with_yolo(self):
        image = None
        if self.from_camera:
            image = self.capture.copy()
        else:
            image = cv2.imread(self.image_path)

        results = self.yolo_model(image)
        image_with_boxes = results[0].plot()
        self.image_result = image_with_boxes
        self.display_image(image_with_boxes)

    def predict_with_ssd(self):
        image = None
        if self.from_camera:
            image = self.capture.copy()
        else:
            image = cv2.imread(self.image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (self.tflite_width, self.tflite_height))
        input_data = np.expand_dims(image_resized, axis=0)

        if self.tflite_model.get_input_details()[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        input_details = self.tflite_model.get_input_details()
        output_details = self.tflite_model.get_output_details()
        self.tflite_model.set_tensor(input_details[0]['index'], input_data)
        self.tflite_model.invoke()

        boxes = self.tflite_model.get_tensor(output_details[1]['index'])[0]
        classes = self.tflite_model.get_tensor(output_details[3]['index'])[0]
        scores = self.tflite_model.get_tensor(output_details[0]['index'])[0]

        for i in range(len(scores)):
            if self.min_conf_threshold < scores[i] <= 1.0:
                ymin = int(max(1, boxes[i][0] * imH))
                xmin = int(max(1, boxes[i][1] * imW))
                ymax = int(min(imH, boxes[i][2] * imH))
                xmax = int(min(imW, boxes[i][3] * imW))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = self.labels[int(classes[i])]
                label = f'{object_name}: {int(scores[i] * 100)}%'
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        self.image_result = image
        self.display_image(image)

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scaled_image = cv2.resize(rgb_image, (self.image_label.width(), self.image_label.height()))
        q_image = QImage(scaled_image.data, scaled_image.shape[1], scaled_image.shape[0],
                         scaled_image.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

class CameraInferencePage(QWidget):
    def __init__(self, tflite_model, yolo_model, labels, min_conf_threshold=0.1):
        super().__init__()
        self.tflite_model = tflite_model
        self.yolo_model = yolo_model
        self.labels = labels
        self.min_conf_threshold = min_conf_threshold
        self.current_model = "SSD"  # Default model is SSD MobileNet

        # Get input details for SSD MobileNet
        input_details = self.tflite_model.get_input_details()
        self.tflite_height = int(input_details[0]['shape'][1])
        self.tflite_width = int(input_details[0]['shape'][2])

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.model_selector = QComboBox()
        self.model_selector.addItems(["SSD MobileNet", "YOLOv8"])
        self.model_selector.currentTextChanged.connect(self.change_model)
        layout.addWidget(self.model_selector)

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.camera_running = False
        self.button = QPushButton("Start Camera")
        self.button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.button)

    def toggle_camera(self):
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Failed to open camera.")
                return

            self.camera_running = True
            self.button.setText('Stop Camera')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.timer.start(30)  # Update frame every 30ms
        else:
            if self.cap:
                self.cap.release()
            self.camera_running = False
            self.button.setText('Start Camera')
            self.timer.stop()

    def change_model(self, model_name):
        self.current_model = "YOLO" if model_name == "YOLOv8" else "SSD"

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.current_model == "YOLO":
                processed_frame = self.process_with_yolo(frame)
            else:
                processed_frame = self.process_with_ssd(frame)

            # Display the processed frame
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_img))

    def process_with_yolo(self, frame):
        results = self.yolo_model(frame)
        return results[0].plot()

    def process_with_ssd(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imH, imW, _ = frame.shape
        resized_frame = cv2.resize(rgb_frame, (self.tflite_width, self.tflite_height))
        input_data = np.expand_dims(resized_frame, axis=0)

        if self.tflite_model.get_input_details()[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        input_details = self.tflite_model.get_input_details()
        output_details = self.tflite_model.get_output_details()
        self.tflite_model.set_tensor(input_details[0]['index'], input_data)
        self.tflite_model.invoke()

        boxes = self.tflite_model.get_tensor(output_details[1]['index'])[0]
        classes = self.tflite_model.get_tensor(output_details[3]['index'])[0]
        scores = self.tflite_model.get_tensor(output_details[0]['index'])[0]

        for i in range(len(scores)):
            if self.min_conf_threshold < scores[i] <= 1.0:
                ymin = int(max(1, boxes[i][0] * imH))
                xmin = int(max(1, boxes[i][1] * imW))
                ymax = int(min(imH, boxes[i][2] * imH))
                xmax = int(min(imW, boxes[i][3] * imW))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                label = f'{self.labels[int(classes[i])]}: {int(scores[i] * 100)}%'
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        create_folder(saved_path)
        self.setFixedSize(700, 700)

        self.tflite_model = tflite.Interpreter(model_path="model.tflite")
        self.tflite_model.allocate_tensors()

        self.yolo_model = YOLO('yolov8s.pt', verbose=False)  # Load YOLO model (nano for lightweight)

        self.radio_image = QRadioButton("Image Inference")
        self.radio_camera = QRadioButton("Camera Inference")
        self.radio_image.setChecked(True)

        self.radio_image.toggled.connect(self.toggle_page)

        self.stack = QStackedWidget()
        self.image_page = ImageInferencePage(self.tflite_model, self.yolo_model, '12345', 0.5)
        # self.camera_page = QLabel("Camera inference not implemented yet.")  # Placeholder for camera inference
        self.camera_page = CameraInferencePage(self.tflite_model, self.yolo_model, '12345', 0.5)

        self.stack.addWidget(self.image_page)
        self.stack.addWidget(self.camera_page)

        layout = QVBoxLayout()
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_image)
        radio_layout.addWidget(self.radio_camera)
        layout.addLayout(radio_layout)
        layout.addWidget(self.stack)

        self.setLayout(layout)
        self.setWindowTitle("Object Detection: SSD and YOLO")

    def toggle_page(self):
        if self.radio_image.isChecked():
            self.stack.setCurrentWidget(self.image_page)
        else:
            self.stack.setCurrentWidget(self.camera_page)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


