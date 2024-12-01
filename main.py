import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QRadioButton, QStackedWidget, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import tensorflow.lite as tflite

class ImageInferencePage(QWidget):
    def __init__(self, model, labels, min_conf_threshold=0.1):
        super().__init__()
        self.model = model
        self.labels = labels 
        self.min_conf_threshold = min_conf_threshold

        input_details = self.model.get_input_details()
        self.height = input_details[0]['shape'][1]
        self.width = input_details[0]['shape'][2]
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.label = QLabel("Select an image to predict")
        layout.addWidget(self.label)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(400, 400)
        layout.addWidget(self.image_label)
        
        select_button = QPushButton("Select Image")
        select_button.clicked.connect(self.load_image)
        layout.addWidget(select_button)
        
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.predict_image)
        layout.addWidget(predict_button)
        
        self.setLayout(layout)
        
        self.image_path = None
        
    def load_image(self):
        options = QFileDialog.Options()
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp *.jpeg)", options=options)
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height()))
    
    def predict_image(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape
            # Resize image for the model while keeping aspect ratio intact
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Normalize pixel values for floating-point model
            if self.model.get_input_details()[0]['dtype'] == np.float32:
                input_data = (np.float32(input_data) - 127.5) / 127.5

            # Perform inference
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()

            # Retrieve detection results
            boxes = self.model.get_tensor(output_details[1]['index'])[0]
            classes = self.model.get_tensor(output_details[3]['index'])[0]
            scores = self.model.get_tensor(output_details[0]['index'])[0]

            # Ensure scores, boxes, and classes are iterable
            if not isinstance(scores, np.ndarray):
                scores = np.expand_dims(scores, axis=0)
                boxes = np.expand_dims(boxes, axis=0)
                classes = np.expand_dims(classes, axis=0)

            # Loop through detections and draw results
            for i in range(len(scores)):
                if self.min_conf_threshold < scores[i] <= 1.0:
                    # Rescale the bounding boxes to match the original image size
                    ymin = int(max(1, boxes[i][0] * imH))
                    xmin = int(max(1, boxes[i][1] * imW))
                    ymax = int(min(imH, boxes[i][2] * imH))
                    xmax = int(min(imW, boxes[i][3] * imW))

                    # Draw bounding box
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = self.labels[int(classes[i])]
                    label = f'{object_name}: {int(scores[i] * 100)}%'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, label_size[1] + 10)
                    cv2.rectangle(image, (xmin, label_ymin - label_size[1] - 10),
                                  (xmin + label_size[0], label_ymin + 5), (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, label, (xmin, label_ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Convert image back to QPixmap and display
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = channel * width

            # Scale the image to fit within the display area (while maintaining aspect ratio)
            scaled_image = cv2.resize(rgb_image, (self.image_label.width(), self.image_label.height()))
            
            # Convert the scaled image to QImage
            q_image = QImage(scaled_image.data, scaled_image.shape[1], scaled_image.shape[0],
                             scaled_image.strides[0], QImage.Format_RGB888)
            
            # Display the image on the label
            self.image_label.setPixmap(QPixmap.fromImage(q_image))
        else:
            self.label.setText("No image selected!")

class CameraInferencePage(QWidget):
    def __init__(self, model, labels, min_conf_threshold=0.1):
        super().__init__()
        self.model = model
        self.labels = labels
        self.min_conf_threshold = min_conf_threshold

        # Get input details from the model (this is where the issue might lie)
        input_details = self.model.get_input_details()
        self.height = input_details[0]['shape'][1]  # Height (height of input image)
        self.width = input_details[0]['shape'][2]   # Width (width of input image)

        # Ensure they are integers (sometimes numpy types cause issues)
        self.height = int(self.height)
        self.width = int(self.width)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label)

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        print("Starting camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Failed to open camera.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer.start(30)  # Update frame every 30ms

    def stop_camera(self):
        print("Stopping camera...")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.camera_label.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize the frame for the model input while maintaining aspect ratio
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(rgb_frame, (self.width, self.height))

            # Normalize the input for the model if necessary
            input_data = np.expand_dims(image_resized, axis=0)
            if self.model.get_input_details()[0]['dtype'] == np.float32:
                input_data = (np.float32(input_data) - 127.5) / 127.5

            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()

            # Retrieve the outputs from the model
            boxes = self.model.get_tensor(output_details[1]['index'])[0]
            classes = self.model.get_tensor(output_details[3]['index'])[0]
            scores = self.model.get_tensor(output_details[0]['index'])[0]

            if not isinstance(scores, np.ndarray):
                scores = np.expand_dims(scores, axis=0)
                boxes = np.expand_dims(boxes, axis=0)
                classes = np.expand_dims(classes, axis=0)

            # Draw bounding boxes and labels on the frame
            for i in range(len(scores)):
                if self.min_conf_threshold < scores[i] <= 1.0:
                    ymin = int(max(1, boxes[i][0] * frame.shape[0]))
                    xmin = int(max(1, boxes[i][1] * frame.shape[1]))
                    ymax = int(min(frame.shape[0], boxes[i][2] * frame.shape[0]))
                    xmax = int(min(frame.shape[1], boxes[i][3] * frame.shape[1]))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                    object_name = self.labels[int(classes[i])]
                    label = f'{object_name}: {int(scores[i] * 100)}%'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(ymin, label_size[1] + 10)
                    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                                  (xmin + label_size[0], label_ymin + 5), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xmin, label_ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Convert the frame to QImage and display on the QLabel
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_img))

        else:
            print("Failed to capture frame.")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.model = tflite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()
        
        self.radio_image = QRadioButton("Image Inference")
        self.radio_camera = QRadioButton("Camera Inference")
        
        self.radio_image.setChecked(True)
        
        self.radio_image.toggled.connect(self.toggle_page)
        
        self.stack = QStackedWidget()
        self.image_page = ImageInferencePage(self.model, '12345', 0.5)
        self.camera_page = CameraInferencePage(self.model, '12345', 0.7)
        
        self.stack.addWidget(self.image_page)
        self.stack.addWidget(self.camera_page)
        
        layout = QVBoxLayout()
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.radio_image)
        radio_layout.addWidget(self.radio_camera)
        layout.addLayout(radio_layout)
        layout.addWidget(self.stack)
        
        self.setLayout(layout)
        self.setWindowTitle("Cassava Disease")
        
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


