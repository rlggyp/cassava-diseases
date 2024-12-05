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
from tflite_runtime.interpreter import Interpreter

class_list = [ "CBB", "CBSD", "CGM", "CMD", "Healthy" ]
class_colors = [
    (0, 128, 255), # Blue - CBB
    (0, 255, 128), # Green - CBSD
    (255, 128, 0), # Orange - CGM
    (128, 0, 255), # Purple - CMD
    (255, 255, 0)  # Yellow - Healthy
]

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{class_list[class_id]} ({confidence:.2f})"
    color = class_colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

class ImageInferencePage(QWidget):
    def __init__(self, tflite_model, yolo_model, labels, min_conf_threshold=0.5):
        super().__init__()
        self.tflite_model = tflite_model
        self.yolo_model = yolo_model
        self.labels = labels
        self.min_conf_threshold = min_conf_threshold

        input_details = self.tflite_model.get_input_details()
        self.tflite_height = input_details[0]['shape'][1]
        self.tflite_width = input_details[0]['shape'][2]

        self.init_ui()
        self.current_model = "SSD"

    def init_ui(self):
        layout = QVBoxLayout()

        self.model_selector = QComboBox()
        self.model_selector.addItems(["SSD MobileNet", "YOLO"])
        self.model_selector.currentTextChanged.connect(self.change_model)
        layout.addWidget(self.model_selector)

        self.image_label = QLabel()
        # self.image_label.setFixedSize(640, 480)
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
        self.current_model = "YOLO" if model_name == "YOLO" else "SSD"

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
            self.timer.start(30) # Update frame every 30ms
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
            self.display_image(cv2.imread(self.image_path))

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

        original_image: np.ndarray = image
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 320

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(320, 320), swapRB=True)
        self.yolo_model.setInput(blob)

        # Perform inference
        outputs = self.yolo_model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": class_list[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        image_with_boxes = original_image
        self.image_result = original_image
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

                
                class_id = int(classes[i]) % len(class_colors)
                color = class_colors[class_id]

                
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)

                
                object_name = self.labels[class_id]
                label = f'{object_name} {scores[i]:.2f}'

                
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_width, text_height = text_size

                
                label_x = xmin
                label_y = ymin - text_height - 10
                if label_y < 0:
                    label_y = ymin + text_height + 10
                if label_x + text_width + 10 > imW:
                    label_x = imW - text_width - 10

                cv2.rectangle(image, (label_x, label_y - text_height - 5), 
                              (label_x + text_width + 10, label_y + 5), color, -1)

                cv2.putText(image, label, (label_x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        self.image_result = image
        self.display_image(image)

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        label_aspect_ratio = label_width / label_height
        
        image_height, image_width, _ = rgb_image.shape
        image_aspect_ratio = image_width / image_height
        
        if image_aspect_ratio > label_aspect_ratio:
            new_width = label_width
            new_height = int(label_width / image_aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * image_aspect_ratio)
        
        resized_image = cv2.resize(rgb_image, (new_width, new_height))
        
        padded_image = 255 * np.ones((label_height, label_width, 3), dtype=np.uint8)
        
        y_offset = (label_height - new_height) // 2
        x_offset = (label_width - new_width) // 2
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        q_image = QImage(padded_image.data, padded_image.shape[1], padded_image.shape[0],
                         padded_image.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

class CameraInferencePage(QWidget):
    def __init__(self, tflite_model, yolo_model, labels, min_conf_threshold=0.5):
        super().__init__()
        self.tflite_model = tflite_model
        self.yolo_model = yolo_model
        self.labels = labels
        self.min_conf_threshold = min_conf_threshold
        self.current_model = "SSD"

        input_details = self.tflite_model.get_input_details()
        self.tflite_height = int(input_details[0]['shape'][1])
        self.tflite_width = int(input_details[0]['shape'][2])

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.model_selector = QComboBox()
        self.model_selector.addItems(["SSD MobileNet", "YOLO"])
        self.model_selector.currentTextChanged.connect(self.change_model)
        layout.addWidget(self.model_selector)

        self.camera_label = QLabel()
        # self.camera_label.setFixedSize(640, 480)
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
        self.current_model = "YOLO" if model_name == "YOLO" else "SSD"

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.current_model == "YOLO":
                processed_frame = self.process_with_yolo(frame)
            else:
                processed_frame = self.process_with_ssd(frame)

            self.display_image(processed_frame)

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_width = self.camera_label.width()
        label_height = self.camera_label.height()
        label_aspect_ratio = label_width / label_height
        
        image_height, image_width, _ = rgb_image.shape
        image_aspect_ratio = image_width / image_height
        
        if image_aspect_ratio > label_aspect_ratio:
            new_width = label_width
            new_height = int(label_width / image_aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * image_aspect_ratio)
        
        resized_image = cv2.resize(rgb_image, (new_width, new_height))
        
        padded_image = 255 * np.ones((label_height, label_width, 3), dtype=np.uint8)
        
        y_offset = (label_height - new_height) // 2
        x_offset = (label_width - new_width) // 2
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        q_image = QImage(padded_image.data, padded_image.shape[1], padded_image.shape[0],
                         padded_image.strides[0], QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_image))

    def process_with_yolo(self, frame):
        original_image: np.ndarray = frame
        [height, width, _] = original_image.shape

        # Prepare a square image for inference
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image

        # Calculate scale factor
        scale = length / 320

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(320, 320), swapRB=True)
        self.yolo_model.setInput(blob)

        # Perform inference
        outputs = self.yolo_model.forward()

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": class_list[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            draw_bounding_box(
                original_image,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        return original_image

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

                class_id = int(classes[i]) % len(class_colors)
                color = class_colors[class_id]

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                object_name = self.labels[class_id]
                label = f'{object_name} {scores[i]:.2f}'

                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_width, text_height = text_size

                label_x = xmin
                label_y = ymin - text_height - 10
                if label_y < 0:
                    label_y = ymin + text_height + 10
                if label_x + text_width + 10 > imW:
                    label_x = imW - text_width - 10

                cv2.rectangle(frame, (label_x, label_y - text_height - 5), 
                              (label_x + text_width + 10, label_y + 5), color, -1)

                cv2.putText(frame, label, (label_x + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return frame

class EnvManager:
    def __init__(self, env_file=".env"):
        self.env_file = env_file
        self.env_vars = {}
        self.load_env()

    def load_env(self):
        """Load existing environment variables from .env file."""
        if os.path.exists(self.env_file):
            with open(self.env_file, "r") as file:
                for line in file:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        self.env_vars[key] = self.cast_value(value)

    def cast_value(self, value):
        """Cast the value to int, float, or keep as a string."""
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def set_env_var(self, key, value):
        """Set or update an environment variable."""
        if isinstance(value, (int, float)):
            value = str(value)
        if key in self.env_vars and str(self.env_vars[key]) == value:
            print(f"No change for '{key}' (already set to '{value}')")
            return
        self.env_vars[key] = self.cast_value(value)
        self.write_env()
        print(f"Updated '{key}' to '{value}'")

    def write_env(self):
        """Write the current environment variables to the .env file."""
        with open(self.env_file, "w") as file:
            for key, value in self.env_vars.items():
                file.write(f"{key}={value}\n")
        print(f"Environment variables written to {self.env_file}")

    def get_env_var(self, key):
        """Get the value of an environment variable."""
        return self.env_vars.get(key)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.env_manager = EnvManager()
        min_conf_threshold = self.env_manager.get_env_var('MIN_CONF_THRESHOLD')
        yolo_model_filepath = self.env_manager.get_env_var('YOLO_MODEL_FILEPATH')
        tflite_model_filepath = self.env_manager.get_env_var('TFLITE_MODEL_FILEPATH')
        saved_path = self.env_manager.get_env_var('SAVED_PATH')

        create_folder(saved_path)

        # self.setFixedSize(700, 700)

        self.tflite_model = Interpreter(model_path=tflite_model_filepath)
        self.tflite_model.allocate_tensors()

        self.yolo_model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(yolo_model_filepath)

        self.radio_image = QRadioButton("Image Inference")
        self.radio_camera = QRadioButton("Camera Inference")
        self.radio_image.setChecked(True)

        self.radio_image.toggled.connect(self.toggle_page)
        
        self.stack = QStackedWidget()
        self.image_page = ImageInferencePage(self.tflite_model, self.yolo_model, class_list, min_conf_threshold)
        self.camera_page = CameraInferencePage(self.tflite_model, self.yolo_model, class_list, min_conf_threshold)

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
