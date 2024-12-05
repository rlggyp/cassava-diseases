# Cassava Leaf Disease Detection GUI
A Python-based desktop application for detecting cassava leaf diseases. Built with PyQt5, the application supports both SSD MobileNet and YOLOv8 models.

## Features
- [x] Image Inference
- [x] Camera Capture & Inference
- [x] Realtime Detection
- [x] Result Saving

Tested on Ubuntu `20.04`.

## Dependencies
- opencv-python
- PyQt5
- tflite-runtime

## Screenshots
<p align="center"> 
  <img alt="screenshots" src="assets/screenshot.jpeg">
</p>

## Installation
### Installing Ubuntu Packages
```bash
sudo apt-get install -y \
    python3 \
    python3-pyqt5* \
    pyqt5* \
    python3-opencv \
    python3-pip \
    qt5-qmake \
    qtbase5-dev \
    qtchooser

```
### Installing Python Packages
```bash
pip3 install tflite-runtime opencv-python PyQt5
```
## Usage
```bash
python3 main.py
```
