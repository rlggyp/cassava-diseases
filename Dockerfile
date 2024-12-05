FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \ 
  && apt-get install -y debconf-utils \
  && echo 'keyboard-configuration keyboard-configuration/layoutcode string us' | debconf-set-selections \
  && echo 'keyboard-configuration keyboard-configuration/modelcode string pc105' | debconf-set-selections \
  && ln -sf /usr/share/zoneinfo/Asia/Jakarta /etc/localtime \
  && apt-get install -y \
    python3 \
    python3-pyqt5* \
    pyqt5* \
    python3-opencv \
    python3-pip \
    qt5-qmake \
    qtbase5-dev \
    qtchooser \
  && apt-get clean \ 
  && rm -rf /var/lib/apt/lists/* \
  && pip3 install --upgrade pip \
  && pip3 install tflite-runtime opencv-python PyQt5

CMD ["bash"]
