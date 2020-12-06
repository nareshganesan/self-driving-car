FROM nvidia/cuda:10.1-devel-ubuntu18.04
LABEL maintainer "nareshkumarganesan@gmail.com"

RUN apt update && apt upgrade -y && \
  apt install -y software-properties-common wget nano && \
  add-apt-repository 'deb http://security.ubuntu.com/ubuntu xenial-security main' && \
  apt update && apt install -y \
  build-essential cmake pkg-config unzip yasm \
  git checkinstall libjpeg-dev libpng-dev \
  libtiff-dev libjasper1 libjasper-dev libavcodec-dev \
  libavformat-dev libswscale-dev libavresample-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libxvidcore-dev x264 libx264-dev libfaac-dev \
  libmp3lame-dev libtheora-dev libfaac-dev \
  libmp3lame-dev libvorbis-dev libopencore-amrnb-dev \
  libopencore-amrwb-dev libdc1394-22 libdc1394-22-dev \
  libxine2-dev libv4l-dev v4l-utils libgtk-3-dev \
  python3-dev python3-pip libtbb-dev \
  libatlas-base-dev gfortran && cd /usr/include/linux && \
  ln -s -f ../libv4l1-videodev.h videodev.h && \
  pip3 install -U pip numpy && \
  pip3 install virtualenv virtualenvwrapper \
  matplotlib scipy scikit-learn && \
  echo "Create a virtual environtment for the python binding module" && \
  echo 'export WORKON_HOME=$HOME/.virtualenvs' >> ~/.bashrc && \
  echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> ~/.bashrc && \
  echo '. /usr/local/bin/virtualenvwrapper.sh' >> ~/.bashrc && \
  echo 'export PKG_CONFIG_PATH=/opencv/opencv/build/unix-install/opencv.pc' >> ~/.bashrc && \
  rm -rf ~/.cache/pip && \
  rm -rf /var/lib/apt/lists/* && cd / && \
  wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip && \
  wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip && \
  unzip opencv.zip && \
  unzip opencv_contrib.zip && \
  echo "Procced with the installation" && \
  cd opencv-4.1.0 && mkdir build && cd build && \
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D INSTALL_PYTHON_EXAMPLES=ON \
  -D INSTALL_C_EXAMPLES=OFF \
  -D WITH_TBB=ON \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=7.5 \
  -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
  -D BUILD_opencv_cudacodec=OFF \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_CUBLAS=1 \
  -D WITH_V4L=ON \
  -D WITH_QT=OFF \
  -D WITH_OPENGL=ON \
  -D WITH_GSTREAMER=ON \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D OPENCV_PC_FILE_NAME=opencv.pc \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.6/dist-packages \
  -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib-4.1.0/modules \
  -D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
  -D BUILD_EXAMPLES=ON .. && \
  make -j"$(nproc)" && \
  make install && \
  ldconfig && \
  pip3 install torch==1.7.0+cu101 \
  torchvision==0.8.1+cu101 \
  torchaudio==0.7.0 \
  -f https://download.pytorch.org/whl/torch_stable.html \
  && apt-get clean \
  && apt-get autoclean \
  && apt-get autoremove \
  && rm -rf /tmp/* /var/tmp/* \
  && rm -rf /var/lib/apt/lists/* \
  && rm -f /var/cache/apt/archives/*.deb \
  && rm -f /var/cache/apt/archives/partial/*.deb \
  && rm -f /var/cache/apt/*.bin

