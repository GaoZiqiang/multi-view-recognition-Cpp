# Introduction
the C++ implement of multi-view recognition.
# Requiremens
## the ditectory structure
├── CMakeLists.txt

├── data

├── main.cpp

├── main.h

├── model

│   ├── ModelManager.cpp

│   ├── ModelManager.h

│   └── modified_resnet.pt

├── README.md

└── util

    ├── DataManager.cpp

    ├── DataManager.h

    ├── MetricLearning.cpp

    └── MetricLearning.h

## requirements
gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~18.04)

opencv

libtorch 1.7.1+cpu
## models and datasets
downloaded and put into diretory "./model/",download from 

链接: https://pan.baidu.com/s/1Fst0TwDY-xsn_GxdfyUTCQ  密码: 6dag;

downloaded and put into directory "./data/",download from 

链接: https://pan.baidu.com/s/1H0ZnIHHA2klRRlGxGUA6gg  密码: 3kek;
# Run
git clone https://github.com/GaoZiqiang/multi-view-recognition-Cpp.git

cd $PROJECT_HOME

mkdir build

cmake -DCMAKE_PREFIX_PATH=/your/path/to/libtorch ..

cmake --build . --config Release

./MultiViewRecogCpp

