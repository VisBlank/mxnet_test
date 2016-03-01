#-------------------------------------------------
#
# Project created by QtCreator 2016-01-13T11:40:08
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = mxnet_test
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app

# 添加头文件路径
CUDA_PATH = /usr/local/cuda

# 添加lib文件路径
CV_LIB = /usr/local/lib
MXNET_LIB = $${PWD}/lib

DESTDIR = $${PWD}/build

INCLUDEPATH += $${PWD}/include \
               $${CUDA_PATH}/include

LIBS += -L$${MXNET_LIB} -lmxnet \
        -L$${CV_LIB} -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

SOURCES += src/predict.cpp

HEADERS += include/predict.h
