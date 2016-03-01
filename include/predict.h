#ifndef PREDICT_H
#define PREDICT_H

#include <stdio.h>

// Path for c_predict_api
#include <mxnet/c_predict_api.h>

#include <opencv2/opencv.hpp>

#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <sys/time.h>
#include <fstream>


// Read file to buffer
class BufferFile {
 public :
    std::string file_path_;
    int length_;
    char* buffer_;

    explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

        std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
        if (!ifs) {
            std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            assert(false);
        }

        ifs.seekg(0, std::ios::end);
        length_ = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
//        std::cout << file_path.c_str() << " ... "<< length_ << " bytes\n";

        buffer_ = new char[sizeof(char) * length_];
        ifs.read(buffer_, length_);
        ifs.close();
    }

    int GetLength() {
        return length_;
    }
    char* GetBuffer() {
        return buffer_;
    }

    ~BufferFile() {
        delete[] buffer_;
        buffer_ = NULL;
    }
};

void GetMeanFile(const std::string image_file, mx_float* image_data,
                const int channels, const cv::Size resize_size) {
    // Read all kinds of file into a BGR color 3 channels image
    cv::Mat im_ori = cv::imread(image_file, 1);

    if (im_ori.empty()) {
        std::cerr << "Can't open the image. Please check " << image_file << ". \n";
        assert(false);
    }

    cv::Mat im;

    resize(im_ori, im, resize_size);

    // Better to be read from a mean.nb file
    float mean = 117.0;

    int size = im.rows * im.cols * 3;

    mx_float* ptr_image_r = image_data;
    mx_float* ptr_image_g = image_data + size / 3;
    mx_float* ptr_image_b = image_data + size / 3 * 2;

    for (int i = 0; i < im.rows; i++) {
        uchar* data = im.ptr<uchar>(i);

        for (int j = 0; j < im.cols; j++) {
            mx_float b = static_cast<mx_float>(*data++) - mean;
            mx_float g = static_cast<mx_float>(*data++) - mean;
            mx_float r = static_cast<mx_float>(*data++) - mean;

            *ptr_image_r++ = r;
            *ptr_image_g++ = g;
            *ptr_image_b++ = b;
        }
    }
}

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(const char *filename) {
    std::ifstream fi(filename);

    if ( !fi.is_open() ) {
        std::cerr << "Error opening file " << filename << std::endl;
        assert(false);
    }

    std::vector<std::string> output;

    std::string synset, lemma;
    while ( fi >> synset ) {
        getline(fi, lemma);
        output.push_back(lemma);
    }

    fi.close();

    return output;
}

int PrintOutputResult(const std::vector<float>& data, const std::vector<std::string>& synset) {
    if (data.size() != synset.size()) {
        std::cerr << "Result data and synset size does not match!" << std::endl;
    }

    float best_accuracy = 0.0;
    int best_idx = 0;

    for ( int i = 0; i < static_cast<int>(data.size()); i++ ) {
//        printf("Accuracy[%d] = %.8f\n", i, data[i]);

        if ( data[i] > best_accuracy ) {
            best_accuracy = data[i];
            best_idx = i;
        }
    }
    std::cout << "Best result: " << synset[best_idx]
              << " id: " << best_idx
              << " accuracy: " << best_accuracy << std::endl;

//    printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
//            synset[best_idx].c_str(), best_idx, best_accuracy);

    return best_idx;
}

#endif // PREDICT_H
