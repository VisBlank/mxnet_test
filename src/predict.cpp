#include "predict.h"

//#define Folder_file 1
#define Txt_file 1
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "No test image here." << std::endl
        << "Usage: ./image-classification-predict apple.jpg" << std::endl;
        return 0;
    }

    std::string test_file;
//    test_file = std::string(argv[1]);

    // Path for your model, you have to modify it
    BufferFile json_data("./210classes/class_210-symbol.json");
    BufferFile param_data("./210classes/class_210_stage5-0007.params");
    std::vector<std::string> synset = LoadSynset("./210classes/label.txt");
    std::ifstream txtFile("./210classes/Test.txt"); //test file

    // Parameters
    int dev_type = 1;  // 1: cpu, 2: gpu
    int dev_id = 0;  // arbitrary.
    mx_uint num_input_nodes = 1;  // 1 for feedforward
    const char* input_key[1] = {"data"};
    const char** input_keys = input_key;
    string test_file_path = "./new";
    int classes = 10;   //total classes
    float count = 0;  	//for accuracy
    float TotalNumberOfTestImages = 0;
    struct timeval tpstart,tpend;
    float timeuse;
    gettimeofday(&tpstart,NULL);

    // Image size and channels
    int width = 28;
    int height = 28;
    int channels = 3;

    const mx_uint input_shape_indptr[2] = { 0, 4 };
    // ( trained_width, trained_height, channel, num)
    const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(width),
                                        static_cast<mx_uint>(height) };
    PredictorHandle out = 0;  // alias for void *

    // Just a big enough memory 1000x1000x3
    int image_size = width * height * channels;
    std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

#ifdef Txt_file
    //read file form TXT
    string strtmp;
    vector<string> vect,vect_class;
    while(getline(txtFile, strtmp, '\n')){
        vect.push_back(strtmp.substr(0, strtmp.find(' ')));
        vect_class.push_back(strtmp.substr(strtmp.find(' '),strtmp.size()));
    }
    for(int j = 0; j < vect.size(); j++){
        string test_file = vect[j];
        string Cuclass = vect_class[j];
        TotalNumberOfTestImages = vect.size();
    //read file form TXT
#endif

        //-- Create Predictor
        MXPredCreate((const char*)json_data.GetBuffer(),
                     (const char*)param_data.GetBuffer(),
                     static_cast<size_t>(param_data.GetLength()),
                     dev_type,
                     dev_id,
                     num_input_nodes,
                     input_keys,
                     input_shape_indptr,
                     input_shape_data,
                     &out);

        std::cout << "------ Prediction for " << test_file << " ------" << std::endl;
        //-- Read Mean Data
        GetMeanFile(test_file, image_data.data(), channels, cv::Size(width, height));

        //-- Set Input Image
        MXPredSetInput(out, "data", image_data.data(), image_size);

        //-- Do Predict Forward
        MXPredForward(out);

        mx_uint output_index = 0;

        mx_uint *shape = 0;
        mx_uint shape_len;

        //-- Get Output Result
        MXPredGetOutputShape(out, output_index, &shape, &shape_len);

        size_t size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

        std::vector<float> data(size);

        MXPredGetOutput(out, output_index, &(data[0]), size);

        // Release Predictor
        MXPredFree(out);

        //-- Print Output Data
        int predicted_class = PrintOutputResult(data, synset);

#ifdef Txt_file
        int current_class;
        stringstream convert1(Cuclass);
        convert1 >> current_class;
        cout << count << "\t" << j+1 << endl;
#endif

        if(predicted_class == current_class)
            count++;

      }
    cout << "count : " << count << "\t" "Numbers: " << TotalNumberOfTestImages << endl;
#ifdef Txt_file
    txtFile.close();
#endif
    gettimeofday(&tpend,NULL);
    timeuse = 1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;
    timeuse /= 1000000;
    std::cout << "Accuracy: " << count << "/" << TotalNumberOfTestImages
              << " = " << count/TotalNumberOfTestImages
              << "\nTime used: " << timeuse << " seconds." << std::endl;
    return 0;
}



//int main(int argc, char* argv[]) {
//    if (argc < 2) {
//        std::cout << "No test image here." << std::endl
//        << "Usage: ./image-classification-predict apple.jpg" << std::endl;
//        return 0;
//    }

//    std::string test_file;
//    test_file = std::string(argv[1]);

//    // Path for your model, you have to modify it
//    BufferFile json_data("./210classes/class_210_stage2-symbol.json");
//    BufferFile param_data("./210classes/class_210_stage2-0005.params");
//    std::vector<std::string> synset = LoadSynset("./210classes/label.txt");
//    std::ifstream txtFile("./210classes/Test.txt");

//    // Parameters
//    int dev_type = 1;  // 1: cpu, 2: gpu
//    int dev_id = 0;  // arbitrary.
//    mx_uint num_input_nodes = 1;  // 1 for feedforward
//    const char* input_key[1] = {"data"};
//    const char** input_keys = input_key;

//    // Image size and channels
//    int width = 28;
//    int height = 28;
//    int channels = 3;

//    const mx_uint input_shape_indptr[2] = { 0, 4 };
//    // ( trained_width, trained_height, channel, num)
//    const mx_uint input_shape_data[4] = { 1,
//                                        static_cast<mx_uint>(channels),
//                                        static_cast<mx_uint>(width),
//                                        static_cast<mx_uint>(height) };
//    PredictorHandle out = 0;  // alias for void *

//    //-- Create Predictor
//    MXPredCreate((const char*)json_data.GetBuffer(),
//                 (const char*)param_data.GetBuffer(),
//                 static_cast<size_t>(param_data.GetLength()),
//                 dev_type,
//                 dev_id,
//                 num_input_nodes,
//                 input_keys,
//                 input_shape_indptr,
//                 input_shape_data,
//                 &out);

//    // Just a big enough memory 1000x1000x3
//    int image_size = width * height * channels;
//    std::vector<mx_float> image_data = std::vector<mx_float>(image_size);

//    //-- Read Mean Data
//    GetMeanFile(test_file, image_data.data(), channels, cv::Size(width, height));

//    //-- Set Input Image
//    MXPredSetInput(out, "data", image_data.data(), image_size);

//    //-- Do Predict Forward
//    MXPredForward(out);

//    mx_uint output_index = 0;

//    mx_uint *shape = 0;
//    mx_uint shape_len;

//    //-- Get Output Result
//    MXPredGetOutputShape(out, output_index, &shape, &shape_len);

//    size_t size = 1;
//    for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

//    std::vector<float> data(size);

//    MXPredGetOutput(out, output_index, &(data[0]), size);

//    // Release Predictor
//    MXPredFree(out);

//    //-- Print Output Data
//    PrintOutputResult(data, synset);

//    return 0;
//}
