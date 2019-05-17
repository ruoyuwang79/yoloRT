#ifndef _CONFIGS_H_
#define _CONFIGS_H_

    // RGB
    #define INPUT_CHANNEL    3

    // network configuration and model file, relative directory of build
    #define ENGINE_DIR       "./models/"
    #define INPUT_PROTOTXT   "./models/yolov3_tiny.prototxt"
    #define INPUT_CAFFEMODEL "./models/yolov3_tiny_train_bn0_iter_340000.caffemodel"

    // input image, relative directory of build
    #define INPUT_IMAGES     "/home/nvidia/images/"
    #define EVAL_NAME        "/home/nvidia/images/result/time/alltime.txt"
    #define OUTPUTS_DIR      "/home/nvidia/images/result/xml/DeepZS/"
    
    // set mode here, support FLOAT32, FLOAT16, INT8
    #define MODE             RUN_MODE::FLOAT16
    #define MODE_NAME        "fp16"

    // network output configuration
    #define OUTPUTS          "tiny-yolo-det"
    #define INPUT_WIDTH      640
    #define INPUT_HEIGHT     360
    
    // control whether load or not
    #define LOAD_FROM_ENGINE 1

#endif
