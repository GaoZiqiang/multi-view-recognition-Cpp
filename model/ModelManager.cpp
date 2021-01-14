//
// Created by gaoziqiang on 2021/1/14.
//
#include "../util/DataManager.h"//注意路径

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;
using namespace cv;

torch::jit::script::Module LoadResnet50(){
    cout << "func:LoadResnet50" << endl;

    //使用torch::jit::script加载resnet50
    torch::jit::script::Module model = torch::jit::load("../model/modified_resnet.pt");

    cout << "resnet50 loaded successfully" << endl;

    return model;
}

torch::Tensor FeatureExtracting(string path){
    cout << "func:FeatureExtracting" << endl;

    LoadResnet50();
    torch::jit::script::Module model = LoadResnet50();

    torch::Tensor img_tensors = ImgToTensor(path);


    torch::Tensor feature = model.forward({img_tensors}).toTensor();


    return feature;
}