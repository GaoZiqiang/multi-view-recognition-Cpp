//
// Created by gaoziqiang on 2021/1/11.

//main.cpp函数直接调用哪一个就在.h中声明哪一个
#pragma once// 防止多次定义
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;

//extern vector<String> GetImgPaths(string path);
extern torch::Tensor ImgToTensor(string path);
