//
// Created by gaoziqiang on 2021/1/14.
//
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

using namespace std;

#pragma once//防止多次定义


extern torch::jit::script::Module LoadResnet50();
extern torch::Tensor FeatureExtracting(string path);