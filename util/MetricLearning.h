//
// Created by gaoziqiang on 2021/1/14.
//
#pragma once

#include <string>

using namespace std;

extern torch::Tensor PreprocessFeature(string path);
extern void EvalObjNum(string path1,string path2);
