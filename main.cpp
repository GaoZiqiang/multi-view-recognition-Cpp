#include "util/DataManager.h"// 包含功能模块头文件
#include "model/ModelManager.h"
#include "util/MetricLearning.h"

#include <iostream>
#include <string>

using namespace std;

int main()
{
    string path1 = "../data/view1/";//注意路径是相对与DataManager.cpp所在的util目录而言的相对路径
    string path2 = "../data/view2/";

    EvalObjNum(path1,path2);
}