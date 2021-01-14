//
// Created by gaoziqiang on 2021/1/14.
//
#include "../model/ModelManager.h"

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

torch::Tensor PreprocessFeature(string path){
    cout << "method:PreprocessFeature" << endl;

//    GetImgPaths(path).size[0];

    //提取特征
    torch::Tensor feature = FeatureExtracting(path);//view1的feature
    cout << "method:PreprocessFeature feature got!" << endl;
    cout << feature.size(0) << " " << feature.size(1) << endl;

    //特征标准化 归一化
    torch::Tensor normalized_feature = feature / norm(feature);//norm():矩阵范数
//
//    //cout << "normalized_feature" << '\n' << normalized_feature << endl;
//    cout << "normalized_feature's size" << normalized_feature.size(0) << " " << normalized_feature.size(1) << endl;
//
//    //统计特征矩阵的行数
    int m1 = normalized_feature.size(0);
//    //int m2 = normalized_feature1.size(1);
//
//    //矩阵中的元素求平方
    normalized_feature = torch::pow(normalized_feature,2);
//    //按行求和,dim=1,变为一个m1*1的向量
    torch::Tensor sum_feature = torch::sum(normalized_feature,1);
//
    //cout << "sum_feature" << sum_feature << endl;

    //这里两个m,n应为m=max(m1,m2),n=min(m1,m2)
    //sum_feature.expand({8,8});

    return sum_feature;

}

void EvalObjNum(string path1,string path2){
    cout << "method:EvalObjNum" << endl;

    int obj_num1 = PreprocessFeature(path1).size(0);
    int obj_num2 = PreprocessFeature(path2).size(0);
    cout << "num of pics" << endl;
    cout << obj_num1 << " " << obj_num2 << endl;


    torch::Tensor feature1 = PreprocessFeature(path1);
    torch::Tensor feature2 = PreprocessFeature(path2);

    //cout << "before expand feature1" << '\n' << feature1 << endl;
    cout << "before expand feature1" << '\n' << feature2 << endl;




    //这里两个m,n应为m=max(m1,m2),n=min(m1,m2)


    feature1 = feature1.expand({obj_num2,obj_num1}).transpose(1,0);//保持列数不变,列数为obj_num
    feature2 = feature2.expand({obj_num1,obj_num2}).transpose(1,0);
    cout << "expanded feature1" << '\n' << feature1 << endl;
    cout << "expanded feature2" << '\n' << feature2 << endl;


    torch::Tensor distmat = feature1 + feature2.transpose(1,0);


    if(obj_num1 < obj_num1)
        torch::Tensor distmat = feature2 + feature1.transpose(1,0);

    int maxnum = max(obj_num1,obj_num2);//最大目标数量


    //计算相似度
    int mm = distmat.size(0);
    int nn = distmat.size(1);

    //tensor矩阵转换为float矩阵
    cv::Mat float_distmat(distmat.size(0), distmat.size(1), CV_32FC1, distmat.data<float>());

    cout << "目标之间的相似度矩阵" << '\n' << float_distmat << endl;

    int min[8] = {1, 1, 1, 1, 1, 1, 1, 1};//因为单兵数量最多为8
    cout << min << endl;
    int num = 0;
    for (int i=0;i<mm;++i){
        float *data =  float_distmat.ptr<float>(i);
        for (int j=0;j<nn;++j){
            if(data[j] < min[i]){
                min[i] = data[j];
            }
        }
        if(min[i]<0.2)
            num += 1;//从某种意义上讲，num取决于循环次数，不科学
    }
    cout << "经多视角识别后的目标数量为:" << endl;
    cout << num << endl;


}