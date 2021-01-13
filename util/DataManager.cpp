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

class DataManager{
    //属性声明
public:
    string name;

    //函数声明
    void manager(void);
};

//成员方法实现
void DataManager::manager(void){
    //绝对路径下类型为.bmp的照片
    string pattern_bmp = "/home/gaoziqiang/project/multi-perspective-detection/PersonReID/data/market1501/view1/*.jpg";

    //创建一个String类型 名字为image_files的vector
    vector<String> image_paths;

    //把pattern_bmp路径下所有文件名存在image_files中
    glob(pattern_bmp, image_paths);
    cout << "image_paths size" << endl;
    cout << image_paths.size() << endl;

    //预定义img_tensors来暂存拼接好的img_tensor
    //先把第一张图片的img_tensor0提取出来，赋值给img_tensors，作为初始值
    cv::Mat image0 = cv::imread(image_paths[0]);
    torch::Tensor img_tensor0 = torch::from_blob(image0.data, {1, image0.rows, image0.cols, 3}, torch::kByte);
    img_tensor0 = img_tensor0.permute({0, 3, 1, 2});
    //img_tensors赋初值为第一张图像img_tensor0
    torch::Tensor img_tensors = img_tensor0;
//    torch::Tensor img_tensors = torch::rand({1,3,128,64});
    //剩下的7张图片
    for (int i = 1;i < image_paths.size();++i){
        cout << image_paths[i] << endl;
        cv::Mat image = cv::imread(image_paths[i]);//读取出来为Mat，而且为BGR格式

        if(!image.data)
            std::cerr << "Problem loading image!!!" << std::endl;

        //转换为tensor处理，每个tensor的shape为[1,3,128,64]
        torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({0, 3, 1, 2});//转换为[N,C,W,H]格式，[1,3,128,64]

        //做img_tensor拼接
        img_tensors = torch::cat({img_tensor,img_tensors},0);
    }

    img_tensors = img_tensors.toType(torch::kFloat);
    img_tensors = img_tensors.div(255);

    //然后使用model进行特征提取
    torch::jit::script::Module module = torch::jit::load("../model/modified_resnet.pt");
    torch::Tensor output = module.forward({img_tensors}).toTensor();
    torch::Tensor output2 = module.forward({img_tensors}).toTensor();

    cout << "------ output ------" << endl;
    cout << "output's type" << endl;
    cout << typeid(output).name() << endl;//结果为N2at6TensorE
//    cout << "print query_feature" << endl;
//    cout << output << endl;


    //下面进行度量学习
    cout << "------start test------" << endl;

    torch::Tensor query_feature = output;
    torch::Tensor gallery_feature = output2;
    //feature normlization　特征标准化
    //torch::data::transforms::Normalize(query_feature,2,dim = -1);
    //norm(query_feature,"f");
    cout << "print norm of query_feature" << endl;
    //先用这个默认范数吧
    cout << norm(query_feature) << endl;
    torch::Tensor normalized_query_feature = query_feature / norm(query_feature);
    cout << "print normalized_query_feature" << endl;
    //cout << normalized_query_feature << endl;
    cout << "print normalized_query_feature's 行数" << endl;
    int m = normalized_query_feature.size(0);
    cout << normalized_query_feature.size(0) << endl;
    cout << "print normalized_query_feature's 列数" << endl;
//    int n = normalized_gallery_feature.size(1);
//    cout << normalized_gallery_feature.size(1) << endl;

    torch::pow(normalized_query_feature,2);
    cout << "print torch::pow()" << endl;
    //平方后，按行求和
    normalized_query_feature = torch::pow(normalized_query_feature,2);

    //cout << normalized_query_feature << endl;
    cout << "sumed feature" << endl;
    //cout << torch::sum(normalized_query_feature,1) << endl;
    torch::Tensor sum_feature = torch::sum(normalized_query_feature,1);
    cout << sum_feature << endl;
    //sum_feature::expand((m,m));//注意：此处应为query_feature和行数和gallery_feature的行数



    //expand
    cout << "print expanded sum_feature" << endl;
    cout << sum_feature.expand({8,8}) << endl;
    cout << "转置" << endl;
    cout << sum_feature.expand({8,8}).transpose(1,0) << endl;
    //求距离
    torch::Tensor distmat = sum_feature.expand({8,8}) + sum_feature.expand({8,8}).transpose(1,0);
    cout << "距离" << endl;
    cout << distmat << endl;

    //计算相似度
    int mm = distmat.size(0);
    int nn = distmat.size(1);
    cout << "mm nn" << endl;
    cout << mm << nn <<endl;

    //cv::Mat float_distmat = cv::Mat(distmat.data<float>());//float* 是个指针变量
    cv::Mat float_distmat(distmat.size(0), distmat.size(1), CV_32FC1, distmat.data<float>());
    cout << "转换后的distmat" << endl;
    cout << float_distmat << endl;

    int min[8] = {1, 1, 1, 1, 1, 1, 1, 1};
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
            num += 1;
    }
    cout << "目标数量为:" << endl;
    cout << num << endl;
}