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
    cout << output << endl;


    //下面进行度量学习
    cout << "------start test------" << endl;

    torch::Tensor query_feature = output;
    torch::Tensor gallery_feature = output2;
    //feature normlization　特征标准化
    //torch::data::transforms::Normalize(query_feature,2,dim = -1,keepdim = true);
    //对dim=0方向进行cat竖向拼接
//    query_feature = torch::cat(query_feature,0);
//    cout << "------cated query_feature------" << endl;
//    cout << query_feature << endl;





}