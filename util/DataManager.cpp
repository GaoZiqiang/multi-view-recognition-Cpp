#include <opencv2/opencv.hpp>
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
    vector<String> image_files;

    //把pattern_bmp路径下所有文件名存在image_files中
    glob(pattern_bmp, image_files);
    cout << "image_files size" << endl;
    cout << image_files.size() << endl;
    for (int i = 0;i < image_files.size();++i){
        cout << image_files[i] << endl;
        cv::Mat image = cv::imread(image_files[i]);//读取出来为Mat，而且为BGR格式
        //建议下面直接处理图像的矩阵Mat 128*64
        cout << image.cols << " " << image.rows << endl;
        cout << image.channels() << " " << image.cols << endl;

        //转换为tensor处理，每个tensor的shape为[1,3,128,64]
        torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
        img_tensor = img_tensor.permute({0, 3, 1, 2});//转换为[N,C,W,H]格式，[1,3,128,64]
        img_tensor = img_tensor.toType(torch::kFloat);
        img_tensor = img_tensor.div(255);

        cout << "img_tensor shape" << endl;
        //cout << img_tensor << endl;

        if(!image.data)
            std::cerr << "Problem loading image!!!" << std::endl;

//        cv::imshow("temp",image);
//        cv::waitKey(0);
    }

    //接下来要将所有的image组成imgs[8,3,128,64]
    //然后使用model进行特征提取
}