#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include "PoseDetect.h"
#include <fstream>

#include <opencv2/core/utils/logger.hpp>
void ConfigureLogging() {
#include <opencv2/core/utils/logger.hpp>
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    //cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);//只输出错误日志
}

bool check_absolute_path(std::string relative_path) {
    char full_path[MAX_PATH];
    std::string absolute_path;
    if (_fullpath(full_path, relative_path.c_str(), MAX_PATH) != NULL) {
        absolute_path = std::string(full_path);
        std::cout << "Absolute path: " << absolute_path << std::endl;

        struct stat buffer;
        if (!(stat(absolute_path.c_str(), &buffer) == 0)) {
            return false;
        }
    }
    return true;
}


int main() {
    // 使denbug下opencv不输出log
    ConfigureLogging();

	std::string path = "./models/human-pose-estimation-3d.onnx";
    bool file_exists  = check_absolute_path(path);
    if (!file_exists) {
        std::cout << "File does not exist or error occurred." << std::endl;
        return 0;
    }

    PoseDetect pose;
    pose.init(path, 1, 0, 0);

    std::string img_path = "./data/R.jpg";
    cv::Mat image = cv::imread(img_path);
    while (true) {
        pose.getInference(image);
    }

}