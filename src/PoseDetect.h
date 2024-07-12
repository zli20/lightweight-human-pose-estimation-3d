#ifndef POSE_3D_H
#define POSE_3D_H

#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "OnnxEngine.h"


const int NUM_POINTS = 18;  // 检测出的点数
const float POINTS_THRESHOLD = 0.1;
const int AVG_PERSON_HEIGHT = 180;  // 人身高
const std::vector<std::vector<int>> LIMBS = {  // 关节之间连接关系
	{18, 17, 1},
	{16, 15, 1},
	{5, 4, 3},
	{8, 7, 6},
	{11, 10, 9},
	{14, 13, 12} };

const float OPENPOSE_ID_TO_PANOPTIC[18] = { 1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18 };  // 模型输出openpsoe，转换为PANOPTIC
/*
------模型输出，对应openpose18点------
		   14   15
		 16   0   17
			  |
		 2----1----5
		/|         |\
	   3 |         | 6
	  /  |         |  \
	4    |         |   7
		 8--------11
		 |         |
		 9        12
		 |         |
		 10       13
----panoptic关键点排列，相对于openpose增加了骨盆点------
			15   16
		  17   1   18
			   |
		  9----0----3
		 /     |     \
	   10      |      4
	  /        |       \
	11         |        5
		  12---2----6
		  |         |
		  13        7
		  |         |
		  14        8
*/

class PoseDetect :public OnnxEngine
{
public:
	PoseDetect();
    ~PoseDetect();
        
	void getInference(const cv::Mat& input_mat);

	void preProcess(cv::Mat& input_mat) override;  // 图像预处理
	void postProcess() override;  // 后处理

	void drawPose2d(cv::Mat& input_mat, const std::vector<std::vector<float>>& pose2d);

	std::vector<std::vector<float>> _pose2d;  // 2D姿态
	std::vector<std::vector<float>> _pose3d;  // 3D姿态
private:
	static std::vector<cv::Mat> wrap_feature_maps(float* heatmapPtr, int num_channels, int height, int width);  // 输出heatmapP转化为mat
	void rotate_poses(std::vector<float>& poses_3d, cv::Mat R, cv::Mat t);  // 使用相机参数，变换输出pose
	float computeTrace(const std::vector<std::vector<float>>& matrix);  // 计算矩阵的迹，用于变换3d点

	//void savePose3dToTxt(std::vector<std::vector<float>>& poses, const std::string& filename);  // 点写入txt中，用于调试
	//void saveDataToTxt(const float* dataPtr, int dataSize, const std::string& filename);
private:  // 预处理和后处理参数
	float _detScale{ 1.0 };  // 图像resize的尺度，用于图像预处理
	//const cv::Size _imgSize{ 256, 448 };  // 模型输入大小		
	const cv::Size _imgSize{ 448, 256 };  // 模型输入大小
	float _fx{1.0}; // 焦距

	// 相机参数
	cv::Mat _Rotation = (cv::Mat_<double>(3, 3) <<
		0.1656794936, 0.0336560618, -0.9856051821,
		- 0.09224101321,0.9955650135, 0.01849052095,
		0.9818563545, 0.08784972047, 0.1680491765);

	cv::Mat _Translation = (cv::Mat_<double>(3, 1) << 17.76193366, 126.741365, 286.3860507);

};

#endif // POSE_3D_H

