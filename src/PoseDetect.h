#ifndef POSE_3D_H
#define POSE_3D_H

#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "OnnxEngine.h"


class PoseDetect :public OnnxEngine
{
public:
	PoseDetect();
    ~PoseDetect();
        
	void getInference(const cv::Mat& input_mat);

	void preProcess(cv::Mat& input_mat) override;  // ͼ��Ԥ����
	void postProcess() override;  // ����

	void drawPose2d(cv::Mat& input_mat, const std::vector<std::vector<float>>& pose2d);

	std::vector<std::vector<float>> _pose2d;  // 2D��̬
	std::vector<std::vector<float>> _pose3d;  // 3D��̬

private:
	float _detScale{ 1.0 };  // ͼ��resize�ĳ߶ȣ�����ͼ��Ԥ����
	//const cv::Size _imgSize{ 256, 448 };  // ģ�������С		
	const cv::Size _imgSize{ 448, 256 };  // ģ�������С
	float _fx{1.0}; // ����

	// �������
	cv::Mat _Rotation = (cv::Mat_<double>(3, 3) <<
		0.1656794936, 0.0336560618, -0.9856051821,
		- 0.09224101321,0.9955650135, 0.01849052095,
		0.9818563545, 0.08784972047, 0.1680491765);

	cv::Mat _Translation = (cv::Mat_<double>(3, 1) << 17.76193366, 126.741365, 286.3860507);

};

#endif // POSE_3D_H

