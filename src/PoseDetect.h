#ifndef POSE_3D_H
#define POSE_3D_H

#include <iostream>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "OnnxEngine.h"


const int NUM_POINTS = 18;  // �����ĵ���
const float POINTS_THRESHOLD = 0.1;
const int AVG_PERSON_HEIGHT = 180;  // �����
const std::vector<std::vector<int>> LIMBS = {  // �ؽ�֮�����ӹ�ϵ
	{18, 17, 1},
	{16, 15, 1},
	{5, 4, 3},
	{8, 7, 6},
	{11, 10, 9},
	{14, 13, 12} };

const float OPENPOSE_ID_TO_PANOPTIC[18] = { 1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18 };  // ģ�����openpsoe��ת��ΪPANOPTIC
/*
------ģ���������Ӧopenpose18��------
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
----panoptic�ؼ������У������openpose�����˹����------
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

	void preProcess(cv::Mat& input_mat) override;  // ͼ��Ԥ����
	void postProcess() override;  // ����

	void drawPose2d(cv::Mat& input_mat, const std::vector<std::vector<float>>& pose2d);

	std::vector<std::vector<float>> _pose2d;  // 2D��̬
	std::vector<std::vector<float>> _pose3d;  // 3D��̬
private:
	static std::vector<cv::Mat> wrap_feature_maps(float* heatmapPtr, int num_channels, int height, int width);  // ���heatmapPת��Ϊmat
	void rotate_poses(std::vector<float>& poses_3d, cv::Mat R, cv::Mat t);  // ʹ������������任���pose
	float computeTrace(const std::vector<std::vector<float>>& matrix);  // �������ļ������ڱ任3d��

	//void savePose3dToTxt(std::vector<std::vector<float>>& poses, const std::string& filename);  // ��д��txt�У����ڵ���
	//void saveDataToTxt(const float* dataPtr, int dataSize, const std::string& filename);
private:  // Ԥ����ͺ������
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

