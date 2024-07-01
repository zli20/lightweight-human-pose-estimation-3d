#include "PoseDetect.h"
#include "extract_poses.hpp"
#include <fstream>
#include <cmath>

PoseDetect::PoseDetect()
{

}

PoseDetect::~PoseDetect()
{

}

void savePose3dToTxt(std::vector<std::vector<float>>& poses, const std::string& filename) {
	// 打开文件，续写
	std::ofstream outfile(filename, std::ios::app);
	if (!outfile.is_open()) {
		std::cerr << "Unable to open file " << filename << std::endl;
		return;
	}
	for (size_t i = 0; i < poses[0].size() / 4; i++) {
		outfile << poses[0][i * 4 + 0] << " " << poses[0][i * 4 + 1] << " " << poses[0][i * 4 + 2]<< std::endl;
		
		}
	outfile.close();
	std::cout << "Pose3d data successfully saved to " << filename << std::endl;
}

static std::vector<cv::Mat> wrap_feature_maps(float* heatmapPtr, int num_channels, int height, int width) {
	std::vector<cv::Mat> feature_maps(num_channels);

	for (int c_id = 0; c_id < num_channels; c_id++) {
		feature_maps[c_id] = cv::Mat(height, width, CV_32FC1, heatmapPtr + c_id * height * width);
	}

	return feature_maps;
}

void saveDataToTxt(const float* dataPtr, int dataSize, const std::string& filename) {
	std::ofstream outfile(filename);
	if (outfile.is_open()) {
		for (int i = 0; i < dataSize; i++) {
			outfile << dataPtr[i] << ",";
			if (i != 0 && i % (32 * 56) == 0) {
				outfile << "\n";
			}
		}
		outfile.close();
		std::cout << "Data successfully saved to " << filename << std::endl;
	}
	else {
		std::cerr << "Unable to open file " << filename << std::endl;
	}
}

float computeTrace(const std::vector<std::vector<float>>& matrix) {
	float trace = 0.0;
	for (size_t i = 0; i < matrix.size(); ++i) {
		trace += matrix[i][i];
	}
	return trace;
}

void rotate_poses(std::vector<float>& poses_3d, cv::Mat R, cv::Mat t) {
	cv::Mat R_inv;
	cv::invert(R, R_inv, cv::DECOMP_SVD); // 计算 R 的逆矩阵

	std::vector<float> poses_3d_rotate;
	int pose_size = 4; // 假设每个姿态有4个元素

	for (size_t pose_id = 0; pose_id < poses_3d.size() / pose_size; ++pose_id) {
		int start_idx = pose_id * pose_size;

		cv::Mat pose_3d_part = cv::Mat::zeros(3, 1, CV_32F); // 创建一个 3x1 的零矩阵

		t.convertTo(t, CV_32F);
		R_inv.convertTo(R_inv, CV_32F);

		for (int i = 0; i < 3; ++i) {
			pose_3d_part.at<float>(i, 0) = poses_3d[start_idx + i]; // 设置姿态的前三个元素
		}

		cv::Mat trans_part = pose_3d_part - t; // 执行 R * pose_3d_part
		// 执行 R_inv * (pose_3d_part - t)
		cv::Mat transformed_part = R_inv * trans_part;

		for (int i = 0; i < 3; ++i) {
			poses_3d[start_idx + i] = transformed_part.at<float>(i, 0);
		}
	}
}

void PoseDetect::getInference(const cv::Mat& input_mat)
{

	cv::Mat input_mat_copy;
	input_mat.copyTo(input_mat_copy);

	this->_fx = 0.8 * input_mat_copy.cols;

	auto start_time = cv::getTickCount();
	// 预处理
	preProcess(input_mat_copy);

	auto pre_time = cv::getTickCount();

	// 推理
	inference(input_mat_copy);

	auto inference_time = cv::getTickCount();

	// 后处理
	postProcess();

	auto end_time = cv::getTickCount();

	// 打印时间
	double pre_cost = (pre_time - start_time) / cv::getTickFrequency() * 1000;
	double inference_cost = (inference_time - pre_time) / cv::getTickFrequency() * 1000;
	double post_cost = (end_time - inference_time) / cv::getTickFrequency() * 1000;
	std::cout << "Pose3d pre-process time cost: " << pre_cost << " ms" << std::endl;
	std::cout << "Pose3d inference time cost: " << inference_cost << " ms" << std::endl;
	std::cout << "Pose3d post-process time cost: " << post_cost << " ms" << std::endl;
	double time_cost = (end_time - start_time) / cv::getTickFrequency() * 1000;
	std::cout << "Pose3d all time cost: " << time_cost << " ms" << std::endl;

	cv::Mat input_mat_draw;
	input_mat.copyTo(input_mat_draw);
	drawPose2d(input_mat_draw, this->_pose2d);
	imshow("Pose3d", input_mat_draw);
	cv::waitKey(1);

	// savePose3dToTxt(this->_pose3d, "pose3d.txt");
}

void PoseDetect::preProcess(cv::Mat& input_mat)
{
	// 图像均值化、标准化
	cv::Scalar standValues(255.0, 255.0, 255.0);
	//cv::Scalar meanValues(123.675, 116.28, 103.53);
	cv::Scalar meanValues(128.0, 128.0, 128.0);

#if 1
	cv::Mat det_im = cv::Mat::zeros(_imgSize, CV_32FC3);
	cv::Mat resized_img;
	int new_height, new_width;
	float im_ratio = float(input_mat.rows) / float(input_mat.cols);
	if (im_ratio > 1) {
		new_height = _imgSize.height;
		new_width = int(new_height / im_ratio);
	}
	else {
		new_width = _imgSize.width;
		new_height = int(new_width * im_ratio);
	}
	_detScale = (float)(new_height) / (float)(input_mat.rows);
	cv::resize(input_mat, resized_img, cv::Size(new_width, new_height));
	resized_img.copyTo(det_im(cv::Rect(0, 0, new_width, new_height)));
	input_mat = det_im;

#else
	cv::resize(input_mat, input_mat, _imgSize);
#endif

#if 0
	// 计算输入比例
	float input_scale = 256.0 / input_mat.rows;
	// 对图像进行缩放
	cv::Mat scaled_img;
	cv::resize(input_mat, scaled_img, cv::Size(), input_scale, input_scale);
	
	//// 裁剪图像宽度使其为8的倍数
	//int new_width = scaled_img.cols - (scaled_img.cols % 8);
	//cv::imshow("scaled_img", scaled_img);
	
	//cv::waitKey(0);
	// 计算需要填充的像素数
	int original_width = scaled_img.cols;
	int new_width = original_width + (8 - (original_width % 8)) % 8;
	int pad = new_width - original_width;
	// 使用 pad 进行填充
	cv::Mat padded_img;
	cv::copyMakeBorder(scaled_img, padded_img, 0, 0, 0, pad, cv::BORDER_CONSTANT, cv::Scalar(0));
	//cv::imshow("padded_img", padded_img);
	//cv::waitKey(0);

	cv::Size newSize(448, 256);
	cv::Mat resizedImg;
	cv::resize(scaled_img, resizedImg, newSize, 0, 0, cv::INTER_LINEAR);
	//cv::imshow("resizedImg", resizedImg);
	//cv::waitKey(0);
#endif
	input_mat.convertTo(input_mat, CV_32F);
	input_mat -= meanValues;
	input_mat = input_mat / standValues;
	input_mat = cv::dnn::blobFromImage(input_mat, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
}

void PoseDetect::postProcess()
{
	auto* heatmapPtr = this->_outDataPtr["heatmaps"];
	auto* pafsPtr = this->_outDataPtr["pafs"];
	auto* featuresPtr = this->_outDataPtr["features"];

	auto shape0 = this->_outputShapes["heatmaps"];
	auto shape1 = this->_outputShapes["pafs"];
	auto shape2 = this->_outputShapes["features"];

	//saveDataToTxt(heatmapPtr, shape0[1] * shape0[2] * shape0[3], "heatmap.csv");
	//saveDataToTxt(pafsPtr, shape1[1] * shape1[2] * shape1[3], "paf.csv");
	//saveDataToTxt(featuresPtr, shape2[1] * shape2[2] * shape2[3], "features.csv");

	//std::cout << "heatmapPtr: " << heatmapPtr << std::endl;
	//std::cout << "pafsPtr: " << pafsPtr << std::endl;
	//std::cout << "featuresPtr: " << featuresPtr << std::endl;

	std::vector<cv::Mat> heatmaps;
	std::vector<cv::Mat> pafs;
	const int ratio = 4;

	heatmaps = wrap_feature_maps(heatmapPtr, shape0[1]-1, shape0[2], shape0[3]);
	pafs = wrap_feature_maps(pafsPtr, shape1[1], shape1[2], shape1[3]);

	std::vector<human_pose_estimation::HumanPose> poses = human_pose_estimation::extractPoses(
		heatmaps, pafs, ratio);

	size_t num_persons = poses.size();
	size_t num_keypoints = 0;
	if (num_persons > 0) {
		num_keypoints = poses[0].keypoints.size();
	}

	//float* out_data = new float[num_persons * ((num_keypoints+1) * 3 + 1)];
	//for (size_t person_id = 0; person_id < num_persons; person_id++)
	//{
	//	for (size_t kpt_id = 0; kpt_id < num_keypoints * 3; kpt_id += 3)
	//	{
	//		out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 0] = poses[person_id].keypoints[kpt_id / 3].x / ratio;
	//		out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 1] = poses[person_id].keypoints[kpt_id / 3].y /	ratio;
	//		out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 2] = poses[person_id].keypoints[kpt_id / 3].z;
	//	}
	//	out_data[person_id * (num_keypoints * 3 + 1) + num_keypoints * 3] = poses[person_id].score;
	//}

	// 2dpose获取
	std::vector<std::vector<float>> mul_pose2d;
	// 序号索引没有2，是骨盆点
	const float map_id_to_panoptic[18] = { 1, 0, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18 };
	for (size_t person_id = 0; person_id < num_persons; person_id++)
	{
		// 如果没有检测到neck点，则跳过
		if (poses[person_id].keypoints[1].z == -1) {
			continue;
		}
		// 2d psoe 19个点，实际检测到的点数为18个，加上了骨盆一个点
		std::vector<float> pose2d(19 * 3 + 1, -1);
		for (size_t kpt_id = 0; kpt_id < num_keypoints; kpt_id++)
		{
			int map_id = map_id_to_panoptic[kpt_id];
			pose2d[map_id * 3 + 0] = poses[person_id].keypoints[kpt_id].x / ratio;
			pose2d[map_id * 3 + 1] = poses[person_id].keypoints[kpt_id].y / ratio;
			pose2d[map_id * 3 + 2] = poses[person_id].keypoints[kpt_id].z;
		}
		pose2d[19 * 3] = poses[person_id].score;
		mul_pose2d.push_back(pose2d);
	}

	// 3dpose获取,需要用到特征图featuresPtr
	const float point_threshold = 0.1;
	const int AVG_PERSON_HEIGHT = 180;
	const std::vector<std::vector<int>> limbs = {
		{18, 17, 1},
		{16, 15, 1},
		{5, 4, 3},
		{8, 7, 6},
		{11, 10, 9},
		{14, 13, 12} };

	std::vector<std::vector<float>> mul_pose3d;
	for (auto pose2d : mul_pose2d)
	{
		if (pose2d[2] > point_threshold) {
			std::vector<float> pose3d(19 * 4 + 1, -1);

			// 先围绕neck点初始化3d坐标
			int neck_2d_x = pose2d[0];
			int neck_2d_y = pose2d[1];
			for (size_t kpt_id = 0; kpt_id < num_keypoints + 1; kpt_id++)
			{
				pose3d[kpt_id * 4 + 0] = featuresPtr[(kpt_id * 3 + 0) * 32 * 56 + neck_2d_y * 56 + neck_2d_x] * AVG_PERSON_HEIGHT;
				pose3d[kpt_id * 4 + 1] = featuresPtr[(kpt_id * 3 + 1) * 32 * 56 + neck_2d_y * 56 + neck_2d_x] * AVG_PERSON_HEIGHT;
				pose3d[kpt_id * 4 + 2] = featuresPtr[(kpt_id * 3 + 2) * 32 * 56 + neck_2d_y * 56 + neck_2d_x] * AVG_PERSON_HEIGHT;
				pose3d[kpt_id * 4 + 3] = pose2d[kpt_id * 3 + 2];
			}

			// 根据关节依赖重新计算3d坐标
			for (auto limb : limbs) {
				for (auto kpt_id_from : limb) {
					if (pose2d[kpt_id_from * 3 + 2] > point_threshold) {
						for (auto kpt_id_to : limb) {
							int x_2d = pose2d[kpt_id_from * 3 + 0];
							int y_2d = pose2d[kpt_id_from * 3 + 1];
							pose3d[kpt_id_to * 4 + 0] = featuresPtr[(kpt_id_to * 3 + 0) * 32 * 56 + y_2d * 56 + x_2d] * AVG_PERSON_HEIGHT;
							pose3d[kpt_id_to * 4 + 1] = featuresPtr[(kpt_id_to * 3 + 1) * 32 * 56 + y_2d * 56 + x_2d] * AVG_PERSON_HEIGHT;
							pose3d[kpt_id_to * 4 + 2] = featuresPtr[(kpt_id_to * 3 + 2) * 32 * 56 + y_2d * 56 + x_2d] * AVG_PERSON_HEIGHT;
						}
						break;
					}
				}
			}

			mul_pose3d.push_back(pose3d);
		}
	}

	// 2d关键点scale到原图尺寸
	std::vector<std::vector<float>> mul_pose2d_scaled;
	for (auto pose2d : mul_pose2d) {
		//cv::Mat image = cv::Mat::zeros(500, 500, CV_8UC3);
		//image.setTo(cv::Scalar(255, 255, 255));
		std::vector<float> pose2d_scaled(19 * 3 + 1, -1);
		//cv::Mat image = cv::imread("C:\\Users\\26522\\Desktop\\R.jpg");
		for (int i = 0; i < 19; i++) {

			pose2d_scaled[i * 3 + 0] = pose2d[i * 3 + 0] * 8 / this->_detScale;
			pose2d_scaled[i * 3 + 1] = pose2d[i * 3 + 1] * 8 / this->_detScale;
			pose2d_scaled[i * 3 + 2] = pose2d[i * 3 + 2];

			//if (pose2d_scaled[i * 3 + 0] >= 0 && pose2d_scaled[i * 3 + 1] >= 0 && pose2d_scaled[i * 3 + 0] < image.cols && pose2d_scaled[i * 3 + 1] < image.rows) {
			//	cv::circle(image, cv::Point(static_cast<int>(pose2d_scaled[i * 3 + 0]), static_cast<int>(pose2d_scaled[i * 3 + 1])), 3, cv::Scalar(0, 0, 255), -1);
			//}
		}
		pose2d_scaled[19 * 3] = pose2d[19 * 3];
		mul_pose2d_scaled.push_back(pose2d_scaled);

		//cv::imshow("Pose Points", image);
		//cv::waitKey(1);
	}

	// trans point
	std::vector<std::vector<float>> mul_pose3d_trans;
	for (size_t point_id = 0; point_id < mul_pose2d.size();point_id++) {
		float mean_2d[] = { 0, 0 };
		float mean_3d[] = { 0, 0 , 0 };
		int num_valid_points = 0;

		std::vector<float> pose3d_trans;

		std::vector<float> pose2d_valid;
		std::vector<float> pose3d_valid;

		for (size_t i = 0; i < 19; i++) {
			if (mul_pose2d[point_id][i * 3 + 2] !=-1) {

				float x_2d = mul_pose2d[point_id][i * 3 + 0] - shape2[3] / 2;
				float y_2d = mul_pose2d[point_id][i * 3 + 1] - shape2[2] / 2;

				mean_2d[0] += x_2d;
				mean_2d[1] += y_2d;

				mean_3d[0] += mul_pose3d[point_id][i * 4 + 0];
				mean_3d[1] += mul_pose3d[point_id][i * 4 + 1];
				mean_3d[2] += mul_pose3d[point_id][i * 4 + 2];

				pose2d_valid.push_back(x_2d);
				pose2d_valid.push_back(y_2d);

				pose3d_valid.push_back(mul_pose3d[point_id][i * 4 + 0]);
				pose3d_valid.push_back(mul_pose3d[point_id][i * 4 + 1]);
				pose3d_valid.push_back(mul_pose3d[point_id][i * 4 + 2]);
				num_valid_points++;
			}
		}
		mean_2d[0] /= num_valid_points;
		mean_2d[1] /= num_valid_points;
		mean_3d[0] /= num_valid_points;
		mean_3d[1] /= num_valid_points;
		mean_3d[2] /= num_valid_points;

		// Subtract mean from each 3D point
		std::vector<std::vector<float>> diff_3d(num_valid_points, std::vector<float>(3));
		for (size_t i = 0; i < num_valid_points; ++i) {
			diff_3d[i][0] = pose3d_valid[i * 3] - mean_3d[0];
			diff_3d[i][1] = pose3d_valid[i * 3 + 1] - mean_3d[1];
			diff_3d[i][2] = pose3d_valid[i * 3 + 2] - mean_3d[2];
		}

		// Compute the dot product of the transpose of diff_3d and diff_3d
		std::vector<std::vector<float>> dot_product_3d(3, std::vector<float>(3, 0.0));
		for (size_t i = 0; i < num_valid_points; ++i) {
			for (size_t j = 0; j < 3; ++j) {
				for (size_t k = 0; k < 3; ++k) {
					dot_product_3d[j][k] += diff_3d[i][j] * diff_3d[i][k];
				}
			}
		}

		// Compute the trace of the dot product matrix for 3D
		float numerator = computeTrace(dot_product_3d);

		// Compute the square root of the trace for 3D
		numerator = std::sqrt(numerator);

		// Subtract mean from each 2D point
		std::vector<std::vector<float>> diff_2d(num_valid_points, std::vector<float>(2));
		for (size_t i = 0; i < num_valid_points; ++i) {
			diff_2d[i][0] = pose2d_valid[i * 2] - mean_2d[0];
			diff_2d[i][1] = pose2d_valid[i * 2 + 1] - mean_2d[1];
		}

		// Compute the dot product of the transpose of diff_2d and diff_2d
		std::vector<std::vector<float>> dot_product_2d(2, std::vector<float>(2, 0.0));
		for (size_t i = 0; i < num_valid_points; ++i) {
			for (size_t j = 0; j < 2; ++j) {
				for (size_t k = 0; k < 2; ++k) {
					dot_product_2d[j][k] += diff_2d[i][j] * diff_2d[i][k];
				}
			}
		}

		// Compute the trace of the dot product matrix for 2D
		float denominator = computeTrace(dot_product_2d);

		// Compute the square root of the trace for 2D
		denominator = std::sqrt(denominator);

		std::vector<float> mean_2d_values = { mean_2d[0], mean_2d[1], this->_fx * this->_detScale / 8 };
		std::vector<float> mean_3d_values = { mean_3d[0], mean_3d[1], 0.0f };
		std::vector<float> translation(3);

		for (size_t i = 0; i < 3; ++i) {
			translation[i] = numerator / denominator * mean_2d_values[i] - mean_3d_values[i];
		}


		for (size_t i = 0; i < 19; i++) {
			pose3d_trans.push_back(mul_pose3d[point_id][i * 4 + 0] + translation[0]);
			pose3d_trans.push_back(mul_pose3d[point_id][i * 4 + 1] + translation[1]);
			pose3d_trans.push_back(mul_pose3d[point_id][i * 4 + 2] + translation[2]);
			pose3d_trans.push_back(mul_pose3d[point_id][i * 4 + 3]);
		}
		rotate_poses(pose3d_trans, this->_Rotation, this->_Translation);
		mul_pose3d_trans.push_back(pose3d_trans);
	}


	this->_pose3d = mul_pose3d_trans;
	this->_pose2d = mul_pose2d_scaled;


}

void PoseDetect::drawPose2d(cv::Mat& input_mat, const std::vector<std::vector<float>>& pose2d)
{
	for (auto pose : pose2d) {
		for (size_t i = 0; i < pose.size() / 3; i++) {
			cv::circle(input_mat, cv::Point(static_cast<int>(pose[i * 3 + 0]), static_cast<int>(pose[i * 3 + 1])), 3, cv::Scalar(0, 0, 255), -1);
		}
	}
}
