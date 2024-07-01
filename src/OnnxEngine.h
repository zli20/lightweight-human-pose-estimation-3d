#ifndef ONNXENGINE_H
#define ONNXENGINE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// onnxruntime推理类
class OnnxEngine
{
public:
	OnnxEngine();
	virtual ~OnnxEngine();
    //OnnxEngine(const OnnxEngine& other);
    /*
    初始化推理框架和引擎
    const std::string& model_path ：模型路径
    int is_cuda ： 是否使用gpu 0:cpu、1:cuda， 默认cpu
    int device_id ： cuda设备id ，用于cuda推理，默认取0
    int threads ： onnx内部运算时线程参数, 影响推理速度，需要根据推理设备进行设置，默认取0，并不是最优值
    */
	int init(const std::string& model_path, int is_cuda=0, int device = 0, int threads=0);

    // 推理，输入预处理后的图像，结果传入 _out_data_ptr
    void inference(const cv::Mat &input);

    // rtmdet和rtmpose的预处理和后处理不同，在子类中重载
    virtual void preProcess(cv::Mat& input_mat) = 0;
    virtual void postProcess() = 0;

public:

    std::map<std::string, std::vector<int64_t>> _inputShapes;  // 输入张量shape
    std::map<std::string, std::vector<int64_t>> _outputShapes;  // 输出张量shape
    size_t _inputNodesNum = 1;        // 模型输入节点数
    size_t _outputNodesNum = 1;       // 模型输出节点数
    std::vector<std::string> _inputNames;   // 模型输入名
    std::vector<std::string> _outputNames;  // 模型输出名

    std::map<std::string, float*> _outDataPtr;  // 存储模型输出数据指针

private:
    // onnxruntime环境构建参数
	std::unique_ptr<Ort::Env> _ortEnv;
	std::unique_ptr<Ort::Session> _ortSession;
    //Ort::MemoryInfo _ortMemoryInfo;
    //Ort::SessionOptions _ortSessionOptions = Ort::SessionOptions();

};

#endif  // ONNXENGINE_H

