#ifndef ONNXENGINE_H
#define ONNXENGINE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// onnxruntime������
class OnnxEngine
{
public:
	OnnxEngine();
	virtual ~OnnxEngine();
    //OnnxEngine(const OnnxEngine& other);
    /*
    ��ʼ�������ܺ�����
    const std::string& model_path ��ģ��·��
    int is_cuda �� �Ƿ�ʹ��gpu 0:cpu��1:cuda�� Ĭ��cpu
    int device_id �� cuda�豸id ������cuda����Ĭ��ȡ0
    int threads �� onnx�ڲ�����ʱ�̲߳���, Ӱ�������ٶȣ���Ҫ���������豸�������ã�Ĭ��ȡ0������������ֵ
    */
	int init(const std::string& model_path, int is_cuda=0, int device = 0, int threads=0);

    // ��������Ԥ������ͼ�񣬽������ _out_data_ptr
    void inference(const cv::Mat &input);

    // rtmdet��rtmpose��Ԥ����ͺ���ͬ��������������
    virtual void preProcess(cv::Mat& input_mat) = 0;
    virtual void postProcess() = 0;

public:

    std::map<std::string, std::vector<int64_t>> _inputShapes;  // ��������shape
    std::map<std::string, std::vector<int64_t>> _outputShapes;  // �������shape
    size_t _inputNodesNum = 1;        // ģ������ڵ���
    size_t _outputNodesNum = 1;       // ģ������ڵ���
    std::vector<std::string> _inputNames;   // ģ��������
    std::vector<std::string> _outputNames;  // ģ�������

    std::map<std::string, float*> _outDataPtr;  // �洢ģ���������ָ��

private:
    // onnxruntime������������
	std::unique_ptr<Ort::Env> _ortEnv;
	std::unique_ptr<Ort::Session> _ortSession;
    //Ort::MemoryInfo _ortMemoryInfo;
    //Ort::SessionOptions _ortSessionOptions = Ort::SessionOptions();

};

#endif  // ONNXENGINE_H

