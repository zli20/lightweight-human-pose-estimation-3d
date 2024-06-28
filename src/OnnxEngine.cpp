#include "OnnxEngine.h"
#include <numeric>
#include <filesystem>
#include <onnxruntime_c_api.h>

OnnxEngine::OnnxEngine() : _ortEnv(nullptr), _ortSession(nullptr)
{
}

OnnxEngine::~OnnxEngine() = default;

//// 拷贝构造函数的实现
//OnnxEngine::OnnxEngine(const OnnxEngine& other)
//    : _inputShapes(other._inputShapes),
//    _outputShapes(other._outputShapes),
//    _inputNodesNum(other._inputNodesNum),
//    _outputNodesNum(other._outputNodesNum),
//    _inputNames(other._inputNames),
//    _outputNames(other._outputNames),
//    _outDataPtr(other._outDataPtr)
//{
///*    if (other._ortEnv) {
//        _ortEnv = std::make_unique<Ort::Env>();
//    }
//    if (other._ortSession) {
//        _ortSession = std::make_unique<Ort::Session>(other._ortSessionp*/
//}

int OnnxEngine::init(const std::string & model_path, int is_cuda, int device_id, int threads)
{

    struct stat buffer;
    // 确认模型路径是否存在
    if (!(stat(model_path.c_str(), &buffer) == 0)) {
        std::cerr << "Model path does not exist: " << model_path << std::endl;
        return EXIT_FAILURE;
    }

    // onnxruntime推理引擎初始化
    OrtThreadingOptions* thOpts{};
    Ort::SessionOptions _ortSessionOptions = Ort::SessionOptions();

    OrtStatusPtr status;
    status = Ort::GetApi().CreateThreadingOptions(&thOpts);
    status = Ort::GetApi().SetGlobalIntraOpNumThreads(thOpts, threads);  // 设置全局的 Intra-Op线程数,Intra-Op 并行用于在单个操作（如矩阵乘法）内并行化计算
    status = Ort::GetApi().SetGlobalInterOpNumThreads(thOpts, threads);  // 设置全局的 Inter-Op线程数,Inter-Op 并行用于并行执行不同的操作（如卷积层和全连接层。

    //_ortSessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
    _ortSessionOptions.SetExecutionMode(ORT_PARALLEL);
    _ortSessionOptions.DisablePerSessionThreads();
    //_ortSessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    _ortSessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    _ortSessionOptions.SetIntraOpNumThreads(threads);
    _ortSessionOptions.SetInterOpNumThreads(threads);

    //ORT_LOGGING_LEVEL_ERROR
    // 创建推理环境
    if (this->_ortEnv == nullptr)
        this->_ortEnv = std::make_unique<Ort::Env>(thOpts, ORT_LOGGING_LEVEL_ERROR, model_path.c_str());
        //this->_ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, model_path.c_str());
    if (!this->_ortEnv) return EXIT_FAILURE;

    // 设置cuda
    if (is_cuda==1) {
        OrtCUDAProviderOptions options;
        options.device_id = device_id;
        //options.arena_extend_strategy = 0;
        //options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
        //options.do_copy_in_default_stream = 1;
        _ortSessionOptions.AppendExecutionProvider_CUDA(options);
    }

    // 加载模型
    wchar_t path[1024];
    mbstowcs_s(nullptr, path, 1024, model_path.c_str(), 1024);
    this->_ortSession = std::make_unique<Ort::Session>(*(this->_ortEnv), path, _ortSessionOptions);
    if (!(this->_ortSession)) {
        std::cout << "Failde to Load Model!" << std::endl;
        return EXIT_FAILURE;
    }

    // 获取模型输入输出层名
    Ort::AllocatorWithDefaultOptions allocator;
    this->_inputNodesNum = this->_ortSession->GetInputCount();
    this->_outputNodesNum = this->_ortSession->GetOutputCount();

    for (int i = 0; i < this->_inputNodesNum; i++) {
        Ort::AllocatedStringPtr input = this->_ortSession->GetInputNameAllocated(i, allocator);
        this->_inputNames.push_back(input.get());
        //input.release(); // 不需要手动释放
    }

    for (int i = 0; i < this->_outputNodesNum; i++) {
        Ort::AllocatedStringPtr output = this->_ortSession->GetOutputNameAllocated(i, allocator);
        _outputNames.push_back(output.get());
        //output.release();
    }

    if (this->_inputNames.size() != this->_inputNodesNum || this->_outputNames.size() != this->_outputNodesNum) return EXIT_FAILURE;


    for (int i = 0; i < this->_inputNames.size(); i++) {
        auto name = _inputNames[i];
        this->_inputShapes.emplace(name, this->_ortSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    for (int i = 0; i < this->_outputNames.size(); i++) {
        auto name = _outputNames[i];
        this->_outputShapes.emplace(name, this->_ortSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    // 初始化完成，释放thOpts
    Ort::GetApi().ReleaseThreadingOptions(thOpts);
    return EXIT_SUCCESS;
}

void OnnxEngine::inference(const cv::Mat& input)
{
    if (!this->_ortEnv || !this->_ortSession)
        throw std::runtime_error("Error: Environment or Session is Not Initialized.");

    Ort::AllocatorWithDefaultOptions allocator;

    //std::vector<int64_t> dims(input.size.p, input.size.p + input.size.dims());
    std::vector<int64_t> dims(this->_inputShapes[this->_inputNames[0]]);
    size_t data_size = 1;
    for (auto dim : dims) data_size *= dim;

    // 创建tensor
    std::vector<float> input_data(data_size);
    input_data.assign((float*)input.datastart, (float*)input.dataend);
    auto _ortMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(_ortMemoryInfo, input_data.data(), input_data.size(), dims.data(), dims.size());
    if (!input_tensor.IsTensor()) {
        std::cerr << "Error: Build Tensor Failed!" << std::endl;
        throw std::runtime_error("Error: Build Tensor Failed!");
    }

    std::vector<const char*> input_names(this->_inputNodesNum);
    for (int i = 0; i < this->_inputNodesNum; i++){
        input_names[i] = this->_inputNames[i].c_str();
    }

    std::vector<const char*> output_names(this->_outputNodesNum);
    for (int i = 0; i < this->_outputNodesNum; i++){
        output_names[i] = this->_outputNames[i].c_str();
    }

    auto time_start = cv::getTickCount();
    auto output_tensor = this->_ortSession->Run(Ort::RunOptions{ nullptr },
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        output_names.size());

    auto time_end = cv::getTickCount();
    double time_cost = (time_end - time_start) / cv::getTickFrequency() * 1000;
    std::cout << "Inference Time Cost: " << time_cost << " ms" << std::endl;
    assert(output_tensor.front().IsTensor());

    for (size_t i = 0; i < output_names.size(); i++) {
        float* output_ptr = output_tensor[i].GetTensorMutableData<float>();
        std::vector<int64_t> _outputTensorShape = output_tensor[i].GetTensorTypeAndShapeInfo().GetShape();
        _outputShapes[output_names[i]] = _outputTensorShape;
        _outDataPtr[output_names[i]] = output_ptr;
    }

    //for (auto& name : input_names) allocator.Free((void*)name);
    //for (auto& name : output_names) allocator.Free((void*)name);
}
