#include <thread>
#include <set>
#include <algorithm>
#include "Matchers/lightglue_onnx.h"

int LightGlueDecoupleOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    try
    {
        env1 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Matcher");

        session_options1 = Ort::SessionOptions();
        
        // Configuración nativa estable para evitar bloqueos en el contenedor Docker
        session_options1.SetIntraOpNumThreads(1);
        session_options1.SetInterOpNumThreads(1);
        session_options1.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options1.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda") {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0; 
            cuda_options.arena_extend_strategy = 1; 
            cuda_options.do_copy_in_default_stream = 1; 
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            session_options1.AppendExecutionProvider_CUDA(cuda_options);
        }

        cfg.lightgluePath = "onnxmodel/lightglue_sim.onnx";
        std::string matcher_modelPath = cfg.lightgluePath;

        MatcherSession = new Ort::Session(env1 , matcher_modelPath.c_str() , session_options1);

        size_t numInputNodes = MatcherSession->GetInputCount();
        MatcherInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0 ; i < numInputNodes ; i++)
        {
            MatcherInputNodeNames.emplace_back(MatcherSession->GetInputNameAllocated(i , allocator).get());
            MatcherInputNodeShapes.emplace_back(MatcherSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        size_t numOutputNodes = MatcherSession->GetOutputCount();
        MatcherOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            MatcherOutputNodeNames.emplace_back(MatcherSession->GetOutputNameAllocated(i , allocator).get());
            MatcherOutputNodeShapes.emplace_back(MatcherSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f> , float*> LightGlueDecoupleOnnxRunner::Extractor_PostProcess(Configuration cfg , std::vector<Ort::Value> tensor)
{
    std::pair<std::vector<cv::Point2f> , float*> extractor_result;
    try{
        std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts = (int64_t*)tensor[0].GetTensorMutableData<void>();

        std::vector<int64_t> score_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
        float* scores = (float*)tensor[1].GetTensorMutableData<void>();

        std::vector<int64_t> descriptors_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
        float* desc = (float*)tensor[2].GetTensorMutableData<void>();

        std::vector<cv::Point2f> kpts_f;
        kpts_f.reserve(kpts_Shape[1]);
        for (int i = 0; i < kpts_Shape[1] * 2; i += 2) 
        {
            kpts_f.emplace_back(cv::Point2f(static_cast<float>(kpts[i]) , static_cast<float>(kpts[i + 1])));
        }

        extractor_result.first = kpts_f;
        extractor_result.second = desc;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] Extractor postprocess failed : " << ex.what() << std::endl;
    }

    return extractor_result;
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::Point2f> kpts, int h , int w)
{
    return NormalizeKeypoints(kpts , h , w);
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::KeyPoint> kpts, int h , int w)
{
    cv::Size size(w, h);
    cv::Point2f shift(static_cast<float>(w) / 2.0f, static_cast<float>(h) / 2.0f);
    float scale = static_cast<float>((std::max)(w, h)) / 2.0f;

    std::vector<cv::Point2f> normalizedKpts;
    normalizedKpts.reserve(kpts.size());
    for (const cv::KeyPoint& kpt : kpts) {
        cv::Point2f normalizedKpt = (kpt.pt - shift) / scale;
        normalizedKpts.push_back(normalizedKpt);
    }

    return normalizedKpts;
}
// =========================================================================
// VARIANTE 1: Inferencia usando cv::Point2f (Tracking normal / Local Mapping)
// =========================================================================
std::vector<Ort::Value> LightGlueDecoupleOnnxRunner::Matcher_Inference(std::vector<cv::Point2f> kpts0 , std::vector<cv::Point2f> kpts1 , float* desc0 , float* desc1)
{
    try
    {
        // FRENO DE SEGURIDAD PARA ALIVIAR LA GPU
        // Truncamos el máximo de puntos a procesar para que las capas de atención del Transformer
        // no colapsen la gráfica, bajando el uso del 100% y acelerando el tracking.
        size_t max_gpu_points = 800;
        size_t size0 = std::min(kpts0.size(), max_gpu_points);
        size_t size1 = std::min(kpts1.size(), max_gpu_points);

        MatcherInputNodeShapes[0] = {1 , static_cast<int>(size0) , 2};
        MatcherInputNodeShapes[1] = {1 , static_cast<int>(size1) , 2};
        MatcherInputNodeShapes[2] = {1 , static_cast<int>(size0) , 256};
        MatcherInputNodeShapes[3] = {1 , static_cast<int>(size1) , 256};
        
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        std::vector<float> kpts0_data(size0 * 2);
        std::vector<float> kpts1_data(size1 * 2);

        for (size_t i = 0; i < size0; ++i) {
            kpts0_data[i * 2] = kpts0[i].x;
            kpts0_data[i * 2 + 1] = kpts0[i].y;
        }
        for (size_t i = 0; i < size1; ++i) {
            kpts1_data[i * 2] = kpts1[i].x;
            kpts1_data[i * 2 + 1] = kpts1[i].y;
        }

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts0_data.data() , kpts0_data.size() * sizeof(float), \
            MatcherInputNodeShapes[0].data() , MatcherInputNodeShapes[0].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts1_data.data() , kpts1_data.size() * sizeof(float), \
            MatcherInputNodeShapes[1].data() , MatcherInputNodeShapes[1].size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc0 , size0 * 256 * sizeof(float), \
            MatcherInputNodeShapes[2].data() , MatcherInputNodeShapes[2].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc1 , size1 * 256 * sizeof(float) , \
            MatcherInputNodeShapes[3].data() , MatcherInputNodeShapes[3].size()
        ));

        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0","mscores0"};

        auto time_start = std::chrono::high_resolution_clock::now();
        auto output_tensor = MatcherSession->Run(Ort::RunOptions{nullptr} , input_names , input_tensors.data() , \
                    input_tensors.size() , output_names , MatcherOutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        matcher_timer += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

        return output_tensor;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueMatcher Inference (Point2f) failed : " << ex.what() << std::endl;
        return std::vector<Ort::Value>();
    }
}

// =========================================================================
// VARIANTE 2: Inferencia usando cv::KeyPoint (Loop Closing / Map Merging)
// =========================================================================
std::vector<Ort::Value> LightGlueDecoupleOnnxRunner::Matcher_Inference(std::vector<cv::KeyPoint> kpts0 , std::vector<cv::KeyPoint> kpts1 , float* desc0 , float* desc1)
{
    try
    {
        // Aplicamos el mismo freno protector aquí para asegurar que las solicitudes de fusión 
        // de bucles compartan el mismo techo de cómputo ligero y liberen la GPU.
        size_t max_gpu_points = 900;
        size_t size0 = std::min(kpts0.size(), max_gpu_points);
        size_t size1 = std::min(kpts1.size(), max_gpu_points);

        MatcherInputNodeShapes[0] = {1 , static_cast<int>(size0) , 2};
        MatcherInputNodeShapes[1] = {1 , static_cast<int>(size1) , 2};
        MatcherInputNodeShapes[2] = {1 , static_cast<int>(size0) , 256};
        MatcherInputNodeShapes[3] = {1 , static_cast<int>(size1) , 256};
        
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        std::vector<float> kpts0_data(size0 * 2);
        std::vector<float> kpts1_data(size1 * 2);

        for (size_t i = 0; i < size0; ++i) {
            kpts0_data[i * 2] = kpts0[i].pt.x;
            kpts0_data[i * 2 + 1] = kpts0[i].pt.y;
        }
        for (size_t i = 0; i < size1; ++i) {
            kpts1_data[i * 2] = kpts1[i].pt.x;
            kpts1_data[i * 2 + 1] = kpts1[i].pt.y;
        }

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts0_data.data() , kpts0_data.size() * sizeof(float), \
            MatcherInputNodeShapes[0].data() , MatcherInputNodeShapes[0].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts1_data.data() , kpts1_data.size() * sizeof(float), \
            MatcherInputNodeShapes[1].data() , MatcherInputNodeShapes[1].size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc0 , size0 * 256 * sizeof(float), \
            MatcherInputNodeShapes[2].data() , MatcherInputNodeShapes[2].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc1 , size1 * 256 * sizeof(float) , \
            MatcherInputNodeShapes[3].data() , MatcherInputNodeShapes[3].size()
        ));

        const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
        const char* output_names[] = {"matches0","mscores0"};

        auto time_start = std::chrono::high_resolution_clock::now();
        auto output_tensor = MatcherSession->Run(Ort::RunOptions{nullptr} , input_names , input_tensors.data() , \
                    input_tensors.size() , output_names , MatcherOutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        std::cout << "[INFO] Matcher loop inference cost time : " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << "ms" << std::endl;
        
        return output_tensor;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueMatcher Inference (KeyPoint) failed : " << ex.what() << std::endl;
        return std::vector<Ort::Value>();
    }
}

int LightGlueDecoupleOnnxRunner::Matcher_PostProcess(Configuration cfg , std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1)
{
    try{
        std::vector<int64_t> matches0_Shape = matcher_outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches0 = (int64_t*)matcher_outputtensors[0].GetTensorMutableData<void>();

        std::vector<int64_t> matches1_Shape = matcher_outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches1 = (int64_t*)matcher_outputtensors[1].GetTensorMutableData<void>();

        std::vector<int64_t> mscore0_Shape = matcher_outputtensors[2].GetTensorTypeAndShapeInfo().GetShape();
        float* mscores0 = (float*)matcher_outputtensors[2].GetTensorMutableData<void>();

        std::vector<cv::Point2f> kpts0_f , kpts1_f;
        kpts0_f.reserve(kpts0.size());
        kpts1_f.reserve(kpts1.size());

        float scale0 = (scales.size() > 0) ? scales[0] : 1.0f;

        for (size_t i = 0; i < kpts0.size(); i++) 
        {
            kpts0_f.emplace_back(cv::Point2f((kpts0[i].x + 0.5f) / scale0 - 0.5f , (kpts0[i].y + 0.5f) / scale0 - 0.5f));
        }
        for (size_t i = 0; i < kpts1.size(); i++) 
        {
            kpts1_f.emplace_back(cv::Point2f((kpts1[i].x + 0.5f) / scale0 - 0.5f , (kpts1[i].y + 0.5f) / scale0 - 0.5f));
        }

        std::vector<int64_t> validIndices;
        for (int i = 0; i < matches0_Shape[1]; ++i) {
            if (matches0[i] > -1 && mscores0[i] > this->matchThresh && matches1[matches0[i]] == i) { 
                validIndices.emplace_back(i);
            }
        }

        std::set<std::pair<int, int>> matches;
        std::vector<cv::Point2f> m_kpts0 , m_kpts1;
        for (int i : validIndices) {
            matches.insert(std::make_pair(i, static_cast<int>(matches0[i])));
        }

        for (const auto& match : matches) {
            m_kpts0.emplace_back(kpts0_f[match.first]);
            m_kpts1.emplace_back(kpts1_f[match.second]);
        }

        keypoints_result.first = m_kpts0;
        keypoints_result.second = m_kpts1;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int LightGlueDecoupleOnnxRunner::Matcher_PostProcess_fused(std::vector<Ort::Value>& output, std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1, std::vector<int>& vnMatches12)
{
    int size = 0;
    try
    {
        std::vector<int64_t> matches_Shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t *matches = (int64_t *)output[0].GetTensorMutableData<void>();

        std::vector<int64_t> mscore_Shape = output[1].GetTensorTypeAndShapeInfo().GetShape();
        float *mscores = (float *)output[1].GetTensorMutableData<void>();

        // DEFICIENCIA CRÍTICA RESUELTA: Inicializamos vnMatches12 con -1 para romper 
        // cualquier enlace basura residual de ejecuciones previas antes de llenarlo
        std::fill(vnMatches12.begin(), vnMatches12.end(), -1);

        // El modelo ONNX de LightGlue devuelve pares entrelazados en matches_Shape[0]
        for (int i = 0; i < matches_Shape[0]; i++)
        {
            if (mscores[i] > this->matchThresh)
            {
                int idx0 = static_cast<int>(matches[i * 2]);
                int idx1 = static_cast<int>(matches[i * 2 + 1]);
                
                // MEJORA ULTRA-SEGURA: Verificación estricta de límites bidireccionales 
                // para evitar violaciones de segmentación (SegFault) en el Atlas del SLAM
                if (idx0 >= 0 && idx0 < vnMatches12.size() && idx1 >= 0 && idx1 < kpts1.size()) {
                    vnMatches12[idx0] = idx1;
                    size++;
                }
            }
        }

        return size;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] PostProcess_fused failed : " << ex.what() << std::endl;
        return size;
    }
}

float LightGlueDecoupleOnnxRunner::GetMatchThresh()
{
    return this->matchThresh;
}

void LightGlueDecoupleOnnxRunner::SetMatchThresh(float thresh)
{
    this->matchThresh = thresh;
}

double LightGlueDecoupleOnnxRunner::GetTimer(std::string name)
{
    if (name == "extractor")
    {
        return this->extractor_timer;
    }else
    {
        return this->matcher_timer;
    }
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}

LightGlueDecoupleOnnxRunner::LightGlueDecoupleOnnxRunner(unsigned int threads) : num_threads(threads)
{
}

LightGlueDecoupleOnnxRunner::~LightGlueDecoupleOnnxRunner()
{
}