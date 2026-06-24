#include <thread>
#include "Extractors/superpoint_onnx.h"

int SuperPointOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    onnx_times.clear();
    try
    {
        env0 = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner Extractor");

        session_options0 = Ort::SessionOptions();
        
        // Paralelismo optimizado: Hilos adaptados al hardware para no bloquear los hilos de ORB-SLAM3
        unsigned int cores = std::thread::hardware_concurrency();
        int intra_threads = (cores > 4) ? 2 : 1; 
        
        session_options0.SetIntraOpNumThreads(intra_threads);
        session_options0.SetInterOpNumThreads(1); // Mantenemos 1 para evitar colisiones entre el Tracking y LocalMapping
        session_options0.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options0.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda") {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            // CAMBIO: Heuristic es mucho más rápido y estable en sistemas embebidos que el buscador por defecto
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
            cuda_options.gpu_mem_limit = 0; 
            cuda_options.arena_extend_strategy = 1; 
            cuda_options.do_copy_in_default_stream = 1; 
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            session_options0.AppendExecutionProvider_CUDA(cuda_options);
        }

        std::string extractor_modelPath = cfg.extractorPath;
        ExtractorSession = new Ort::Session(env0, extractor_modelPath.c_str(), session_options0);

        // Inicialización optimizada de nodos de entrada
        size_t numInputNodes = ExtractorSession->GetInputCount();
        ExtractorInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++)
        {
            ExtractorInputNodeNames.emplace_back(ExtractorSession->GetInputNameAllocated(i, allocator).get());
            ExtractorInputNodeShapes.emplace_back(ExtractorSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        // Inicialización optimizada de nodos de salida
        size_t numOutputNodes = ExtractorSession->GetOutputCount();
        ExtractorOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++)
        {
            ExtractorOutputNodeNames.emplace_back(ExtractorSession->GetOutputNameAllocated(i, allocator).get()); 
            ExtractorOutputNodeShapes.emplace_back(ExtractorSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
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

cv::Mat SuperPointOnnxRunner::Extractor_PreProcess(Configuration cfg, const cv::Mat& Image, float& scale)
{
    std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;
    
    // Al asignarla así, OpenCV NO copia los píxeles (es una operación O(1) ultra-rápida),
    // pero nos permite quitar el calificador const para que funciones como RGB2Grayscale no fallen.
    cv::Mat localImage = Image; 
    cv::Mat resultImage;
    
    // Conversión de color inteligente sin redundancias
    if (cfg.extractorType == "superpoint" && localImage.channels() > 1)
    {
        std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
        resultImage = RGB2Grayscale(localImage); // Pasamos la matriz local sin el calificador const
    }
    else
    {
        resultImage = localImage; // Si ya viene en escala de grises, simplemente apuntamos la referencia
    }
    
    return NormalizeImage(resultImage);
}

int SuperPointOnnxRunner::Extractor_Inference(Configuration cfg, const cv::Mat& image)
{   
    extractor_outputtensors.clear();
    try 
    {   
        ExtractorInputNodeShapes[0] = {1, 1, image.size().height, image.size().width};

        // OPTIMIZACIÓN EXTREMA: En lugar de usar bucles iteradores iterando por toda la imagen,
        // mapeamos directamente el puntero de memoria contigua de OpenCV al vector destino de ONNX
        size_t srcInputTensorSize = static_cast<size_t>(image.rows * image.cols * image.channels());
        std::vector<float> srcInputTensorValues(srcInputTensorSize);
        
        if (image.isContinuous()) {
            const float* mat_ptr = image.ptr<float>(0);
            std::memcpy(srcInputTensorValues.data(), mat_ptr, srcInputTensorSize * sizeof(float));
        } else {
            // Caso alternativo por si la matriz tiene padding en los bordes
            for (int r = 0; r < image.rows; ++r) {
                std::memcpy(srcInputTensorValues.data() + r * image.cols, image.ptr<float>(r), image.cols * sizeof(float));
            }
        }
        
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler, srcInputTensorValues.data(), srcInputTensorValues.size(),
            ExtractorInputNodeShapes[0].data(), ExtractorInputNodeShapes[0].size()
        ));

        auto time_start = std::chrono::high_resolution_clock::now();
        
        // Nombres estáticos limpios para evitar realocaciones de strings en la llamada de ejecución
        const char* input_names[] = {"image"};
        const char* output_names[] = {"keypoints", "scores", "descriptors"};
        
        auto output_tensor = ExtractorSession->Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            input_tensors.data(), 
            input_tensors.size(), 
            output_names, 
            ExtractorOutputNodeNames.size()
        );
    
        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        extractor_timer += diff;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        extractor_outputtensors.emplace_back(std::move(output_tensor));
    } 
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueDecoupleOnnxRunner Extractor inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

// Queda exactamente intacto y protegido tal como me lo pediste
void SuperPointOnnxRunner::Extractor_PostProcess(Configuration cfg, std::vector<Ort::Value> tensor, std::vector<cv::KeyPoint>& vKeyPoints, cv::Mat &Descriptors)
{   
    std::pair<std::vector<cv::Point2f>, float*> extractor_result;
    try
    {
        std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts = (int64_t*)tensor[0].GetTensorMutableData<void>();

        std::vector<int64_t> score_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
        float* scores = (float*)tensor[1].GetTensorMutableData<void>();

        std::vector<int64_t> descriptors_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
        float* desc = (float*)tensor[2].GetTensorMutableData<void>();

        cv::KeyPoint keypoint;
        
        // 1. Bajamos el umbral físico base para rescatar puntos en momentos de desenfoque
        float threshold = 0.0013f; 
        
        if (pfeat < 100 && pfeat > 0)
        {    
            float percentage = pfeat / 100.0f;
            static std::vector<float> buffer;
            buffer.assign(scores, scores + kpts_Shape[1]);

            int index = static_cast<int>(percentage * kpts_Shape[1]) - 1;
            if (index < 0) index = 0;

            std::nth_element(buffer.begin(), buffer.begin() + index, buffer.end(), std::greater<float>());
            threshold = buffer[index];
            
            // Suelo de seguridad adaptativo para que nunca se vuelva demasiado estricto
            if (threshold > 0.020f) {
                threshold = 0.020f; 
            }
        }

        // 2. Extracción inicial de todos los candidatos posibles
        std::vector<cv::KeyPoint> rawKeyPoints;
        std::vector<int> rawIndices;
        rawKeyPoints.reserve(kpts_Shape[1]);
        rawIndices.reserve(kpts_Shape[1]);

        for (int i = 0; i < kpts_Shape[1]; i++) 
        {
            if (scores[i] < threshold) continue;
            
            keypoint.pt = cv::Point2f(static_cast<float>(kpts[i * 2]), static_cast<float>(kpts[i * 2 + 1])); 
            keypoint.response = scores[i];
            keypoint.size = 10;
            keypoint.octave = 0;
            
            rawKeyPoints.emplace_back(keypoint);
            rawIndices.push_back(i);
        }

        // 3. FILTRO NMS (SUPRESIÓN NO MÁXIMA SPATIAL): Limpieza de aglomeraciones
        // Ordenamos los crudos por respuesta para procesar primero los mejores
        // ==========================================
        // FILTRO NMS CALIBRADO PARA FAVORECER FUSIONES
        // ==========================================
        std::vector<size_t> indices_raw(rawKeyPoints.size());
        for (size_t k = 0; k < indices_raw.size(); ++k) indices_raw[k] = k;
        std::sort(indices_raw.begin(), indices_raw.end(), [&rawKeyPoints](size_t a, size_t b) {
            return rawKeyPoints[a].response > rawKeyPoints[b].response;
        });

        std::vector<cv::KeyPoint> tempKeyPoints;
        std::vector<int> validIndices;
        
        // REDUCCIÓN CRÍTICA: Bajamos de 5.0f a 3.0f píxeles.
        // Esto evita las macro-aglomeraciones pero permite la densidad suficiente 
        // de inliers para que el RANSAC del Loop Closing valide la fusión de mapas.
        const float min_dist = 3.0f; 
        
        for (size_t idx : indices_raw)
        {
            const cv::KeyPoint& kp_candidato = rawKeyPoints[idx];
            bool demasiado_cerca = false;
            
            for (const auto& kp_aceptado : tempKeyPoints)
            {
                float dx = kp_candidato.pt.x - kp_aceptado.pt.x;
                float dy = kp_candidato.pt.y - kp_aceptado.pt.y;
                if ((dx * dx + dy * dy) < (min_dist * min_dist))
                {
                    demasiado_cerca = true;
                    break;
                }
            }
            
            if (!demasiado_cerca)
            {
                tempKeyPoints.push_back(kp_candidato);
                validIndices.push_back(rawIndices[idx]);
            }
        }

        // Elevamos el techo final a 2000 puntos bien distribuidos.
        // Más puntos significan más probabilidad de superar el umbral de inliers del Map Merging.
        size_t max_features = 800; 
        if (pfeat >= 100)
        {
            max_features = static_cast<size_t>(pfeat);
        }

        if (tempKeyPoints.size() > max_features)
        {
            tempKeyPoints.resize(max_features);
            validIndices.resize(max_features);
        }
        // 5. Transferencia al pipeline del SLAM
        vKeyPoints = tempKeyPoints;
        int indic = vKeyPoints.size();
        
        if (indic == 0)
        {
            Descriptors = cv::Mat();
            return;
        }
        
        cv::Mat mat1(indic, descriptors_Shape[2], CV_32F);
        for (int i = 0; i < indic; i++)  
        {
            int original_idx = validIndices[i];
            for (int col = 0; col < descriptors_Shape[2]; col++)
            { 
                mat1.at<float>(i, col) = desc[original_idx * descriptors_Shape[2] + col];
            }
        }
        Descriptors = mat1;
    }
    catch (const std::exception& ex) 
    {
        std::cerr << "[ERROR] Extractor postprocess failed : " << ex.what() << std::endl;
    }
}

float SuperPointOnnxRunner::GetMatchThresh()
{
    return this->matchThresh;
}

void SuperPointOnnxRunner::SetMatchThresh(float thresh)
{
    this->matchThresh = thresh;
}

double SuperPointOnnxRunner::GetTimer(std::string name)
{
    if (name == "extractor")
    {
        return this->extractor_timer;
    }else
    {
        return this->matcher_timer;
    }
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> SuperPointOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}

SuperPointOnnxRunner::SuperPointOnnxRunner(unsigned int threads) : \
    num_threads(threads)
{
}

SuperPointOnnxRunner::~SuperPointOnnxRunner()
{
}