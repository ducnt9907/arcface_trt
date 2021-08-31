#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
#include <NvOnnxParser.h>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

#define USE_FP16
#define DEVICE 0
#define BATCH_SIZE 1

static const int INPUT_H = 112;
static const int INPUT_W = 112;
static const int OUTPUT_SIZE = 512;
const char *INPUT_BLOB_NAME;
const char *OUTPUT_BLOB_NAME;

using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    assert(engine.getNbBindings() == 2);
    void *buffers[2];
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

int main(int argc, char **argv){
    cudaSetDevice(DEVICE);

    if (std::string(argv[1]) == "-s"){
        std::cout<<"PREPARING THE ENGINE ...\n";
        std::string model_path = "../backbone_r100.onnx";

        IBuilder *builder = createInferBuilder(gLogger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
        IParser *parser = createParser(*network, gLogger);
        IBuilderConfig *config = builder->createBuilderConfig();

        if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
        {
            std::cerr << "ERROR: could not parse the model.\n";
            return 0;
        }

        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 112 , 112});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 112 , 112});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{1, 3, 112 , 112});
        config->addOptimizationProfile(profile);

        config->setMaxWorkspaceSize(1 << 20);
//        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        builder->setMaxBatchSize(1);

        ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);

        std::cout<<"ENGINE DONE...\n";

        IHostMemory *modelStream = engine->serialize();

        std::ofstream p("arcface_r100_onnx.engine", std::ios::binary);
        if (!p){
            std::cerr << "could not open plan output file\n";
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout<<network->getInput(0)->getName()<<' '<<network->getOutput(0)->getName()<<'\n';
        modelStream->destroy();
        return 1;
    }

    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("arcface_r100_onnx.engine", std::ios::binary);
    if (file.good()){
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    IRuntime *runtime = createInferRuntime(gLogger);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext *context = engine->createExecutionContext();

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    cv::Mat img = cv::imread("../vietth.jpg");
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));

    INPUT_BLOB_NAME = "input.1";
    OUTPUT_BLOB_NAME = "1333";

    for (int b=0; b<BATCH_SIZE; b++){
        float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
        for (int i=0; i<INPUT_H*INPUT_W; i++){
            p_data[i] = (img.at<cv::Vec3b>(i)[2]/255.0 - 0.5)/0.5;
            p_data[i + INPUT_H*INPUT_W] = (img.at<cv::Vec3b>(i)[1]/255.0 - 0.5)/0.5;
            p_data[i + 2*INPUT_H*INPUT_W] = (img.at<cv::Vec3b>(i)[0]/255.0 - 0.5)/0.5;
        }
    }

    for(int i=0; i<10; i++){
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    std::freopen("mtp_feat_onnx.txt", "w", stdout);
    for(int i=0; i<OUTPUT_SIZE; i++){
         std::cout<<std::fixed<<std::setprecision(32)<<prob[i]<<' ';
    }
    std::cout<<'\n';
    return 0;
}
