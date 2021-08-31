#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <string>
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
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
static inline std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

static inline IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* addPRelu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    int len = weightMap[lname + ".weight"].count;

    float *scval_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    float *scval_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval_1[i] = -1.0;
        scval_2[i] = -gamma[i];
    }
    Weights scale_1{ DataType::kFLOAT, scval_1, len };
    Weights scale_2{ DataType::kFLOAT, scval_2, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = 0.0;
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    auto relu1 = network->addActivation(input, ActivationType::kRELU);
    IScaleLayer* scale1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale_1, power);
    auto relu2 = network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
    IScaleLayer* scale2 = network->addScale(*relu2->getOutput(0), ScaleMode::kCHANNEL, shift, scale_2, power);
    IElementWiseLayer* ew1 = network->addElementWise(*relu1->getOutput(0), *scale2->getOutput(0), ElementWiseOperation::kSUM);
    return ew1;
}

ILayer *IBasicBlock(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int stride, std::string lname, int downdim=0){
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, input, lname + ".bn1", 1e-5);
    IConvolutionLayer *conv1 = network->addConvolutionNd(*bn1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv1.weight"], emptywts);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn2", 1e-5);
    ILayer *prelu = addPRelu(network, weightMap, *bn2->getOutput(0), lname + ".prelu");

    IConvolutionLayer *conv2 = network->addConvolutionNd(*prelu->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn3", 1e-5);

    if (downdim == 0){
        IElementWiseLayer *ew = network->addElementWise(*bn3->getOutput(0), input, ElementWiseOperation::kSUM);
        return ew;
    }
    else {
        IConvolutionLayer *conv = network->addConvolutionNd(input, downdim, DimsHW{1, 1}, weightMap[lname + ".downsample.0.weight"], emptywts);
        conv->setStrideNd(DimsHW{2, 2});
        conv->setPaddingNd(DimsHW{0, 0});
        IScaleLayer *bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".downsample.1", 1e-5);

        IElementWiseLayer *ew = network->addElementWise(*bn3->getOutput(0), *bn->getOutput(0), ElementWiseOperation::kSUM);

        return ew;
    }
}

ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt){
    INetworkDefinition *network = builder->createNetworkV2(0U);

    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});

    std::map<std::string, Weights> weightMap = loadWeights("../arcface_r50.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer *conv1 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["conv1.weight"], emptywts);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);
    ILayer *prelu = addPRelu(network, weightMap, *bn1->getOutput(0), "prelu");

    ILayer *x = IBasicBlock(network, weightMap, *prelu->getOutput(0), 64, 2, "layer1.0", 64);
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 64, 1, "layer1.1");
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 64, 1, "layer1.2");
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 128, 2, "layer2.0", 128);
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 128, 1, "layer2.1");
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 128, 1, "layer2.2");
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 128, 1, "layer2.3");
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 256, 2, "layer3.0", 256);
    for(int i=1; i<=13; i++){
        std::string lname = "layer3." + std::to_string(i);
        x = IBasicBlock(network, weightMap, *x->getOutput(0), 256, 1, lname);
    }
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 512, 2, "layer4.0", 512);
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 512, 1, "layer4.1");
    x = IBasicBlock(network, weightMap, *x->getOutput(0), 512, 1, "layer4.2");

    IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *x->getOutput(0), "bn2", 1e-5);

    IFullyConnectedLayer *fc = network->addFullyConnected(*bn2->getOutput(0), 512, weightMap["fc.weight"], weightMap["fc.bias"]);
    IScaleLayer *features = addBatchNorm2d(network, weightMap, *fc->getOutput(0), "features", 1e-5);

    features->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*features->getOutput(0));

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1<<20);
//    config->setFlag(BuilderFlag::kFP16);

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    network->destroy();
    for (auto &mem: weightMap){
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);
    (*modelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
}

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

    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s"){
        IHostMemory *modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("arcface_r50.engine", std::ios::binary);
        if (!p){
            std::cerr << "could not open plan output file\n";
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    }
    else
        if (std::string(argv[1]) == "-d"){
            std::ifstream file("arcface_r50.engine", std::ios::binary);
            if (file.good()){
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream = new char[size];
                assert(trtModelStream);
                file.read(trtModelStream, size);
                file.close();
            }
        }
        else return -1;

    // prepare input data ---------------------------
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    IRuntime *runtime = createInferRuntime(gLogger);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext *context = engine->createExecutionContext();
    delete[] trtModelStream;

    cv::Mat img = cv::imread("../mtp.png");
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));

    for (int b=0; b<BATCH_SIZE; b++){
        float *p_data = &data[b * 3 * INPUT_H * INPUT_W];
        for (int i=0; i<INPUT_H*INPUT_W; i++){
            p_data[i] = (img.at<cv::Vec3b>(i)[2]/255.0 - 0.5)/0.5;
            p_data[i + INPUT_H*INPUT_W] = (img.at<cv::Vec3b>(i)[1]/255.0 - 0.5)/0.5;
            p_data[i + 2*INPUT_H*INPUT_W] = (img.at<cv::Vec3b>(i)[0]/255.0 - 0.5)/0.5;
        }
    }

    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::freopen("mtp_feat.txt", "w", stdout);
    for(int i=0; i<OUTPUT_SIZE; i++){
        std::cout<<prob[i]<<' ';
    }
    std::cout<<'\n';
    return 0;
}
