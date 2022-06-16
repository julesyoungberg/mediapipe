#include "mediagraph.h"
#include "mediagraph_impl.h"

namespace mediagraph {

Detector* Detector::Create(const char* graph_config, const Output* outputs, uint8_t num_outputs) {
    DetectorImpl* mediagraph = new DetectorImpl();

    absl::Status status = mediagraph->Init(graph_config, outputs, num_outputs);
    if (status.ok()) {
        return mediagraph;
    } else {
        LOG(INFO) << "Error initializing graph " << status.ToString();
        delete mediagraph;
        return nullptr;
    }
}

Detector::~Detector() {
    dynamic_cast<DetectorImpl*>(this)->~DetectorImpl();
}

Landmark* Detector::Process(uint8_t* data, int width, int height, uint8_t* num_features) {
    return dynamic_cast<DetectorImpl*>(this)->Process(data, width, height, num_features);
}

Effect* Effect::Create(const char* graph_config, const char* output_node) {
    EffectImpl* mediagraph = new EffectImpl();

    absl::Status status = mediagraph->Init(graph_config, output_node);
    if (status.ok()) {
        return mediagraph;
    } else {
        LOG(INFO) << "Error initializing graph " << status.ToString();
        delete mediagraph;
        return nullptr;
    }
}

Effect::~Effect() {
    dynamic_cast<EffectImpl*>(this)->~EffectImpl();
}

uint8_t* Effect::Process(uint8_t* data, int width, int height) {
    return dynamic_cast<EffectImpl*>(this)->Process(data, width, height);
}

}
