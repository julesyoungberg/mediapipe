#include "mediagraph.h"
#include "mediagraph_impl.h"

namespace mediagraph {

Mediagraph* Mediagraph::Create(GraphType graph_type, const char* graph_config, const char* output_node) {
    MediagraphImpl* mediagraph = new MediagraphImpl();

    absl::Status status = mediagraph->Init(graph_type, graph_config, output_node);
    if (status.ok()) {
        return mediagraph;
    } else {
        LOG(INFO) << "Error initializing graph " << status.ToString();
        delete mediagraph;
        return nullptr;
    }
}

Landmark* Mediagraph::Process(uint8_t* data, int width, int height) {
    return dynamic_cast<MediagraphImpl*>(this)->Process(data, width, height);
}

}
