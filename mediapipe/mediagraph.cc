#include "mediagraph.h"
#include "mediagraph_impl.h"

namespace mediagraph {

Mediagraph* Mediagraph::Create(GraphType graph_type, const char* graph_config, const char* output_node) {
    MediagraphImpl* mediagraph;
    switch (graph_type) {
        case GraphType::POSE:
            mediagraph = new PoseGraph();
            break;
        case GraphType::HANDS:
            mediagraph = new HandsGraph();
            break;
        case GraphType::FACE:
            mediagraph = new FaceMeshGraph();
            break;
        default:
            return nullptr;
    }

    absl::Status status = mediagraph->Init(graph_config, output_node);
    if (status.ok()) {
        return mediagraph;
    } else {
        LOG(INFO) << "Error initializing graph " << status.ToString();
        delete mediagraph;
        return nullptr;
    }
}

}
