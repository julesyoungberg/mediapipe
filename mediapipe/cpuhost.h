#ifndef CPU_HOST_H
#define CPU_HOST_H

#include <opencv2/core.hpp>

struct Landmark {
    float x;
    float y;
    float z;
    float visibility;
    float presence;
};

struct Pose {
    Landmark data[33];
};

struct Hand {
    Landmark data[21];
};

struct FaceMesh {
    Landmark data[478];
};

class PoseGraph {
    public: 
        PoseGraph(const char* graph_config, const char* output_node);
        bool process(const cv::Mat *input, Pose &output);
        ~PoseGraph();
};

class HandsGraph {
    public:
        HandsGraph(const char* graph_config, const char* output_node);
        bool process(const cv::Mat *input, Hand &left, Hand &right);
        ~HandsGraph();
};

class FaceMeshGraph {
    public:
        FaceMeshGraph(const char* graph_config, const char* output_node);
        bool process(const cv::Mat *input, FaceMesh &mesh);
        ~FaceMeshGraph();
};

#endif
