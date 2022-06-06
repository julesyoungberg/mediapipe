// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <string>

// #include "absl/flags/flag.h"
// #include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] = "input_video";

// -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt
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
  private: 
    std::unique_ptr<mediapipe::OutputStreamPoller> poller;
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
};

class HandsGraph {
  public:
    HandsGraph(const char* graph_config, const char* output_node);
    bool process(const cv::Mat *input, Hand &left, Hand &right);
    ~HandsGraph();
  private:
    std::unique_ptr<mediapipe::OutputStreamPoller> poller;
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
};

class FaceMeshGraph {
  public:
    FaceMeshGraph(const char* graph_config, const char* output_node);
    bool process(const cv::Mat *input, FaceMesh &mesh);
    ~FaceMeshGraph();
  private:
    std::unique_ptr<mediapipe::OutputStreamPoller> poller;
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
};
