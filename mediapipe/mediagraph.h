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

#ifndef MEDIAGRAPH_H
#define MEDIAGRAPH_H

#include <cstdlib>
#include <string>

namespace mediagraph {

// -calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt
struct Landmark {
  float x;
  float y;
  float z;
  float visibility;
  float presence;
};

enum GraphType {
    POSE,
    HANDS,
    FACE,
};

class Mediagraph {
public:
    // Create and initialize using provided graph
    // Returns nullptr if initialization failed
    static Mediagraph* Create(GraphType graph_type, const char* graph_config, const char* output_node);
    virtual ~Mediagraph();

    // Processes one frame and blocks until finished
    // Input data is expected to be ImageFormat::SRGB (24bits)
    // Returns nullptr if failed to run graph
    // Returns pointer to image whose size is the same as input image
    // and whose format is ImageFormat::SRGB
    // ImageFormat::SRGB is QImage::Format_RGB888 in Qt
    // Function does not take ownership of input data
    virtual Landmark* Process(uint8_t* data, int width, int height);
};

}

#endif
