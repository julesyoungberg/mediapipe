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

enum FeatureType {
    FACE,
    FACES,
    HAND,
    HANDS,
    POSE,
    POSES,
};

struct Output {
    FeatureType type;
    char* name;
};

class Detector {
public:
    // Create and initialize using provided graph
    // Returns nullptr if initialization failed
    static Detector* Create(const char* graph_config, const Output* outputs, uint8_t num_outputs);
    virtual ~Detector();

    // Processes one frame and returns immediately.
    // If a result is available it is returned.
    // Input data is expected to be ImageFormat::SRGB (24bits)
    // Returns an empty vector if nothing is detected.
    virtual Landmark* Process(uint8_t* data, int width, int height, uint8_t* num_features);
};

class Effect {
public:
    // Create and initialize using provided graph
    // Returns nullptr if initialization failed
    static Effect* Create(const char* graph_config, const char* output_node);

    virtual ~Effect();

    // Processes one frame and blocks until finished
    // Input data is expected to be ImageFormat::SRGB (24bits)
    // Returns nullptr if failed to run graph
    // Returns pointer to image whose size is the same as input image
    // and whose format is ImageFormat::SRGB
    // ImageFormat::SRGB is QImage::Format_RGB888 in Qt
    // Function does not take ownership of input data
    virtual uint8_t* Process(uint8_t* data, int width, int height);
};

}

#endif
