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

#ifndef MEDIAGRAPH_IMPL
#define MEDIAGRAPH_IMPL

#include <cstdlib>
#include <string>

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_framework.h"

#include "mediagraph.h"

namespace mediagraph {

class MediagraphImpl : public Mediagraph {
public:
    MediagraphImpl(){}
    ~MediagraphImpl();

    absl::Status Init(const char* graph, const char* output_node);

    Landmark* Process(uint8_t* data, int width, int height) override;
private:
    mediapipe::CalculatorGraph m_graph;
    absl::StatusOr<mediapipe::OutputStreamPoller> m_poller;
    size_t m_frame_timestamp = 0;

    Landmark* parsePacket(const mediapipe::Packet& packet);
};

class PoseGraph : public MediagraphImpl {
private:
    // returns 33 landmarks
    Landmark* parsePacket(const mediapipe::Packet& packet);
};

class HandsGraph : public MediagraphImpl {
private:
    // returns 42 landmarks, the first 21 are for the left hand, the last 21 are for the right hand
    Landmark* parsePacket(const mediapipe::Packet& packet);
};

class FaceMeshGraph : public MediagraphImpl {
private:
    // returns 478 landmarks
    Landmark* parsePacket(const mediapipe::Packet& packet);
};

}

#endif
