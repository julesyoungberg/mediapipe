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
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/calculator_framework.h"

#include "mediagraph.h"

namespace mediagraph {

class DetectorImpl : public Detector {
public:
    DetectorImpl(){}
    ~DetectorImpl() override;

    absl::Status Init(const char* graph, const Output* outputs_, uint8_t num_outputs_);

    FeatureList* Process(uint8_t* data, int width, int height) override;
private:
    mediapipe::CalculatorGraph m_graph;
    size_t m_frame_timestamp = 0;
    std::vector<Output> outputs;
    std::vector<std::vector<mediapipe::Packet>> out_packets;
    std::vector<absl::Mutex> out_mutexes;
    uint8_t num_outputs;
};

class EffectImpl : public Effect {
public:
    EffectImpl() {}
    ~EffectImpl() override;

    absl::Status Init(const char* graph, const char* output_node);

    uint8_t* Process(uint8_t* data, int width, int height) override;
private:
    mediapipe::CalculatorGraph m_graph;
    absl::StatusOr<mediapipe::OutputStreamPoller> m_poller;
    size_t m_frame_timestamp = 0;
};

}

#endif
