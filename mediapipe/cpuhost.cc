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

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
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

PoseGraph::PoseGraph(const char* graph_config, const char* output_node) {
  mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
      graph_config);

  graph = std::make_unique<mediapipe::CalculatorGraph>();
  
  absl::Status status = graph->Initialize(config);
  if (!status.ok()) {
    std::cerr << "Graph Init" << status.message() << std::endl;
    throw std::runtime_error("Failed to create the graph");
  }

  mediapipe::StatusOrPoller sop = graph->AddOutputStreamPoller(output_node);

  if (!sop.ok()) {
    std::cerr << "Failed to add poller to graph: [" << sop.status().message() << "]" << std::endl;
    throw std::runtime_error("Failed to add poller to graph");
  }

  assert(sop.ok());

  poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));
  
  graph->StartRun({});
}

PoseGraph::~PoseGraph() {
  graph->CloseInputStream(kInputStream);
  graph->WaitUntilDone();
}

bool PoseGraph::process(const cv::Mat *input, Pose &output) {
  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, input->cols, input->rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  input->copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

  absl::Status run_status = graph->AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));

  if (!run_status.ok()) {
    std::cerr << "Add Packet error: [" << run_status.message() << "]" << std::endl;
    return false;
  }

  // Get the graph result packet, or stop if that fails.
  mediapipe::Packet packet;
  if (poller->QueueSize() > 0) {
    if (poller->Next(&packet)) {
      auto& landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();
      
      for (int idx = 0; idx < landmarks.landmark_size(); ++idx) { 
        const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(idx);
    
        output.data[idx] = {
	  .x = landmark.x(),
          .y = landmark.y(),
          .z = landmark.z(),
          .visibility = landmark.visibility(),
          .presence = landmark.presence(),
        };
      }
    } else {
	std::cerr << "No packet from poller" << std::endl; 
	return false;
    }
  }

  return true;
}

class HandsGraph {
  public:
    HandsGraph(const char* graph_config, const char* output_node);
    bool process(const cv::Mat *input, Hand &left, Hand &right);
    ~HandsGraph();
  private:
    std::unique_ptr<mediapipe::OutputStreamPoller> poller;
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
};

HandsGraph::HandsGraph(const char* graph_config, const char* output_node) {
  mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
      graph_config);

  graph = std::make_unique<mediapipe::CalculatorGraph>();

  absl::Status status = graph->Initialize(config);
  if (!status.ok()) {
    std::cerr << "Graph Init" << status.message() << std::endl;
    throw std::runtime_error("Failed to create the graph");
  }

  mediapipe::StatusOrPoller sop = graph->AddOutputStreamPoller(output_node);
  // if (!status_or_poller.ok()) {
  //   throw std::runtime_error("hi");
  // }

   if (!sop.ok()) {
     std::cerr << "Failed to add poller to graph: [" << sop.status().message() << "]" << std::endl;
     throw std::runtime_error("Failed to add poller to graph");
     //LOG(ERROR) << "Failed to run the graph: " << run_status.message();
     //return EXIT_FAILURE;
   }

  assert(sop.ok());

  poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));

  graph->StartRun({});
}

HandsGraph::~HandsGraph() {
  graph->CloseInputStream(kInputStream);
  graph->WaitUntilDone();
}

bool HandsGraph::process(const cv::Mat *input, Hand &left, Hand &right) {
  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, input->cols, input->rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);

  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  input->copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

  absl::Status run_status = graph->AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));

  if (!run_status.ok()) {
    std::cerr << "Add Packet error: [" << run_status.message() << "]" << std::endl;
    return false;
  }

  // Get the graph result packet, or stop if that fails.
  mediapipe::Packet packet;
  if (poller->QueueSize() > 0) {
    if (poller->Next(&packet)) {
      auto& hands = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

      // left
      if (hands.size() > 0) {
          const mediapipe::NormalizedLandmarkList &left_hand = hands.at(0);
          for (int idx = 0; idx < left_hand.landmark_size(); ++idx) {
            const mediapipe::NormalizedLandmark& landmark = left_hand.landmark(idx);

            left.data[idx] = {
              .x = landmark.x(),
              .y = landmark.y(),
              .z = landmark.z(),
              .visibility = landmark.visibility(),
              .presence = landmark.presence(),
            };
          }
      }

      // right
      if (hands.size() > 1) {
          const mediapipe::NormalizedLandmarkList &right_hand = hands.at(1);
          for (int idx = 0; idx < right_hand.landmark_size(); ++idx) {
            const mediapipe::NormalizedLandmark& landmark = right_hand.landmark(idx);

            right.data[idx] = {
              .x = landmark.x(),
              .y = landmark.y(),
              .z = landmark.z(),
              .visibility = landmark.visibility(),
              .presence = landmark.presence(),
            };
          }
      }
    } else {
        std::cerr << "No packet from poller" << std::endl;
        return false;
    }
  }

  return true;
}

class FaceMeshGraph {
  public:
    FaceMeshGraph(const char* graph_config, const char* output_node);
    bool process(const cv::Mat *input, FaceMesh &mesh);
    ~FaceMeshGraph();
  private:
    std::unique_ptr<mediapipe::OutputStreamPoller> poller;
    std::unique_ptr<mediapipe::CalculatorGraph> graph;
};

FaceMeshGraph::FaceMeshGraph(const char* graph_config, const char* output_node) {
  mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
      graph_config);

  graph = std::make_unique<mediapipe::CalculatorGraph>();

  absl::Status status = graph->Initialize(config);
  if (!status.ok()) {
    std::cerr << "Graph Init" << status.message() << std::endl;
    throw std::runtime_error("Failed to create the graph");
  }

  mediapipe::StatusOrPoller sop = graph->AddOutputStreamPoller(output_node);
  // if (!status_or_poller.ok()) {
  //   throw std::runtime_error("hi");
  // }

   if (!sop.ok()) {
     std::cerr << "Failed to add poller to graph: [" << sop.status().message() << "]" << std::endl;
     throw std::runtime_error("Failed to add poller to graph");
     //LOG(ERROR) << "Failed to run the graph: " << run_status.message();
     //return EXIT_FAILURE;
   }

  assert(sop.ok());

  poller = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));

  graph->StartRun({});
}

FaceMeshGraph::~FaceMeshGraph() {
  graph->CloseInputStream(kInputStream);
  graph->WaitUntilDone();
}

bool FaceMeshGraph::process(const cv::Mat *input, FaceMesh &mesh) {
  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, input->cols, input->rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);

  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  input->copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

  absl::Status run_status = graph->AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));

  if (!run_status.ok()) {
    std::cerr << "Add Packet error: [" << run_status.message() << "]" << std::endl;
    return false;
  }

  // Get the graph result packet, or stop if that fails.
  mediapipe::Packet packet;
  if (poller->QueueSize() > 0) {
    if (poller->Next(&packet)) {
      auto& faces = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

      // left
      if (faces.size() > 0) {
          const mediapipe::NormalizedLandmarkList &face = faces.at(0);
          // 478 landmarks with irises, 468 without
          for (int idx = 0; idx < face.landmark_size(); ++idx) {
            const mediapipe::NormalizedLandmark& landmark = face.landmark(idx);

            mesh.data[idx] = {
              .x = landmark.x(),
              .y = landmark.y(),
              .z = landmark.z(),
              .visibility = landmark.visibility(),
              .presence = landmark.presence(),
            };
          }
      }
    } else {
        std::cerr << "No packet from poller" << std::endl;
        return false;
    }
  }

  return true;
}
