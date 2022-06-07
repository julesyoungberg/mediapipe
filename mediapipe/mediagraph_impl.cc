#include "mediagraph_impl.h"
// #include "absl/flags/flag.h"
// #include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

namespace mediagraph {

constexpr char kInputStream[] = "input_video";

MediagraphImpl::~MediagraphImpl() {
    LOG(INFO) << "Shutting down.";
    absl::Status status = m_graph.CloseInputStream(kInputStream);
    if (status.ok()){
    	absl::Status status1 = m_graph.WaitUntilDone();
        if (!status1.ok()) {
            LOG(INFO) << "Error in WaitUntilDone(): " << status1.ToString();
        }
    } else {
        LOG(INFO) << "Error in CloseInputStream(): " << status.ToString();
    }
}

absl::Status MediagraphImpl::Init(GraphType graph_type, const char* graph, const char* output_node) {
    m_graph_type = graph_type;
    LOG(INFO) << "Parsing graph config " << graph;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

    LOG(INFO) << "Initialize the calculator graph.";
    MP_RETURN_IF_ERROR(m_graph.Initialize(config));

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(m_poller, m_graph.AddOutputStreamPoller(output_node));
    MP_RETURN_IF_ERROR(m_graph.StartRun({}));

    return absl::OkStatus();
}

Landmark* parsePosePacket(const mediapipe::Packet& packet) {
    LOG(INFO) << "parsePosePacket()\n";
    Landmark output[33];

    auto& landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();

    assert(landmarks.landmark_size() == 33);
      
    for (int idx = 0; idx < 33; ++idx) { 
        const mediapipe::NormalizedLandmark& landmark = landmarks.landmark(idx);
    
        output[idx] = {
	        .x = landmark.x(),
            .y = landmark.y(),
            .z = landmark.z(),
            .visibility = landmark.visibility(),
            .presence = landmark.presence(),
        };
    }

    return output;
}

Landmark* parseHandsPacket(const mediapipe::Packet& packet) {
    LOG(INFO) << "parseHandsPacket()\n";
    Landmark output[42];

    auto& hands = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    // left
    if (hands.size() > 0) {
        const mediapipe::NormalizedLandmarkList &left_hand = hands.at(0);
        assert(left_hand.landmark_size() == 21);

        for (int idx = 0; idx < 21; ++idx) {
            const mediapipe::NormalizedLandmark& landmark = left_hand.landmark(idx);

            output[idx] = {
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
        assert(right_hand.landmark_size() == 21);

        for (int idx = 0; idx < 21; ++idx) {
            const mediapipe::NormalizedLandmark& landmark = right_hand.landmark(idx);

            output[idx + 21] = {
                .x = landmark.x(),
                .y = landmark.y(),
                .z = landmark.z(),
                .visibility = landmark.visibility(),
                .presence = landmark.presence(),
            };
        }
    }

    return output;
}

Landmark* parseFacePacket(const mediapipe::Packet& packet) {
    LOG(INFO) << "parseFacePacket()\n";
    Landmark output[478];

    auto& faces = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    if (faces.size() > 0) {
        const mediapipe::NormalizedLandmarkList &face = faces.at(0);
        // 478 landmarks with irises, 468 without
        for (int idx = 0; idx < face.landmark_size(); ++idx) {
            const mediapipe::NormalizedLandmark& landmark = face.landmark(idx);

            output[idx] = {
                .x = landmark.x(),
                .y = landmark.y(),
                .z = landmark.z(),
                .visibility = landmark.visibility(),
                .presence = landmark.presence(),
            };
        }
    }

    return output;
}

Landmark* MediagraphImpl::parsePacket(const mediapipe::Packet& packet) {
    LOG(INFO) << "MediagraphImpl::parsepacket()\n";
    switch (m_graph_type) {
        case GraphType::POSE:
            return parsePosePacket(packet);
        case GraphType::HANDS:
            return parseHandsPacket(packet);
        case GraphType::FACE:
            return parseFacePacket(packet);
        default:
            LOG(INFO) << "NO MATCH\n";
            return nullptr;
    }
}

Landmark* MediagraphImpl::Process(uint8_t* data, int width, int height) {
    if (data == nullptr){
        LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
        return nullptr;
    }

    int width_step = width * mediapipe::ImageFrame::ByteDepthForFormat(mediapipe::ImageFormat::SRGB) * mediapipe::ImageFrame::NumberOfChannelsForFormat(mediapipe::ImageFormat::SRGB);

    auto input_frame_for_input = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, width, height, width_step, 
                                                                (uint8*)data, mediapipe::ImageFrame::PixelDataDeleter::kNone);

    m_frame_timestamp++;

    if (!m_graph.AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame_for_input.release()).At(mediapipe::Timestamp(m_frame_timestamp))).ok()) {
        LOG(INFO) << "Failed to add packet to input stream. Call m_graph.WaitUntilDone() to see error (or destroy Example object)";
        return nullptr;
    }

    mediapipe::Packet packet;
    if (!m_poller->Next(&packet)){
        LOG(INFO) << "Poller didnt give me a packet, sorry. Call m_graph.WaitUntilDone() to see error (or destroy Example object). Error probably is that models are not available under mediapipe/models or mediapipe/modules";
        return nullptr;
    }

    return parsePacket(packet);
}

}
