#include <vector>

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

DetectorImpl::~DetectorImpl() {
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

absl::Status DetectorImpl::Init(const char* graph, const std::vector<Output> outputs_) {
    outputs = outputs_;
    LOG(INFO) << "Parsing graph config " << graph;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

    LOG(INFO) << "Initialize the calculator graph.";
    MP_RETURN_IF_ERROR(m_graph.Initialize(config));

    LOG(INFO) << "Start running the calculator graph.";

    num_outputs = outputs.size();
    out_packets = std::vector<std::vector<mediapipe::Packet>>(num_outputs);
    out_mutexes = std::vector<absl::Mutex>(num_outputs);

    for (uint i = 0; i < num_outputs; ++i) {
        auto out_cb = [&](const mediapipe::Packet& p) {
            absl::MutexLock lock(&out_mutexes[i]);
            out_packets[i].push_back(p);
            return absl::OkStatus();
        };

        MP_RETURN_IF_ERROR(m_graph.ObserveOutputStream(outputs[i].name, out_cb));
    }

    MP_RETURN_IF_ERROR(m_graph.StartRun({}));

    return absl::OkStatus();
}

FeatureList parseHandsPacket(const mediapipe::Packet& packet) {
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

    Feature f = { data: output };
    FeatureList l { num_features: 1, features: &f };
    return l;
}

FeatureList parseFacePacket(const mediapipe::Packet& packet) {
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

    Feature f = { data: output };
    FeatureList l { num_features: 1, features: &f };
    return l;
}

FeatureList parsePosePacket(const mediapipe::Packet& packet) {
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

    Feature f = { data: output };
    FeatureList l { num_features: 1, features: &f };
    return l;
}

FeatureList parsePacket(const mediapipe::Packet& packet, const FeatureType type) {
    switch (type) {
        case FeatureType::POSE:
            return parsePosePacket(packet);
        case FeatureType::HANDS:
            return parseHandsPacket(packet);
        case FeatureType::FACE:
            return parseFacePacket(packet);
        default:
            LOG(INFO) << "NO MATCH\n";
            return { num_features: 1, features: nullptr };
    }
}

FeatureList* DetectorImpl::Process(uint8_t* data, int width, int height) {
    if (data == nullptr){
        LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
        return nullptr;
    }

    int width_step = width * mediapipe::ImageFrame::ByteDepthForFormat(mediapipe::ImageFormat::SRGB)
        * mediapipe::ImageFrame::NumberOfChannelsForFormat(mediapipe::ImageFormat::SRGB);

    auto input_frame_for_input = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, width, height, width_step,
        (uint8*)data, mediapipe::ImageFrame::PixelDataDeleter::kNone
    );

    m_frame_timestamp++;

    mediapipe::Status run_status = m_graph.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame_for_input.release()).At(mediapipe::Timestamp(m_frame_timestamp))
    );

    if (!run_status.ok()) {
        LOG(INFO) << "Add Packet error: [" << run_status.message() << "]" << std::endl;
        return nullptr;
    }

    FeatureList results[num_outputs];
    mediapipe::Packet packet;

    for (uint i = 0; i < num_outputs; ++i) {
        absl::MutexLock lock(&out_mutexes[i]);

        if (out_packets[i].size() == 0) {
            results[i] = { num_features: 0, features: nullptr };
            continue;
        }

        packet = out_packets[i].back();
        out_packets[i].clear();

        results[i] = parsePacket(packet, outputs[i].type);
    }
    
    return results;
}

EffectImpl::~EffectImpl() {
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

absl::Status EffectImpl::Init(const char* graph, const char* output_node) {
    LOG(INFO) << "Parsing graph config " << graph;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

    LOG(INFO) << "Initialize the calculator graph.";
    MP_RETURN_IF_ERROR(m_graph.Initialize(config));

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(m_poller, m_graph.AddOutputStreamPoller(output_node));
    MP_RETURN_IF_ERROR(m_graph.StartRun({}));

    return absl::OkStatus();
}

uint8_t* EffectImpl::Process(uint8_t* data, int width, int height) {
    if (data == nullptr){
        LOG(INFO) << __FUNCTION__ << " input data is nullptr!";
        return nullptr;
    }

    int width_step = width * mediapipe::ImageFrame::ByteDepthForFormat(mediapipe::ImageFormat::SRGB)
        * mediapipe::ImageFrame::NumberOfChannelsForFormat(mediapipe::ImageFormat::SRGB);

    auto input_frame_for_input = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, width, height, width_step,
        (uint8*)data, mediapipe::ImageFrame::PixelDataDeleter::kNone
    );

    m_frame_timestamp++;

    mediapipe::Status run_status = m_graph.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame_for_input.release()).At(mediapipe::Timestamp(m_frame_timestamp))
    );

    if (!run_status.ok()) {
        LOG(INFO) << "Add Packet error: [" << run_status.message() << "]" << std::endl;
        return nullptr;
    }

    mediapipe::Packet packet;
    if (!m_poller->Next(&packet)){
        LOG(INFO) << "No packet from poller";
        return nullptr;
    }

    const mediapipe::ImageFrame &output_frame = packet.Get<mediapipe::ImageFrame>();
    size_t output_bytes = output_frame.PixelDataSizeStoredContiguously();

    // This could be optimized to not copy but return output_frame.PixelData()
    uint8_t* out_data = new uint8_t[output_bytes];
    output_frame.CopyToBuffer((uint8*)out_data, output_bytes);
    return out_data; 
}

}
