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
    absl::Status status = graph_.CloseInputStream(kInputStream);
    if (status.ok()){
    	absl::Status status1 = graph_.WaitUntilDone();
        if (!status1.ok()) {
            LOG(INFO) << "Error in WaitUntilDone(): " << status1.ToString();
        }
    } else {
        LOG(INFO) << "Error in CloseInputStream(): " << status.ToString();
    }
}

absl::Status DetectorImpl::Init(const char* graph, const Output* outputs, uint8_t num_outputs) {
    num_outputs_ = num_outputs;
    outputs_ = std::vector<Output>(outputs, outputs + num_outputs_);
    LOG(INFO) << "Parsing graph config " << graph;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph);

    LOG(INFO) << "Initialize the calculator graph.";
    MP_RETURN_IF_ERROR(graph_.Initialize(config));

    LOG(INFO) << "Start running the calculator graph.";

    out_packets_ = std::vector<std::vector<mediapipe::Packet>>(num_outputs_);
    out_mutexes_ = std::vector<absl::Mutex>(num_outputs_);

    for (uint i = 0; i < num_outputs_; ++i) {
        auto out_cb = [&](const mediapipe::Packet& p) {
            absl::MutexLock lock(&out_mutexes_[i]);
            out_packets_[i].push_back(p);
            return absl::OkStatus();
        };

        MP_RETURN_IF_ERROR(graph_.ObserveOutputStream(outputs_[i].name, out_cb));
    }

    MP_RETURN_IF_ERROR(graph_.StartRun({}));

    return absl::OkStatus();
}

std::vector<Landmark> parseHandsPacket(const mediapipe::Packet& packet, uint8_t* num_features) {
    auto& hands = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    // @todo remove
    if (hands.size() < 2) {
        *num_features = 0;
        return std::vector<Landmark>();
    }

    std::vector<Landmark> output(42);

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

    *num_features = 1;

    return output;
}

std::vector<Landmark> parseFacePacket(const mediapipe::Packet& packet, uint8_t* num_features) {
    auto& faces = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    if (faces.size() < 1) {
        *num_features = 0;
        return std::vector<Landmark>();
    }

    std::vector<Landmark> output(478);
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

    *num_features = 1;

    return output;
}

std::vector<Landmark> parsePosePacket(const mediapipe::Packet& packet, uint8_t* num_features) {
    auto& landmarks = packet.Get<mediapipe::NormalizedLandmarkList>();

    assert(landmarks.landmark_size() == 33);

    std::vector<Landmark> output(33);
      
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

    *num_features = 1;

    return output;
}

std::vector<Landmark> parsePacket(const mediapipe::Packet& packet, const FeatureType type, uint8_t* num_features) {
    switch (type) {
        case FeatureType::POSE:
            return parsePosePacket(packet, num_features);
        case FeatureType::HANDS:
            return parseHandsPacket(packet, num_features);
        case FeatureType::FACE:
            return parseFacePacket(packet, num_features);
        default:
            LOG(INFO) << "NO MATCH\n";
            *num_features = 0;
            return std::vector<Landmark>(0);
    }
}

Landmark* DetectorImpl::Process(uint8_t* data, int width, int height, uint8_t* num_features) {
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

    frame_timestamp_++;

    mediapipe::Status run_status = graph_.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame_for_input.release()).At(mediapipe::Timestamp(frame_timestamp_))
    );

    if (!run_status.ok()) {
        LOG(INFO) << "Add Packet error: [" << run_status.message() << "]" << std::endl;
        return nullptr;
    }

    std::vector<Landmark> landmarks;
    mediapipe::Packet packet;

    for (uint i = 0; i < num_outputs_; ++i) {
        {
            absl::MutexLock lock(&out_mutexes_[i]);

            if (out_packets_[i].size() == 0) {
                num_features[i] = 0;
                continue;
            }

            packet = out_packets_[i].back();
            out_packets_[i].clear();
        }

        auto result = parsePacket(packet, outputs_[i].type, num_features + i);

        if (result.size() > 0) {
            landmarks.insert(landmarks.end(), result.begin(), result.end());
        }
    }
    
    return landmarks.data();
}

EffectImpl::~EffectImpl() {
    LOG(INFO) << "Shutting down.";
    absl::Status status = graph_.CloseInputStream(kInputStream);
    if (status.ok()){
    	absl::Status status1 = graph_.WaitUntilDone();
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
    MP_RETURN_IF_ERROR(graph_.Initialize(config));

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(poller_, graph_.AddOutputStreamPoller(output_node));
    MP_RETURN_IF_ERROR(graph_.StartRun({}));

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

    frame_timestamp_++;

    mediapipe::Status run_status = graph_.AddPacketToInputStream(
        kInputStream,
        mediapipe::Adopt(input_frame_for_input.release()).At(mediapipe::Timestamp(frame_timestamp_))
    );

    if (!run_status.ok()) {
        LOG(INFO) << "Add Packet error: [" << run_status.message() << "]" << std::endl;
        return nullptr;
    }

    mediapipe::Packet packet;
    if (!poller_->Next(&packet)){
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
