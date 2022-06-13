# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])  # Apache 2.0

# Note: yes, these need to use "//external:android/crosstool", not
# @androidndk//:default_crosstool.

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_x86_64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86_64",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_armeabi",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
    visibility = ["//visibility:public"],
)

# Note: this cannot just match "apple_platform_type": "macos" because that option
# defaults to "macos" even when building on Linux!
alias(
    name = "macos",
    actual = select({
        ":macos_i386": ":macos_i386",
        ":macos_x86_64": ":macos_x86_64",
        "//conditions:default": ":macos_i386",  # Arbitrarily chosen from above.
    }),
    visibility = ["//visibility:public"],
)

# Note: this also matches on crosstool_top so that it does not produce ambiguous
# selectors when used together with "android".
config_setting(
    name = "ios",
    values = {
        "crosstool_top": "@bazel_tools//tools/cpp:toolchain",
        "apple_platform_type": "ios",
    },
    visibility = ["//visibility:public"],
)

alias(
    name = "apple",
    actual = select({
        ":macos": ":macos",
        ":ios": ":ios",
        "//conditions:default": ":ios",  # Arbitrarily chosen from above.
    }),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_i386",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_x86_64",
    },
    visibility = ["//visibility:public"],
)

[
    config_setting(
        name = arch,
        values = {"cpu": arch},
        visibility = ["//visibility:public"],
    )
    for arch in [
        "ios_i386",
        "ios_x86_64",
        "ios_armv7",
        "ios_arm64",
        "ios_arm64e",
    ]
]

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

exports_files(
    ["provisioning_profile.mobileprovision"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mediagraph_lib",
    srcs = [
	    "mediagraph.cc",
        "mediagraph_impl.cc",
    ],
    hdrs = [
        "mediagraph.h",
        "mediagraph_impl.h",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
    deps = [
	    "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        
        "@com_google_absl//absl/flags:parse",

        # Selfie segmentation
        # "//mediapipe/graphs/selfie_segmentation:selfie_segmentation_cpu_deps",

        # Hand tracking
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",

        # Face detection
        "//mediapipe/graphs/face_detection:face_detection_full_range_desktop_live_deps",

        # Face mesh
        "//mediapipe/graphs/face_mesh:desktop_calculators",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",

        # Hair segmentation
        # "//mediapipe/graphs/hair_segmentation:desktop_calculators",

        # Holistic tracking
        # "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps",

        # Iris tracking
        # "//mediapipe/graphs/iris_tracking:iris_tracking_cpu_deps",

        # Object detection
        # "//mediapipe/calculators/core:concatenate_vector_calculator",
        # "//mediapipe/calculators/core:flow_limiter_calculator",
        # "//mediapipe/calculators/core:previous_loopback_calculator",
        # "//mediapipe/calculators/core:split_vector_calculator",
        # "//mediapipe/calculators/image:image_transformation_calculator",
        # "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        # "//mediapipe/calculators/tflite:tflite_converter_calculator",
        # "//mediapipe/calculators/tflite:tflite_inference_calculator",
        # "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        # "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        # "//mediapipe/calculators/util:non_max_suppression_calculator",

        # Pose tracking
        "//mediapipe/graphs/pose_tracking:pose_tracking_cpu_deps",
	],
)

cc_binary(
    name = "mediagraph",
    linkshared = 1,
    deps = [
        ":mediagraph_lib",
    ],
)

cc_binary(
    name = "libmediagraph.dylib",
    linkshared = 1,
    deps = [
        ":mediagraph_lib",
    ],
)
