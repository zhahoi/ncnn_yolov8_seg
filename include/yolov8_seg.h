// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLOV8_SEG_H
#define YOLOV8_SEG_H

#include <opencv2/core/core.hpp>
#include <benchmark.h>
#include <net.h>

#define PARAM_PATH "/home/hit/Project/yolov8Seg/weights/yolov8s-seg-sim-opt-fp16.param"
#define BIN_PATH "/home/hit/Project/yolov8Seg/weights/yolov8s-seg-sim-opt-fp16.bin"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_feat;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class Yolo
{
public:
    Yolo();

    ~Yolo();

    int detect(const cv::Mat& rgb, std::vector<Object>& objects);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

    int draw_unsupported(cv::Mat& rgb);

    int draw_fps(cv::Mat& rgb);

private:
    ncnn::Net yolo;

    const int target_size = 320;
    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    const float prob_threshold = 0.4f;
    const float nms_threshold = 0.5f;
    const bool use_gpu = false;

    int image_w;
    int image_h;
    int in_w;
    int in_h;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // YOLOV8_SEG_H
