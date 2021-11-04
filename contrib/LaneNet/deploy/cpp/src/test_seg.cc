#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/include/paddle_inference_api.h"
#include "yaml-cpp/yaml.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include "lanenet_model.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
using namespace std;
using namespace chrono;


DEFINE_string(model_dir, "/Users/huangshenghui/PP/PaddleSeg/contrib/LaneNet/output/", "Directory of the inference model. "
                             "It constains deploy.yaml and infer models");
DEFINE_string(img_path, "/Users/huangshenghui/PP/PaddleSeg/contrib/LaneNet/data/test_images/3.jpg", "Path of the test image.");
DEFINE_bool(use_cpu, true, "Wether use CPU. Default: use GPU.");
DEFINE_string(save_dir, "", "Directory of the output image.");

typedef struct YamlConfig {
  std::string model_file;
  std::string params_file;
  bool is_normalize;
}YamlConfig;

YamlConfig load_yaml(const std::string& yaml_path) {
  YAML::Node node = YAML::LoadFile(yaml_path);
  std::string model_file = node["Deploy"]["model"].as<std::string>();
  std::string params_file = node["Deploy"]["params"].as<std::string>();
  bool is_normalize = false;
  if (node["Deploy"]["transforms"] &&
    node["Deploy"]["transforms"][0]["type"].as<std::string>() == "Normalize") {
      is_normalize = true;
  }

  YamlConfig yaml_config = {model_file, params_file, is_normalize};
  return yaml_config;
}

std::shared_ptr<paddle_infer::Predictor> create_predictor(const YamlConfig& yaml_config) {
  std::string& model_dir = FLAGS_model_dir;

  paddle_infer::Config infer_config;
  infer_config.SetModel(model_dir + "/" + yaml_config.model_file,
                  model_dir + "/" + yaml_config.params_file);
  infer_config.EnableMemoryOptim();

  auto predictor = paddle_infer::CreatePredictor(infer_config);
  return predictor;
}

void hwc_img_2_chw_data(const cv::Mat& hwc_img, float* data) {
  int rows = hwc_img.rows;
  int cols = hwc_img.cols;
  int chs = hwc_img.channels();
  for (int i = 0; i < chs; ++i) {
    cv::extractChannel(hwc_img, cv::Mat(rows, cols, CV_32FC1, data + i * rows * cols), i);
  }
}

cv::Mat read_process_image(bool is_normalize) {
    cv::Mat img = cv::imread(FLAGS_img_path, cv::IMREAD_COLOR);
    cv::Mat seg_resized;

    cv::resize(img, seg_resized, cv::Size(512, 256), cv::INTER_NEAREST);
    seg_resized.convertTo(seg_resized, CV_32F, 1.0 / 255, 0);
    seg_resized = (seg_resized - 0.5) / 0.5;

    return seg_resized;
}

int main(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model_dir == "") {
    LOG(FATAL) << "The model_dir should not be empty.";
    }
    
    // Load yaml
    std::string yaml_path = FLAGS_model_dir + "/deploy.yaml";
    YamlConfig yaml_config = load_yaml(yaml_path);
    
    // Create predictor
    auto predictor = create_predictor(yaml_config);

    int rows = 256;
    int cols = 512;
    int chs = 3;
    
    // Set input
    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputHandle(input_names[0]);
    std::vector<int> input_shape = {1, chs, rows, cols};
    input_t->Reshape(input_shape);
   
    std::vector<float> input_data(1 * chs * rows * cols, 0.0f);
    std::vector<float> out_seg_data(1 * chs * rows * cols * 2);
    std::vector<float> out_emb_data(1 * chs * rows * cols * 4);
    
    beec_task::lane_detection::LaneNet* laneNet = new beec_task::lane_detection::LaneNet(cv::Size(cols, rows));

    string path = "/Users/huangshenghui/PP/PaddleSeg/contrib/LaneNet/data/tusimple_lane_detection/training/gt_image/0000.png";
    cv::Mat image_origin;
    cv::Mat image_final;
    image_origin = cv::imread(path.c_str(), cv::IMREAD_COLOR);
    
    // Prepare data
    cv::resize(image_origin, image_final, cv::Size(cols, rows));
    
    image_final.convertTo(image_final, CV_32F, 1.0 / 255, 0);
    image_final = (image_final - 0.5) / 0.5;
    cv::Mat img = read_process_image(yaml_config.is_normalize);
  
    hwc_img_2_chw_data(image_final, input_data.data());
    input_t->CopyFromCpu(input_data.data());

    // Run
    predictor->Run();
    
    // Set output
    auto output_names = predictor->GetOutputNames();
    auto output_seg_handle = predictor->GetOutputHandle(output_names[0]);
    auto output_emb_handle = predictor->GetOutputHandle(output_names[1]);
    
    cv::Size size = cv::Size(cols, rows);
    int skip_index = size.height * size.width;
    cv::Mat ins_tmp;
    ins_tmp.create(size.height,size.width, CV_32FC(4));
    cv::Mat seg_planes[2];
    for(int i = 0; i < 2; i++) {
        seg_planes[i].create(size.height, size.width, CV_32FC(1));
    }

    cv::Mat ins_planes[4];
    for(int i = 0; i < 4; i++) {
       ins_planes[i].create(size.height, size.width, CV_32FC(1));
    }
    
    cv::Mat binary_image = cv::Mat::zeros(size.height, size.width, CV_8UC1);

    // Get output
    output_seg_handle->CopyToCpu(out_seg_data.data());
    output_emb_handle->CopyToCpu(out_emb_data.data());

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float xx1 = out_seg_data[row * cols + col];
            float xx2 = out_seg_data[row * cols + rows * cols + col];
            binary_image.at<uchar>(row,col)  = xx1 > xx2 ? 0 : 255;
        }
    }

    if (out_seg_data.size() > 0) {
        for(int i = 0; i < 4; i++) {
            ::memcpy(ins_planes[i].data, out_emb_data.data() + i*skip_index, skip_index * sizeof(float)); //内存拷贝
        }
        cv::merge(ins_planes, 4, ins_tmp);
    }

    cv::Mat mask;
    laneNet->detect(binary_image, ins_tmp, mask);
    cv::imshow("mask", mask);
    cv::waitKey();
}

