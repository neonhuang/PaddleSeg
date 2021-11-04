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
#include "lanenet_model.h"


DEFINE_string(model_dir, "", "Directory of the inference model. "
                             "It constains deploy.yaml and infer models");
DEFINE_string(img_path, "", "Path of the test image.");
DEFINE_string(save_dir, "", "Directory of the output image.");

typedef struct YamlConfig {
  std::string model_file;
  std::string params_file;
  bool is_normalize;
  bool is_resize;
  int resize_width;
  int resize_height;
}YamlConfig;

YamlConfig load_yaml(const std::string& yaml_path) {
  YAML::Node node = YAML::LoadFile(yaml_path);
  std::string model_file = node["Deploy"]["model"].as<std::string>();
  std::string params_file = node["Deploy"]["params"].as<std::string>();
  YamlConfig yaml_config = {model_file, params_file};
  if (node["Deploy"]["transforms"]) {
    const YAML::Node& transforms = node["Deploy"]["transforms"];
    for (size_t i = 0; i < transforms.size(); i++) {
      if (transforms[i]["type"].as<std::string>() == "Normalize") {
        yaml_config.is_normalize = true;
      } else if (transforms[i]["type"].as<std::string>() == "Resize") {
        yaml_config.is_resize = true;
        const YAML::Node& target_size = transforms[i]["target_size"];
        yaml_config.resize_width = target_size[0].as<int>();
        yaml_config.resize_height = target_size[1].as<int>();
      }
    }
  }
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

cv::Mat read_process_image(const YamlConfig& yaml_config) {
  cv::Mat img = cv::imread(FLAGS_img_path, cv::IMREAD_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  if (yaml_config.is_resize) {
    cv::resize(img, img, cv::Size(yaml_config.resize_width, yaml_config.resize_height));
  }
  if (yaml_config.is_normalize) {
    img.convertTo(img, CV_32F, 1.0 / 255, 0);
    img = (img - 0.5) / 0.5;
  }
  return img;
}

int main(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model_dir == "") {
      LOG(FATAL) << "The model_dir should not be empty.";
    }

    // Load yaml
    std::string yaml_path = FLAGS_model_dir + "/deploy.yaml";
    YamlConfig yaml_config = load_yaml(yaml_path);

    // Prepare data
    cv::Mat img = read_process_image(yaml_config);
    int rows = img.rows;
    int cols = img.cols;
    int chs = img.channels();
    std::vector<float> input_data(1 * chs * rows * cols, 0.0f);
    hwc_img_2_chw_data(img, input_data.data());

    // Create predictor
    auto predictor = create_predictor(yaml_config);

    // Set input
    auto input_names = predictor->GetInputNames();
    auto input_t = predictor->GetInputHandle(input_names[0]);
    std::vector<int> input_shape = {1, chs, rows, cols};
    input_t->Reshape(input_shape);
    input_t->CopyFromCpu(input_data.data());

    // Run
    predictor->Run();
    
    // Get output
    auto output_names = predictor->GetOutputNames();
    auto output_seg_handle = predictor->GetOutputHandle(output_names[0]);
    auto output_emb_handle = predictor->GetOutputHandle(output_names[1]);
    
    std::vector<float> out_seg_data(1 * chs * rows * cols * 2);
    std::vector<float> out_emb_data(1 * chs * rows * cols * 4);
    output_seg_handle->CopyToCpu(out_seg_data.data());
    output_emb_handle->CopyToCpu(out_emb_data.data());

    cv::Size size = cv::Size(cols, rows);
    cv::Mat binary_image = cv::Mat::zeros(size.height, size.width, CV_8UC1);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float xx1 = out_seg_data[row * cols + col];
            float xx2 = out_seg_data[row * cols + rows * cols + col];
            binary_image.at<uchar>(row,col)  = xx1 > xx2 ? 0 : 255;
        }
    }
    
    int skip_index = size.height * size.width;
    cv::Mat ins_tmp;
    ins_tmp.create(size.height,size.width, CV_32FC(4));
    cv::Mat ins_planes[4];
    for(int i = 0; i < 4; i++) {
       ins_planes[i].create(size.height, size.width, CV_32FC(1));
    }
    
    if (out_seg_data.size() > 0) {
        for(int i = 0; i < 4; i++) {
            ::memcpy(ins_planes[i].data, out_emb_data.data() + i*skip_index, skip_index * sizeof(float)); //内存拷贝
        }
        cv::merge(ins_planes, 4, ins_tmp);
    }

    cv::Mat mask;
    beec_task::lane_detection::LaneNet* laneNet = new beec_task::lane_detection::LaneNet(cv::Size(cols, rows));
    laneNet->detect(binary_image, ins_tmp, mask);
    cv::imshow("mask", mask);
    cv::waitKey();
}


