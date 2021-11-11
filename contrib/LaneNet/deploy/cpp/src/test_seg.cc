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
#include "lane_postprocess.hpp"

using namespace std;
using namespace cv;

DEFINE_string(model_dir, "/Users/huangshenghui/PP/PaddleSeg/contrib/LaneNet/output", "Directory of the inference model. "
                             "It constains deploy.yaml and infer models");
DEFINE_string(img_path, "/Users/huangshenghui/PP/PaddleSeg/contrib/LaneNet/data/test_images/0.jpg", "Path of the test image.");
DEFINE_bool(use_cpu, false, "Wether use CPU. Default: use GPU.");

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

std::shared_ptr<paddle_infer::Predictor> create_predictor(
    const YamlConfig& yaml_config) {
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

void process_image(const YamlConfig& yaml_config, cv::Mat& img) {
    if (yaml_config.is_resize) {
        cv::resize(img, img, cv::Size(yaml_config.resize_width, yaml_config.resize_height));
    }
    if (yaml_config.is_normalize) {
        img.convertTo(img, CV_32F, 1.0 / 255, 0);
        img = (img - 0.5) / 0.5;
    }
}

int main(int argc, char *argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model_dir == "") {
    LOG(FATAL) << "The model_dir should not be empty.";
    }
    
    LanePostProcess* laneNet = new LanePostProcess();

    // Load yaml
    std::string yaml_path = FLAGS_model_dir + "/deploy.yaml";
    YamlConfig yaml_config = load_yaml(yaml_path);

    // Prepare data
    cv::Mat img = cv::imread(FLAGS_img_path, cv::IMREAD_COLOR);
    cv::Mat image_ori = img.clone();
    int cut_height = 160;
    int input_width = img.cols;
    int input_height = img.rows;
    cv::Rect roi = {0, cut_height, input_width, input_height-cut_height};

    img = img(roi);
    process_image(yaml_config, img);
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
    auto output_t = predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());
    std::vector<float> out_data(out_num);
    output_t->CopyToCpu(out_data.data());
    
    cv::Size size = cv::Size(cols, rows);
    int skip_index = size.height * size.width;
    
    auto lane_coords = laneNet->lane_process(7,  size ,out_num,  skip_index);
    cv::Scalar colors[] = {cv::Scalar(255,0,0),cv::Scalar(0,255,0),
        cv::Scalar(0,0,255), cv::Scalar(255,255,0),
        cv::Scalar(255,0,255), cv::Scalar(0,255,255),
        cv::Scalar(50, 100, 50), cv::Scalar(100, 50, 100)};
    int lane_id = 0;
    for(auto& coords: lane_coords) {
        for(auto& coord: coords) {
            int x = coord.first;
            int y = coord.second;
            if (x <= 0 || y <= 0)
                continue;
            cv::circle(image_ori, cv::Point(x, y), 4, colors[lane_id % 6], 2);
        }
        lane_id++;
    }
    cv::imshow("ttt", image_ori);
    cv::waitKey();
    return;
    const int num_class = 7;
    cv::Mat seg_result;
    seg_result.create(size, CV_32FC(num_class));

    cv::Mat ins_planes[num_class];
    for(int i = 0; i < num_class; i++) {
       ins_planes[i].create(size, CV_32FC(1));
    }

    for(int i = 0; i < num_class; i++) {
        ::memcpy(ins_planes[i].data, out_data.data() + i*skip_index, skip_index * sizeof(float)); //内存拷贝
    }
    cv::merge(ins_planes, num_class, seg_result);
    
    cv::Size output_size = cv::Size(1280, 560);
    int output_nums = output_size.width * output_size.height;
  
    cv::Mat image_final;
    cv::resize(seg_result, image_final, output_size);

    cv::Mat binary_image=cv::Mat::zeros(output_size, CV_8UC1);
    vector<int> result(output_nums, 0);
    int i = 0;
    for (int y = 0; y < image_final.rows; y++){
        for (int x = 0; x< image_final.cols; x++) {
            vector<float> tmp(num_class, 0);
            for (int idx = 0; idx < num_class; idx++) {
                tmp[idx] = image_final.at<cv::Vec<float, num_class>>(y,x)[idx];
            }
            int aa = max_element(tmp.begin(),tmp.end()) - tmp.begin();
            binary_image.at<uchar>(y,x)= aa;
            result[i++] = aa;
        }
    }

    cv::imshow("test", binary_image * 255);
    cv::waitKey();

    // Get pseudo image
    std::vector<uint8_t> out_data_u8(output_nums);
    for (int i = 0; i < output_nums; i++) {
        out_data_u8[i] = static_cast<uint8_t>(result[i]);
    }
    cv::Mat out_gray_img(output_size, CV_8UC1, out_data_u8.data());
    cv::Mat out_eq_img;
    cv::equalizeHist(out_gray_img, out_eq_img);
    cv::imwrite("out_img.jpg", out_eq_img);

    LOG(INFO) << "Finish";
}