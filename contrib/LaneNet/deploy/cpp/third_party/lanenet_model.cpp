/************************************************
* Copyright 2019 Baidu Inc. All Rights Reserved.
* Author: MaybeShewill-CV
* File: lanenetModel.cpp
* Date: 2019/11/5 下午5:19
************************************************/

#include "lanenet_model.h"
#include <glog/logging.h>
#include "dbscan.hpp"
#include <numeric>

namespace beec_task {
namespace lane_detection {

/******************Public Function Sets***************/

/***
 * Constructor. Using config file to setup lanenet model. Mainly defined object are as follows:
 * 1.Init mnn model file path
 * 2.Init lanenet model pixel embedding feature dims
 * 3.Init dbscan cluster search radius eps threshold
 * 4.Init dbscan cluster min pts which are supposed to belong to a core object.
 * @param config
 */
LaneNet::LaneNet(cv::Size sz) {
    _m_dbscan_eps = 0.4;
    _m_dbscan_min_pts = 500;
    _m_input_node_size_host = sz;

    return;
}

/***
 * Destructor
 */
LaneNet::~LaneNet() {
}

/***
 * Detect lanes on image using lanenet model
 * @param binary_seg_result
 * @param pix_embedding_result
 */
void LaneNet::detect(cv::Mat &binary_seg_result, cv::Mat &instance_seg_result, cv::Mat &mask) {
    // gather pixel embedding features
    std::vector<cv::Point> coords;
    std::vector<DBSCAMSample> pixel_embedding_samples;
    gather_pixel_embedding_features(binary_seg_result, instance_seg_result,coords, pixel_embedding_samples);

    // simultaneously random shuffle embedding vector and coord vector inplace
    simultaneously_random_shuffle<cv::Point, DBSCAMSample >(coords, pixel_embedding_samples);

    // normalize pixel embedding features
    normalize_sample_features(pixel_embedding_samples, pixel_embedding_samples);

    // cluster samples
    std::vector<std::vector<uint> > cluster_ret;
    std::vector<uint> noise;
    {
        cluster_pixem_embedding_features(pixel_embedding_samples, cluster_ret, noise);
    }

    // visualize instance segmentation
    mask = cv::Mat(_m_input_node_size_host, CV_8UC3, cv::Scalar(0, 0, 0));
    visualize_instance_segmentation_result(cluster_ret, coords, mask);
}

/***************Private Function Sets*******************/
/***
 * Gather pixel embedding features via binary segmentation result
 * @param binary_mask
 * @param pixel_embedding
 * @param coords
 * @param embedding_features
 */
void LaneNet::gather_pixel_embedding_features(const cv::Mat &binary_mask, const cv::Mat &pixel_embedding,
        std::vector<cv::Point> &coords,
        std::vector<DBSCAMSample> &embedding_samples) {

    CHECK_EQ(binary_mask.size(), pixel_embedding.size());
    auto image_rows = _m_input_node_size_host.height;
    auto image_cols = _m_input_node_size_host.width;

    for (auto row = 0; row < image_rows; ++row) {
        auto binary_image_row_data = binary_mask.ptr<uchar>(row);
        auto embedding_image_row_data = pixel_embedding.ptr<cv::Vec4f>(row);
        for (auto col = 0; col < image_cols; ++col) {
            auto binary_image_pix_value = binary_image_row_data[col];
            if (binary_image_pix_value == 255) {
                coords.emplace_back(cv::Point(col, row));
                Feature embedding_features;
                for (auto index = 0; index < 4; ++index) {
                    embedding_features.push_back(embedding_image_row_data[col][index]);
                }
                DBSCAMSample sample(embedding_features, CLASSIFY_FLAGS::NOT_CALSSIFIED);
                embedding_samples.push_back(sample);
            }
        }
    }
}

/***
 *
 * @param embedding_samples
 * @param cluster_ret
 */
void LaneNet::cluster_pixem_embedding_features(std::vector<DBSCAMSample> &embedding_samples,
        std::vector<std::vector<uint> > &cluster_ret, std::vector<uint>& noise) {

    if (embedding_samples.empty()) {
        LOG(INFO) << "Pixel embedding samples empty";
        return;
    }

    // dbscan cluster
    auto dbscan = DBSCAN<DBSCAMSample, float>();
    dbscan.Run(&embedding_samples, _m_lanenet_pix_embedding_feature_dims, _m_dbscan_eps, _m_dbscan_min_pts);
    cluster_ret = dbscan.Clusters;
    noise = dbscan.Noise;
}

/***
 * Visualize instance segmentation result
 * @param cluster_ret
 * @param coords
 */
void LaneNet::visualize_instance_segmentation_result(
    const std::vector<std::vector<uint> > &cluster_ret,
    const std::vector<cv::Point> &coords,
    cv::Mat& intance_segmentation_result) {

    LOG(INFO) << "Cluster nums: " << cluster_ret.size();

    std::map<int, cv::Scalar> color_map = {
        {0, cv::Scalar(0, 0, 255)},
        {1, cv::Scalar(0, 255, 0)},
        {2, cv::Scalar(255, 0, 0)},
        {3, cv::Scalar(255, 0, 255)},
        {4, cv::Scalar(0, 255, 255)},
        {5, cv::Scalar(255, 255, 0)},
        {6, cv::Scalar(125, 0, 125)},
        {7, cv::Scalar(0, 125, 125)}
    };

//    omp_set_num_threads(4);
    for (int class_id = 0; class_id < cluster_ret.size(); ++class_id) {
        auto class_color = color_map[class_id];
        #pragma omp parallel for
        for (auto index = 0; index < cluster_ret[class_id].size(); ++index) {
            auto coord = coords[cluster_ret[class_id][index]];
            auto image_col_data = intance_segmentation_result.ptr<cv::Vec3b>(coord.y);
            image_col_data[coord.x][0] = class_color[0];
            image_col_data[coord.x][1] = class_color[1];
            image_col_data[coord.x][2] = class_color[2];
        }
    }
}

/***
 * Calculate the mean feature vector among a vector of DBSCAMSample samples
 * @param input_samples
 * @return
 */
Feature LaneNet::calculate_mean_feature_vector(const std::vector<DBSCAMSample> &input_samples) {

    if (input_samples.empty()) {
        return Feature();
    }

    auto feature_dims = input_samples[0].get_feature_vector().size();
    auto sample_nums = input_samples.size();
    Feature mean_feature_vec;
    mean_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (auto index = 0; index < feature_dims; ++index) {
            mean_feature_vec[index] += sample[index];
        }
    }
    for (auto index = 0; index < feature_dims; ++index) {
        mean_feature_vec[index] /= sample_nums;
    }

    return mean_feature_vec;
}

/***
 *
 * @param input_samples
 * @param mean_feature_vec
 * @return
 */
Feature LaneNet::calculate_stddev_feature_vector(
        const std::vector<DBSCAMSample> &input_samples,
        const Feature& mean_feature_vec) {

    if (input_samples.empty()) {
        return Feature();
    }

    auto feature_dims = input_samples[0].get_feature_vector().size();
    auto sample_nums = input_samples.size();

    // calculate stddev feature vector
    Feature stddev_feature_vec;
    stddev_feature_vec.resize(feature_dims, 0.0);
    for (const auto& sample : input_samples) {
        for (auto index = 0; index < feature_dims; ++index) {
            auto sample_feature = sample.get_feature_vector();
            auto diff = sample_feature[index] - mean_feature_vec[index];
            diff = std::pow(diff, 2);
            stddev_feature_vec[index] += diff;
        }
    }
    for (auto index = 0; index < feature_dims; ++index) {
        stddev_feature_vec[index] /= sample_nums;
        stddev_feature_vec[index] = std::sqrt(stddev_feature_vec[index]);
    }

    return stddev_feature_vec;
}

/***
 * Normalize input samples' feature. Each sample's feature is normalized via function as follows:
 * feature[i] = (feature[i] - mean_feature_vector[i]) / stddev_feature_vector[i].
 * @param input_samples
 * @param output_samples
 */
void LaneNet::normalize_sample_features(const std::vector<DBSCAMSample> &input_samples,
                                        std::vector<DBSCAMSample> &output_samples) {
    // calcualte mean feature vector
    Feature mean_feature_vector = calculate_mean_feature_vector(input_samples);

    // calculate stddev feature vector
    Feature stddev_feature_vector = calculate_stddev_feature_vector(input_samples, mean_feature_vector);

    std::vector<DBSCAMSample> input_samples_copy = input_samples;
    for (auto& sample : input_samples_copy) {
        auto feature = sample.get_feature_vector();
        for (auto index = 0; index < feature.size(); ++index) {
            feature[index] = (feature[index] - mean_feature_vector[index]) / stddev_feature_vector[index];
        }
        sample.set_feature_vector(feature);
    }
    output_samples = input_samples_copy;
}

/***
 * simultaneously random shuffle two vector inplace. The two input source vector should have the same size.
 * @tparam T
 * @param src1
 * @param src2
 */
template <typename T1, typename T2>
void LaneNet::simultaneously_random_shuffle(std::vector<T1> src1, std::vector<T2> src2) {

    CHECK_EQ(src1.size(), src2.size());
    if (src1.empty() || src2.empty()) {
        return;
    }

    // construct index vector of two input src
    std::vector<uint> indexes;
    indexes.reserve(src1.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::random_shuffle(indexes.begin(), indexes.end());

    // make copy of two input vector
    std::vector<T1> src1_copy(src1);
    std::vector<T2> src2_copy(src2);

    // random two source input vector via random shuffled index vector
    for (uint i = 0; i < indexes.size(); ++i) {
        src1[i] = src1_copy[indexes[i]];
        src2[i] = src2_copy[indexes[i]];
    }
}

}
}
