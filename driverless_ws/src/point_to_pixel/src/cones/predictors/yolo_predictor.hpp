#pragma once

#include "general_predictor.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace point_to_pixel
{
    class YoloPredictor : public general_predictor
    {
    public:
        YoloPredictor(
            const std::string &model_path
        ) {
            net_ = init_model(model_path);
        }

        ConeClass predict_color(
            std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
            std::pair<cv::Mat, cv::Mat> frame_pair,
            double confidence_threshold) override;

        // Additional YOLO-specific methods
        void process_frame(const cv::Mat &frame, bool is_left_frame);
        void draw_bounding_boxes(
            cv::Mat &frame,
            cv::Mat &canvas,
            std::vector<cv::Mat> detections,
            int rows,
            int cols,
            double confidence_threshold);

    private:
        cv::dnn::Net net_;
        int yolo_width_ = 640;
        int yolo_height_ = 640;
        std::vector<cv::Mat> detections_l;
        std::vector<cv::Mat> detections_r;

        std::pair<ConeClass, double> get_color(
            Eigen::Vector3d &pixel,
            std::vector<cv::Mat> detections,
            int cols,
            int rows,
            double confidence_threshold);

        float depth_to_box_height(float depth);
        cv::dnn::Net init_model(const std::string &model_path);
    };
} // namespace point_to_pixel