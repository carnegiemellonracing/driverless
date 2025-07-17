#pragma once

#include "general_predictor.hpp"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace point_to_pixel
{
    class HSVPredictor : public general_predictor
    {
    public:
        HSVPredictor(
            const cv::Scalar &yellow_filter_low,
            const cv::Scalar &yellow_filter_high,
            const cv::Scalar &blue_filter_low,
            const cv::Scalar &blue_filter_high,
            const cv::Scalar &orange_filter_low,
            const cv::Scalar &orange_filter_high) 
          : yellow_filter_low_(yellow_filter_low),
            yellow_filter_high_(yellow_filter_high),
            blue_filter_low_(blue_filter_low),
            blue_filter_high_(blue_filter_high),
            orange_filter_low_(orange_filter_low),
            orange_filter_high_(orange_filter_high) {}

        ConeClass predict_color(
            std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
            std::pair<cv::Mat, cv::Mat> frame_pair,
            double confidence_threshold) override;

    private:
        cv::Scalar yellow_filter_low_;
        cv::Scalar yellow_filter_high_;
        cv::Scalar blue_filter_low_;
        cv::Scalar blue_filter_high_;
        cv::Scalar orange_filter_low_;
        cv::Scalar orange_filter_high_;

        std::pair<ConeClass, double> get_color(
            Eigen::Vector3d &pixel,
            cv::Mat img,
            double confidence_threshold);
    };
} // namespace point_to_pixel