#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <utility>

namespace point_to_pixel
{
    enum class ConeClass
    {
        ORANGE, // 0
        YELLOW, // 1
        BLUE,   // 2
        UNKNOWN // -1
    };

    class GeneralPredictor
    {
    public:

        /**
         * @brief default destructor
         */
        virtual ~GeneralPredictor() = default;

        /**
         * @param pixel_pair 
         * @param frame_pair
         * @param confidence_threshold
         */
        virtual ConeClass predict_color(
            std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
            std::pair<cv::Mat, cv::Mat> frame_pair,
            double confidence_threshold) = 0;
    };
} // namespace point_to_pixel