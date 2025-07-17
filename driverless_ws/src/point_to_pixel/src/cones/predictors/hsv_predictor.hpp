#pragma once

#include "general_predictor.hpp"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace point_to_pixel
{
    class HSVPredictor : public GeneralPredictor
    {
    public:
        /**
         * @brief constructor
         * 
         * @param yellow_filter_low min clipping val for yellow filter
         * @param yellow_filter_high max clipping val for yellow filter
         * @param blue_filter_low min clipping val for blue filter
         * @param blue_filter_high max clipping val for blue filter
         * @param orange_filter_low min clipping val for orange filter
         * @param orange_filter_high max clipping val for orange filter
         * @return HSVPredictor object
         */
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
        
        /**
         * @brief applies HSV predictor on images and classifies pixels based on the colors (in HSV) of the pixels around.
         * 
         * @param pixel_pair pair of two pixels (left and right image) representing a cone centroid transformed to image space.
         * @param frame_pair pair of image frames
         * @param confidence_threshold YOLO min confidence threshold for classification
         * @return ConeClass of the pixel_pair representing a cone centroid. ORANGE == 0; YELLOW == 1; BLUE == 2; UNKNOWN == -1
         */
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

        /**
         * @brief uses filters to create pixel masks for yellow, blue, and orange. If the pixel is in an area 
         * of high average of one color, it is classified as such
         * 
         * @param pixel pixel corresponding to cone_centroid
         * @param img image matrix
         * @param confidence_threshold min confidence threshold for classification
         * @return pair of ConeClass and confidence of the pixel
         */
        std::pair<ConeClass, double> get_color(
            Eigen::Vector3d &pixel,
            cv::Mat img,
            double confidence_threshold);
    };
} // namespace point_to_pixel