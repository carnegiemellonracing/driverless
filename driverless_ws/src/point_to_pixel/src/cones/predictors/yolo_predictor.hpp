#pragma once

#include "general_predictor.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace point_to_pixel
{
    class YoloPredictor : public GeneralPredictor
    {
    public:
        /**
         * @brief constructor
         * 
         * @param model_path string path to .onnx file
         * @return YoloPredictor object
         */
        YoloPredictor(
            const std::string &model_path
        ) {
            net_ = init_model(model_path);
        }

        /**
         * @brief applies YOLO model on images and classifies pixels based on the bounding box classes they fall in. If 
         * a pixel fall in multiple bounding boxes, a heuristic is applied that compares pixel depth and bounding box area.
         * The idea is that the smaller the bounding box, the greater the depth of the cone it corresponds to.
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

        // Additional YOLO-specific methods

        /**
         * @brief applies YOLO model to image
         * 
         * @param frame camera frame
         * @param is_left_frame boolean flag that decides which instance variable to save YOLO output to
         */
        void process_frame(const cv::Mat &frame, bool is_left_frame);

        /**
         * @brief uses cv2 to draw filled bounding boxes on the frames for visualization
         * 
         * @param frame camera frame
         * @param canvas canvas frame
         * @param detections output from YOLO model
         * @param rows rows of detection matrix to parse detections
         * @param cols cols of detection matrix to parse detections
         * @param confidence_threshold YOLO min confidence threshold for classification
         */
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
        
        /**
         * @brief parse YOLO detections on a frame and return the most likely cone color given a pixel in frame
         * 
         * @param pixel pixel corresponding to cone_centroid
         * @param detections output from YOLO model
         * @param rows cols of detection matrix to parse detections
         * @param cols rows of detection matrix to parse detections
         * @param confidence_threshold YOLO min confidence threshold for classification
         * @return pair of ConeClass and confidence of the pixel
         */
        std::pair<ConeClass, double> get_color(
            Eigen::Vector3d &pixel,
            std::vector<cv::Mat> detections,
            int rows,
            int cols,
            double confidence_threshold);
        
        /**
         * @brief depth heuristic implementation to classify cone centroids that appear in multiple bounding boxes
         * 
         * @param depth depth of cone centroids
         * @return estimated box height of cone
         */
        float depth_to_box_height(float depth);
        cv::dnn::Net init_model(const std::string &model_path);
    };
} // namespace point_to_pixel