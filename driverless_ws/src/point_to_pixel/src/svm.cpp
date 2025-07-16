#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cassert>
#include <chrono>
#include <thread>
#include <cstring>

#include "../include/svm.hpp"

namespace point_to_pixel {
namespace recoloring {
    double node_predictor(const std::vector<double> &cone, const svm_model *model) {
        svm_node *node = new svm_node[cone.size() + 1];
        for (size_t i = 0; i < cone.size(); ++i) {
            node[i].index = i + 1;
            node[i].value = cone[i];
        }
        node[cone.size()].index = -1;
        double value = svm_predict(model, node);
        delete[] node;
        return value;
    }
    
    TrackBounds recolor_cones(TrackBounds track_bounds, double C) {
        auto total_start = std::chrono::high_resolution_clock::now();
    
        // check if there are no blue or yellow cones
        int original_blue_cone_count = track_bounds.blue.size();
        int original_yellow_cone_count = track_bounds.yellow.size();
    
        if (track_bounds.blue.empty() && track_bounds.yellow.empty()) {
            return TrackBounds();
        }
    
        auto prep_start = std::chrono::high_resolution_clock::now();
        
        
        // augment dataset to make it better for SVM training
        TrackBounds augmented_cones = track_bounds;
        supplement_cones(augmented_cones);
    
        // acquire the feature matrix and label vector
        std::pair<std::vector<std::vector<double>>, std::vector<double>> xy = cones_to_xy(augmented_cones);
        std::vector<std::vector<double>> X = xy.first;
        std::vector<double> y = xy.second;
    
        auto prep_end = std::chrono::high_resolution_clock::now();
        auto prep_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prep_end - prep_start);
    
        // SVM setup timing
        auto setup_start = std::chrono::high_resolution_clock::now();
    
        // prepare SVM data
        svm_problem prob;
        prob.l = X.size();               // number of training examples
        prob.y = new double[prob.l];     // labels
        prob.x = new svm_node *[prob.l]; // feature vectors
    
        for (int i = 0; i < prob.l; ++i) {
            prob.y[i] = y[i]; // set the label for each example
    
            // create the feature vector
            prob.x[i] = new svm_node[X[i].size() + 1]; // +1 for the end marker
            for (size_t j = 0; j < X[i].size(); ++j) {
                prob.x[i][j].index = j + 1; // 1-based indexing for libSVM
                prob.x[i][j].value = X[i][j];
            }
            prob.x[i][X[i].size()].index = -1; // End marker
        }
    
        /* this section of code used to SVM parameters to match scikit-learn parameters */
        // first, compute mean of all elements in X
        double sum_all = 0.0;
    
        int N = static_cast<int>(X.size());
        if (N == 0 || X[0].empty()) {
            std::cerr << "No training data available for SVM.\n";
            return TrackBounds();
        }
        int d = static_cast<int>(X[0].size());
    
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d; ++j) {
                sum_all += X[i][j];
            }
        }
        double mean_all = sum_all / (N * d);
    
        // then, compute variance over all elements in X
        double sum_var = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d; ++j) {
                double diff = X[i][j] - mean_all;
                sum_var += diff * diff;
            }
        }
        double var_all = sum_var / (N * d);
    
        // calculate gamma_scale to match scikit-learn
        double gamma_scale = 1.0 / (d * var_all);
    
        // set up svm
        svm_parameter param;
        memset(&param, 0, sizeof(param));
        param.svm_type = C_SVC;
        param.kernel_type = POLY;
        param.degree = 3;
        param.C = C; // represents trade-off between maximizing margin and minimizing misclassification [Lower = more margin, more misclassification (good for recoloring)]
        param.coef0 = 1.0;
        param.gamma = gamma_scale;
        param.cache_size = 200;
        param.eps = 0.001;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = NULL;
        param.weight = NULL;
    
        const char *error_msg = svm_check_parameter(&prob, &param);
        if (error_msg) {
            std::cerr << "Error in SVM parameters: " << error_msg << std::endl;
            return TrackBounds();
        }
    
        auto setup_end = std::chrono::high_resolution_clock::now();
        auto setup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start);
    
        // SVM training timing
        auto train_start = std::chrono::high_resolution_clock::now();
    
        // train the SVM model
        svm_model *model = svm_train(&prob, &param);
    
        auto train_end = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
    
        // Recoloring timing
        auto recolor_start = std::chrono::high_resolution_clock::now();
        TrackBounds recolored_track_bounds;   

        // Process original blue cones
        for (const auto& cone : track_bounds.blue) {
            // Use SVM to predict the class (blue=1.0, yellow=0.0)
            double predicted_class = node_predictor({cone.point.x, cone.point.y}, model);
            
            if (predicted_class == 1.0) {  // Blue
                recolored_track_bounds.blue.push_back(cone);
            } else if (predicted_class == 0.0) {  // Yellow
                recolored_track_bounds.yellow.push_back(cone);
            }
        }
        
        // Process original yellow cones
        for (const auto& cone : track_bounds.yellow) {
            // Use SVM to predict the class (blue=1.0, yellow=0.0)
            double predicted_class = node_predictor({cone.point.x, cone.point.y}, model);

            if (predicted_class == 1.0) {  // Blue
                recolored_track_bounds.blue.push_back(cone);
            } else if (predicted_class == 0.0) {  // Yellow
                recolored_track_bounds.yellow.push_back(cone);
            }
        }
        
        auto recolor_end = std::chrono::high_resolution_clock::now();
        auto recolor_ms = std::chrono::duration_cast<std::chrono::milliseconds>(recolor_end - recolor_start);
    
        // free allocated memory
        for (int i = 0; i < prob.l; ++i) {
            delete[] prob.x[i];
        }
        delete[] prob.x;
        delete[] prob.y;
    
        svm_free_and_destroy_model(&model);
    
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
        std::cout << "\n=== SVM Recoloring Timing ===\n";
        std::cout << "Original blue cones: " << original_blue_cone_count 
                    << ", yellow cones: " << original_yellow_cone_count << std::endl;
        std::cout << "Recolored blue cones: " << recolored_track_bounds.blue.size() 
                    << ", yellow cones: " << recolored_track_bounds.yellow.size() << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Data preparation: " << prep_ms.count() << " ms (" 
                    << (prep_ms.count() * 100.0 / total_ms.count()) << "%)\n";
        std::cout << "SVM setup:       " << setup_ms.count() << " ms (" 
                    << (setup_ms.count() * 100.0 / total_ms.count()) << "%)\n";
        std::cout << "Training:        " << train_duration.count() << " ms (" 
                    << (train_duration.count() * 100.0 / total_ms.count()) << "%)\n";
        std::cout << "Recoloring:      " << recolor_ms.count() << " ms (" 
                    << (recolor_ms.count() * 100.0 / total_ms.count()) << "%)\n";
        std::cout << "Total execution: " << total_ms.count() << " ms (100%)\n";
        std::cout << "===========================\n\n";
    
        return recolored_track_bounds;
    }
} // namespace recoloring
} // namespace point_to_pixel