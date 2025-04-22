#include "svm_recoloring.hpp"

namespace cones {
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
    }
}