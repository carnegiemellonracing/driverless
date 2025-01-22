#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <numbers>
#include <utility> 

#include "cones.hpp"
namespace controls {
    namespace midline {
            
        std::string Cones::reprCones(const std::vector<std::vector<double>>& cones) const {
            if (cones.empty()) {
                return "\tNo cones\n";
            }
            std::ostringstream oss;
            for (const auto& cone : cones) {
                oss << "\tx: " << std::fixed << std::setprecision(2)
                    << std::setw(6) << cone[0]
                    << ", y: " << std::setw(6) << cone[1]
                    << ", z: " << std::setw(6) << cone[2] << "\n";
            }
            return oss.str();
        }

        void Cones::addBlueCone(double x, double y, double z) {
            blue_cones.push_back({x, y, z});
        }

        void Cones::addYellowCone(double x, double y, double z) {
            yellow_cones.push_back({x, y, z});
        }

        void Cones::addOrangeCone(double x, double y, double z) {
            orange_cones.push_back({x, y, z});
        }

        void Cones::addMultipleBlue(std::vector<std::vector<double>> blue_list) {
            blue_cones.insert(blue_cones.end(), blue_list.begin(), blue_list.end());
        }

        void Cones::addMultipleYellow(std::vector<std::vector<double>> yellow_list) {
            yellow_cones.insert(yellow_cones.end(), yellow_list.begin(), yellow_list.end());
        }

        void Cones::addCones(const Cones& other) {
            blue_cones.insert(blue_cones.end(), other.blue_cones.begin(), other.blue_cones.end());
            yellow_cones.insert(yellow_cones.end(), other.yellow_cones.begin(), other.yellow_cones.end());
            orange_cones.insert(orange_cones.end(), other.orange_cones.begin(), other.orange_cones.end());
        }

        const std::vector<std::vector<double>>& Cones::getBlueCones() const {
            return blue_cones;
        }

        const std::vector<std::vector<double>>& Cones::getYellowCones() const {
            return yellow_cones;
        }

        void Cones::map(const std::function<std::vector<double>(const std::vector<double>&)>& mapper) {
            auto mapFunc = [&](std::vector<std::vector<double>>& cones) {
                for (auto& cone : cones) {
                    cone = mapper(cone);
                }
            };
            mapFunc(blue_cones);
            mapFunc(yellow_cones);
            mapFunc(orange_cones);
        }

        void Cones::supplementCones() {
            addBlueCone(-2.0, -1.0, 0.0);
            addYellowCone(2.0, -1.0, 0.0);
        }

        std::string Cones::toString() const {
            std::ostringstream oss;
            oss << std::string(20, '-') << "Cones" << std::string(20, '-') << "\n";
            oss << "Blue (" << blue_cones.size() << " cones)\n" << reprCones(blue_cones);
            oss << "Yellow (" << yellow_cones.size() << " cones)\n" << reprCones(yellow_cones);
            oss << "Orange (" << orange_cones.size() << " cones)\n" << reprCones(orange_cones);
            return oss.str();
        }

        size_t Cones::size() const {
            return blue_cones.size() + yellow_cones.size() + orange_cones.size();
        }

        Cones Cones::copy() const {
            Cones copied;
            copied.addCones(*this);
            return copied;
        }

        /* TODO: This code comes from python using from numpy and to numpy
        Use Eigen3 library for any vectors or matrix operations
        Decide whether this function is necessary
        */
        Cones::ConeData Cones::toStruct() const {
            ConeData data;
            data.blue_cones = blue_cones;       
            data.yellow_cones = yellow_cones; 
            data.orange_cones = orange_cones;  
            return data;
        }

        // add circles around the original cones
        std::vector<std::vector<double>> Cones::augmentDatasetCircle(std::vector<std::vector<double>> &X, int deg, double radius) {
            double radian = deg * (M_PI) / 180;

            // build vector of all the angles
            std::vector<double> angles;
            for (double angle = 0; angle < 2 * M_PI; angle += radian) {
                angles.push_back(angle);
            }

            int N = X.size();

            int num_angles = angles.size();
            std::vector<std::vector<double>> X_extra;

            /* Keep this structure instead of two nested for-loops 
            * In case the code is too slow, it's easier to perform vectorization 
            * from this code than with the double for-loops
            */

            // copies X num_angles times to rotate all of the points around a circle
            for (int i = 0; i < num_angles; ++i) {
                X_extra.insert(X_extra.end(), X.begin(), X.end());
            }

            std::vector<double> repeated_angles(num_angles * N);
            for (size_t i = 0; i < num_angles; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    repeated_angles[i * N + j] = angles[i];
                }
            }

            // rotate each point around the circle
            for (size_t i = 0; i < num_angles; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    size_t index = i * N + j;
                    X_extra[index][0] = X[j][0] + radius * std::cos(repeated_angles[index]);
                    X_extra[index][1] = X[j][1] + radius * std::sin(repeated_angles[index]);
                }
            }


            // add the original cones back to the dataset
            X.insert(X.end(), X_extra.begin(), X_extra.end());

            return X;
        }

        // TODO: Decide whether to remove this function, getter functions have been built
        Cones Cones::fromStruct(const ConeData& data) {
            Cones newCones;

            for (const auto& cone : data.blue_cones) {
                newCones.addBlueCone(cone[0], cone[1], cone[2]);
            }
            for (const auto& cone : data.yellow_cones) {
                newCones.addYellowCone(cone[0], cone[1], cone[2]);
            }
            for (const auto& cone : data.orange_cones) {
                newCones.addOrangeCone(cone[0], cone[1], cone[2]);
            }

            return newCones;
        }

        // Augment the cones dataset by adding circles around each cone
        Cones Cones::augmentConesCircle(Cones& cones, int deg, double radius) {
            std::vector<std::vector<double>> blue = augmentDatasetCircle(cones.blue_cones, deg, radius);
            std::vector<std::vector<double>> yellow = augmentDatasetCircle(cones.yellow_cones, deg, radius);
            std::vector<std::vector<double>> orange = augmentDatasetCircle(cones.orange_cones, deg, radius);

            Cones newCones;
            for (const std::vector<double>& cone : blue) {
                newCones.addBlueCone(cone[0], cone[1], cone[2]);
            }
            for (const std::vector<double>& cone : yellow) {
                newCones.addYellowCone(cone[0], cone[1], cone[2]);
            }
            for (const std::vector<double>& cone : orange) {
                newCones.addOrangeCone(cone[0], cone[1], cone[2]);
            }

            return newCones;
        }

        // builds feature matrix and label vertex
        std::pair<std::vector<std::vector<double>>, std::vector<double>> Cones::conesToXY(const Cones& cones) {
            std::vector<std::vector<double>> blue_cones = cones.blue_cones;
            std::vector<std::vector<double>> yellow_cones = cones.yellow_cones;

            // assigns blue cones a label of 0
            for (std::vector<double>& cone : blue_cones) {
                if (cone.size() > 2) {
                    cone[2] = 0.0;
                }
            }

            // assigns yellow cones a label of 1
            for (std::vector<double>& cone : yellow_cones) {
                if (cone.size() > 2) {
                    cone[2] = 1.0; 
                }
            }

            // combines the cones dataset into one vector
            std::vector<std::vector<double>> combined_data;
            combined_data.insert(combined_data.end(), blue_cones.begin(), blue_cones.end());
            combined_data.insert(combined_data.end(), yellow_cones.begin(), yellow_cones.end());

            std::vector<std::vector<double>> X;
            std::vector<double> y;

            // builds feature matrix and label vertex
            for (const std::vector<double>& cone : combined_data) {
                if (cone.size() >= 3) {
                    X.push_back({cone[0], cone[1]});
                    y.push_back(cone[2]);        
                }
            }

            return {X, y};
        }
    }
}