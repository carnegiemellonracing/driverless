#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <numbers>
#include <utility> 
#include <cmath>

#include "cones.h"

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

void Cones::addCones(const Cones& other) {
    blue_cones.insert(blue_cones.end(), other.blue_cones.begin(), other.blue_cones.end());
    yellow_cones.insert(yellow_cones.end(), other.yellow_cones.begin(), other.yellow_cones.end());
    orange_cones.insert(orange_cones.end(), other.orange_cones.begin(), other.orange_cones.end());
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
    addBlueCone(-4.0, 0.0, 0.0); 
    addYellowCone(4.0, 0.0, 0.0);
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

Cones::ConeData Cones::toStruct() const {
    ConeData data;
    data.blue_cones = blue_cones;       
    data.yellow_cones = yellow_cones; 
    data.orange_cones = orange_cones;  
    return data;
}

std::vector<std::vector<double>> Cones::augmentDatasetCircle(std::vector<std::vector<double>> X, int deg, int radius) {
    double radian = deg * (M_PI) / 180;

    std::vector<double> angles;
    for (double angle = 0; angle < 2 * M_PI; angle += radian) {
        angles.push_back(angle);
    }

    int N = X.size();

    int num_angles = angles.size();
    std::vector<std::vector<double>> X_extra;

    for (int i = 0; i < num_angles; ++i) {
        X_extra.insert(X_extra.end(), X.begin(), X.end());
    }

    std::vector<double> repeated_angles;
    repeated_angles.reserve(num_angles * N); 
    for (double angle : angles) {
        for (int j = 0; j < N; ++j) {
            repeated_angles.push_back(angle);
        }
    }

    radius = static_cast<double>(radius);
    for (size_t i = 0; i < X_extra.size(); ++i) {
        X_extra[i][0] += radius * std::cos(repeated_angles[i]); 
        X_extra[i][1] += radius * std::sin(repeated_angles[i]); 
    }

    X.insert(X.end(), X_extra.begin(), X_extra.end());

    return X;
}

Cones::Cones(const ConeData& data) {
    blue_cones = data.blue_cones;
    yellow_cones = data.yellow_cones;
    orange_cones = data.orange_cones;
    // Cones newCones;

    // for (const auto& cone : data.blue_cones) {
    //     newCones.addBlueCone(cone[0], cone[1], cone[2]);
    // }
    // for (const auto& cone : data.yellow_cones) {
    //     newCones.addYellowCone(cone[0], cone[1], cone[2]);
    // }
    // for (const auto& cone : data.orange_cones) {
    //     newCones.addOrangeCone(cone[0], cone[1], cone[2]);
    // }

    // return newCones;
}

Cones Cones::augmentConesCircle(const Cones& cones, int deg, double radius) {
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

std::pair<std::vector<std::vector<double>>, std::vector<double>> Cones::conesToXY(const Cones& cones) {
    std::vector<std::vector<double>> blue_cones = cones.blue_cones;
    std::vector<std::vector<double>> yellow_cones = cones.yellow_cones;

    for (std::vector<double>& cone : blue_cones) {
        if (cone.size() > 2) {
            cone[2] = 0.0;
        }
    }

    for (std::vector<double>& cone : yellow_cones) {
        if (cone.size() > 2) {
            cone[2] = 1.0; 
        }
    }

    std::vector<std::vector<double>> combined_data;
    combined_data.insert(combined_data.end(), blue_cones.begin(), blue_cones.end());
    combined_data.insert(combined_data.end(), yellow_cones.begin(), yellow_cones.end());

    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (const std::vector<double>& cone : combined_data) {
        if (cone.size() >= 3) {
            X.push_back({cone[0], cone[1]});
            y.push_back(cone[2]);        
        }
    }

    return {X, y};
}


