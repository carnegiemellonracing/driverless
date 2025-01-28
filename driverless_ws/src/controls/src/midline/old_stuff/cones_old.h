#ifndef CONES_H
#define CONES_H

#include <vector>
#include <string>
#include <functional>
#include <numbers>

class Cones {
private:
    std::vector<std::vector<double>> blue_cones;
    std::vector<std::vector<double>> yellow_cones;
    std::vector<std::vector<double>> orange_cones;

    std::string reprCones(const std::vector<std::vector<double>>& cones) const;

public:
    struct ConeData
    {
        std::vector<std::vector<double>> blue_cones;
        std::vector<std::vector<double>> yellow_cones;
        std::vector<std::vector<double>> orange_cones;
    };

    Cones() = default;
    Cones(const ConeData& data);

    void addBlueCone(double x, double y, double z);
    void addYellowCone(double x, double y, double z);
    void addOrangeCone(double x, double y, double z);

    void addCones(const Cones& other);

    void map(const std::function<std::vector<double>(const std::vector<double>&)>& mapper);

    void supplementCones();

    std::string toString() const;

    size_t size() const;

    Cones copy() const;

    std::vector<std::vector<double>> augmentDatasetCircle(std::vector<std::vector<double>> X, int deg, int radius);
    Cones augmentConesCircle(const Cones& cones, int deg = 20, double radius = 2.0);

    ConeData toStruct() const;

    // Cones fromStruct(const ConeData& data);

    std::pair<std::vector<std::vector<double>>, std::vector<double>> conesToXY(const Cones& cones);
};

#endif
