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
    Cones() = default;

    void addBlueCone(double x, double y, double z);
    void addYellowCone(double x, double y, double z);
    void addOrangeCone(double x, double y, double z);

    void addMultipleBlue(std::vector<std::vector<double>> blue_list);
    void addMultipleYellow(std::vector<std::vector<double>> yellow_list);

    void addCones(const Cones& other);

    const std::vector<std::vector<double>>& getBlueCones() const;
    const std::vector<std::vector<double>>& getYellowCones() const;

    void map(const std::function<std::vector<double>(const std::vector<double>&)>& mapper);

    void supplementCones();

    std::string toString() const;

    size_t size() const;

    Cones copy() const;

    std::vector<std::vector<double>> augmentDatasetCircle(std::vector<std::vector<double>> &X, int deg, double radius);
    Cones augmentConesCircle(Cones& cones, int deg = 20, double radius = 2.0);

    struct ConeData {
        std::vector<std::vector<double>> blue_cones;
        std::vector<std::vector<double>> yellow_cones;
        std::vector<std::vector<double>> orange_cones;
    };

    ConeData toStruct() const;

    Cones fromStruct(const ConeData& data);

    std::pair<std::vector<std::vector<double>>, std::vector<double>> conesToXY(const Cones& cones);
};

#endif
