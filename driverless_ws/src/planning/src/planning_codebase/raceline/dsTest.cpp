#include "dynamicSpline.cpp"
#include <math.h>

// angle to rads?
double ator(int a){
    return (double) a * M_PI / 180.0;
}

// feet to m
double ftom(int a){
    return (double) a * 0.3048;
}

std::vector<std::pair<double,double>> yellow_cones = {
    std::make_pair(0, 0),
    std::make_pair(0, 2),
    std::make_pair(0, 4),
    std::make_pair(0, 6),
    std::make_pair(0, 8),
    std::make_pair(0, 10),
    std::make_pair(0, 12),
    std::make_pair(0, 14),
    std::make_pair(0, 16),
    std::make_pair(0, 18),
    std::make_pair(0, 20),
    std::make_pair(0, 22),
    std::make_pair(0 - 6 + 6 * cos(ator(30)), 22 + 6 * sin(ator(30))),
    std::make_pair(0 - 6 + 6 * cos(ator(60)), 22 + 6 * sin(ator(60))),
    std::make_pair(-6, 28),
    std::make_pair(-8, 28),
    std::make_pair(-10, 28),
    std::make_pair(-12, 28),
    std::make_pair(-14, 28),
    std::make_pair(-14 + 6 * cos(ator(120)), 28 - 6 + 6 * sin(ator(120))),
    std::make_pair(-14 + 6 * cos(ator(150)), 28 - 6 + 6 * sin(ator(150))),
    std::make_pair(-20, 22),
    std::make_pair(-20 + ftom(4), 20),
    std::make_pair(-20 + ftom(6), 18),
    std::make_pair(-20 + ftom(2), 16),
    std::make_pair(-20 - ftom(2), 14),
    std::make_pair(-20 - ftom(6), 12),
    std::make_pair(-20 - ftom(4), 10),
    std::make_pair(-20, 8),
    std::make_pair(-20, 6),
    std::make_pair(-20, 4),
    std::make_pair(-16+2 - 6*cos(ator(30)),4-6*sin(ator(30))),
    std::make_pair(-16+2 - 6*cos(ator(60)),4-6*sin(ator(90))),
    std::make_pair(-14,-2),
    std::make_pair(-12,-2),
    std::make_pair(-10,-2),
    std::make_pair(-8,-2),
    std::make_pair(-6,-2),
    std::make_pair(-4,-2)
};

std::vector<std::pair<double,double>> blue_cones = {
    std::make_pair(-4, 0),
    std::make_pair(-4, 2),
    std::make_pair(-4, 4),
    std::make_pair(-4, 6),
    std::make_pair(-4, 8),
    std::make_pair(-4, 10),
    std::make_pair(-4, 12),
    std::make_pair(-4, 14),
    std::make_pair(-4, 16),
    std::make_pair(-4, 18),
    std::make_pair(-4, 20),
    std::make_pair(-4, 22),
    std::make_pair(-4-2 + 2 * cos(ator(30)), 22 + 2 * sin(ator(30))),
    std::make_pair(-4-2 + 2 * cos(ator(60)), 22 + 2 * sin(ator(60))),
    std::make_pair(-6, 24),
    std::make_pair(-8, 24),
    std::make_pair(-10, 24),
    std::make_pair(-12, 24),
    std::make_pair(-14, 24),
    std::make_pair(-14 + 2 * cos(ator(120)), 24 - 2 + 2 * sin(ator(120))),
    std::make_pair(-14 + 2 * cos(ator(150)), 24 - 2 + 2 * sin(ator(150))),
    std::make_pair(-16, 22),
    std::make_pair(-16 + ftom(4), 20),
    std::make_pair(-16 + ftom(6), 18),
    std::make_pair(-16 + ftom(2), 16),
    std::make_pair(-16 - ftom(2), 14),
    std::make_pair(-16 - ftom(6), 12),
    std::make_pair(-16 - ftom(4), 10),
    std::make_pair(-16, 8),
    std::make_pair(-16, 6),
    std::make_pair(-16, 4),
    std::make_pair(-16+2 - 2*cos(ator(30)),4-2*sin(ator(30))),
    std::make_pair(-16+2 - 2*cos(ator(60)),4-2*sin(ator(90))),
    std::make_pair(-14,2),
    std::make_pair(-12,2),
    std::make_pair(-10,2),
    std::make_pair(-8,2),
    std::make_pair(-6,2),
    std::make_pair(-4,2)
}

void main() {
    // translate these
    plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], c="orange"),
    plt.scatter(blue_cones[:, 0], blue_cones[:, 1], c="blue")

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    plt.show()  
}