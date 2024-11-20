#include <math.h>
#include <vector>
#include <iostream>
#include "racelineChunk.hpp"
#include <cassert>
#include <fstream>

// angle to rads?
double ator(int a){
    return (double) a * M_PI / 180.0;
}

// feet to m
double ftom(int a){
    return (double) a * 0.3048;
}

// std::vector<std::pair<double,double>> yellow_cones = {
//     std::make_pair(0, 0),
//     std::make_pair(0, 2),
//     std::make_pair(0, 4),
//     std::make_pair(0, 6),
//     std::make_pair(0, 8),
//     std::make_pair(0, 10),
//     std::make_pair(0, 12),
//     std::make_pair(0, 14),
//     std::make_pair(0, 16),
//     std::make_pair(0, 18),
//     std::make_pair(0, 20),
//     std::make_pair(0, 22),
//     std::make_pair(0 - 6 + 6 * cos(ator(30)), 22 + 6 * sin(ator(30))),
//     std::make_pair(0 - 6 + 6 * cos(ator(60)), 22 + 6 * sin(ator(60))),
//     std::make_pair(-6, 28),
//     std::make_pair(-8, 28),
//     std::make_pair(-10, 28),
//     std::make_pair(-12, 28),
//     std::make_pair(-14, 28),
//     std::make_pair(-14 + 6 * cos(ator(120)), 28 - 6 + 6 * sin(ator(120))),
//     std::make_pair(-14 + 6 * cos(ator(150)), 28 - 6 + 6 * sin(ator(150))),
//     std::make_pair(-20, 22),
//     std::make_pair(-20 + ftom(4), 20),
//     std::make_pair(-20 + ftom(6), 18),
//     std::make_pair(-20 + ftom(2), 16),
//     std::make_pair(-20 - ftom(2), 14),
//     std::make_pair(-20 - ftom(6), 12),
//     std::make_pair(-20 - ftom(4), 10),
//     std::make_pair(-20, 8),
//     std::make_pair(-20, 6),
//     std::make_pair(-20, 4),
//     std::make_pair(-16+2 - 6*cos(ator(30)),4-6*sin(ator(30))),
//     std::make_pair(-16+2 - 6*cos(ator(60)),4-6*sin(ator(90))),
//     std::make_pair(-14,-2),
//     std::make_pair(-12,-2),
//     std::make_pair(-10,-2),
//     std::make_pair(-8,-2),
//     std::make_pair(-6,-2),
//     std::make_pair(-4,-2)
// };

// std::vector<std::pair<double,double>> blue_cones = {
//     std::make_pair(-4, 0),
//     std::make_pair(-4, 2),
//     std::make_pair(-4, 4),
//     std::make_pair(-4, 6),
//     std::make_pair(-4, 8),
//     std::make_pair(-4, 10),
//     std::make_pair(-4, 12),
//     std::make_pair(-4, 14),
//     std::make_pair(-4, 16),
//     std::make_pair(-4, 18),
//     std::make_pair(-4, 20),
//     std::make_pair(-4, 22),
//     std::make_pair(-4-2 + 2 * cos(ator(30)), 22 + 2 * sin(ator(30))),
//     std::make_pair(-4-2 + 2 * cos(ator(60)), 22 + 2 * sin(ator(60))),
//     std::make_pair(-6, 24),
//     std::make_pair(-8, 24),
//     std::make_pair(-10, 24),
//     std::make_pair(-12, 24),
//     std::make_pair(-14, 24),
//     std::make_pair(-14 + 2 * cos(ator(120)), 24 - 2 + 2 * sin(ator(120))),
//     std::make_pair(-14 + 2 * cos(ator(150)), 24 - 2 + 2 * sin(ator(150))),
//     std::make_pair(-16, 22),
//     std::make_pair(-16 + ftom(4), 20),
//     std::make_pair(-16 + ftom(6), 18),
//     std::make_pair(-16 + ftom(2), 16),
//     std::make_pair(-16 - ftom(2), 14),
//     std::make_pair(-16 - ftom(6), 12),
//     std::make_pair(-16 - ftom(4), 10),
//     std::make_pair(-16, 8),
//     std::make_pair(-16, 6),
//     std::make_pair(-16, 4),
//     std::make_pair(-16+2 - 2*cos(ator(30)),4-2*sin(ator(30))),
//     std::make_pair(-16+2 - 2*cos(ator(60)),4-2*sin(ator(90))),
//     std::make_pair(-14,2),
//     std::make_pair(-12,2),
//     std::make_pair(-10,2),
//     std::make_pair(-8,2),
//     std::make_pair(-6,2),
//     std::make_pair(-4,2)
// };

std::vector<std::pair<double,double>> yellow_cones = {
    std::make_pair(0, 0),
    std::make_pair(20, 0),
    std::make_pair(40, 40),
    std::make_pair(60, 40)
};

std::vector<std::pair<double,double>> blue_cones = {
    std::make_pair(0, 20),
    std::make_pair(20, 0),
    std::make_pair(13.24, -14.99),
    std::make_pair(0, -20)
};

void testMakeSplinesVector() {
    // make a std::vector<std::pair<double,double>> with 2 different polynomials
    // one point overlap, have 7 points in total 
    std::vector<std::pair<double,double>> blue_cones = {
        std::make_pair(400, 400),
        std::make_pair(300, 300),
        std::make_pair(200, 200),
        std::make_pair(100, 100)
    };

    // std::make_pair(589, 167.513),
    // std::make_pair(455, 101.664),
    // std::make_pair(320, 94.88),
    // std::make_pair(201, 21.194), 

    // std::make_pair(1.382, 9.425), // y = 2x^3 + 3x
    // std::make_pair(1.192, 6.963),
    // std::make_pair(0.494, 1.723),
    // std::make_pair(0, 0), // y = 2x^3 + 5x^2 + x
    // std::make_pair(-0.68, 1.003), 
    // std::make_pair(-1.78, 2.782),
    // std::make_pair(-2.56, -3.346)

    // pass these 7 points into makeSplinesVector
    std::pair<std::vector<Spline>,std::vector<double>> slVectors = make_splines_vector(blue_cones);

    // should return a vector of 2 splines, each splines should have one of the polynomials
    std::vector<Spline> splines = slVectors.first;
    // std::cout << "size of splines: " << deg << std::endl;

    // should return a vecotr of cumulativeLengths
    std::vector<double> cumulativeLengths = slVectors.second;

    assert(splines.size() == 1);
    assert(cumulativeLengths.size() == 1);

    for (int i = 0; i < splines.size(); i++){
        polynomial poly = splines[i].get_SplPoly();
        int deg = poly.deg;
        Eigen::VectorXd coeffs = poly.nums;
        // std::cout << "degree: " << deg << std::endl;
        // std::cout << "coefficients: " << coeffs << std::endl;
        // std::cout << "length: " << cumulativeLengths[i] << std::endl;
    }
}



int main() {
    
    // std::cout << "Hello, world!" << std::endl;

    testMakeSplinesVector();

    std::vector<Chunk*> chunks = *generateChunks(blue_cones, yellow_cones);

    // if (chunks == nullptr) {
    //     std::cout << "CHUNKS VECTOR IS NULL" << std::endl;
    // }

    // if ((*chunks)[0] == nullptr) {
    //     std::cout << "FIRST CHUNK IS NULL" << std::endl;
    // }

    // outputting chunks
    // std::ofstream Blue("blue_points.txt");
    // std::ofstream Yellow("yellow_points.txt");

    for (int i  = 0; i < chunks.size(); i++) {
        // std::cout << "start, end: " << chunks[i]->startProdgress << ", " << chunks[i]->endProgress << std::endl;
        // std::cout << "average curvature: " << chunks[i]->avgCurvature << std::endl;

        for (int j = 0; j < chunks[i]->yellowPoints.size(); j++) {
            std::cout << chunks[i]->bluePoints[j].first << "," << chunks[i]->bluePoints[j].second << std::endl;
        }

        // for (int j = 0; j < chunks[i]->yellowPoints.size(); j++) {
        //     Yellow << chunks[i]->yellowPoints[j].first << "," << chunks[i]->yellowPoints[j].second << std::endl;
        // }

        std::cout << "#" << std::endl;
        // Yellow << "#" << std::endl;
    }

    // Blue.close();
    // Yellow.close();

    // for (int j = 0; j < chunks.size(); j++) {
    //     for (int i = 0; i < chunks[j]->bluePoints.size(); i++) {
    //         std::cout << "(" << chunks[j]->bluePoints[i].first << "," << chunks[j]->bluePoints[i].second << ")" << std::endl;
    //     }
    // }    
    
    return 0;
}