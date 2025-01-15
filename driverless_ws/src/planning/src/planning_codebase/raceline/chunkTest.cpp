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

/* Squidward track */
void createSquidwardTrack(std::vector<std::pair<double,double>> &blue_cones,
                            std::vector<std::pair<double,double>> &yellow_cones) {
    yellow_cones = {
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
         std::make_pair(-6,-2)
         std::make_pair(-4,-2)
    };

    blue_cones = {
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
        std::make_pair(-6,2)
        std::make_pair(-4,2)
    };
}

// semi circle with straight line after, should gen 3 splines, 2 for semicircle, 1 for straight
// std::vector<std::pair<double,double>> yellow_cones = {
//     std::make_pair(0, 20),
//     std::make_pair(9.25, 17.73),
//     std::make_pair(17.5, 9.68),
//     std::make_pair(20, 0),
//     std::make_pair(18.25, -8.18),
//     std::make_pair(13.24, -14.99),
//     std::make_pair(0, -20), 
    // std::make_pair(-5, -20),
    // std::make_pair(-10, -20),
    // std::make_pair(0, -20)  
// };

// wave 1 spline
/* Wave spline */

void createWaveTrack(std::vector<std::pair<double, double>> &blue_cones, 
                        std::vector<std::pair<double, double>> &yellow_cones) {


    yellow_cones = {
        std::make_pair(0, 20),
        std::make_pair(20, 0),
        std::make_pair(40, 40),
        std::make_pair(60, 40)
    };

    blue_cones = {
        std::make_pair(0, 20),
        std::make_pair(20, 0),
        std::make_pair(40, 40),
        std::make_pair(60, 40)
    };
}

void createSemiCircleTrack(std::vector<std::pair<double, double>> &blue_cones, 
                        std::vector<std::pair<double, double>> &yellow_cones) {
    yellow_cones = {
        std::make_pair(0, 20),
        std::make_pair(9.25, 17.73),
        std::make_pair(17.5, 9.68),
        std::make_pair(20, 0),
        std::make_pair(18.25, -8.18),
        std::make_pair(13.24, -14.99),
        std::make_pair(0, -20), 
        std::make_pair(-5, -20),
        std::make_pair(-10, -20),
        std::make_pair(-15, -20)
    };

    blue_cones = {
        std::make_pair(0, 20),
        std::make_pair(9.25, 17.73),
        std::make_pair(17.5, 9.68),
        std::make_pair(20, 0),
        std::make_pair(18.25, -8.18),
        std::make_pair(13.24, -14.99),
        std::make_pair(0, -20), 
        std::make_pair(-5, -20),
        std::make_pair(-10, -20),
        std::make_pair(-15, -20)
    };                
}

void createParabToStraightTrack(std::vector<std::pair<double, double>> &blue_cones, 
                        std::vector<std::pair<double, double>> &yellow_cones) {
    yellow_cones = {
        std::make_pair(0, 25),
        std::make_pair(2, 16),
        std::make_pair(4, 9),
        std::make_pair(6, 4),
        std::make_pair(7, 2.25),
        std::make_pair(8, 1),
        std::make_pair(10, 0),
        std::make_pair(15, 0),
        std::make_pair(20, 0),
        std::make_pair(25, 0)
    };

    blue_cones = {
        std::make_pair(0, 25),
        std::make_pair(2, 16),
        std::make_pair(4, 9),
        std::make_pair(6, 4),
        std::make_pair(7, 2.25),
        std::make_pair(8, 1),
        std::make_pair(10, 0),
        std::make_pair(15, 0),
        std::make_pair(20, 0),
        std::make_pair(25, 0)
    };                
}


// a(t) = 0+0t +0t^2 +0t^3
// b(t) = 0+6.00266t +0.263389t^2 +-0.266049t^3
// c(t) = 0+0t +0t^2 +0t^3
// d(t) = 6+6.00266t +0.263389t^2 +-0.266049t^3
// f(t) = 0+0t +0t^2 +0t^3
// g(t) = 12+6.00266t +0.263389t^2 +-0.266049t^3
// h(t) = 0+-0.76856t +3.49346t^2 +-3.52874t^3
// i(t) = 18+6.95876t +-4.08253t^2 +4.12377t^3
// j(t) = -0.803848+-3.59019t +-12.0856t^2 +8.47965t^3
// k(t) = 25+9.36502t +-9.12193t^2 +2.75691t^3
// l(t) = -8+-6.00266t +-0.263389t^2 +0.266049t^3
// m(t) = 28+2.02505*10^(-13)t +-6.82121*10^(-13)t^2 +4.40536*10^(-13)t^3
// n(t) = -14+-9.71789t +1.00455t^2 +2.71334t^3
// o(t) = 28+0.361024t +-9.51701t^2 +3.15598t^3
// p(t) = -20+3.4616t +2.45829t^2 +-5.31029t^3
// q(t) = 22+-6.00266t +-0.263389t^2 +0.266049t^3
// r(t) = -19.3904+-1.9107t +-8.10838t^2 +8.19028t^3
// s(t) = 16+-6.00266t +-0.263389t^2 +0.266049t^3
// u(t) = -21.2192+6.76102t +-11.1396t^2 +5.5978t^3
// v(t) = 10+-6.00266t +-0.263389t^2 +0.266049t^3
// w(t) = 20+-0.361024t +9.51701t^2 +-3.15598t^3
// C(t) = 4+-6.13569t +-13.4328t^2 +13.5685t^3
// D(t) = -14+6.00266t +0.263389t^2 +-0.266049t^3
// z(t) = -2+-1.33227*10^(-14)t +4.9738*10^(-14)t^2 +-3.01981*10^(-14)t^3
// A(t) = -10+6.00266t +0.263389t^2 +-0.266049t^3
// B(t) = -2+-1.33227*10^(-14)t +4.9738*10^(-14)t^2 +-3.01981*10^(-14)t^3


// void testMakeSplinesVector() {
//     // make a std::vector<std::pair<double,double>> with 2 different polynomials
//     // one point overlap, have 7 points in total 
//     std::vector<std::pair<double,double>> blue_cones = {
//         std::make_pair(400, 400),
//         std::make_pair(300, 300),
//         std::make_pair(200, 200),
//         std::make_pair(100, 100)
//     };

//     // std::make_pair(589, 167.513),
//     // std::make_pair(455, 101.664),
//     // std::make_pair(320, 94.88),
//     // std::make_pair(201, 21.194), 

//     // std::make_pair(1.382, 9.425), // y = 2x^3 + 3x
//     // std::make_pair(1.192, 6.963),
//     // std::make_pair(0.494, 1.723),
//     // std::make_pair(0, 0), // y = 2x^3 + 5x^2 + x
//     // std::make_pair(-0.68, 1.003), 
//     // std::make_pair(-1.78, 2.782),
//     // std::make_pair(-2.56, -3.346)

//     // pass these 7 points into makeSplinesVector
//     std::pair<std::vector<Spline>,std::vector<double>> slVectors = make_splines_vector(blue_cones);

//     // should return a vector of 2 splines, each splines should have one of the polynomials
//     std::vector<Spline> splines = slVectors.first;
//     // std::cout << "size of splines: " << deg << std::endl;

//     // should return a vecotr of cumulativeLengths
//     std::vector<double> cumulativeLengths = slVectors.second;

//     assert(splines.size() == 1);
//     assert(cumulativeLengths.size() == 1);

//     for (int i = 0; i < splines.size(); i++){
//         polynomial poly = splines[i].get_SplPoly();
//         int deg = poly.deg;
//         Eigen::VectorXd coeffs = poly.nums;
//         // std::cout << "degree: " << deg << std::endl;
//         // std::cout << "coefficients: " << coeffs << std::endl;
//         // std::cout << "length: " << cumulativeLengths[i] << std::endl;
//     }
// }

void print_poly(Spline x, Spline y) {
    std::cout << "(["<< x.spl_poly.nums(0) << "," << x.spl_poly.nums(1) << ","
     << x.spl_poly.nums(2) << "," << x.spl_poly.nums(3) << "]," 
     << "[" << y.spl_poly.nums(0) << "," << y.spl_poly.nums(1) << ","
     << y.spl_poly.nums(2) << "," << y.spl_poly.nums(3) << "])"<< std::endl;
}

int main() {
    std::vector<std::pair<double, double>> blue_cones = {};
    std::vector<std::pair<double, double>> yellow_cones = {};

    //createSquidwardTrack(blue_cones, yellow_cones);
    // std::vector<std::pair<double,double>> blue_cones = {
    //     std::make_pair(400, 400),
    //     std::make_pair(300, 300),
    //     std::make_pair(200, 200),
    //     std::make_pair(100, 100)
    // };

    // std::vector<std::pair<double,double>> yellow_cones = {
    //     std::make_pair(400, 400),
    //     std::make_pair(300, 300),
    //     std::make_pair(200, 200),
    //     std::make_pair(100, 100)
    // };

    createSquidwardTrack(blue_cones, yellow_cones);

    std::vector<Chunk*> chunks = *generateChunks(blue_cones, yellow_cones);

    // if (chunks == nullptr) {
    //     std::cout << "CHUNKS VECTOR IS NULL" << std::endl;
    // }

    // if ((*chunks)[0] == nullptr) {
    //     std::cout << "FIRST CHUNK IS NULL" << std::endl;
    // }

    // outputting chunks
    // std::string blue_chunk_file = "/root/driverless/driverless_ws/src/planning/src/planning_codebase/raceline/chunk_vis_blue.txt";
    // std::string yellow_chunk_file = "/root/driverless/driverless_ws/src/planning/src/planning_codebase/raceline/chunk_vis_yellow.txt";
    // std::ofstream Blue;
    // Blue.open(blue_chunk_file, std::ios::out);
    // std::ofstream Yellow;
    // Yellow.open(yellow_chunk_file, std::ios::out);

    
    for (int i  = 0; i < chunks.size(); i++) {
        // std::cout << "start, end: " << chunks[i]->startProdgress << ", " << chunks[i]->endProgress << std::endl;
        // std::cout << "average curvature: " << chunks[i]->avgCurvature << std::endl;

        // for (int j = 0; j < chunks[i]->bluePoints.size(); j++) {
        //     Blue << chunks[i]->bluePoints[j].first << "," << chunks[i]->bluePoints[j].second << std::endl;
        //     std::cout << chunks[i]->bluePoints[j].first << "," << chunks[i]->bluePoints[j].second << std::endl;
        // }


        std::cout << "([" << std::endl;

        // for (int j = 0; j < chunks[i]->blueSplines.size(); j++) {
        //     // Yellow << chunks[i]->yellowSplines[j].first << "," << chunks[i]->yellowSplines[j].second << std::endl;
        //     if (j != 0) {
        //         std::cout << ",";
        //     }
        //     print_poly(chunks[i]->blueSplines[j].spline_x, chunks[i]->blueSplines[j].spline_y);

        // }

        for (int j = 0; j < chunks[i]->yellowSplines.size(); j++) {
            // Yellow << chunks[i]->yellowSplines[j].first << "," << chunks[i]->yellowSplines[j].second << std::endl;
            if (j != 0) {
                std::cout << "," << std::endl;
            }
            print_poly(chunks[i]->yellowSplines[j].spline_x, chunks[i]->yellowSplines[j].spline_y);

        }

        std::cout << "]," << std::endl;

        // std::cout << " (" << "0" <<  ", " << "1" << ")" << std::endl;
        std::cout << " (" << chunks[i]->tStart <<  ", " << chunks[i]->tEnd << ")" << std::endl;

        // Blue << "#" << std::endl;
        std::cout << ")," << std::endl;


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

