//Raceline Algorithm

#include <math.h>
#include <vector>
#include <iostream>
#include "racelineChunk.hpp"
#include <cassert>
#include <fstream>
#include <chrono>

typedef std::pair<double, double> Point;

// Function to calculate a point at a specific distance along a line (half for now)
Point pointAlongLine(const Point& blue_point, const Point& yellow_point, double distance) {
  double dx = blue_point.first - yellow_point.first;
  double dy = blue_point.second - yellow_point.second;
  double length = sqrt(dx * dx + dy * dy);
  double unit_dx = dx / length;
  double unit_dy = dy / length;
  return {yellow_point.first + length * distance * unit_dx, yellow_point.second + length * distance * unit_dy};
}

// Function to solve the large matrix equation
std::vector<Eigen::VectorXd> solve(const std::vector<Point>& points, double k_x, double l_x, double k_y, double l_y) {
    Eigen::VectorXd b_x(8), b_y(8);
    b_x << points[0].first, points[1].first, points[1].first, points[2].first, k_x, l_x, 0, 0;
    b_y << points[0].second, points[1].second, points[1].second, points[2].second, k_y, l_y, 0, 0;

    // Hardcoded inverse matrix
    Eigen::MatrixXd A_inverse(8, 8);
    A_inverse << 10, -10, -6, 6, 3, -1, 2, 0.25,
                 -9, 9, 3, -3, -3.5, 0.5, -1, -0.125,
                 0, 0, 0, 0, 1, 0, 0, 0,
                 1, 0, 0, 0, 0, 0, 0, 0,
                 -6, 6, 10, -10, -1, 3, -2, 0.25,
                 6, -6, -6, 6, 1, -1, 2, -0.25,
                 -1.5, 1.5, -1.5, 1.5, -0.25, -0.25, -0.5, 0.0625,
                 0, 0, 1, 0, 0, 0, 0, 0;

    Eigen::VectorXd X = A_inverse * b_x;
    Eigen::VectorXd Y = A_inverse * b_y;

    return {X.segment(0, 4), X.segment(4, 4), Y.segment(0, 4), Y.segment(4, 4)};
}

// coeffs is coefficients fo the spline ax^3 + bx^2 + cx + d
bool is_curve(const std::vector<double>& coeffs, double x, double threshold) {
    double a = coeffs[0], b = coeffs[1], c = coeffs[2];

    // Compute first derivative: f'(x) = 3ax^2 + 2bx + c
    double f_prime = 3 * a * x * x + 2 * b * x + c;

    // Compute second derivative: f''(x) = 6ax + 2b
    double f_double_prime = 6 * a * x + 2 * b;

    // Compute curvature: kappa = |f''(x)| / (1 + (f'(x))^2)^(3/2)
    double denominator = std::pow(1 + f_prime * f_prime, 1.5);

    return (std::abs(f_double_prime) / denominator) < threshold;
}

//Function to find raceline
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
runOptimizer(Chunk& chunk, double d1, double d2, double d3) {
    Point blueStart = {chunk.blueStartX, chunk.blueStartY};
    Point yellowStart = {chunk.yellowStartX, chunk.yellowStartY};
    Point blueMid = {chunk.blueMidX, chunk.blueMidY};
    Point yellowMid = {chunk.yellowMidX, chunk.yellowMidY};
    Point blueEnd = {chunk.blueEndX, chunk.blueEndY};
    Point yellowEnd = {chunk.yellowEndX, chunk.yellowEndY};

    double blueFirstDerXStart = chunk.blueFirstDerXStart;
    double blueFirstDerXEnd = chunk.blueFirstDerXEnd;
    double blueFirstDerYStart = chunk.blueFirstDerYStart;
    double blueFirstDerYEnd = chunk.blueFirstDerYEnd;
    double yellowFirstDerXStart = chunk.yellowFirstDerXStart;
    double yellowFirstDerXEnd = chunk.yellowFirstDerXEnd;
    double yellowFirstDerYStart = chunk.yellowFirstDerYStart;
    double yellowFirstDerYEnd = chunk.yellowFirstDerYEnd;

    std::vector<Point> points;
    points.push_back(pointAlongLine(blueStart, yellowStart, d1));
    points.push_back(pointAlongLine(blueMid, yellowMid, d2));
    points.push_back(pointAlongLine(blueEnd, yellowEnd, d3));

    double k_x = (blueFirstDerXStart + blueFirstDerXEnd) / 2;
    double k_y = (blueFirstDerYStart + blueFirstDerYEnd) / 2;
    double l_x = (yellowFirstDerXStart + yellowFirstDerXEnd) / 2;
    double l_y = (yellowFirstDerYStart + yellowFirstDerYEnd) / 2;

    auto splines = solve(points, k_x, l_x, k_y, l_y);

    // Return the full 16 coefficients: 4 for X1, 4 for X2, 4 for Y1, 4 for Y2
    return std::make_tuple(
        std::vector<double>(splines[0].data(), splines[0].data() + splines[0].size()),  // First X spline (first half)
        std::vector<double>(splines[1].data(), splines[1].data() + splines[1].size()),  // Second X spline (second half)
        std::vector<double>(splines[2].data(), splines[2].data() + splines[2].size()),  // First Y spline (first half)
        std::vector<double>(splines[3].data(), splines[3].data() + splines[3].size())   // Second Y spline (second half)
    );
}

double calculateEnd(Chunk& chunk, double start) {
    return 0.5;
}

void fillSplines(Chunk& chunks, std::vector<std::vector<double>> racelineSplines, double dstart) {
    int max_indiv_raceline_gen_time = std::numeric_limits<int>::min();
    auto start_raceline_gen = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < chunks.size(); ++i) {
        // Define param2 and param4 dynamically for each chunk
        double dend = calculateEnd(*chunks[i],dstart);
        auto start_cur_raceline_gen =std::chrono::high_resolution_clock::now();
        auto [X1, X2, Y1, Y2] = runOptimizer(*chunks[i], dstart, 0.5, dend);	
        // Combine all coefficients into one vector (16 coefficients per chunk)
        racelineSplines[i] = {
            X1[0], X1[1], X1[2], X1[3],  // First X spline (first half)
            X2[0], X2[1], X2[2], X2[3],  // Second X spline (second half)
            Y1[0], Y1[1], Y1[2], Y1[3],  // First Y spline (first half)
            Y2[0], Y2[1], Y2[2], Y2[3]   // Second Y spline (second half)
        };
        dstart = dend;
        auto end_cur_raceline_gen = std::chrono::high_resolution_clock::now();
        auto dur_cur_raceline_gen = std::chrono::duration_cast<std::chrono::microseconds>(end_cur_raceline_gen - start_cur_raceline_gen);
    
        if (max_indiv_raceline_gen_time < dur_cur_raceline_gen.count()) {
            max_indiv_raceline_gen_time = dur_cur_raceline_gen.count();
        }
        // std::cout << "\t Current raceline gen time: " << dur_cur_raceline_gen.count() << " microseconds" << std::endl;
    }
}

int main() {
    std::vector<std::tuple<double, double, int>> blue_cones = {};
    std::vector<std::tuple<double, double, int>> yellow_cones = {};
    
    createSquidwardTrack(blue_cones, yellow_cones);
    
    int max_total_raceline_gen_time = std::numeric_limits<int>::min();
    std::vector<std::vector<double>> sample_raceline_splines;
    for (size_t i = 0; i<100; ++i) {
        std::cout << "===========================" << std::endl;
    	auto start_chunking = std::chrono::high_resolution_clock::now();
        std::vector<Chunk*> chunks = *generateChunks(blue_cones, yellow_cones);
        auto end_chunking = std::chrono::high_resolution_clock::now();
        auto dur_chunking = std::chrono::duration_cast<std::chrono::microseconds>(end_chunking - start_chunking);
        std::cout << "Chunking time: " << dur_chunking.count() << std::endl;

        // write splines to file
        // std::ofstream blue_splines_outputFile("src/planning/src/planning_codebase/raceline/blue.txt");
        std::ofstream blue_splines_outputFile("blue.txt");
        if (blue_splines_outputFile.is_open()) {
            for (int i  = 0; i < chunks.size(); i++) {
                blue_splines_outputFile << "([";

                for (int j = 0; j < chunks[i]->blueSplines.size(); j++) {
                    if (j != 0) {
                        blue_splines_outputFile << ",";
                    }
                    print_poly(blue_splines_outputFile, chunks[i]->blueSplines[j].spline_x, chunks[i]->blueSplines[j].spline_y);
                }

                blue_splines_outputFile << "], \n";

                blue_splines_outputFile << " (0,1) ";

                blue_splines_outputFile << "), \n" ;

                // std::cout << "start blue cone ids for chunk" << i << std::endl;
                // for (int k = 0; k < chunks[i]->blueConeIds.size(); k++) {
                //     std::cout << chunks[i]->blueConeIds[k] << std::endl;
                // }
                // std::cout << "end yellow cone ids for chunk" << i << std::endl;

            }

            blue_splines_outputFile.close();
            std::cout << "Blue race bound polynomials have been written to blue.txt" << std::endl;
        } else {
            std::cerr << "Unable to open file for writing!" << std::endl;
        }

        // write splines to file
        // std::ofstream yellow_splines_outputFile("src/planning/src/planning_codebase/raceline/yellow.txt");
        std::ofstream yellow_splines_outputFile("yellow.txt");
        if (yellow_splines_outputFile.is_open()) {
            for (int i  = 0; i < chunks.size(); i++) {
                yellow_splines_outputFile << "([";

                for (int j = 0; j < chunks[i]->yellowSplines.size(); j++) {
                    if (j != 0) {
                        yellow_splines_outputFile << ",";
                    }
                    print_poly(yellow_splines_outputFile, chunks[i]->yellowSplines[j].spline_x, chunks[i]->yellowSplines[j].spline_y);
                }

                yellow_splines_outputFile << "], \n";

                yellow_splines_outputFile << " (" << chunks[i]->tStart <<  ", " << chunks[i]->tEnd << ")";

                yellow_splines_outputFile << "), \n" ;

                // std::cout << "start yellow cone ids for chunk" << i << std::endl;
                // for (int k = 0; k < chunks[i]->yellowConeIds.size(); k++) {
                //     std::cout << chunks[i]->yellowConeIds[k] << std::endl;
                // }
                // std::cout << "end yellow cone ids for chunk" << i << std::endl;

                std::cout << "(" << chunks[i]->blueMidX << "," << chunks[i]->blueMidY << ")," << std::endl;
                std::cout << "(" << chunks[i]->yellowMidX << "," << chunks[i]->yellowMidY << ")," << std::endl;

            }

            yellow_splines_outputFile.close();
            std::cout << "Yellow race bound polynomials have been written to blue.txt" << std::endl;
        } else {
            std::cerr << "Unable to open file for writing!" << std::endl;
        }


    	double dstart = 0.5; 
    	// Vector to hold results: one vector for each chunk, with 16 coefficients (4 X1, 4 X2, 4 Y1, 4 Y2)
        std::vector<std::vector<double>> racelineSplines(chunks.size());
        // Function that takes in dstart and racelineSplines
    	racelineSplines = fillSplines(chunks, racelineSplines, dstart);

        if (i == 0) {
            sample_raceline_splines = racelineSplines;
        }
    	auto end_raceline_gen = std::chrono::high_resolution_clock::now();
    	auto dur_raceline_gen = std::chrono::duration_cast<std::chrono::microseconds>(end_raceline_gen - start_raceline_gen);

        if (max_total_raceline_gen_time < dur_raceline_gen.count() + dur_chunking.count()) {
            max_total_raceline_gen_time = dur_raceline_gen.count() + dur_chunking.count();
        }
        std::cout << "Raceline gen longest individual run time: " << max_indiv_raceline_gen_time << " microseconds" << std::endl;
    	std::cout << "Raceline gen entire track time: " << dur_raceline_gen.count() << " microseconds" << std::endl;
        std::cout << "===========================\n\n" << std::endl;
    }

    std::cout << "===========================" << std::endl;
    std::cout << "End results: " << std::endl;
    std::cout << "Longest pipeline time: " << max_total_raceline_gen_time << std::endl;
    std::cout << "===========================" << std::endl;

    // Write output to a text file
    // std::ofstream outputFile("src/planning/src/planning_codebase/raceline/splines.txt");
    std::ofstream outputFile("splines.txt");
    if (outputFile.is_open()) {
        for (const auto& spline : sample_raceline_splines ){
            // Write coefficients to file
            outputFile << spline[0] << " " << spline[1] << " " << spline[2] << " " << spline[3] << " "; // First Half X
            outputFile << spline[4] << " " << spline[5] << " " << spline[6] << " " << spline[7] << " "; // Second Half X
            outputFile << spline[8] << " " << spline[9] << " " << spline[10] << " " << spline[11] << " "; // First Half Y
            outputFile << spline[12] << " " << spline[13] << " " << spline[14] << " " << spline[15] << "\n"; // Second Half Y
        }
        outputFile.close();
        std::cout << "Spline coefficients have been written to splines.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }

    // std::ofstream splines_outputFile("src/planning/src/planning_codebase/raceline/splines.txt");
    std::ofstream splines_outputFile("splines.txt");
    if (splines_outputFile.is_open()) {
        for (const auto& spline : sample_raceline_splines ){
            // Write coefficients to file
            splines_outputFile << spline[0] << " " << spline[1] << " " << spline[2] << " " << spline[3] << " "; // First Half X
            splines_outputFile << spline[4] << " " << spline[5] << " " << spline[6] << " " << spline[7] << " "; // Second Half X
            splines_outputFile << spline[8] << " " << spline[9] << " " << spline[10] << " " << spline[11] << " "; // First Half Y
            splines_outputFile << spline[12] << " " << spline[13] << " " << spline[14] << " " << spline[15] << "\n"; // Second Half Y
        }
        splines_outputFile.close();
        std::cout << "Spline coefficients have been written to splines.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing!" << std::endl;
    }
    return 0;
}