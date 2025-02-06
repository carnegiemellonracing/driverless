#include "racelineChunk.hpp"

#include <math.h>
#include <chrono>
// #include "../midline/generator.hpp"


// tunable params for chunks
// thresholds are arclength in meters
#define CHUNK_LEN_MAX_THRESH 1000
#define CHUNK_LEN_MIN_THRESH 15
#define CHUNK_FIRST_DER_THRESH 0.01
#define CHUNK_SECOND_DER_THRESH 100
#define CHUNK_THIRD_DER_THRESH 100
#define CHUNK_INFINITY_THRESH 500.0

/**
 * Constructor for chunks.
 */
Chunk::Chunk() = default;

/** 
 * Returns true if the thresholds for derivatives, otherwise returns false.
 * 
 * The 1st, 2nd, and 3rd derivatives for spline1 and spline2 should differ below
 * certain threshold as declared above.
 * Otherwise returns false.
 * 
 * @param spline1 the previous spline we check against
 * @param spline2 the current spline 
 *  
 */
bool Chunk::continueChunk(ParameterizedSpline spline1, ParameterizedSpline spline2) {
    double spline1_first_der = spline1.get_first_der(1);
    double spline2_first_der = spline2.get_first_der(0);
    double spline1_second_der = spline1.get_second_der(1);
    double spline2_second_der = spline2.get_second_der(0);
    // double spline1_third_der = spline1.get_third_der(1);
    double spline2_third_der = spline2.get_third_der(0);



    bool inffirst = false;
    if ((spline1_first_der == std::numeric_limits<double>::infinity() && abs(spline2_first_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_first_der == std::numeric_limits<double>::infinity()) && abs(spline1_first_der) >= CHUNK_INFINITY_THRESH){
            inffirst = true;
        }
    bool checkFirstDer = inffirst || (abs(spline1_first_der - spline2_first_der) <= CHUNK_FIRST_DER_THRESH);

    bool infsec = false;
    if ((spline1_second_der == std::numeric_limits<double>::infinity() && abs(spline2_second_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_second_der == std::numeric_limits<double>::infinity()) && abs(spline1_second_der) >= CHUNK_INFINITY_THRESH){
            infsec = true;
        }
    bool checkSecondDer = infsec || (abs(spline1_second_der - spline2_second_der) <= CHUNK_SECOND_DER_THRESH);

    bool infthird = false;
    if (((this->minThirdDer == std::numeric_limits<double>::infinity() && abs(spline2_third_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_third_der == std::numeric_limits<double>::infinity()) && abs(this->minThirdDer) >= CHUNK_INFINITY_THRESH )&&
        ((this->maxThirdDer == std::numeric_limits<double>::infinity() && abs(spline2_third_der) >= CHUNK_INFINITY_THRESH) || 
        (spline2_third_der == std::numeric_limits<double>::infinity()) && abs(this->maxThirdDer) >= CHUNK_INFINITY_THRESH )){
            infthird = true;
        }
    bool checkThirdDerMin = infthird || (abs(this->minThirdDer - spline2_third_der) <= CHUNK_THIRD_DER_THRESH);
    bool checkThirdDerMax = infthird || (abs(this->maxThirdDer - spline2_third_der) <= CHUNK_THIRD_DER_THRESH);

    // if (checkFirstDer == false) {
    //     std::cout << "check first failed " << std::endl;
    // } else if (checkSecondDer == false) {
    //     std::cout << "check second failed " << std::endl;
    // } else if (checkThirdDerMin == false) {
    //     std::cout << "check third min failed " << std::endl;
    //     std::cout << "3rd min" << this->minThirdDer << std::endl;
    //     std::cout << "curr 3rd" << spline2_third_der << std::endl;
    // } else if (checkThirdDerMax == false) {
    //     std::cout << "check third max failed " << std::endl;
    //     std::cout << "3rd max" << this->maxThirdDer << std::endl;
    //     std::cout << "curr 3rd" << spline2_third_der << std::endl;
    // }

    bool res = checkFirstDer && checkSecondDer && checkThirdDerMax && checkThirdDerMin;

    if (res) {
        if (spline2_third_der < this->minThirdDer) {
            this->minThirdDer = spline2_third_der;
        }
        else if (spline2_third_der > this->maxThirdDer) {
            this->maxThirdDer = spline2_third_der;
        }

    } else {
        // std::cout << "3rd diff with min" << abs(this->minThirdDer - spline2_third_der) << std::endl;
        // std::cout << "3rd diff with max" << abs(this->maxThirdDer - spline2_third_der) << std::endl;

        // std::cout << "2nd diff with prev spline" << abs(spline1_second_der - spline2_second_der) << std::endl;
    }

    return res;
}

/**
 * Return the best match t that corresponds to the arclength in the spline.
 * 
 * Arclength from start of spline to t should be targetArclength.
 * 
 * @param spline spline that t should reside in
 * @param targetArclength arclength from start of spline to t should be targetArclength
 */
double tInterpolate(ParameterizedSpline spline, double targetArclength) {
    std::pair<polynomial, polynomial> splinePair = std::make_pair(spline.spline_x.first_der, spline.spline_y.first_der);
    double low = 0;
    double high = 1;
    double mid;
    double curArclength;
    while (high-low >= 0.001) {
        double mid = low + (high-low) / 2;
        double curArclength = arclength(splinePair, 0, mid);
        if (abs(curArclength - targetArclength) <= 0.001) {
            return mid;
        }
        else if (curArclength < targetArclength) {
            low = mid + 0.001;
        }
        else {
            high = mid;
        }
    }
    return low + (high-low) / 2;
}

/** 
 * Generates a vector of raceline chunks based on track boundaries.
 *
 * @param blueCones Vector of blue cone points.
 * @param yellowCones Vector of yellow cone points.
 * 
 * @return Vector of raceline chunks.
 */

//TODO: should be returning by reference not value
std::vector<Chunk*>* generateChunks(std::vector<std::pair<double,double>> blueCones,
                                  std::vector<std::pair<double,double>> yellowCones) {

    // create chunk vector that stores chunks
    //TODO: use new keyword to create vector in heap not stack
    std::vector<Chunk*>* chunkVector = new std::vector<Chunk*>();

    /* Getting the polynomials/splines for each track bound*/
    /* Pass in all of the blue and yellow cones, */
    auto start = std::chrono::high_resolution_clock::now();
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> blue = make_splines_vector(blueCones);
    std::pair<std::vector<ParameterizedSpline>,std::vector<double>> yellow = make_splines_vector(yellowCones);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Make splines vector time " << duration.count() << " microseconds" << std::endl;


    std::vector<ParameterizedSpline> blueRacetrackSplines = blue.first;
    std::vector<double> blueCumulativeLen = blue.second;

    std::vector<ParameterizedSpline> yellowRacetrackSplines = yellow.first;
    std::vector<double> yellowCumulativeLen = yellow.second;

    // create a chunk
    Chunk* chunk = new Chunk();
    chunk->minThirdDer = blueRacetrackSplines[0].get_third_der(0);
    chunk->maxThirdDer = chunk->minThirdDer;
    chunk->blueSplines.push_back(blueRacetrackSplines[0]);
    chunk->tStart = 0;
    chunk->blueArclengthStart = 0;
    chunk->blueFirstSplineArclength = blueCumulativeLen[0];
    
    double bluePercentProgress;
    int yellowSplineIdx = 0;

    int blueIdx = 0;
    int yellowIdx = 0;

    /* Track the max time it takes for determining whether to continue to chunk */
    auto max_continue_chunk_time = std::numeric_limits<int>::min();
    auto max_get_yellow_time = std::numeric_limits<int>::min();
    auto max_split_yellow_time = std::numeric_limits<int>::min();
    auto max_define_chunk_fields_time = std::numeric_limits<int>::min();

    auto start_chunking = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= blueRacetrackSplines.size(); i++) {
        // add spline to chunk
        auto start_continue_chunk = std::chrono::high_resolution_clock::now();
        bool check_continue = i < blueRacetrackSplines.size() && (chunk->continueChunk(blueRacetrackSplines[i-1], blueRacetrackSplines[i]));
        auto end_continue_chunk = std::chrono::high_resolution_clock::now();
        auto dur_continue_chunk = std::chrono::duration_cast<std::chrono::microseconds>(end_continue_chunk - start_continue_chunk);
        if (max_continue_chunk_time < dur_continue_chunk.count()) {
            max_continue_chunk_time = dur_continue_chunk.count();
        }

        if (check_continue) {
            chunk->blueSplines.push_back(blueRacetrackSplines[i]);
        }
        // stop current chunk, add to vector, start new chunk
        else { 
            chunk->blueArclengthEnd = blueCumulativeLen[i - 1];
            bluePercentProgress = blueCumulativeLen[i - 1] / blueCumulativeLen[blueCumulativeLen.size() - 1];
            
            // GET CORRESPONDING YELLOW CHUNK
            // yellowindex is greater than yellowRacetrackSplines or 
            // cumsum is greater than cumsum of blue;yellowSplineIdx

            auto start_get_yellow = std::chrono::high_resolution_clock::now();
            while ((yellowSplineIdx < yellowRacetrackSplines.size()) && 
                (yellowCumulativeLen[yellowSplineIdx] < yellowCumulativeLen[yellowCumulativeLen.size() - 1] * bluePercentProgress)) {
                chunk->yellowSplines.push_back(yellowRacetrackSplines[yellowSplineIdx]);
                yellowSplineIdx++;
            }
            auto end_get_yellow = std::chrono::high_resolution_clock::now();
            auto dur_get_yellow = std::chrono::duration_cast<std::chrono::microseconds>(end_get_yellow - start_get_yellow);
            if (max_get_yellow_time < dur_get_yellow.count()) {
                max_get_yellow_time = dur_get_yellow.count();
            }


            double nextTStart = 0;

            // split yellow if yellow spline included in chunk is longer than curr blue

            if (yellowSplineIdx < yellowRacetrackSplines.size()) {
                ParameterizedSpline splitSpline = yellowRacetrackSplines[yellowSplineIdx];

                double yellowStartLen = 0;

                if (yellowSplineIdx != 0) {
                    yellowStartLen = yellowCumulativeLen[yellowSplineIdx - 1];
                }
                
                auto start_split_yellow = std::chrono::high_resolution_clock::now();
                double splitT = tInterpolate(splitSpline, (bluePercentProgress * yellowCumulativeLen[yellowCumulativeLen.size() - 1]) - yellowStartLen);
                auto end_split_yellow = std::chrono::high_resolution_clock::now();
                auto dur_split_yellow = std::chrono::duration_cast<std::chrono::microseconds>(end_split_yellow - start_split_yellow);
                if (max_split_yellow_time < dur_split_yellow.count()) {
                    max_split_yellow_time = dur_split_yellow.count();
                }



                chunk->tEnd = splitT;

                chunk->yellowSplines.push_back(splitSpline);

                if (splitT == 1) {
                    nextTStart = 0;
                    yellowSplineIdx++;
                }
                else {
                    nextTStart = splitT;
                }
            }
            



            // FIND ALL FIELDS NEEDED FOR A CHUNK

            auto start_define_chunk_fields = std::chrono::high_resolution_clock::now();
            chunkVector->emplace_back(chunk);

            chunk->blueArclength = chunk->blueArclengthEnd - chunk->blueArclengthStart;
            chunk->yellowArclength = chunk->blueArclength * yellowCumulativeLen[yellowCumulativeLen.size() - 1] / blueCumulativeLen[blueCumulativeLen.size() - 1];

            chunk->blueFirstDerXStart = poly_eval(chunk->blueSplines[0].spline_x.first_der, 0);
            chunk->blueFirstDerXEnd = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_x.first_der, 1);
            chunk->blueFirstDerYStart = poly_eval(chunk->blueSplines[0].spline_y.first_der, 0);
            chunk->blueFirstDerYEnd = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_y.first_der, 1);

            chunk->yellowFirstDerXStart = poly_eval(chunk->yellowSplines[0].spline_x.first_der, chunk->tStart);
            chunk->yellowFirstDerXEnd = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_x.first_der, chunk->tEnd);
            chunk->yellowFirstDerYStart = poly_eval(chunk->yellowSplines[0].spline_y.first_der, chunk->tStart);
            chunk->yellowFirstDerYEnd = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_y.first_der, chunk->tEnd);
            auto end_define_chunk_fields = std::chrono::high_resolution_clock::now();
            auto dur_define_chunk_fields = std::chrono::duration_cast<std::chrono::microseconds>(end_define_chunk_fields - start_define_chunk_fields);
            if (max_define_chunk_fields_time < dur_define_chunk_fields.count()) {
                max_define_chunk_fields_time = dur_define_chunk_fields.count();
            }

            double blueArcStart = chunk->blueArclengthEnd;

            // FIND MIDPOINTS
            while (blueCumulativeLen[blueIdx] < (chunk->blueArclength/2 + chunk->blueArclengthStart)) {
                blueIdx += 1;
            }

            double midFromMidSpline = (chunk->blueArclength/2 + chunk->blueArclengthStart);
            if (blueIdx != 0) {
                midFromMidSpline = midFromMidSpline - blueCumulativeLen[blueIdx-1];
            }

            // binary search from start of blueIdx spline 
            double midT = tInterpolate(blueRacetrackSplines[blueIdx], midFromMidSpline);

            chunk->blueMidX = poly_eval(blueRacetrackSplines[blueIdx].spline_x.spl_poly, midT);
            chunk->blueMidY = poly_eval(blueRacetrackSplines[blueIdx].spline_y.spl_poly, midT);
            chunk->blueStartX = poly_eval(chunk->blueSplines[0].spline_x.spl_poly, 0);
            chunk->blueStartY = poly_eval(chunk->blueSplines[0].spline_y.spl_poly, 0);
            chunk->blueEndX = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_x.spl_poly, 1);
            chunk->blueEndY = poly_eval(chunk->blueSplines[chunk->blueSplines.size() - 1].spline_y.spl_poly, 1);
            chunk->blueFirstDerMidX = poly_eval(blueRacetrackSplines[blueIdx].spline_x.first_der, midT);
            chunk->blueFirstDerMidY = poly_eval(blueRacetrackSplines[blueIdx].spline_y.first_der, midT);

            double yellowEndLength = yellowCumulativeLen[yellowCumulativeLen.size() - 1];

            if (i != blueRacetrackSplines.size()) {
                // yellow midpoint and tangent
                ParameterizedSpline yellowSpline = yellowRacetrackSplines[yellowSplineIdx];
                yellowEndLength = arclength(std::make_pair(yellowSpline.spline_x.first_der, yellowSpline.spline_y.first_der), 0, chunk->tEnd);
                if (yellowSplineIdx > 0) {
                    yellowEndLength += yellowCumulativeLen[yellowSplineIdx - 1];
                }
            }

            while (yellowCumulativeLen[yellowIdx] < (yellowEndLength - chunk->yellowArclength/2)) {
                yellowIdx += 1;
            }

            // spline containing
            midFromMidSpline = (yellowEndLength - chunk->yellowArclength/2);
            if (yellowIdx != 0) {
                midFromMidSpline = midFromMidSpline - yellowCumulativeLen[yellowIdx-1];
            }
            
            // binary search from start of yellowIdx spline 
            midT = tInterpolate(yellowRacetrackSplines[yellowIdx], midFromMidSpline);

            chunk->yellowMidX = poly_eval(yellowRacetrackSplines[yellowIdx].spline_x.spl_poly, midT);
            chunk->yellowMidY = poly_eval(yellowRacetrackSplines[yellowIdx].spline_y.spl_poly, midT);
            chunk->yellowStartX = poly_eval(chunk->yellowSplines[0].spline_x.spl_poly, chunk->tStart);
            chunk->yellowStartY = poly_eval(chunk->yellowSplines[0].spline_y.spl_poly, chunk->tStart);
            chunk->yellowEndX = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_x.spl_poly, chunk->tEnd);
            chunk->yellowEndY = poly_eval(chunk->yellowSplines[chunk->yellowSplines.size() - 1].spline_y.spl_poly, chunk->tEnd);
            chunk->yellowFirstDerMidX = poly_eval(yellowRacetrackSplines[yellowIdx].spline_x.first_der, midT);
            chunk->yellowFirstDerMidY = poly_eval(yellowRacetrackSplines[yellowIdx].spline_y.first_der, midT);
            if (chunk->yellowSplines.size() == 1) {
                chunk->yellowFirstSplineArclength = arclength(std::make_pair(chunk->yellowSplines[0].spline_x.spl_poly, chunk->yellowSplines[0].spline_y.spl_poly), chunk->tStart, chunk->tEnd);
                chunk->yellowLastSplineArclength = arclength(std::make_pair(chunk->yellowSplines[0].spline_x.spl_poly, chunk->yellowSplines[0].spline_y.spl_poly), chunk->tStart, chunk->tEnd);
            }
            else {
                chunk->yellowFirstSplineArclength = arclength(std::make_pair(chunk->yellowSplines[0].spline_x.spl_poly, chunk->yellowSplines[0].spline_y.spl_poly), chunk->tStart, 1);
                chunk->yellowLastSplineArclength = arclength(std::make_pair(chunk->yellowSplines[chunk->yellowSplines.size()-1].spline_x.spl_poly, chunk->yellowSplines[chunk->yellowSplines.size()-1].spline_y.spl_poly), 0, chunk->tEnd);
            }

            if (i != blueRacetrackSplines.size()) {
                if (i > 1) {
                    chunk->blueLastSplineArclength = blueCumulativeLen[i-1] - blueCumulativeLen[i-2];
                }
                else {
                     chunk->blueLastSplineArclength = blueCumulativeLen[0];
                }

                // std::cout << "blue first spline leng" << chunk->blueFirstSplineArclength << std::endl;
                // std::cout << "blue last spline leng" << chunk->blueLastSplineArclength << std::endl;
                // std::cout << "yellow first spline leng" << chunk->yellowFirstSplineArclength << std::endl;
                // std::cout << "yellow last spline leng" << chunk->yellowLastSplineArclength << std::endl;

                chunk = new Chunk();
                chunk->tStart = nextTStart;
                chunk->blueArclengthStart = blueArcStart;
                // init chunk and add curr spline
                chunk->blueSplines.push_back(blueRacetrackSplines[i]);
                chunk->minThirdDer = blueRacetrackSplines[i].get_third_der(0);
                chunk->maxThirdDer = chunk->minThirdDer;
                chunk->blueFirstSplineArclength = blueCumulativeLen[i]-blueCumulativeLen[i-1];
            }
        }
    }
    auto end_chunking = std::chrono::high_resolution_clock::now();
    auto dur_chunking = std::chrono::duration_cast<std::chrono::microseconds>(end_chunking - start_chunking);
    std::cout << "*************************************" << std::endl;
    std::cout << "generateChunks: chunking loop time: " << dur_chunking.count() << " microseconds" << std::endl;
    std::cout << "Per iteration: " << std::endl;
    std::cout << "generateChunks: max_continue_chunk_time: " << max_continue_chunk_time << " microseconds;" << std::endl;
    std::cout << "generateChunks: max_get_yellow_time: " << max_get_yellow_time << " microseconds" << std::endl;
    std::cout << "generateChunks: max_split_yellow_time tInterpolate: " << max_split_yellow_time <<  " microseconds" << std::endl;
    std::cout << "generateChunks: max_define_chunk_fields_time: " << max_define_chunk_fields_time << " microseconds" << std::endl;
    std::cout << "\nWorst case one whole run" << std::endl;
    std::cout << "generateChunks: max_continue_chunk_time: " << max_continue_chunk_time * blueRacetrackSplines.size() << " microseconds;" << std::endl;
    std::cout << "generateChunks: max_get_yellow_time tInterpolate: " << max_get_yellow_time * blueRacetrackSplines.size()<< " microseconds" << std::endl;
    std::cout << "generateChunks: max_split_yellow_time: " << max_split_yellow_time* blueRacetrackSplines.size() <<  " microseconds" << std::endl;
    std::cout << "generateChunks: max_define_chunk_fields_time: " << max_define_chunk_fields_time* blueRacetrackSplines.size() << " microseconds" << std::endl;

    std::cout << "Num blueRacetrackSplines: " << blueRacetrackSplines.size() << std::endl;
    std::cout << "*************************************" << std::endl;


    return chunkVector;
}


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

//Function to find raceline


double calculateEnd(Chunk& chunk, double start) {
    return 0.5;
}

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

/**
 * Evaluates a cubic polymomial at some input 
 * 0th index is the 3rd order term
 * 1st index is the 2nd order term
 * 2nd index is the 1st order term 
 * 3rd index is the 0th order term (constant)
 * 
 */
double poly_at(std::vector<double> p, double t) {
    return p.at(0) * pow(t, 3) + p.at(1) * pow(t, 2) + p.at(2) * t + p.at(3);
}

std::vector<Point> interpolatePoints (std::vector<double> x1_polynomial, std::vector<double> x2_polynomial,
                                        std::vector<double> y1_polynomial, std::vector<double> y2_polynomial) {
    double increments = 0.001;
    std::vector<Point> result = {};
    for (double i = 0; i < 0.5; i += increments ) {
        double x = poly_at(x1_polynomial, i);
        double y = poly_at(y1_polynomial, i);
        result.emplace_back(x, y);


    }
    
    for (double i = 0.5; i < 1; i += increments ) {
        double x = poly_at(x2_polynomial, i);
        double y = poly_at(y2_polynomial, i);
        result.emplace_back(x, y);

    }

    return result;
}


std::vector<Point> controlsGenerateMidline(std::vector<Point> blue_cones, std::vector<Point> yellow_cones, double d1, double d2, double d3) {
    Point blueStart = blue_cones.at(0);
    Point yellowStart = yellow_cones.at(0);
    Point blueStartNext = blue_cones.at(1);
    Point yellowStartNext = yellow_cones.at(1);

    Point blueMid = blue_cones.at(blue_cones.size() / 2);
    Point yellowMid = yellow_cones.at(yellow_cones.size() / 2);

    Point blueEnd = blue_cones.at(blue_cones.size() - 1);
    Point yellowEnd = yellow_cones.at(yellow_cones.size() - 1);
    Point blueEndPrev = blue_cones.at(blue_cones.size() - 2);
    Point yellowEndPrev = yellow_cones.at(yellow_cones.size() - 2);

    double dt = 1.0 / blue_cones.size();

    double blueFirstDerXStart = ((double)(blueStartNext.first - blueStart.first)) / dt;
    double blueFirstDerXEnd =   ((double)(blueEnd.first - blueEndPrev.first)) / dt;

    double blueFirstDerYStart = ((double)(blueStartNext.first - blueStart.first)) /dt;
    double blueFirstDerYEnd = ((double)(blueEnd.second - blueEndPrev.second)) / dt;

    double yellowFirstDerXStart = ((double)(yellowStartNext.first - yellowStart.first)) / dt;
    double yellowFirstDerXEnd =   ((double)(yellowEnd.first - yellowEndPrev.first)) / dt;
    
    double yellowFirstDerYStart = ((double)(yellowStartNext.first - yellowStart.first)) / dt;
    double yellowFirstDerYEnd =   ((double)(yellowEnd.second - yellowEndPrev.second)) / dt;

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
    
    std::vector<double> x1_polynomial = std::vector<double>(splines[0].data(), splines[0].data() + splines[0].size());   // First X spline (first half)
    std::vector<double> x2_polynomial = std::vector<double>(splines[1].data(), splines[1].data() + splines[1].size());   // Second X spline (second half)
    std::vector<double> y1_polynomial = std::vector<double>(splines[2].data(), splines[2].data() + splines[2].size());   // First Y spline (first half)
    std::vector<double> y2_polynomial = std::vector<double>(splines[3].data(), splines[3].data() + splines[3].size());   // Second Y spline (second half)
    
    std::vector<Point> result = interpolatePoints(x1_polynomial, x2_polynomial, y1_polynomial, y2_polynomial);
    return result;

}
