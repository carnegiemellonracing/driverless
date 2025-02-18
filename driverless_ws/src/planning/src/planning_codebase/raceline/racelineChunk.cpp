#include "racelineChunk.hpp"

#include <math.h>
#include <chrono>
// #include "../midline/generator.hpp"

// 0 if use tEstimate, 1 if use tInterpolate
#define USE_T_INTERPOLATE 0

// tunable params for chunks
// thresholds are arclength in meters
#define CHUNK_LEN_MAX_THRESH 1000
#define CHUNK_LEN_MIN_THRESH 0
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

    bool res = checkFirstDer && checkSecondDer && checkThirdDerMax && checkThirdDerMin;

    if (res) {
        if (spline2_third_der < this->minThirdDer) {
            this->minThirdDer = spline2_third_der;
        }
        else if (spline2_third_der > this->maxThirdDer) {
            this->maxThirdDer = spline2_third_der;
        }

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

// TODO arc, tEstimate
// takes arclength of spline (linear method) is the current spline arclength
// take in goal arclength, and return t value that
double tEstimate(double currArclength, double targetArclength) {
    if (currArclength == 0) {
        return 0;
    }
    return targetArclength/currArclength;
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
std::vector<Chunk*>* generateChunks(std::vector<std::tuple<double,double,int>> blueCones,
                                  std::vector<std::tuple<double,double,int>> yellowCones) {
    
    auto start_generate_chunks = std::chrono::high_resolution_clock::now();
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

    // swap yellow and blue if curr blue is not the inside
    bool blueIsInside = blueCumulativeLen[blueCumulativeLen.size() - 1] <= yellowCumulativeLen[yellowCumulativeLen.size() - 1];

    if (!blueIsInside) {
        std::vector<ParameterizedSpline> tempRaceSplines = blueRacetrackSplines;
        std::vector<double> tempLen = blueCumulativeLen;
        blueRacetrackSplines = yellowRacetrackSplines;
        blueCumulativeLen = yellowCumulativeLen;
        yellowRacetrackSplines = tempRaceSplines;
        yellowCumulativeLen = tempLen;
    }

    // create a chunk
    Chunk* chunk = new Chunk();
    chunk->minThirdDer = blueRacetrackSplines[0].get_third_der(0);
    chunk->maxThirdDer = chunk->minThirdDer;
    chunk->blueSplines.push_back(blueRacetrackSplines[0]);
    chunk->blueConeIds.push_back(blueRacetrackSplines[0].start_cone_id);
    chunk->tStart = 0;
    chunk->blueArclengthStart = 0;
    chunk->blueFirstSplineArclength = blueCumulativeLen[0];
    
    double bluePercentProgress;
    int yellowSplineIdx = 0;

    int blueIdx = 0;
    int yellowIdx = 0;

    bool yellowSplineIdxChange = true;

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
            chunk->blueConeIds.push_back(blueRacetrackSplines[i].start_cone_id);
        }
        // stop current chunk, add to vector, start new chunk
        else { 
            chunk->blueArclengthEnd = blueCumulativeLen[i - 1];
            bluePercentProgress = blueCumulativeLen[i - 1] / blueCumulativeLen[blueCumulativeLen.size() - 1];
            
            // GET CORRESPONDING YELLOW CHUNK
            // yellowindex is greater than yellowRacetrackSplines or 
            // cumsum is greater than cumsum of blue;yellowSplineIdx

            auto start_get_yellow = std::chrono::high_resolution_clock::now();
            bool addToChunk = yellowSplineIdxChange;
            while ((yellowSplineIdx < yellowRacetrackSplines.size()) && (yellowCumulativeLen[yellowSplineIdx] < yellowCumulativeLen[yellowCumulativeLen.size() - 1] * bluePercentProgress)) {
                if (addToChunk) {
                    chunk->yellowConeIds.push_back(yellowRacetrackSplines[yellowSplineIdx].start_cone_id);
                } else {
                    addToChunk = true;
                }
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
                
                // TODO arc, tEstimate
                // takes arclength of spline (linear method) is the current spline arclength
                // take in goal arclength, and return t value that

                double splitT = 0;
                
                auto start_split_yellow = std::chrono::high_resolution_clock::now();
                if (USE_T_INTERPOLATE) {
                    splitT = tInterpolate(splitSpline, (bluePercentProgress * yellowCumulativeLen[yellowCumulativeLen.size() - 1]) - yellowStartLen);
                } else {
                    if (yellowSplineIdx > 0) {
                        splitT = tEstimate(yellowCumulativeLen[yellowSplineIdx] - yellowCumulativeLen[yellowSplineIdx - 1], (bluePercentProgress * yellowCumulativeLen[yellowCumulativeLen.size() - 1]) - yellowStartLen);
                    }
                    else {
                        splitT = tEstimate(yellowCumulativeLen[yellowSplineIdx], (bluePercentProgress * yellowCumulativeLen[yellowCumulativeLen.size() - 1]) - yellowStartLen);
                    }
                }

                auto end_split_yellow = std::chrono::high_resolution_clock::now();

                auto dur_split_yellow = std::chrono::duration_cast<std::chrono::microseconds>(end_split_yellow - start_split_yellow);
                if (max_split_yellow_time < dur_split_yellow.count()) {
                    max_split_yellow_time = dur_split_yellow.count();
                }
                
                chunk->tEnd = splitT;

                chunk->yellowSplines.push_back(splitSpline);
                chunk->yellowConeIds.push_back(splitSpline.start_cone_id);

                if (splitT == 1) {
                    nextTStart = 0;
                    yellowSplineIdx++;
                    yellowSplineIdxChange = true;
                }
                else {
                    nextTStart = splitT;
                    yellowSplineIdxChange = false;
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
            // TODO arc, use linear arclength
            double midT = 0;

            if (USE_T_INTERPOLATE) {
                midT = tInterpolate(blueRacetrackSplines[blueIdx], midFromMidSpline);
            } else {
                if (blueIdx > 0) {
                    midT = tEstimate(blueCumulativeLen[blueIdx] - blueCumulativeLen[blueIdx - 1], midFromMidSpline);
                }
                else {
                    midT = tEstimate(blueCumulativeLen[blueIdx], midFromMidSpline);
                }
            }
            

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

                if (USE_T_INTERPOLATE) {
                    yellowEndLength = arclength(std::make_pair(yellowSpline.spline_x.first_der, yellowSpline.spline_y.first_der), 0, chunk->tEnd);
                } else {
                    if (yellowSplineIdx > 0) {
                        yellowEndLength = (yellowCumulativeLen[yellowSplineIdx] - yellowCumulativeLen[yellowSplineIdx - 1]) * chunk->tEnd;
                    }
                    else {
                        yellowEndLength = (yellowCumulativeLen[yellowSplineIdx]) * chunk->tEnd;
                    }
                }

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
            if (USE_T_INTERPOLATE) {
                midT = tInterpolate(yellowRacetrackSplines[yellowIdx], midFromMidSpline);
            } else {
                if (yellowIdx > 0) {
                    midT = tEstimate(yellowCumulativeLen[yellowIdx] - yellowCumulativeLen[yellowIdx - 1], midFromMidSpline);
                }
                else {
                    midT = tEstimate(yellowCumulativeLen[yellowIdx], midFromMidSpline);
                }
            }
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

                chunk = new Chunk();
                chunk->tStart = nextTStart;
                chunk->blueArclengthStart = blueArcStart;
                // init chunk and add curr spline
                chunk->blueSplines.push_back(blueRacetrackSplines[i]);
                chunk->blueConeIds.push_back(blueRacetrackSplines[i].start_cone_id);
                chunk->minThirdDer = blueRacetrackSplines[i].get_third_der(0);
                chunk->maxThirdDer = chunk->minThirdDer;
                chunk->blueFirstSplineArclength = blueCumulativeLen[i]-blueCumulativeLen[i-1];
            }
        }
    }
    auto end_chunking = std::chrono::high_resolution_clock::now();
    auto dur_chunking = std::chrono::duration_cast<std::chrono::microseconds>(end_chunking - start_chunking);

    auto end_generate_chunks = std::chrono::high_resolution_clock::now();
    auto dur_generate_chunks = std::chrono::duration_cast<std::chrono::microseconds>(end_generate_chunks - start_generate_chunks);
    std::cout << "*************************************" << std::endl;
    std::cout << "generateChunks: chunking loop time: " << dur_chunking.count() << " microseconds" << std::endl;
    std::cout << "generateChunks: total time: " << dur_generate_chunks.count() << " microseconds" << std::endl;
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