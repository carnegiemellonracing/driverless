#include <vector>
#include "raceline.hpp"

/**
 * Chunks represent segments of the track where all points have similar curvature. 
 */
class Chunk
{
    public:
        // t start and end are only representative of yellow splines, 
        double tStart; // t start of the first spline in yellowSplines
        double tEnd; // t end for the last spline in yellowSplines
        std::vector<ParameterizedSpline> blueSplines;
        std::vector<ParameterizedSpline> yellowSplines;

        Chunk();

        /** 
         * Checks if given chunk should be terminated, i.e. derivatives from endpoint of 
         * previous spline match the first point of current spline
         *
         * @param newCurvature Curvature of new point.
         * 
         * @return True if this chunk should be terminated and not include 
         * the curr spline, false otherwise.
         */
        bool checkContinueChunk(ParameterizedSpline spline1, ParameterizedSpline spline2);
};

/** 
 * Generates a vector of raceline chunks based on track boundaries.
 *
 * @param blueCones Vector of blue cone points.
 * @param yellowCones Vector of yellow cone points.
 * 
 * @return Vector of raceline chunks.
 */
std::vector<Chunk*>* generateChunks(std::vector<std::pair<double,double>> blueCones,
                                  std::vector<std::pair<double,double>> yellowCones);
    