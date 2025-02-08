#include <vector>
#include "raceline.hpp"
#define CHUNK_FILENAME "src/planning/src/planning_codebase/raceline"

/**
 * Chunks represent segments of the track where all points have similar curvature. 
 */
class Chunk
{
    public:
        // t start and end are only representative of yellow splines, 
        double tStart; // t start of the first spline in yellowSplines
        double tEnd; // t end for the last spline in yellowSplines

        std::vector<int> blueConeIds;
        std::vector<int> yellowConeIds;

        double minThirdDer;
        double maxThirdDer;

        double blueArclengthStart;
        double blueArclengthEnd;
        double blueArclength;
        double yellowArclength;

        // first derivative at endpoints
        double blueFirstDerXStart;
        double blueFirstDerXEnd;
        double blueFirstDerYStart;
        double blueFirstDerYEnd;

        double yellowFirstDerXStart;
        double yellowFirstDerXEnd;
        double yellowFirstDerYStart;
        double yellowFirstDerYEnd;

        double blueStartX;
        double blueStartY;
        double blueMidX;
        double blueMidY;
        double blueEndX;
        double blueEndY;
        double blueFirstDerMidX;
        double blueFirstDerMidY;

        double yellowStartX;
        double yellowStartY;
        double yellowMidX;
        double yellowMidY;
        double yellowEndX;
        double yellowEndY;
        double yellowFirstDerMidX;
        double yellowFirstDerMidY;

        double blueFirstSplineArclength;
        double blueLastSplineArclength;
        double yellowFirstSplineArclength;
        double yellowLastSplineArclength;

        

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
        bool continueChunk(ParameterizedSpline spline1, ParameterizedSpline spline2);
};

/** 
 * Generates a vector of raceline chunks based on track boundaries.
 *
 * @param blueCones Vector of blue cone points.
 * @param yellowCones Vector of yellow cone points.
 * 
 * @return Vector of raceline chunks.
 */
std::vector<Chunk*>* generateChunks(std::vector<std::tuple<double,double,int>> blueCones,
                                  std::vector<std::tuple<double,double,int>> yellowCones);
    