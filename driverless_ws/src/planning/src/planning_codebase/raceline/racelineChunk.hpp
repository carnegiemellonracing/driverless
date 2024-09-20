#include <vector>

/**
 * Chunks represent segments of the track where all points have similar curvature. 
 */
class Chunk
{
    public:
        double startProgress; // start of the chunk
        double endProgress; // end of the chunk
        double sumCurvature; // sum of curvatures at sample points

        double avgCurvature; // running avg of chunk
        std::vector<std::pair<double, double>> bluePoints; // interpolated points on the blue cone spline
        std::vector<std::pair<double, double>> yellowPoints; // interpolated points on the yellow cone spline  

        Chunk();

        /** 
         * Calculate curvature running average.
         */
        double calcRunningAvgCurvature();

        /** 
         * Checks if given chunk should be terminated, i.e. running average of
         * given chunk is significantly different from the given curvature
         * sample or the chunk is too long. 
         *
         * @param newCurvature Curvature of new point.
         * 
         * @return True if this chunk should be terminated and should not
         *         include given curvature point, false otherwise.
         */
        bool checkStopChunk(double newCurvature);

        /** 
         * Populate the point vectors in the chunk by interpolating points along
         * the given splines.
         *
         * @param blueRaceline Vectors of splines and cumulative lengths for blue cones.
         * @param yellowRaceline Vectors of splines and cumulative lengths for yellow
         *                       cones.
         */
        void generateConePoints(std::pair<std::vector<Spline>,std::vector<double>> blueRaceline,
                                std::pair<std::vector<Spline>,std::vector<double>> yellowRaceline);
}

/** 
 * Generates a vector of raceline chunks based on track boundaries.
 *
 * @param blueCones Vector of blue cone points.
 * @param yellowCones Vector of yellow cone points.
 * 
 * @return Vector of raceline chunks.
 */
std::vector<Chunk> generateChunks(std::vector<std::pair<double,double>> blueCones,
                                  std::vector<std::pair<double,double>> yellowCones);
    