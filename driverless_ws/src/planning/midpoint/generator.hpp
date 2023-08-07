#include "raceline.hpp"
#include "random.h"

#ifndef MIDPOINTGENERATOR
#define MIDPOINTGENERATOR

struct perceptionsData{

    std::vector<std::pair<double,double>> bluecones,yellowcones,orangecones;
};

class MidpointGenerator
{
private:
    /* data */



public:
    int PERCEP_COLOR = 2;
    int BLUE = 1;
    int YELLOW = 2;
    int ORANGE = 3;
    int interpolation_number;
    std::vector<Spline> cumulated_splines;
    std::vector<int> cumulated_lengths;


    MidpointGenerator(int interpolation_number=30);

    ~MidpointGenerator();
    gsl_matrix *sorted_by_norm(gsl_matrix *list);

    std::vector<Spline> generate_splines(gsl_matrix *midpoints);    
    gsl_matrix* generate_points(perceptionsData perceptions_data);  
    // gsl_matrix *generate_points(perceptionsData *perceptions_data);

    std::vector<float> interpolate_cones(gsl_matrix *perceptions_data);
    Spline spline_from_cones(perceptionsData *perceptions_data);

};

gsl_matrix *midpoint(gsl_matrix *inner,gsl_matrix *outer);

#endif


