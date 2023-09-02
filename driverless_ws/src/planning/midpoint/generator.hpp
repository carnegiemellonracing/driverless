#include "raceline.hpp"
#include "random.h"
#include <math.h>
#include <algorithm>

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
    std::vector<double> cumulated_lengths;


    MidpointGenerator(int interpolation_number=30);

    ~MidpointGenerator();
    std::vector<std::pair<double,double>> sorted_by_norm(std::vector<std::pair<double,double>> inp);
    // gsl_matrix *sorted_by_norm(gsl_matrix *list);

    std::vector<Spline> generate_splines(gsl_matrix *midpoints);    
    gsl_matrix* generate_points(perceptionsData perceptions_data);  
    gsl_matrix* interpolate_cones(perceptionsData perceptions_data, int interpolation_number = -1);
    Spline spline_from_cones(perceptionsData perceptions_data);
    Spline spline_from_cone_side(std::vector<std::pair<double,double>> side);

};

gsl_matrix *midpoint(gsl_matrix *inner,gsl_matrix *outer);

#endif


