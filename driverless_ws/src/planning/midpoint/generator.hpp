#include "raceline.hpp"

#ifndef MIDPOINTGENERATOR
#define MIDPOINTGENERATOR

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

    gsl_matrix *midpoint(gsl_matrix *inner,gsl_matrix *outer);
    std::vector<Spline> generate_splines(gsl_matrix *midpoints);
    gsl_matrix *generate_points(gsl_matrix *perceptions_data);

    std::vector<float> interpolate_cones(gsl_matrix *perceptions_data);
    Spline spline_from_cones(gsl_matrix *perceptions_data);

};

#endif


