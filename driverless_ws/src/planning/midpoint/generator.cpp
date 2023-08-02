#include "generator.hpp"

MidpointGenerator::MidpointGenerator(int interpolation_num){
    interpolation_number=interpolation_num;

}

gsl_matrix *sorted_by_norm(gsl_matrix *list){
    gsl_matrix *res = gsl_matrix_alloc(list->size1,list->size2);
    for(int i=0;i<list->size1;i++)
        for(int j=0;j<list->size1;j++)
        gsl_matrix_set(res,i,j,gsl_matrix_get(list,i,j));

// IMPLEMENT SORT
}


gsl_matrix *midpoint(gsl_matrix *inner,gsl_matrix *outer){
    gsl_matrix *midpt = gsl_matrix_alloc(inner->size1,inner->size2);
    for(int i=0;i<midpt->size1;i++)
        for(int j=0;j<midpt->size1;j++)
        gsl_matrix_set(midpt,i,j,(gsl_matrix_get(inner,i,j)+gsl_matrix_get(outer,i,j))/2);

    return midpt;
}


std::vector<Spline> generate_splines(gsl_matrix *midpoints){
    std::pair<std::vector<Spline>,std::vector<int>> a= raceline_gen(midpoints,midpoints->size2,false);

    
    


}



gsl_matrix *generate_points(gsl_matrix *perceptions_data){

    

}