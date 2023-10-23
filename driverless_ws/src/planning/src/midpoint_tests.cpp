#include "generator.cpp"
#include <cassert>
#include <stdio.h>

int main(){
    //test case 1:
    printf("test case 1: \n");
    gsl_matrix *left_1 = gsl_matrix_alloc(2,1);
    gsl_matrix *right_1 = gsl_matrix_alloc(2,1);
    gsl_matrix_set(left_1, 0, 0, 0);
    gsl_matrix_set(left_1, 1, 0, 0);
    gsl_matrix_set(right_1, 0, 0, 0);
    gsl_matrix_set(right_1, 1, 0, 1);
    gsl_matrix *result = midpoint(left_1, right_1);
    assert(gsl_matrix_get(result, 0, 0, 0));
    assert(gsl_matrix_get(result, 1, 0, 0.5));

    //test case 2:
    printf("test case 2: \n");
    gsl_matrix *left_1 = gsl_matrix_alloc(2,2);
    gsl_matrix *right_1 = gsl_matrix_alloc(2,1);
    gsl_matrix_set(left_1, 0, 0, 0);
    gsl_matrix_set(left_1, 1, 0, 0);
    gsl_matrix_set(left_1, 0, 1, 1);
    gsl_matrix_set(left_1, 1, 1, 0);

    gsl_matrix_set(right_1, 0, 0, 0);
    gsl_matrix_set(right_1, 1, 0, 1);
    gsl_matrix *result = midpoint(left_1, right_1);
    assert(gsl_matrix_get(result, 0, 0, 0));
    assert(gsl_matrix_get(result, 1, 0, 0.5));

}
