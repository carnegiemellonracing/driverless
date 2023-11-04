#include "generator.hpp"
#include <cassert>
#include <stdio.h>

int main(){
    //test case 1:
    printf("test case 1: \n");
    gsl_matrix *left = gsl_matrix_alloc(2,1);
    gsl_matrix *right = gsl_matrix_alloc(2,1);
    gsl_matrix_set(left, 0, 0, 0);
    gsl_matrix_set(left, 1, 0, 0);
    gsl_matrix_set(right, 0, 0, 0);
    gsl_matrix_set(right, 1, 0, 1);
    gsl_matrix *result = midpoint(left, right);
    assert(gsl_matrix_get(result, 0, 0, 0));
    assert(gsl_matrix_get(result, 1, 0, 0.5));
    free(left);
    free(right);
    free(result);


    //test case 2:
    printf("test case 2: \n");
    gsl_matrix *left = gsl_matrix_alloc(2,2);
    gsl_matrix *right = gsl_matrix_alloc(2,1);
    gsl_matrix_set(left, 0, 0, 0);
    gsl_matrix_set(left, 1, 0, 0);
    gsl_matrix_set(left, 0, 1, 1);
    gsl_matrix_set(left, 1, 1, 0);

    gsl_matrix_set(right, 0, 0, 0);
    gsl_matrix_set(right, 1, 0, 1);

    gsl_matrix *result = midpoint(left, right);

    assert(result->size2 == 2);
    assert(gsl_matrix_get(result, 0, 0, 0));
    assert(gsl_matrix_get(result, 1, 0, 0.5));
    assert(gsl_matrix_get(result, 0, 1, 0.5));
    assert(gsl_matrix_get(result, 1, 1, 0.5));
    free(left);
    free(right);
    free(result);

    //test case 3:
    printf("test case 3: \n");
    gsl_matrix *left = gsl_matrix_alloc(2,2);
    gsl_matrix *right = gsl_matrix_alloc(2,2);
    gsl_matrix_set(left, 0, 0, 0);
    gsl_matrix_set(left, 1, 0, 0);
    gsl_matrix_set(left, 0, 1, 1);
    gsl_matrix_set(left, 1, 1, 0);

    gsl_matrix_set(right, 0, 0, 0);
    gsl_matrix_set(right, 1, 0, 1);
    gsl_matrix_set(right, 0, 0, 1);
    gsl_matrix_set(right, 1, 0, 1);

    gsl_matrix *result = midpoint(left, right);

    assert(result->size2 == 3);
    assert(gsl_matrix_get(result, 0, 0, 0));
    assert(gsl_matrix_get(result, 1, 0, 0.5));
    assert(gsl_matrix_get(result, 0, 1, 0.5));
    assert(gsl_matrix_get(result, 1, 1, 0.5));
    assert(gsl_matrix_get(result, 0, 2, 1));
    assert(gsl_matrix_get(result, 1, 2, 0.5));
    free(left);
    free(right);
    free(result);

    //test case 4:
    printf("test case 4: \n");
    gsl_matrix *left = gsl_matrix_alloc(2,3);
    gsl_matrix *right = gsl_matrix_alloc(2,3);
    gsl_matrix_set(left, 0, 0, 0);
    gsl_matrix_set(left, 1, 0, 2);
    gsl_matrix_set(left, 0, 1, 1.414);
    gsl_matrix_set(left, 1, 1, 1.414);
    gsl_matrix_set(left, 0, 1, 2);
    gsl_matrix_set(left, 1, 1, 0);

    gsl_matrix_set(left, 0, 0, 0);
    gsl_matrix_set(left, 1, 0, 3);
    gsl_matrix_set(left, 0, 1, 1.414);
    gsl_matrix_set(left, 1, 1, 2.236);
    gsl_matrix_set(left, 0, 1, 3);
    gsl_matrix_set(left, 1, 1, 0);


    gsl_matrix *result = midpoint(left, right);

    assert(result->size2 == 5);
    assert(gsl_matrix_get(result, 0, 0, 0));
    assert(gsl_matrix_get(result, 1, 0, 2.5));
    assert(gsl_matrix_get(result, 0, 1, 0.707));
    assert(gsl_matrix_get(result, 1, 1, 2.118));
    assert(gsl_matrix_get(result, 0, 2, 1.414));
    assert(gsl_matrix_get(result, 1, 2, 1.825));
    assert(gsl_matrix_get(result, 0, 3, 2.207));
    assert(gsl_matrix_get(result, 1, 3, 0.707));
    assert(gsl_matrix_get(result, 0, 4, 2.5));
    assert(gsl_matrix_get(result, 1, 4, 0));
    free(left);
    free(right);
    free(result);

}
