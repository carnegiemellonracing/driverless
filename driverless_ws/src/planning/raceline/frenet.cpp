#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <vector>

#include "frenet.hpp"
#include "raceline.hpp"

gsl_matrix *mat_mul(gsl_matrix *A, gsl_matrix *B) {
  assert(A->size2 == B->size1);
  gsl_matrix_view A_view = gsl_matrix_submatrix(A, 0, 0, A->size1, A->size2);
  gsl_matrix_view B_view = gsl_matrix_submatrix(A, 0, 0, B->size1, B->size2);
  gsl_matrix *C = gsl_matrix_calloc(A->size1, B->size2);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A_view.matrix,
                 &B_view.matrix, 0.0, C);
  return C;
}

double poly_eval(polynomial poly, double x) {
  return gsl_poly_eval(poly.nums->data, poly.deg + 1, x);
}
//Curvature at point(s) `min_x` based on 2d curvature equation https://mathworld.wolfram.com/Curvature.html
double get_curvature(polynomial poly_der_1, polynomial poly_der_2,
                     double min_x) {
  return poly_eval(poly_der_2, min_x) /
         (1 + pow(pow(poly_eval(poly_der_1, min_x), 2), 3 / 2));
}

//Rotates points based on the rotation and transformation matrices
std::vector<gsl_matrix *> rotate_points(
    std::vector<gsl_matrix *> points, std::vector<gsl_matrix *> poly_Qs,
    std::vector<gsl_matrix *> poly_transMats) {
  size_t n = points.size();
  assert(n == poly_Qs.size());
  assert(n == poly_transMats.size());
  std::vector<gsl_matrix *> rotated_points;
  for (size_t i = 0; i < n; ++i) {
    gsl_matrix *point = points[i];
    gsl_matrix_sub(point, poly_transMats[i]);
    point = mat_mul(point, poly_transMats[i]);
    rotated_points.push_back(point);
  }
  return rotated_points;
}

std::pair<double, double> get_closest_distance(
    double x, double y,
    std::vector<double> poly_coeffs,
    std::vector<double> poly_roots,
    int precision, int samples)
{

  size_t n = poly_coeffs->size1;
  assert(n == poly_roots->size1);


  // Extracting coefficients
  gsl_vector *a = gsl_vector_alloc(n);
  gsl_vector *b = gsl_vector_alloc(n);
  gsl_vector *c = gsl_vector_alloc(n);
  gsl_vector *d = gsl_vector_alloc(n);

  gsl_matrix_get_col(*a, *poly_coeffs, 0);
  gsl_matrix_get_col(*b, *poly_coeffs, 1);
  gsl_matrix_get_col(*c, *poly_coeffs, 2);
  gsl_matrix_get_col(*d, *poly_coeffs, 3);


  // Computing distance function's coefficients

  /*
    c1 = x**2 + y**2 - 2*y*a + a**2
    c2 = 2*(-x - y*b + b*a)
    c3 = 1 - 2*y*c + 2*c*a + b**2
    c4 = 2*(d*a + b*c - y*d)
    c5 = 2*b*d + c**2
    c6 = 2*c*d
    c7 = d**2
  */

  gsl_vector *c1 = gsl_vector_alloc(n);
  gsl_vector *c2 = gsl_vector_alloc(n);
  gsl_vector *c3 = gsl_vector_alloc(n);
  gsl_vector *c4 = gsl_vector_alloc(n);
  gsl_vector *c5 = gsl_vector_alloc(n);
  gsl_vector *c6 = gsl_vector_alloc(n);
  gsl_vector *c7 = gsl_vector_alloc(n);

  gsl_vector_memcpy(*c7, *d);
  gsl_vector_mul(*c7, *d);

  gsl_vector_memcpy(*c6, *c);
  gsl_vector_mul(*c6, *d);
  gsl_vector_scale(*c6, 2.0);

  gsl_vector_memcpy(*c5, *b);
  gsl_vector_mul(*c5, *d);
  gsl_vector_scale(*c5, 2.0);
  gsl_vector *tmp1 = gsl_vector_alloc(n);
  gsl_vector_memcpy(*tmp1, *c);
  gsl_vector_mul(*tmp1, *c);
  gsl_vector_add(*c5, *tmp1);

  gsl_vector *tmp2 = gsl_vector_alloc(n);
  gsl_vector_memcpy(*c4, *d);
  gsl_vector_memcpy(*tmp1, *b);
  gsl_vector_memcpy(*tmp2, *d);
  gsl_vector_mul(*c4, *a);
  gsl_vector_mul(*tmp1, *c);
  gsl_vector_scale(*tmp2, y);
  gsl_vector_add(*c4, *tmp1);
  gsl_vector_sub(*c4, *tmp2);
  gsl_vector_scale(*c4, 2.0);

  gsl_vector_memcpy(*c3, *c);
  gsl_vector_memcpy(*tmp1, *c);
  gsl_vector_memcpy(*tmp2, *b);
  gsl_vector_scale(*c3, -2.0*y);
  gsl_vector_mul(*tmp1, *a);
  gsl_vector_scale(*tmp1, 2.0);
  gsl_vector_mul(*tmp2, *b);
  gsl_vector_add(*c3, *tmp1);
  gsl_vector_add(*c3, *tmp2);
  gsl_vector_add_constant(*c3, 1.0);

  gsl_vector_memcpy(*c2, *b);
  gsl_vector_memcpy(*tmp1, *b);
  gsl_vector_scale(*c2, -y);
  gsl_vector_mul(*tmp1, *a);
  gsl_vector_add(*c2, *tmp1);
  gsl_vector_add_constant(*c2, -x);
  gsl_vector_scale(*c2, 2.0);

  gsl_vector_memcpy(*c1, *a);
  gsl_vector_memcpy(*tmp1, *a);
  gsl_vector_scale(*c1, -2*y);
  gsl_vector_mul(*tmp1, *a);
  gsl_vector_add(*c1, *tmp1);
  gsl_vector_add_constant(*c1, (x*x + y*y));

  // gsl_vector_free(*tmp1);
  gsl_vector_free(*tmp2);
  gsl_vector_free(*a);
  gsl_vector_free(*b);
  gsl_vector_free(*c);
  gsl_vector_free(*d);

  
  // Setting coefficient matrices
  gsl_matrix *dist_coeffs = gsl_matrix_alloc(n, 7);
  gsl_matrix_set_row(*dist_coeffs, 0, *c1);
  gsl_matrix_set_row(*dist_coeffs, 1, *c2);
  gsl_matrix_set_row(*dist_coeffs, 2, *c3);
  gsl_matrix_set_row(*dist_coeffs, 3, *c4);
  gsl_matrix_set_row(*dist_coeffs, 4, *c5);
  gsl_matrix_set_row(*dist_coeffs, 5, *c6);
  gsl_matrix_set_row(*dist_coeffs, 6, *c7);

  gsl_matrix *d_dist_coeffs = gsl_matrix_alloc(n, 7);
  gsl_vector_set_zero(*tmp1);
  gsl_vector_scale(*c3, 2);
  gsl_vector_scale(*c4, 3);
  gsl_vector_scale(*c5, 4);
  gsl_vector_scale(*c6, 5);
  gsl_vector_scale(*c7, 6);
  gsl_matrix_set_row(*d_dist_coeffs, 0, *c2);
  gsl_matrix_set_row(*d_dist_coeffs, 1, *c3);
  gsl_matrix_set_row(*d_dist_coeffs, 2, *c4);
  gsl_matrix_set_row(*d_dist_coeffs, 3, *c5);
  gsl_matrix_set_row(*d_dist_coeffs, 4, *c6);
  gsl_matrix_set_row(*d_dist_coeffs, 5, *c7);
  gsl_matrix_set_row(*d_dist_coeffs, 6, *tmp1);

  gsl_matrix *dd_dist_coeffs = gsl_matrix_alloc(n, 7);
  gsl_vector_set_zero(*tmp1);
  gsl_vector_scale(*c4, 2);
  gsl_vector_scale(*c5, 3);
  gsl_vector_scale(*c6, 4);
  gsl_vector_scale(*c7, 5);
  gsl_matrix_set_row(*dd_dist_coeffs, 0, *c3);
  gsl_matrix_set_row(*dd_dist_coeffs, 1, *c4);
  gsl_matrix_set_row(*dd_dist_coeffs, 2, *c5);
  gsl_matrix_set_row(*dd_dist_coeffs, 3, *c6);
  gsl_matrix_set_row(*dd_dist_coeffs, 4, *c7);
  gsl_matrix_set_row(*dd_dist_coeffs, 5, *tmp1);
  gsl_matrix_set_row(*dd_dist_coeffs, 6, *tmp1);




}

// Finds the progress (length) and curvature of point on a raceline generated from splines
projection frenet(float x, float y, std::vector<Spline> path,
                  std::vector<float> lengths, float prev_progress, float v_x,
                  float v_y) {
  assert(path.size() == lengths.size());
  const size_t num_points = prefered_degree + 1;
  const size_t n = path.size();
  size_t index_offset = 0;
  size_t size = n;
  std::vector<size_t> indexes;
  if (prev_progress != prev_progress_flag) {  //what does this flag do?
    //Lengths must be sorted for bisect to work since its binary search
    assert(is_sorted(lengths.begin(), lengths.end()));
    index_offset =
        std::lower_bound(lengths.begin(), lengths.end(), prev_progress) -
        lengths.begin();
    size = std::min(n, index_offset + 30);
  }
  //get index where all elements in lengths[index:] are >= prev_progress
  for (size_t i = index_offset; i < size; ++i) indexes.push_back(i % n);

  std::vector<Spline> explore_space;
  for (float element : indexes) {
    explore_space.push_back(path[element]);
  }
  size_t explore_space_n = explore_space.size();
  gsl_matrix *poly_coeffs = gsl_matrix_alloc(explore_space_n, num_points);
  gsl_matrix *poly_roots = gsl_matrix_alloc(explore_space_n, num_points);
  std::vector<gsl_matrix *> poly_Qs;
  std::vector<gsl_matrix *> poly_transMats;
  for (size_t i = 0; i < explore_space_n; ++i) { //used to be size: check if this is right
    Spline spline = path[i];
    gsl_vector *nums = spline.get_SplPoly().nums;
    gsl_matrix *rotated_points = spline.get_rotated_points();
    poly_Qs.push_back(spline.get_Q());
    gsl_vector *transMatVec = spline.get_translation();
    gsl_matrix *transMat = gsl_matrix_alloc(2, 1);
    gsl_matrix_set(transMat, 0, 0, gsl_vector_get(transMatVec, 0));
    gsl_matrix_set(transMat, 1, 0, gsl_vector_get(transMatVec, 1));
    poly_transMats.push_back(transMat);
    for (size_t j = 0; j < num_points; ++j) {
      gsl_matrix_set(poly_coeffs, i, j,
                     gsl_vector_get(nums, (num_points - 1 - j)));
      gsl_matrix_set(poly_roots, i, j, gsl_matrix_get(rotated_points, 0, i));
    }
  }
  gsl_matrix *point = gsl_matrix_alloc(2, 1);
  gsl_matrix_set(point, 0, 0, x);
  gsl_matrix_set(point, 0, 1, y);
  std::vector<gsl_matrix *> points = {point};
  std::vector<gsl_matrix *> rotated_points =
      rotate_points(points, poly_Qs, poly_transMats);
  
  //Dimensions are explore_space_n: CHECK if this is correct
  gsl_matrix *opt_xs = gsl_matrix_alloc(explore_space_n,1);
  gsl_matrix *distances = gsl_matrix_alloc(explore_space_n,1);
  gsl_matrix *x_point = gsl_matrix_alloc(explore_space_n,1);
  gsl_matrix *y_point = gsl_matrix_alloc(explore_space_n,1);

  //find how to get the columns from the vector of matrices
  std::pair< gsl_matrix *,gsl_matrix *> dist = get_closest_distance(x_point, y_point,
                                            poly_coeffs,poly_roots);
  opt_xs = dist.first;
  distances = dist.second;

  int i = argmin(distances); //argmin for gsl vector
  int min_index = (i+index_offset)%n;
  polynomial min_polynomial = path[i].get_SplPoly();
  double min_x = gsl_matrix_get(opt_xs,i,1);
  double curvature = get_curvature(path[i].get_first_der(),path[i].get_second_der(),min_x);
  //assert min_index in indexes

  double extra_length = arclength(min_polynomial, 0,min_x);
  double mu = gsl_matrix_get(distances,i,1);

  double velocity = (v_x * cos(mu) - v_y *sin(mu)) / (1 - mu*curvature);

  projection result = projection(float(min_index == 0 ? 0 : lengths[min_index-1]) + extra_length,
                            min_index,mu,curvature,velocity);
  return result;
}
