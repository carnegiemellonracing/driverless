#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_min.h>

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
// Curvature at point(s) `min_x` based on 2d curvature equation
// https://mathworld.wolfram.com/Curvature.html
double get_curvature(polynomial poly_der_1, polynomial poly_der_2,
                     double min_x) {
  return poly_eval(poly_der_2, min_x) /
         (1 + pow(pow(poly_eval(poly_der_1, min_x), 2), 3 / 2));
}

// Rotates points based on the rotation and transformation matrices
// Why aren't the matrices converted into vectors?
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

gsl_matrix *matrix_nonzero(gsl_matrix *m, double nonzero = 0.0001) {
  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
      if (gsl_matrix_get(m, i, j) == 0) {
        gsl_matrix_set(m, i, j, nonzero);
      }
    }
  }
  return m;
}

struct minimization_params {
  double car_x;
  double car_y;
  polynomial poly;
};
double minimization_f(double x, void *p) {
  struct minimization_params *params = (struct minimization_params *)p;
  return pow((params->car_x - x), 2) +
         pow((params->car_y - poly_eval(params->poly, x)), 2);
}

static const size_t n_coeffs = 7;
// x and y are a set of points
std::pair<std::vector<double>, std::vector<double>> get_closest_distance(
    double x, double y, std::vector<polynomial> polys, gsl_matrix *poly_coeffs,
    gsl_matrix *poly_roots, double epsabs = 0.001, double epsrel = 0,
    size_t max_iter = 5) {
  size_t n = poly_coeffs->size1;
  assert(n == poly_roots->size1);

  const gsl_min_fminimizer_type *T = gsl_min_fminimizer_brent;
  gsl_min_fminimizer *s = gsl_min_fminimizer_alloc(T);
  std::pair<std::vector<double>, std::vector<double>> results;
  for (size_t i = 0; i < n; ++i) {
    polynomial poly = polys[i];
    int status;
    size_t iter = 0;
    double x_min = gsl_matrix_get(poly_roots, i, poly.deg + 1), x_lower = 0,
           x_upper = x_min;
    gsl_function f;
    struct minimization_params params = {x, y, poly};
    f.function = &minimization_f;
    f.params = &params;

    gsl_min_fminimizer_set(s, &f, x_min, x_lower, x_upper);

    do {
      ++iter;
      status = gsl_min_fminimizer_iterate(s);
      x_min = gsl_min_fminimizer_x_minimum(s);
      x_lower = gsl_min_fminimizer_x_lower(s);
      x_upper = gsl_min_fminimizer_x_upper(s);
      status = gsl_min_test_interval(x_lower, x_upper, epsabs, epsrel);
    } while (status == GSL_CONTINUE && iter < max_iter);

    results.first.push_back(x_min);
    results.second.push_back(minimization_f(x_min, &params));
  }

  gsl_min_fminimizer_free(s);
  return results;
}

// Argmin in row,col form assuming 2d matrix
std::pair<int, int> argmin(gsl_matrix *m) {
  double min = 9999999;
  std::pair<int, int> minIndex = {-1, -1};
  for (size_t i = 0; i < m->size1; i++) {
    for (size_t j = 0; j < m->size2; j++) {
      if (gsl_matrix_get(m, i, j) < min) {
        min = gsl_matrix_get(m, i, j);
        minIndex = {i, j};
      }
    }
  }
  return minIndex;
}

// Vector of

// Finds the progress (length) and curvature of point on a raceline generated
// from splines
projection frenet(float x, float y, std::vector<Spline> path,
                  std::vector<double> lengths, float prev_progress, float v_x,
                  float v_y) {
  assert(path.size() == lengths.size());
  const size_t num_points = prefered_degree + 1;
  const size_t n = path.size();
  size_t index_offset = 0;
  size_t size = n;
  std::vector<size_t> indexes;
  if (prev_progress != prev_progress_flag) {  // what does this flag do?
    // Lengths must be sorted for bisect to work since its binary search
    assert(is_sorted(lengths.begin(), lengths.end()));
    index_offset =
        std::lower_bound(lengths.begin(), lengths.end(), prev_progress) -
        lengths.begin();
    size = std::min(n, index_offset + 30);
  }
  // get index where all elements in lengths[index:] are >= prev_progress
  for (size_t i = index_offset; i < size; ++i) indexes.push_back(i % n);

  std::vector<Spline> explore_space;
  for (float element : indexes) {
    explore_space.push_back(path[element]);
  }
  size_t explore_space_n = explore_space.size();
  gsl_matrix *poly_coeffs = gsl_matrix_alloc(explore_space_n, num_points);
  gsl_matrix *poly_roots = gsl_matrix_alloc(explore_space_n, num_points);
  std::vector<polynomial> polys;
  std::vector<gsl_matrix *> poly_Qs;
  std::vector<gsl_matrix *> poly_transMats;
  for (size_t i = 0; i < explore_space_n;
       ++i) {  // used to be size: check if this is right
    Spline spline = path[i];
    polynomial poly = spline.get_SplPoly();
    polys.push_back(poly);
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
  // returns a single rotated point
  std::vector<gsl_matrix *> rotated_points =
      rotate_points(points, poly_Qs, poly_transMats);

  // gsl_matrix *x_point = gsl_matrix_alloc(explore_space_n,1);
  // gsl_matrix *y_point = gsl_matrix_alloc(explore_space_n,1);
  // Just x and y from the one row in the matrix
  double x_point = gsl_matrix_get(rotated_points[0], 0, 0);
  double y_point = gsl_matrix_get(rotated_points[0], 0, 1);

  // find how to get the columns from the vector of matrices
  // std::pair< gsl_matrix *,gsl_matrix *> dist = get_closest_distance(x_point,
  // y_point, poly_coeffs,poly_roots);
  std::pair<std::vector<double>, std::vector<double>> optimization_result =
      get_closest_distance(x_point, y_point, polys, poly_coeffs, poly_roots);
  std::vector<double> opt_xs = optimization_result.first;
  assert(opt_xs.size() > 0);
  std::vector<double> distances = optimization_result.second;
  assert(distances.size() > 0);

  size_t i =
      std::min_element(distances.begin(), distances.end()) - distances.begin();
  size_t min_index = (i + index_offset) % n;
  polynomial min_polynomial = path[i].get_SplPoly();
  double min_x = opt_xs[i];
  double curvature =
      get_curvature(path[i].get_first_der(), path[i].get_second_der(), min_x);
  // assert min_index in indexes

  double extra_length = arclength(min_polynomial, 0, min_x);
  double mu = distances[i];

  double velocity = (v_x * cos(mu) - v_y * sin(mu)) / (1 - mu * curvature);

  projection result = projection(
      float(min_index == 0 ? 0 : lengths[min_index - 1]) + extra_length,
      min_index, mu, curvature, velocity);
  return result;
}
