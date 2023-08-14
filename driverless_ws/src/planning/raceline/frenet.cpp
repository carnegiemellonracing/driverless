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

double get_curvature(polynomial poly_der_1, polynomial poly_der_2,
                     double min_x) {
  return poly_eval(poly_der_2, min_x) /
         (1 + pow(pow(poly_eval(poly_der_1, min_x), 2), 3 / 2));
}

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
    gsl_matrix *x_point, gsl_matrix *y_point,
    std::vector<gsl_matrix *> poly_coeffs,
    std::vector<gsl_matrix *> poly_roots) {
  size_t n = poly_coeffs.size();
  assert(n == poly_roots.size());
}

projection frenet(float x, float y, std::vector<Spline> path,
                  std::vector<double> lengths, float prev_progress, float v_x,
                  float v_y) {
  assert(path.size() == lengths.size());
  const size_t num_points = prefered_degree + 1;
  const size_t n = path.size();
  size_t index_offset = 0;
  size_t size = n;
  std::vector<size_t> indexes;
  if (prev_progress != prev_progress_flag) {
    assert(is_sorted(lengths.begin(), lengths.end()));
    index_offset =
        std::lower_bound(lengths.begin(), lengths.end(), prev_progress) -
        lengths.begin();
    size = std::min(n, index_offset + 30);
  }

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
  for (size_t i = 0; i < size; ++i) {
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
}
