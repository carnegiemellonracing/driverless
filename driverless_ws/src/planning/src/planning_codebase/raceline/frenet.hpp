#ifndef FRENET_H
#define FRENET_H

#include <gsl/gsl_vector.h>

#include <limits>
#include <vector>

#include "raceline.hpp"
const float prev_progress_flag = -std::numeric_limits<float>::max();

//Projection class for storing result from minimization and related data
struct projection {
  float progress;
  int min_index;       // For testing
  float min_distance;  // For testing
  float curvature;
  float velocity;

  projection(float progress,int min_index,float min_distance,float curvature,float velocity){
    this->progress = progress;
    this->min_index = min_index;
    this->min_distance = min_distance;
    this->curvature = curvature;
    this->velocity = velocity;
  }
};

projection frenet(float x, float y, std::vector<Spline> path,
                  std::vector<double> lengths,
                  float prev_progress = prev_progress_flag, float v_x = 0,
                  float v_y = 0);

#endif
