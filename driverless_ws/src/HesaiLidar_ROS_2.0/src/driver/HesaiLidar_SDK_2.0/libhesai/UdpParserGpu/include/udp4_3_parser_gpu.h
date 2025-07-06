/************************************************************************************************
Copyright (C) 2023 Hesai Technology Co., Ltd.
Copyright (C) 2023 Original Authors
All rights reserved.

All code in this repository is released under the terms of the following Modified BSD License. 
Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and 
  the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
  the following disclaimer in the documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its contributors may be used to endorse or 
  promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
************************************************************************************************/
#ifndef UDP4_3_PARSER_GPU_H_
#define UDP4_3_PARSER_GPU_H_
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <iostream>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "general_parser_gpu.h"

namespace hesai
{
namespace lidar
{
  // Forward declarations
  template <typename T_Point>
  class Udp4_3ParserGpu;
  template <typename T_Point>
  struct DeviceMemoryMaster;

  // Struct declarations
  struct Config
  {
    // GNC Params
    static constexpr float MAX_DISTANCE = 25.0f; // Maximum allowable distance (meters)
    static constexpr float ERROR_MARGIN = 0.2f;  // Ground outlier threshold (meters)
    static constexpr float MAX_HEIGHT = 0.4f;    // Maximum object height above ground (meters)
    static constexpr float ALPHA_DEFAULT = 1.5f; // Angular resolution for segmentation (radians)
    static constexpr int NUM_BINS_DEFAULT = 10;  // Number of range bins
    static constexpr float RANGE_MIN = 0.0f;     // Minimum range (meters)
    static constexpr float INVALID_POINT_THRESHOLD = 0.01f; // Min x,y for valid points
    static constexpr int MAX_SEGMENTS = 360;                // Maximum angular segments
    static constexpr int MAX_BINS = 100;                    // Maximum range bins

    // DBSCAN Params
    static constexpr float EPS_DEFAULT = 0.3f;    // DBSCAN epsilon (meters)
    static constexpr int MIN_SAMPLES_DEFAULT = 2; // Minimum points per cluster
    static constexpr int MAX_ITERATIONS = 10;     // Union-find iterations

    // Memory Params
    static constexpr int BLOCK_SIZE = 256;              // CUDA block size
    static constexpr int MAX_POINTS_PER_FRAME = 100000; // Maximum points to process
    static constexpr int SHARED_MEMORY_SIZE = 1024;     // Shared memory per block (floats)
    static constexpr size_t DEVICE_MEMORY_POOL_SIZE = 1024 * 1024 * 1024; // 1GB pool
    static constexpr bool USE_MANAGED_MEMORY = false;                     // Use unified memory
    static constexpr bool ENABLE_MEMORY_PREFETCH = true;                  // Prefetch data
  };

  template <typename T_Point>
  struct DeviceMemoryMaster
  {
    // Device vectors for reuse across frames
    thrust::device_vector<T_Point> d_points_buffer;
    thrust::device_vector<int> d_indices_buffer;
    thrust::device_vector<thrust::tuple<float, float, int>> d_polar_buffer;
    thrust::device_vector<thrust::tuple<int, int, int>> d_grid_buffer;
    thrust::device_vector<int> d_cell_indices_buffer;
    thrust::device_vector<int> d_point_indices_buffer;
    thrust::device_vector<bool> d_outlier_flags_buffer;
    thrust::device_vector<int> d_outlier_indices_buffer;
    thrust::device_vector<T_Point> d_outliers_buffer;

    // DBSCAN buffers
    thrust::device_vector<thrust::tuple<int, int>> d_pairs_buffer;
    thrust::device_vector<thrust::tuple<int, int>> d_close_pairs_buffer;
    thrust::device_vector<int> d_parent_buffer;
    thrust::device_vector<int> d_labels_buffer;
    thrust::device_vector<int> d_unique_labels_buffer;
    thrust::device_vector<int> d_cluster_sizes_buffer;

    // Ground plane fitting
    thrust::device_vector<float> d_segment_params_buffer;
    thrust::device_vector<int> d_unique_cells_buffer;
    thrust::device_vector<int> d_lowest_indices_buffer;

    // Temporary arrays for reductions
    thrust::device_vector<float> d_temp_floats;
    thrust::device_vector<int> d_temp_ints;

    void resize_buffers(size_t max_points, int max_segments, int max_bins)
    {
      size_t max_pairs = (max_points * (max_points - 1)) / 2;
      size_t max_cells = max_segments * max_bins;

      d_points_buffer.resize(max_points);
      d_indices_buffer.resize(max_points);
      d_polar_buffer.resize(max_points);
      d_grid_buffer.resize(max_points);
      d_cell_indices_buffer.resize(max_points);
      d_point_indices_buffer.resize(max_points);
      d_outlier_flags_buffer.resize(max_points);
      d_outlier_indices_buffer.resize(max_points);
      d_outliers_buffer.resize(max_points);

      d_pairs_buffer.resize(max_pairs);
      d_close_pairs_buffer.resize(max_pairs);
      d_parent_buffer.resize(max_points);
      d_labels_buffer.resize(max_points);
      d_unique_labels_buffer.resize(max_points);
      d_cluster_sizes_buffer.resize(max_points);

      d_segment_params_buffer.resize(max_segments * 2);
      d_unique_cells_buffer.resize(max_cells);
      d_lowest_indices_buffer.resize(max_cells);

      d_temp_floats.resize(max_points);
      d_temp_ints.resize(max_points);
    }
  };

  struct TimerManager
  {
    cudaEvent_t total_start, total_stop;
    cudaEvent_t gnc_start, ground_filter_stop;
    cudaEvent_t clustering_start, clustering_stop;
    cudaEvent_t memcpy_start, memcpy_stop;

    // Stage-specific timers
    float polar_conversion_time = 0.0f;
    float grid_assignment_time = 0.0f;
    float ground_fitting_time = 0.0f;
    float outlier_detection_time = 0.0f;
    float pair_generation_time = 0.0f;
    float distance_filtering_time = 0.0f;
    float union_find_time = 0.0f;
    float centroid_computation_time = 0.0f;

    // Memory transfer times
    float host_to_device_time = 0.0f;
    float device_to_host_time = 0.0f;

    // Frame statistics
    uint32_t total_points_processed = 0;
    uint32_t filtered_points_count = 0;
    uint32_t clusters_found = 0;

    TimerManager()
    {
      cudaEventCreate(&total_start);
      cudaEventCreate(&total_stop);
      cudaEventCreate(&gnc_start);
      cudaEventCreate(&ground_filter_stop);
      cudaEventCreate(&clustering_start);
      cudaEventCreate(&clustering_stop);
      cudaEventCreate(&memcpy_start);
      cudaEventCreate(&memcpy_stop);
    }

    ~TimerManager()
    {
      cudaEventDestroy(total_start);
      cudaEventDestroy(total_stop);
      cudaEventDestroy(gnc_start);
      cudaEventDestroy(ground_filter_stop);
      cudaEventDestroy(clustering_start);
      cudaEventDestroy(clustering_stop);
      cudaEventDestroy(memcpy_start);
      cudaEventDestroy(memcpy_stop);
    }

    void print_timing() const
    {
      std::cout << "\n================= Timing =================" << std::endl;
      std::cout     << "Total Points Processed: " << total_points_processed << std::endl;
      std::cout << "Filtered Points: " << filtered_points_count << std::endl;
      std::cout << "Clusters Found: " << clusters_found << std::endl;
      std::cout << "\nTiming Breakdown (ms):" << std::endl;
      std::cout << "  Polar Conversion: " << polar_conversion_time << std::endl;
      std::cout << "  Grid Assignment: " << grid_assignment_time << std::endl;
      std::cout << "  Ground Fitting: " << ground_fitting_time << std::endl;
      std::cout << "  Outlier Detection: " << outlier_detection_time << std::endl;
      std::cout << "  Pair Generation: " << pair_generation_time << std::endl;
      std::cout << "  Distance Filtering: " << distance_filtering_time << std::endl;
      std::cout << "  Union-Find: " << union_find_time << std::endl;
      std::cout << "  Centroid Computation: " << centroid_computation_time << std::endl;
      std::cout << "  Host→Device Transfer: " << host_to_device_time << std::endl;
      std::cout << "  Device→Host Transfer: " << device_to_host_time << std::endl;
      std::cout << "=============================================" << std::endl;
    }
  };

  struct GraceAndConradParams
  {
    float alpha;
    int num_bins;
    float angle_min;
    float angle_max;
    float range_min;
    float range_max;
    int num_segments;

    GraceAndConradParams()
        : alpha(Config::ALPHA_DEFAULT),
          num_bins(Config::NUM_BINS_DEFAULT),
          angle_min(0.0f),
          angle_max(2.0f * M_PI),
          range_min(Config::RANGE_MIN),
          range_max(Config::MAX_DISTANCE),
          num_segments(0) {}
  };

  struct DbscanParams
  {
    float eps;
    int min_samples;
    int max_iterations;
    bool use_fast_union_find;

    DbscanParams()
        : eps(Config::EPS_DEFAULT),
          min_samples(Config::MIN_SAMPLES_DEFAULT),
          max_iterations(Config::MAX_ITERATIONS),
          use_fast_union_find(true) {}
  };

  struct ProcessingStats
  {
    // From Lidar
    size_t input_points = 0;
    size_t valid_points = 0;
    size_t points_in_range = 0;

    // Grace & Conrad stats
    size_t grid_cells_used = 0;
    size_t ground_points_found = 0;
    size_t outlier_points = 0;
    float average_ground_slope = 0.0f;

    // DBSCAN stats
    size_t pairs_generated = 0;
    size_t close_pairs = 0;
    size_t initial_clusters = 0;
    size_t final_clusters = 0;
    size_t noise_points = 0;

    // Performance metrics
    float points_per_second = 0.0f;
    float memory_usage_mb = 0.0f;
    float gpu_utilization = 0.0f;

    void reset()
    {
      *this = ProcessingStats{};
    }

    void print_stats() const
    {
      std::cout << "\n=== Processing Summary ===" << std::endl;
      std::cout << "Input Points: " << input_points << std::endl;
      std::cout << "Valid Points: " << valid_points << " ("
                << (100.0f * valid_points / input_points) << "%)" << std::endl;
      std::cout << "Ground Points: " << ground_points_found << std::endl;
      std::cout << "Outlier Points: " << outlier_points << std::endl;
      std::cout << "Final Clusters: " << final_clusters << std::endl;
      std::cout << "Processing Rate: " << points_per_second << " points/sec" << std::endl;
      std::cout << "Memory Usage: " << memory_usage_mb << " MB" << std::endl;
      std::cout << "=============================" << std::endl;
    }
  };

  // Main GPU class
  template <typename T_Point>
  class Udp4_3ParserGpu : public GeneralParserGpu<T_Point>
  {
  private:
    // Raw lidar point processing variables
    bool corrections_loaded_;
    int32_t *channel_azimuths_cu_;
    int32_t *channel_elevations_cu_;
    int8_t *dazis_cu;
    int8_t *deles_cu;
    PointDecodeData *point_data_cu_;
    uint64_t *sensor_timestamp_cu_;
    uint32_t *mirror_azi_begins_cu;
    uint32_t *mirror_azi_ends_cu;

    // Central assets
    std::unique_ptr<DeviceMemoryMaster<T_Point>> device_memory_;
    std::unique_ptr<TimerManager> timers_;
    GraceAndConradParams gnc_params_;
    DbscanParams dbscan_params_;
    ProcessingStats stats_;
    Config config_;

    // Clustering and ground filtering functions
    T_Point *grace_and_conrad(const T_Point *points_ptr, size_t num_points, int *num_filtered);
    std::vector<T_Point> dbscan(const T_Point *points_ptr, size_t num_points);
    void update_angle_range(const T_Point *points_ptr, size_t num_points);
    void optimize_processing_parameters(size_t num_points);

  public:
    Udp4_3ParserGpu();
    ~Udp4_3ParserGpu();

    // Core processing functions - Fixed method name
    virtual int ComputeXYZI(LidarDecodedFrame<T_Point> &frame);
    virtual int load_correction_file(std::string correction_path);
    virtual int load_correction_string(char *correction_string);

    // Config setters and getters
    void set_grace_and_conrad_params(const GraceAndConradParams &params) { gnc_params_ = params; }
    void set_dbscan_params(const DbscanParams &params) { dbscan_params_ = params; }
    const GraceAndConradParams &get_grace_and_conrad_params() const { return gnc_params_; }
    const DbscanParams &get_dbscan_params() const { return dbscan_params_; }

    // Performance monitoring
    const ProcessingStats &get_stats() const { return stats_; }
    void print_summary() const
    {
      if (timers_)
        timers_->print_timing();
      stats_.print_stats();
    }

    // Memory management
    void preallocate_buffers(size_t max_points, int max_segments = Config::MAX_SEGMENTS,
                              int max_bins = Config::MAX_BINS)
    {
      if (device_memory_)
      {
        device_memory_->resize_buffers(max_points, max_segments, max_bins);
      }
    }

    // Algorithm params
    PandarATCorrections m_PandarAT_corrections;
  };

  // Thrust functors
  /// Converts 3D Cartesian coordinates to polar coordinates (angle, range)
  /// Filters invalid points based on distance thresholds and coordinate validity
  template <typename T_Point>
  struct point_to_polar_functor
  {
    __device__
        thrust::tuple<float, float, int>
        operator()(const thrust::tuple<T_Point, int> &input) const
    {
      T_Point p = thrust::get<0>(input);
      int idx = thrust::get<1>(input);

      float angle = atan2f(p.y, p.x);
      if (angle < 0)
        angle += 2 * M_PI;
      float range = sqrtf(p.x * p.x + p.y * p.y);

      // Filter out invalid points using centralized config
      if (fabsf(p.x) > Config::MAX_DISTANCE ||
          fabsf(p.y) > Config::MAX_DISTANCE ||
          range > Config::MAX_DISTANCE ||
          (p.y < Config::INVALID_POINT_THRESHOLD &&
            p.x < Config::INVALID_POINT_THRESHOLD))
      {
        return thrust::make_tuple(-1.0f, -1.0f, -1);
      }

      return thrust::make_tuple(angle, range, idx);
    }
  };

  /// Assigns points to spatial grid cells based on angular segments and range bins
  /// Maps continuous polar coordinates to discrete grid indices for spatial processing
  struct grid_assignment_functor
  {
    float angle_min, angle_max, range_min, range_max;
    int num_segments, num_bins;

    grid_assignment_functor(float amin, float amax, float rmin, float rmax, int nseg, int nbins)
        : angle_min(amin), angle_max(amax), range_min(rmin), range_max(rmax),
          num_segments(nseg), num_bins(nbins) {}

    __device__
        thrust::tuple<int, int, int>
        operator()(const thrust::tuple<float, float, int> &polar_data) const
    {
      float angle = thrust::get<0>(polar_data);
      float range = thrust::get<1>(polar_data);
      int idx = thrust::get<2>(polar_data);

      if (idx == -1)
        return thrust::make_tuple(-1, -1, -1);

      int segment = min(static_cast<int>((angle - angle_min) / (angle_max - angle_min) * num_segments), num_segments - 1);
      int bin = min(static_cast<int>((range - range_min) / (range_max - range_min) * num_bins), num_bins - 1);

      return thrust::make_tuple(segment, bin, idx);
    }
  };

  /// Compares two points by their Z-coordinate (height) for finding lowest points
  /// Used in reduce operations to identify ground-level points in each grid cell
  template <typename T_Point>
  struct height_comparison_functor
  {
    const T_Point *points;

    height_comparison_functor(const T_Point *pts) : points(pts) {}

    __device__ bool operator()(int idx1, int idx2) const
    {
      return points[idx1].z < points[idx2].z;
    }
  };

  /// Converts grid coordinates (segment, bin) to linear cell index
  /// Enables efficient sorting and grouping of points by spatial location
  struct cell_index_functor
  {
    int num_bins;

    cell_index_functor(int nbins) : num_bins(nbins) {}

    __device__ int operator()(const thrust::tuple<int, int, int> &grid_data) const
    {
      int segment = thrust::get<0>(grid_data);
      int bin = thrust::get<1>(grid_data);

      if (segment == -1 || bin == -1)
        return -1;

      return segment * num_bins + bin;
    }
  };

  /// Identifies outlier points that are significantly above the fitted ground plane
  /// Compares actual point height to predicted ground height within error margins
  template <typename T_Point>
  struct outlier_detection_functor
  {
    const T_Point *points;
    const float *segment_params;
    int num_segments;

    outlier_detection_functor(const T_Point *pts, const float *params, int nseg)
        : points(pts), segment_params(params), num_segments(nseg) {}

    __device__ bool operator()(const thrust::tuple<int, int, int> &grid_data) const
    {
      int segment = thrust::get<0>(grid_data);
      int bin = thrust::get<1>(grid_data);
      int idx = thrust::get<2>(grid_data);

      if (segment == -1 || idx == -1 || segment >= num_segments)
        return false;

      T_Point p = points[idx];
      float slope = segment_params[segment * 2];
      float intercept = segment_params[segment * 2 + 1];
      float range = sqrtf(p.x * p.x + p.y * p.y);
      float predicted_z = slope * range + intercept;

      return (p.z - predicted_z) > Config::ERROR_MARGIN &&
              (p.z - predicted_z) < Config::MAX_HEIGHT;
    }
  };

  /// Computes Euclidean distance between point pairs for DBSCAN clustering
  /// Returns true if distance is within epsilon threshold for cluster formation
  template <typename T_Point>
  struct distance_functor
  {
    const T_Point *points;
    float eps_squared;

    distance_functor(const T_Point *pts, float eps) : points(pts), eps_squared(eps * eps) {}

    __device__ bool operator()(const thrust::tuple<int, int> &pair) const
    {
      int i = thrust::get<0>(pair);
      int j = thrust::get<1>(pair);

      T_Point p1 = points[i];
      T_Point p2 = points[j];

      float dx = p1.x - p2.x;
      float dy = p1.y - p2.y;
      float dz = p1.z - p2.z;

      return (dx * dx + dy * dy + dz * dz) <= eps_squared;
    }
  };

} // namespace lidar
} // namespace hesai

#include "udp4_3_parser_gpu.cu"
#endif // UDP4_3_PARSER_GPU_H_