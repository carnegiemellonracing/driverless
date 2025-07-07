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
#include <udp4_3_parser_gpu.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace hesai::lidar;

// Constructor for GPU
template <typename T_Point>
Udp4_3ParserGpu<T_Point>::Udp4_3ParserGpu()
{
    corrections_loaded_ = false;
    cudaSafeMalloc(point_data_cu_, POINT_DATA_LEN);
    cudaSafeMalloc(sensor_timestamp_cu_, SENSOR_TIMESTAMP_LEN);

    // Preallocate buffers as persistent memory
    device_memory_ = std::make_unique<DeviceMemoryMaster<T_Point>>();
    // Timing suite
    timers_ = std::make_unique<TimerManager>();

    // Start with sensible defaults that work for most scenarios
    gnc_params_ = GraceAndConradParams();
    dbscan_params_ = DbscanParams();
    stats_.reset();
}

// Destructor
template <typename T_Point>
Udp4_3ParserGpu<T_Point>::~Udp4_3ParserGpu()
{
    cudaSafeFree(point_data_cu_);
    cudaSafeFree(sensor_timestamp_cu_);

    if (corrections_loaded_)
    {
        // Clean up LiDAR-specific calibration data
        cudaSafeFree(deles_cu);
        cudaSafeFree(dazis_cu);
        cudaSafeFree(channel_elevations_cu_);
        cudaSafeFree(channel_azimuths_cu_);
        cudaSafeFree(mirror_azi_begins_cu);
        cudaSafeFree(mirror_azi_ends_cu);
        corrections_loaded_ = false;
    }
}

// Modified implementation sections without thrust::back_inserter
// Replace these sections in your udp4_3_parser_gpu.cu file

template <typename T_Point>
T_Point *Udp4_3ParserGpu<T_Point>::grace_and_conrad(const T_Point *points_ptr,
                                                    size_t num_points,
                                                    int *num_filtered)
{
    cudaEventRecord(timers_->gnc_start);

    stats_.input_points = num_points;

    // Resize to ensure buffers can handle amount of points
    device_memory_->resize_buffers(num_points, gnc_params_.num_segments, gnc_params_.num_bins);

    // Copy points to device
    cudaEventRecord(timers_->memcpy_start);
    device_memory_->d_points_buffer.assign(points_ptr, points_ptr + num_points);
    thrust::sequence(device_memory_->d_indices_buffer.begin(),
                     device_memory_->d_indices_buffer.begin() + num_points);
    cudaEventRecord(timers_->memcpy_stop);
    cudaEventSynchronize(timers_->memcpy_stop);
    cudaEventElapsedTime(&timers_->host_to_device_time, timers_->memcpy_start, timers_->memcpy_stop);

    // Convert 3D world frame into polar coordinates (azimuth, range)
    auto start_time = std::chrono::high_resolution_clock::now();
    auto polar_iter = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(device_memory_->d_points_buffer.begin(),
                                                     device_memory_->d_indices_buffer.begin())),
                                                     point_to_polar_functor<T_Point>());

    device_memory_->d_polar_buffer.resize(num_points);
    thrust::copy(polar_iter, polar_iter + num_points, device_memory_->d_polar_buffer.begin());

    // Discard points that are too close, too far, or invalid
    auto new_end = thrust::remove_if(device_memory_->d_polar_buffer.begin(),
                                     device_memory_->d_polar_buffer.end(),
                                     [] __device__(const thrust::tuple<float, float, int> &t)
                                     {
                                         return thrust::get<2>(t) == -1;
                                     });
    device_memory_->d_polar_buffer.erase(new_end, device_memory_->d_polar_buffer.end());

    auto end_time = std::chrono::high_resolution_clock::now();
    timers_->polar_conversion_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    stats_.valid_points = device_memory_->d_polar_buffer.size();

    // Find the angular span of the point cloud
    start_time = std::chrono::high_resolution_clock::now();
    auto angle_iter = thrust::make_transform_iterator(device_memory_->d_polar_buffer.begin(),
                                                      [] __device__(const thrust::tuple<float, float, int> &t)
                                                      {
                                                          return thrust::get<0>(t);
                                                      });

    auto minmax_angles = thrust::minmax_element(angle_iter, angle_iter + device_memory_->d_polar_buffer.size());
    gnc_params_.angle_min = *minmax_angles.first;
    gnc_params_.angle_max = *minmax_angles.second;

    // Divide the angular space into segments
    gnc_params_.num_segments = static_cast<int>((gnc_params_.angle_max - gnc_params_.angle_min) / gnc_params_.alpha) + 1;
    gnc_params_.num_segments = std::min(gnc_params_.num_segments, config_.MAX_SEGMENTS);

    // Assign each point to a "grid cell"
    auto grid_iter = thrust::make_transform_iterator(device_memory_->d_polar_buffer.begin(),
                                                     grid_assignment_functor(gnc_params_.angle_min, gnc_params_.angle_max,
                                                                             gnc_params_.range_min, gnc_params_.range_max,
                                                                             gnc_params_.num_segments, gnc_params_.num_bins));

    device_memory_->d_grid_buffer.resize(device_memory_->d_polar_buffer.size());
    thrust::copy(grid_iter, grid_iter + device_memory_->d_polar_buffer.size(),
                 device_memory_->d_grid_buffer.begin());

    // Convert 2D grid coordinates to a singular "address" for efficient sorting
    auto cell_iter = thrust::make_transform_iterator(device_memory_->d_grid_buffer.begin(),
                                                     cell_index_functor(gnc_params_.num_bins));

    device_memory_->d_cell_indices_buffer.resize(device_memory_->d_grid_buffer.size());
    thrust::copy(cell_iter, cell_iter + device_memory_->d_grid_buffer.size(),
                 device_memory_->d_cell_indices_buffer.begin());

    // Extract point indices
    device_memory_->d_point_indices_buffer.resize(device_memory_->d_grid_buffer.size());
    thrust::transform(device_memory_->d_grid_buffer.begin(), device_memory_->d_grid_buffer.end(),
                      device_memory_->d_point_indices_buffer.begin(),
                      [] __device__(const thrust::tuple<int, int, int> &t)
                      {
                          return thrust::get<2>(t);
                      });

    end_time = std::chrono::high_resolution_clock::now();
    timers_->grid_assignment_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Filter points in each grid cell
    start_time = std::chrono::high_resolution_clock::now();
    thrust::sort_by_key(device_memory_->d_cell_indices_buffer.begin(),
                        device_memory_->d_cell_indices_buffer.end(),
                        device_memory_->d_point_indices_buffer.begin());

    // For each cell, find the lowest point - this represents the ground level there
    // Pre-allocate maximum possible output size
    size_t max_cells = gnc_params_.num_segments * gnc_params_.num_bins;
    device_memory_->d_unique_cells_buffer.resize(max_cells);
    device_memory_->d_lowest_indices_buffer.resize(max_cells);

    auto unique_size = thrust::reduce_by_key(
        device_memory_->d_cell_indices_buffer.begin(),
        device_memory_->d_cell_indices_buffer.end(),
        device_memory_->d_point_indices_buffer.begin(),
        device_memory_->d_unique_cells_buffer.begin(),
        device_memory_->d_lowest_indices_buffer.begin(),
        thrust::equal_to<int>(),
        height_comparison_functor<T_Point>(thrust::raw_pointer_cast(device_memory_->d_points_buffer.data())));

    // Resize to actual size
    size_t actual_unique_count = unique_size.first - device_memory_->d_unique_cells_buffer.begin();
    device_memory_->d_unique_cells_buffer.resize(actual_unique_count);
    device_memory_->d_lowest_indices_buffer.resize(actual_unique_count);

    stats_.grid_cells_used = actual_unique_count;

    end_time = std::chrono::high_resolution_clock::now();
    timers_->ground_fitting_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Prepare to fit ground planes--one for each angular segment
    start_time = std::chrono::high_resolution_clock::now();
    device_memory_->d_segment_params_buffer.resize(gnc_params_.num_segments * 2);
    thrust::fill(device_memory_->d_segment_params_buffer.begin(),
                 device_memory_->d_segment_params_buffer.end(), 0.0f);

    // Fit a line through the lowest points in each angular slice
    dim3 block(1);
    dim3 grid_kernel(gnc_params_.num_segments);

    fit_line_in_segment_kernel<<<grid_kernel, block>>>(
        thrust::raw_pointer_cast(device_memory_->d_points_buffer.data()),
        thrust::raw_pointer_cast(device_memory_->d_lowest_indices_buffer.data()),
        thrust::raw_pointer_cast(device_memory_->d_unique_cells_buffer.data()),
        thrust::raw_pointer_cast(device_memory_->d_segment_params_buffer.data()),
        device_memory_->d_unique_cells_buffer.size(),
        gnc_params_.num_segments,
        gnc_params_.num_bins);
    cudaDeviceSynchronize();

    end_time = std::chrono::high_resolution_clock::now();
    timers_->ground_fitting_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    start_time = std::chrono::high_resolution_clock::now();

    // Identify points that are significantly above predicted ground level
    auto outlier_iter = thrust::make_transform_iterator(device_memory_->d_grid_buffer.begin(),
                                                        outlier_detection_functor<T_Point>(
                                                            thrust::raw_pointer_cast(device_memory_->d_points_buffer.data()),
                                                            thrust::raw_pointer_cast(device_memory_->d_segment_params_buffer.data()),
                                                            gnc_params_.num_segments));

    device_memory_->d_outlier_flags_buffer.resize(device_memory_->d_grid_buffer.size());
    thrust::copy(outlier_iter, outlier_iter + device_memory_->d_grid_buffer.size(),
                 device_memory_->d_outlier_flags_buffer.begin());

    // Get the indices of points that are above ground plane
    // First count how many outliers we have
    size_t outlier_count = thrust::count(device_memory_->d_outlier_flags_buffer.begin(),
                                         device_memory_->d_outlier_flags_buffer.end(),
                                         true);

    // Resize and copy outlier indices
    device_memory_->d_outlier_indices_buffer.resize(outlier_count);
    thrust::copy_if(device_memory_->d_point_indices_buffer.begin(),
                    device_memory_->d_point_indices_buffer.end(),
                    device_memory_->d_outlier_flags_buffer.begin(),
                    device_memory_->d_outlier_indices_buffer.begin(),
                    thrust::identity<bool>());

    *num_filtered = device_memory_->d_outlier_indices_buffer.size();
    stats_.outlier_points = *num_filtered;

    // Get the actual non-ground points from the indices
    device_memory_->d_outliers_buffer.resize(*num_filtered);
    thrust::gather(device_memory_->d_outlier_indices_buffer.begin(),
                   device_memory_->d_outlier_indices_buffer.end(),
                   device_memory_->d_points_buffer.begin(),
                   device_memory_->d_outliers_buffer.begin());

    end_time = std::chrono::high_resolution_clock::now();
    timers_->outlier_detection_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Copy filtered points back to the host
    cudaEventRecord(timers_->memcpy_start);
    T_Point *h_outliers = (T_Point *)malloc(*num_filtered * sizeof(T_Point));
    thrust::copy(device_memory_->d_outliers_buffer.begin(),
                 device_memory_->d_outliers_buffer.end(), h_outliers);
    cudaEventRecord(timers_->memcpy_stop);
    cudaEventSynchronize(timers_->memcpy_stop);
    cudaEventElapsedTime(&timers_->device_to_host_time, timers_->memcpy_start, timers_->memcpy_stop);

    cudaEventRecord(timers_->ground_filter_stop);

    return h_outliers;
}

template <typename T_Point>
std::vector<T_Point> Udp4_3ParserGpu<T_Point>::dbscan(const T_Point *points_ptr,
                                                      size_t num_points)
{
    cudaEventRecord(timers_->clustering_start);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Ensure sure we have enough workspace for clustering operations
    device_memory_->resize_buffers(num_points, gnc_params_.num_segments, gnc_params_.num_bins);

    // Start with fresh data and assume all points are noise (label = -1)
    device_memory_->d_points_buffer.assign(points_ptr, points_ptr + num_points);
    device_memory_->d_labels_buffer.resize(num_points);
    thrust::fill(device_memory_->d_labels_buffer.begin(), device_memory_->d_labels_buffer.end(), -1);

    // Check every pair of points--this is expensive but necessary for DBSCAN
    size_t num_pairs = (num_points * (num_points - 1)) / 2;
    stats_.pairs_generated = num_pairs;

    device_memory_->d_pairs_buffer.resize(num_pairs);

    // Convert linear indices to (i,j) pairs efficiently
    auto pair_generator = [num_points] __device__(size_t k)
    {
        size_t i = 0;
        size_t remaining = k;
        while (true)
        {
            size_t row_size = num_points - 1 - i;
            if (remaining < row_size)
            {
                return thrust::make_tuple(static_cast<int>(i),
                                          static_cast<int>(i + 1 + remaining));
            }
            remaining -= row_size;
            i++;
        }
    };

    thrust::transform(thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(num_pairs),
                      device_memory_->d_pairs_buffer.begin(),
                      pair_generator);

    auto end_time = std::chrono::high_resolution_clock::now();
    timers_->pair_generation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    start_time = std::chrono::high_resolution_clock::now();

    // Filter to only pairs that are close enough to potentially be in the same cluster
    auto distance_predicate = distance_functor<T_Point>(
        thrust::raw_pointer_cast(device_memory_->d_points_buffer.data()),
        dbscan_params_.eps);

    // Count close pairs first
    size_t close_pair_count = thrust::count_if(device_memory_->d_pairs_buffer.begin(),
                                               device_memory_->d_pairs_buffer.end(),
                                               distance_predicate);

    // Resize and copy close pairs
    device_memory_->d_close_pairs_buffer.resize(close_pair_count);
    thrust::copy_if(device_memory_->d_pairs_buffer.begin(),
                    device_memory_->d_pairs_buffer.end(),
                    device_memory_->d_close_pairs_buffer.begin(),
                    distance_predicate);

    stats_.close_pairs = device_memory_->d_close_pairs_buffer.size();

    end_time = std::chrono::high_resolution_clock::now();
    timers_->distance_filtering_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    start_time = std::chrono::high_resolution_clock::now();

    // Initialize union-find: each point starts as its own cluster leader
    device_memory_->d_parent_buffer.resize(num_points);
    thrust::sequence(device_memory_->d_parent_buffer.begin(),
                     device_memory_->d_parent_buffer.end());

    // Iteratively merge clusters by processing all close pairs
    for (int iter = 0; iter < dbscan_params_.max_iterations; ++iter)
    {
        thrust::for_each(device_memory_->d_close_pairs_buffer.begin(),
                         device_memory_->d_close_pairs_buffer.end(),
                         [parent = thrust::raw_pointer_cast(device_memory_->d_parent_buffer.data())] __device__(const thrust::tuple<int, int> &pair)
                         {
                             int i = thrust::get<0>(pair);
                             int j = thrust::get<1>(pair);

                             // Find the canonical representative of each point's cluster
                             int root_i = i;
                             while (parent[root_i] != root_i)
                             {
                                 int old_parent = parent[root_i];
                                 parent[root_i] = parent[old_parent];
                                 root_i = old_parent;
                             }

                             int root_j = j;
                             while (parent[root_j] != root_j)
                             {
                                 int old_parent = parent[root_j];
                                 parent[root_j] = parent[old_parent];
                                 root_j = old_parent;
                             }

                             // If they have different canonical representatives, merge the clusters
                             if (root_i != root_j)
                             {
                                 atomicMin(&parent[max(root_i, root_j)], min(root_i, root_j));
                             }
                         });
    }

    // Final cleanup: make sure every point points directly to its canonical representative
    thrust::transform(device_memory_->d_parent_buffer.begin(),
                      device_memory_->d_parent_buffer.end(),
                      device_memory_->d_parent_buffer.begin(),
                      [parent = thrust::raw_pointer_cast(device_memory_->d_parent_buffer.data())] __device__(int idx)
                      {
                          int root = idx;
                          while (parent[root] != root)
                              root = parent[root];
                          return root;
                      });

    end_time = std::chrono::high_resolution_clock::now();
    timers_->union_find_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    start_time = std::chrono::high_resolution_clock::now();

    // Find all unique cluster IDs and count how many points belong to each
    auto sorted_parent = device_memory_->d_parent_buffer;
    thrust::sort(sorted_parent.begin(), sorted_parent.end());

    // Use temp buffer for unique_copy
    size_t unique_count = thrust::unique_copy(sorted_parent.begin(), sorted_parent.end(),
                                              device_memory_->d_temp_ints.begin()) -
                          device_memory_->d_temp_ints.begin();
    device_memory_->d_unique_labels_buffer.resize(unique_count);
    thrust::copy_n(device_memory_->d_temp_ints.begin(), unique_count,
                   device_memory_->d_unique_labels_buffer.begin());

    // Create constant array for counting
    device_memory_->d_temp_floats.resize(sorted_parent.size());
    thrust::fill(device_memory_->d_temp_floats.begin(), device_memory_->d_temp_floats.end(), 1.0f);

    device_memory_->d_cluster_sizes_buffer.resize(unique_count);
    thrust::reduce_by_key(sorted_parent.begin(), sorted_parent.end(),
                          device_memory_->d_temp_floats.begin(),
                          device_memory_->d_temp_ints.begin(), // discard keys
                          device_memory_->d_cluster_sizes_buffer.begin());

    // Move data to host for centroid computation
    thrust::host_vector<int> h_unique_labels = device_memory_->d_unique_labels_buffer;
    thrust::host_vector<int> h_cluster_sizes = device_memory_->d_cluster_sizes_buffer;
    thrust::host_vector<int> h_parent = device_memory_->d_parent_buffer;
    thrust::host_vector<T_Point> h_points = device_memory_->d_points_buffer;

    // Build, final output with centroids of clusters that have enough points
    std::vector<T_Point> centroids;
    stats_.final_clusters = 0;

    for (size_t i = 0; i < h_unique_labels.size(); ++i)
    {
        // Only accept clusters with enough points--small clusters are likely noise
        if (h_cluster_sizes[i] >= dbscan_params_.min_samples)
        {
            T_Point centroid = {0, 0, 0, 0, 0, 0};
            int count = 0;

            // Compute the average position of all points in this cluster
            for (size_t j = 0; j < num_points; ++j)
            {
                if (h_parent[j] == h_unique_labels[i])
                {
                    centroid.x += h_points[j].x;
                    centroid.y += h_points[j].y;
                    centroid.z += h_points[j].z;
                    centroid.intensity += h_points[j].intensity;
                    count++;
                }
            }

            if (count > 0)
            {
                centroid.x /= count;
                centroid.y /= count;
                centroid.z /= count;
                centroid.intensity /= count;
                centroids.push_back(centroid);
                stats_.final_clusters++;
            }
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    timers_->centroid_computation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    cudaEventRecord(timers_->clustering_stop);

    return centroids;
}

template <typename T_Point>
__global__ void fit_line_in_segment_kernel(
    const T_Point *points, const int *lowest_indices, const int *cell_indices,
    float *segment_params, int num_cells, int num_segments, int num_bins)
{

    int segment = blockIdx.x;
    if (segment >= num_segments)
        return;

    // Accumulate data for linear regression: y = mx + b where y=height, x=distance
    float x_sum = 0, z_sum = 0, x2_sum = 0, xz_sum = 0;
    int count = 0;

    // Find all the lowest points that belong to this angular segment
    for (int i = 0; i < num_cells; ++i)
    {
        int cell_idx = cell_indices[i];
        if (cell_idx / num_bins == segment)
        {
            int point_idx = lowest_indices[i];
            T_Point p = points[point_idx];

            float range = sqrtf(p.x * p.x + p.y * p.y);
            x_sum += range;
            z_sum += p.z;
            x2_sum += range * range;
            xz_sum += range * p.z;
            count++;
        }
    }

    // Compute slope and intercept using least squares formula
    // We need at least 2 points to fit a line
    if (count > 1)
    {
        float denominator = count * x2_sum - x_sum * x_sum;
        if (fabsf(denominator) > 1e-6f)
        {
            // Standard least squares formulas
            float slope = (count * xz_sum - x_sum * z_sum) / denominator;
            float intercept = (z_sum - slope * x_sum) / count;
            segment_params[segment * 2] = slope;
            segment_params[segment * 2 + 1] = intercept;
        }
        else
        {
            // Degenerate case: all points at same distance, assume flat ground
            segment_params[segment * 2] = 0;
            segment_params[segment * 2 + 1] = 0;
        }
    }
    else
    {
        // Not enough data points, assume flat ground at height 0
        segment_params[segment * 2] = 0;
        segment_params[segment * 2 + 1] = 0;
    }
}

template <typename T_Point>
__global__ void compute_xyzs_v4_3_impl(
    T_Point *xyzs, const int32_t *channel_azimuths,
    const int32_t *channel_elevations, const int8_t *dazis, const int8_t *deles,
    const uint32_t *raw_azimuth_begin, const uint32_t *raw_azimuth_end, const uint8_t raw_correction_resolution,
    const PointDecodeData *point_data, const uint64_t *sensor_timestamp, const double raw_distance_unit, Transform transform, const uint16_t blocknum, const uint16_t lasernum, uint16_t packet_index)
{
    auto iscan = blockIdx.x;
    auto ichannel = threadIdx.x;
    if (iscan >= packet_index || ichannel >= blocknum * lasernum)
        return;
    int point_index = iscan * blocknum * lasernum + (ichannel % (lasernum * blocknum));
    float azimuth = point_data[point_index].azimuth / kFineResolutionInt;
    int Azimuth = point_data[point_index].azimuth;
    int count = 0, field = 0;
    while (count < 3 &&
           (((Azimuth + (kFineResolutionInt * kCircle) - raw_azimuth_begin[field]) % (kFineResolutionInt * kCircle) +
             (raw_azimuth_end[field] + kFineResolutionInt * kCircle - Azimuth) % (kFineResolutionInt * kCircle)) !=
            (raw_azimuth_end[field] + kFineResolutionInt * kCircle -
             raw_azimuth_begin[field]) %
                (kFineResolutionInt * kCircle)))
    {
        field = (field + 1) % 3;
        count++;
    }
    if (count >= 3)
        return;
    float m = azimuth / 200.f;
    int i = m;
    int j = i + 1;
    float alpha = m - i;    // k
    float beta = 1 - alpha; // 1-k
    // getAziAdjustV3
    auto dazi =
        beta * dazis[(ichannel % lasernum) * (kHalfCircleInt / kResolutionInt) + i] + alpha * dazis[(ichannel % lasernum) * (kHalfCircleInt / kResolutionInt) + j];
    auto theta = ((azimuth + kCircle - raw_azimuth_begin[field] / kFineResolutionFloat) * 2 -
                  channel_azimuths[(ichannel % lasernum)] * raw_correction_resolution / kFineResolutionFloat + dazi * raw_correction_resolution) /
                 kHalfCircleFloat * M_PI;
    // getEleAdjustV3
    auto dele =
        beta * deles[(ichannel % lasernum) * (kHalfCircleInt / kResolutionInt) + i] + alpha * deles[(ichannel % lasernum) * (kHalfCircleInt / kResolutionInt) + j];
    auto phi = (channel_elevations[(ichannel % lasernum)] / kFineResolutionFloat + dele) * raw_correction_resolution / kHalfCircleFloat * M_PI;

    auto rho = point_data[point_index].distances * raw_distance_unit;
    float z = rho * sin(phi);
    auto r = rho * cosf(phi);
    float x = r * sin(theta);
    float y = r * cos(theta);

    float cosa = std::cos(transform.roll);
    float sina = std::sin(transform.roll);
    float cosb = std::cos(transform.pitch);
    float sinb = std::sin(transform.pitch);
    float cosc = std::cos(transform.yaw);
    float sinc = std::sin(transform.yaw);

    float x_ = cosb * cosc * x + (sina * sinb * cosc - cosa * sinc) * y +
               (sina * sinc + cosa * sinb * cosc) * z + transform.x;
    float y_ = cosb * sinc * x + (cosa * cosc + sina * sinb * sinc) * y +
               (cosa * sinb * sinc - sina * cosc) * z + transform.y;
    float z_ = -sinb * x + sina * cosb * y + cosa * cosb * z + transform.z;
    gpu::setTimestamp(xyzs[point_index], double(sensor_timestamp[iscan]) / kMicrosecondToSecond);
    gpu::setRing(xyzs[point_index], ichannel % lasernum);
    gpu::setConfidence(xyzs[point_index], point_data[point_index].confidence);
}

template <typename T_Point>
int Udp4_3ParserGpu<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame)
{
    if (!corrections_loaded_)
        return int(ReturnCode::CorrectionsUnloaded);

    cudaEventRecord(timers_->total_start);

    // Reset our performance counters for this frame
    stats_.reset();
    timers_->total_points_processed = frame.block_num * frame.laser_num * frame.packet_num;

    auto t1 = std::chrono::high_resolution_clock::now();

    // Move raw sensor data to GPU for parallel processing
    cudaSafeCall(cudaMemcpy(point_data_cu_, frame.pointData,
                            frame.block_num * frame.laser_num * frame.packet_num * sizeof(PointDecodeData),
                            cudaMemcpyHostToDevice),
                 ReturnCode::CudaMemcpyHostToDeviceError);

    cudaSafeCall(cudaMemcpy(sensor_timestamp_cu_, frame.sensor_timestamp,
                            frame.packet_num * sizeof(uint64_t),
                            cudaMemcpyHostToDevice),
                 ReturnCode::CudaMemcpyHostToDeviceError);

    // Transform raw LiDAR measurements into 3D coordinates
    // This handles all the complex sensor geometry, calibration, and coordinate transforms
    compute_xyzs_v4_3_impl<<<frame.packet_num, frame.block_num * frame.laser_num>>>(
        this->frame_.gpu()->points, (const int32_t *)channel_azimuths_cu_,
        (const int32_t *)channel_elevations_cu_, (const int8_t *)dazis_cu, deles_cu,
        mirror_azi_begins_cu, mirror_azi_ends_cu, m_PandarAT_corrections.header.resolution,
        point_data_cu_, sensor_timestamp_cu_, frame.distance_unit, this->transform_,
        frame.block_num, frame.laser_num, frame.packet_num);

    cudaDeviceSynchronize();
    this->frame_.DeviceToHost(0, frame.block_num * frame.laser_num * frame.packet_num * sizeof(T_Point));
    std::memcpy(frame.points, this->frame_.cpu()->points,
                frame.block_num * frame.laser_num * frame.packet_num * sizeof(T_Point));

    auto t2 = std::chrono::high_resolution_clock::now();

    // Apply Grace and Conrad algorithm to separate ground from objects
    int *num_filtered = (int *)malloc(sizeof(int));
    auto filtered_points = grace_and_conrad(frame.points,
                                            frame.block_num * frame.laser_num * frame.packet_num,
                                            num_filtered);

    frame.filtered_points_num = static_cast<uint32_t>(*num_filtered);
    std::memcpy(frame.filtered_points, filtered_points, *num_filtered * sizeof(T_Point));
    timers_->filtered_points_count = *num_filtered;

    auto t3 = std::chrono::high_resolution_clock::now();

    // Cluster the remaining object points to identify individual cones
    auto cone_clusters = dbscan(filtered_points, *num_filtered);

    frame.cones_num = cone_clusters.size();
    std::memcpy(frame.cones, cone_clusters.data(), cone_clusters.size() * sizeof(T_Point));
    timers_->clusters_found = cone_clusters.size();

    auto t4 = std::chrono::high_resolution_clock::now();

    cudaEventRecord(timers_->total_stop);
    cudaEventSynchronize(timers_->total_stop);

    // Calculate throughput metrics for performance monitoring
    float total_time_ms;
    cudaEventElapsedTime(&total_time_ms, timers_->total_start, timers_->total_stop);
    stats_.points_per_second = (stats_.input_points * 1000.0f) / total_time_ms;

    print_summary();

    free(num_filtered);
    free(filtered_points);

    return 0;
}

template <typename T_Point>
void Udp4_3ParserGpu<T_Point>::optimize_processing_parameters(size_t num_points)
{
    // Adapt our algorithm parameters based on the workload size
    // Large point clouds: sacrifice some accuracy for speed
    // Small point clouds: we can afford to be more thorough
    if (num_points > 50000)
    {
        gnc_params_.num_bins = std::min(gnc_params_.num_bins, 8);
        dbscan_params_.max_iterations = std::min(dbscan_params_.max_iterations, 5);
    }
    else if (num_points < 5000)
    {
        gnc_params_.num_bins = std::max(gnc_params_.num_bins, 15);
        dbscan_params_.max_iterations = std::max(dbscan_params_.max_iterations, 15);
    }

    // Make sure our memory buffers can handle the expected workload
    preallocate_buffers(num_points, gnc_params_.num_segments, gnc_params_.num_bins);
}

template <typename T_Point>
int Udp4_3ParserGpu<T_Point>::load_correction_string(char *data)
{
    try
    {
        char *p = data;
        PandarATCorrectionsHeader header = *(PandarATCorrectionsHeader *)p;

        // Verify this is a valid correction file format we understand
        if (0xee == header.delimiter[0] && 0xff == header.delimiter[1])
        {
            switch (header.version[1])
            {
            case 5:
            {
                m_PandarAT_corrections.header = header;
                auto frame_num = m_PandarAT_corrections.header.frame_number;
                auto channel_num = m_PandarAT_corrections.header.channel_number;
                p += sizeof(PandarATCorrectionsHeader);

                // Sanity check the data sizes to prevent buffer overflows
                if (frame_num > 8 || channel_num > AT128_LASER_NUM)
                {
                    LogError("correction error, frame_num: %u, channel_num: %u", frame_num, channel_num);
                    return -1;
                }

                // Parse the correction data in the expected order
                // These corrections compensate for manufacturing tolerances in the sensor
                memcpy((void *)&m_PandarAT_corrections.l.start_frame, p,
                       sizeof(uint32_t) * frame_num);
                p += sizeof(uint32_t) * frame_num;
                memcpy((void *)&m_PandarAT_corrections.l.end_frame, p,
                       sizeof(uint32_t) * frame_num);
                p += sizeof(uint32_t) * frame_num;
                memcpy((void *)&m_PandarAT_corrections.l.azimuth, p,
                       sizeof(int32_t) * channel_num);
                p += sizeof(int32_t) * channel_num;
                memcpy((void *)&m_PandarAT_corrections.l.elevation, p,
                       sizeof(int32_t) * channel_num);
                p += sizeof(int32_t) * channel_num;
                auto adjust_length = channel_num * CORRECTION_AZIMUTH_NUM;
                memcpy((void *)&m_PandarAT_corrections.azimuth_offset, p,
                       sizeof(int8_t) * adjust_length);
                p += sizeof(int8_t) * adjust_length;
                memcpy((void *)&m_PandarAT_corrections.elevation_offset, p,
                       sizeof(int8_t) * adjust_length);
                p += sizeof(int8_t) * adjust_length;
                memcpy((void *)&m_PandarAT_corrections.SHA256, p,
                       sizeof(uint8_t) * 32);
                p += sizeof(uint8_t) * 32;

                // Apply resolution scaling to frame boundaries
                for (int i = 0; i < frame_num; ++i)
                {
                    m_PandarAT_corrections.l.start_frame[i] =
                        m_PandarAT_corrections.l.start_frame[i] *
                        m_PandarAT_corrections.header.resolution;
                    m_PandarAT_corrections.l.end_frame[i] =
                        m_PandarAT_corrections.l.end_frame[i] *
                        m_PandarAT_corrections.header.resolution;
                }

                // Upload all correction data to GPU memory for fast access during processing
                CUDACheck(cudaMalloc(&channel_azimuths_cu_, sizeof(m_PandarAT_corrections.l.azimuth)));
                CUDACheck(cudaMalloc(&channel_elevations_cu_, sizeof(m_PandarAT_corrections.l.elevation)));
                CUDACheck(cudaMalloc(&dazis_cu, sizeof(m_PandarAT_corrections.azimuth_offset)));
                CUDACheck(cudaMalloc(&deles_cu, sizeof(m_PandarAT_corrections.elevation_offset)));
                CUDACheck(cudaMalloc(&mirror_azi_begins_cu, sizeof(m_PandarAT_corrections.l.start_frame)));
                CUDACheck(cudaMalloc(&mirror_azi_ends_cu, sizeof(m_PandarAT_corrections.l.end_frame)));
                CUDACheck(cudaMemcpy(channel_azimuths_cu_, m_PandarAT_corrections.l.azimuth, sizeof(m_PandarAT_corrections.l.azimuth), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemcpy(channel_elevations_cu_, m_PandarAT_corrections.l.elevation, sizeof(m_PandarAT_corrections.l.elevation), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemcpy(dazis_cu, m_PandarAT_corrections.azimuth_offset, sizeof(m_PandarAT_corrections.azimuth_offset), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemcpy(deles_cu, m_PandarAT_corrections.elevation_offset, sizeof(m_PandarAT_corrections.elevation_offset), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemcpy(mirror_azi_begins_cu, m_PandarAT_corrections.l.start_frame, sizeof(m_PandarAT_corrections.l.start_frame), cudaMemcpyHostToDevice));
                CUDACheck(cudaMemcpy(mirror_azi_ends_cu, m_PandarAT_corrections.l.end_frame, sizeof(m_PandarAT_corrections.l.end_frame), cudaMemcpyHostToDevice));
                corrections_loaded_ = true;
                return 0;
            }
            break;
            default:
                break;
            }
        }
        return -1;
    }
    catch (const std::exception &e)
    {
        LogFatal("load correction error: %s", e.what());
        return -1;
    }
    return -1;
}

template <typename T_Point>
int Udp4_3ParserGpu<T_Point>::load_correction_file(std::string lidar_correction_file)
{
    LogInfo("load correction file from local correction.csv now!");
    std::ifstream fin(lidar_correction_file);
    if (fin.is_open())
    {
        LogDebug("Open correction file success");

        // Read the entire file into memory for parsing
        int length = 0;
        fin.seekg(0, std::ios::end);
        length = fin.tellg();
        fin.seekg(0, std::ios::beg);
        char *buffer = new char[length];
        fin.read(buffer, length);
        fin.close();

        // Parse the correction data from the loaded buffer
        int ret = load_correction_string(buffer);
        delete[] buffer;

        if (ret != 0)
        {
            LogError("Parse local Correction file Error");
        }
        else
        {
            LogInfo("Parse local Correction file Success!!!");
            return 0;
        }
    }
    else
    {
        LogError("Open correction file failed");
        return -1;
    }
    return -1;
}