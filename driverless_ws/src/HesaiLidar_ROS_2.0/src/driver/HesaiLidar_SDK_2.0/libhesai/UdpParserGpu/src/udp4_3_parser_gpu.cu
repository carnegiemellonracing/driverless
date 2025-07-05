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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <unordered_map>

#include "udp4_3_parser_gpu.h"
#include "safe_call.cuh"
#include "return_code.h"

#define MAX_DISTANCE 25.0f // Maximum allowable distance
#define ERROR_MARGIN 0.2f  // Outlier threshold
#define MAX_HEIGHT 0.4f


// CUDA kernel for union-find initialization
template <typename T_Point>
__global__ void initializeClusters(T_Point* forced, int* parent, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    parent[idx] = idx; // Each point starts as its own cluster
}

// Device function to find the root of a cluster with path compression
template <typename T_Point>
__device__ int findRoot(T_Point* forced, int* parent, int idx) {
    while (idx != parent[idx]) {
        parent[idx] = parent[parent[idx]]; // Path compression
        idx = parent[idx];
    }
    return idx;
}

template <typename T_Point>
__device__ void pairIndexToIJ(const T_Point* points, long long k, int n, int *i_out, int *j_out) {
    int i = 0;
    // For row i, there are (n - 1 - i) pairs: (i,i+1), (i,i+2), ..., (i,n-1)
    while (true) {
        long long rowCount = (long long)(n - 1 - i);
        if (k < rowCount) {
            *i_out = i;
            *j_out = i + 1 + (int)k;  // j runs from i+1 to n-1
            break;
        }
        k -= rowCount;
        i++;
    }
}

template <typename T_Point>
__global__ void newFindAndUnionClusters(
    const T_Point* points, int* parent, int num_points, float eps) {
    long long totalPairs = ((long long)num_points * (num_points - 1)) / 2;
    long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (k >= totalPairs) return;

    int i, j;
    pairIndexToIJ(points, k, num_points, &i, &j);

    // Compute the Euclidean distance between points[i] and points[j]
    float dx = points[i].x - points[j].x;
    float dy = points[i].y - points[j].y;
    float dz = points[i].z - points[j].z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    if (dist <= eps) {
        // Perform union–find union if the points are within eps.
        int root1 = findRoot(points, parent, i);
        int root2 = findRoot(points, parent, j);
        if (root1 != root2) {
            // Attach the tree with the higher index to the lower one.
            int high = root1 > root2 ? root1 : root2;
            int low  = root1 < root2 ? root1 : root2;
            atomicMin(&parent[high], low);
         }
     }
}

// CUDA kernel to find neighbors and union clusters
template <typename T_Point>
__global__ void findAndUnionClusters(
    const T_Point* points, int* parent, int num_points, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    for (int j = idx+1; j < num_points; ++j) {
        if (idx == j) continue;

        float dx = points[idx].x - points[j].x;
        float dy = points[idx].y - points[j].y;
        float dz = points[idx].z - points[j].z;
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        if (dist <= eps) {
            // Union clusters
            int root1 = findRoot(points, parent, idx);
            int root2 = findRoot(points, parent, j);
            if (root1 != root2) {
                atomicMin(&parent[root1], root2); // Union by assigning the smaller root
                atomicMin(&parent[root2], root1);
            }
        }
    }
}

// CUDA kernel to compute centroids for clusters
template <typename T_Point>
__global__ void computeCentroids(
    const T_Point* points, const int* parent, T_Point* centroids, int* cluster_sizes, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int root = findRoot(points, const_cast<int*>(parent), idx); // Find cluster root
    atomicAdd(&centroids[root].x, points[idx].x);
    atomicAdd(&centroids[root].y, points[idx].y);
    atomicAdd(&centroids[root].z, points[idx].z);
    atomicAdd(&centroids[root].intensity, points[idx].intensity);
    atomicAdd(&cluster_sizes[root], 1);
}

// CUDA kernel to finalize centroids
template <typename T_Point>
__global__ void finalizeCentroids(T_Point* centroids, int* cluster_sizes, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int size = cluster_sizes[idx];
    if (size > 0) {
        centroids[idx].x /= size;
        centroids[idx].y /= size;
        centroids[idx].z /= size;
        centroids[idx].intensity /= size;
    }
}

template <typename T_Point>
__global__ void flattenClusters(T_Point* forced, int* parent, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    parent[idx] = findRoot(forced, parent, idx); // Compress paths fully
}



template <typename T_Point>
std::vector<T_Point> runDBSCAN(const T_Point* points_ptr, size_t num_points, float eps, int min_samples) {

    // Create CUDA events for timing
    cudaEvent_t total_start, total_stop, start, stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time_initialize = 0.0f, time_union = 0.0f, time_flatten = 0.0f, time_compute_centroids = 0.0f, time_finalize_centroids = 0.0f, total_time = 0.0f;

    // Record total start
    cudaEventRecord(total_start);

    // Allocate device memory
    T_Point* d_points;
    cudaMalloc(&d_points, num_points * sizeof(T_Point));
    cudaMemcpy(d_points, points_ptr, num_points * sizeof(T_Point), cudaMemcpyHostToDevice);

    int* d_parent;
    cudaMalloc(&d_parent, num_points * sizeof(int));
    cudaMemset(d_parent, 0, num_points * sizeof(int));

    T_Point* d_centroids;
    cudaMalloc(&d_centroids, num_points * sizeof(T_Point));
    cudaMemset(d_centroids, 0, num_points * sizeof(T_Point));

    int* d_cluster_sizes;
    cudaMalloc(&d_cluster_sizes, num_points * sizeof(int));
    cudaMemset(d_cluster_sizes, 0, num_points * sizeof(int));




    // Step 1: Initialize clusters
    cudaEventRecord(start);
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    initializeClusters<<<grid, block>>>(d_points, d_parent, num_points);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_initialize, start, stop);

    long long totalPairs = ((long long)num_points * (num_points - 1)) / 2;
    int pairBlock = 256;
    int pairGrid = (totalPairs + pairBlock - 1) / pairBlock;

    // Step 2: Find and union clusters
    cudaEventRecord(start);
    newFindAndUnionClusters<<<pairGrid, pairBlock>>>(
        d_points,
        d_parent,
        num_points, eps);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_union, start, stop);

    // Step 3: Flatten clusters
    cudaEventRecord(start);
    flattenClusters<<<grid, block>>>(
        d_points,
        d_parent,
        num_points);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_flatten, start, stop);

    // Copy flattened parent array back to host
    int* h_parent = (int*)malloc(num_points * sizeof(int));
    cudaMemcpy(h_parent, d_parent, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    

    // Identify unique roots
    std::unordered_map<int, int> root_to_index;
    std::vector<int> unique_roots;
    for (int i = 0; i < num_points; ++i) {
        int root = h_parent[i];
        if (root_to_index.find(root) == root_to_index.end()) {
            root_to_index[root] = unique_roots.size();
            unique_roots.push_back(root);
        }
    }


    // Step 4: Compute centroids
    cudaEventRecord(start);
    computeCentroids<<<grid, block>>>(
        d_points,
        d_parent,
        d_centroids,
        d_cluster_sizes,
        num_points);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_compute_centroids, start, stop);

    // Step 5: Finalize centroids
    cudaEventRecord(start);
    finalizeCentroids<<<grid, block>>>(
        d_centroids,
        d_cluster_sizes,
        num_points);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_finalize_centroids, start, stop);

    // Record total stop
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&total_time, total_start, total_stop);

    // Copy centroids back to host
    T_Point* h_centroids = (T_Point*)malloc(num_points * sizeof(T_Point));
    cudaMemcpy(h_centroids, d_centroids, num_points * sizeof(T_Point), cudaMemcpyDeviceToHost);

    int* h_cluster_sizes = (int*)malloc(num_points * sizeof(int));
    cudaMemcpy(h_cluster_sizes, d_cluster_sizes, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // Filter out valid clusters using unique roots
    std::vector<T_Point> centroids;
    for (int root : unique_roots) {
        if (h_cluster_sizes[root] > 0) {
            centroids.push_back(h_centroids[root]);
        }
    }

    free(h_parent);
    free(h_centroids);
    free(h_cluster_sizes);


    // Print timing for each step
    std::cout << "Timing Results (ms):" << std::endl;
    std::cout << " - Initialize Clusters: " << time_initialize << std::endl;
    std::cout << " - Find and Union Clusters: " << time_union << std::endl;
    std::cout << " - Flatten Clusters: " << time_flatten << std::endl;
    std::cout << " - Compute Centroids: " << time_compute_centroids << std::endl;
    std::cout << " - Finalize Centroids: " << time_finalize_centroids << std::endl;
    std::cout << " - Total Time: " << total_time << std::endl;

    cudaFree(d_points);
    cudaFree(d_parent);
    cudaFree(d_centroids);
    cudaFree(d_cluster_sizes);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);

    return centroids;
}


template <typename T_Point>
__global__ void assignToGrid(
    const T_Point* points, int* segments, int* bins, int num_points,
    float angle_min, float angle_max, float range_min, float range_max,
    int num_segments, int num_bins) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) return;

    T_Point p = points[idx];
    float angle = atan2f(p.y, p.x);
    if (angle < 0) angle += 2 * M_PI;

    float range = sqrtf((p.x * p.x) + (p.y * p.y));

    if (fabsf(p.x) > MAX_DISTANCE || fabsf(p.y) > MAX_DISTANCE ) {
        bins[idx] = 0;
        segments[idx] = 0;
        return;
    }

    if (range > MAX_DISTANCE || (p.y < 0.01f && p.x < 0.01f)) return; // Discard points further than MAX_DISTANCE


    int segment_value = min(static_cast<int>((angle - angle_min) / (angle_max - angle_min) * num_segments), num_segments - 1);
    segments[idx] = segment_value;
    bins[idx] = min(static_cast<int>((range - range_min) / (range_max - range_min) * num_bins), num_bins - 1);
    // if (idx % 1000 == 0){
    //     printf("Point %d is assigned to bin %d and segment %d \n", idx, bins[idx], segments[idx]);

    // }
}


__device__ __inline__ float atomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}


__device__ __inline__ unsigned long long packFloatAndInt(float z, int point_index) {
    int32_t scaled_z = __float2int_rz((z + 50.0f) * 100.0f);
    return (static_cast<unsigned long long>(scaled_z) << 32 | (point_index & 0xFFFFFFFF));
}

__device__ __inline__ void unpackFloatAndInt(unsigned long long packed_value, float& z, int& point_index) {
    int32_t scaled_z = (packed_value >> 32) & 0xFFFFFFFF;
    point_index = static_cast<int>(packed_value & 0xFFFFFFFF);
    z = (static_cast<float>(scaled_z) / 100.0f) - 50.0f;
    
}

template <typename T_Point>
__global__ void shitLowestPointInBin(
    T_Point* points, const int* segments, const int* bins, 
    unsigned long long* bin_min_z, int* bin_min_indices,
    int num_points, int num_segments, int num_bins) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) return;

    int segment = segments[idx];
    int bin = bins[idx];
    int cell_idx = segment * num_bins + bin;


    if (segment < 0 || segment >= num_segments || bin < 0 || bin >= num_bins){
        // printf("[ERROR] Invalid segment/bin: segment=%d, bin=%d, idx=%d\n", segment, bin, idx);
        return;
    }

    float z = points[idx].z;

    if (isnan(z) || z > 100.0f) {
        printf("[SKIP] Invalid z value: z=%f at idx=%d\n", z, idx);
        return;
    }

    unsigned long long packed_z = packFloatAndInt(z, idx);

    float test_z;
    int test_idx;
    unpackFloatAndInt(packed_z, test_z, test_idx);
    
    assert(test_z == z && "Error: Packed and unpacked z values are different!");
    assert(test_idx == idx && "Error: Packed and unpacked indices are different!");

    unsigned long long old_value;
    float old_z;
    int old_idx;
    
    atomicMin(&bin_min_z[cell_idx], packed_z);
}


template <typename T_Point>
__global__ void findLowestPointInBin(
    const T_Point* points, const int* segments, const int* bins, 
    unsigned long long* bin_min_z, int* bin_min_indices,
    int num_points, int num_segments, int num_bins) {

    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) return;

    int segment = segments[idx];
    int bin = bins[idx];
    int cell_idx = segment * num_bins + bin;

    float z = points[idx].z;

    float z_test;
    int idx_test;
    unpackFloatAndInt(packFloatAndInt(z, idx), z_test, idx_test);
    assert(z_test == z);
    assert(idx_test == idx);

    static_assert(sizeof(unsigned long long) == sizeof(uint64_t));


    assert(points[idx].z == z);
    unsigned long long packed_z = packFloatAndInt(z, idx);

    if (isnan(z) || z > 100.0f) return;
    
    unsigned long long value_actually_there = bin_min_z[cell_idx];
    unsigned long long prev_value_actually_there = value_actually_there;
    while (true) {
        float float_actually_there;
        int _death;
        unpackFloatAndInt(value_actually_there, float_actually_there, _death);
        prev_value_actually_there = value_actually_there; // just for the check in 3 lines
        if (z < float_actually_there){
            value_actually_there = atomicCAS(&bin_min_z[cell_idx], value_actually_there, packed_z);
            if (value_actually_there == prev_value_actually_there) {
                // printf("Successful cas for %d\n", cell_idx);
                break;

            } else {
                // printf("unsuccessful cas for %d\n", cell_idx);
                // float vat_float;
                // int vat_index;
                // unpackFloatAndInt(value_actually_there, vat_float, vat_index);

                // printf("vat_float: %f, vat_index: %d, old_z_unpacked: %f, _death: %d, z: %f, idx: %d\nval_actually_there: %lld, old_min_z_packed:%lld, packed_z: %lld\n",
                // vat_float, vat_index, old_z_unpacked, _death, z, idx, value_actually_there, old_min_z_packed, packed_z);

            }
        } else {
            // printf("cas: z value too high for %d\n", cell_idx);

            break;
        }
    }


    // TODO: missing the loop

    // printf("FUCKY WUCKY %f \n", &bin_min_z[cell_idx]);


    

    // if (atomicMinFloat(&bin_min_z[cell_idx], z) == z) {
    //     atomicExch(&bin_min_indices[cell_idx], idx);
    // }
}


// template <typename T_Point>
// __global__ void findLowestPointInBin(
//     const T_Point* points, const int* segments, const int* bins, T_Point* bin_lowest_points,
//     int num_points, int num_segments, int num_bins) {

//     int segment = blockIdx.x;
//     int bin = threadIdx.x;

//     int cell_idx = segment * num_bins + bin;
//     float min_z = FLT_MAX;
//     T_Point lowest_point;

//     for (int i = 0; i < num_points; i++) {
//         if (segments[i] == segment && bins[i] == bin) {
//             if (points[i].z < min_z) {
//                 min_z = points[i].z;
//                 lowest_point = points[i];
//             }
//         }
//     }

//     if (min_z < FLT_MAX) {
//         bin_lowest_points[cell_idx] = lowest_point;
//     } else {
//         T_Point default_point = {0, 0, FLT_MAX, 0, 0, 0};
//         bin_lowest_points[cell_idx] = default_point; // No valid points in bin
//     }
// }

template <typename T_Point>
__global__ void fitLineInSegment(
    const T_Point* bin_lowest_points, float* segment_line_params, int num_segments, int num_bins) {

    int segment = blockIdx.x;

    float x_sum = 0, z_sum = 0, x2_sum = 0, xz_sum = 0;
    int count = 0;

    for (int bin = 0; bin < num_bins; bin++) {
        int idx = segment * num_bins + bin;
        T_Point p = bin_lowest_points[idx];
        if (p.z != FLT_MAX) { // Ignore empty bins
            float range = sqrtf(p.x * p.x + p.y * p.y);
            x_sum += range;
            z_sum += p.z;
            x2_sum += range * range;
            xz_sum += range * p.z;
            count++;
        }
    }

    if (count > 1) {
        float slope = (count * xz_sum - x_sum * z_sum) / (count * x2_sum - x_sum * x_sum);
        float intercept = (z_sum - slope * x_sum) / count;
        segment_line_params[segment * 2 + 0] = slope;
        segment_line_params[segment * 2 + 1] = intercept;
    } else {
        segment_line_params[segment * 2 + 0] = 0; // Default flat line
        segment_line_params[segment * 2 + 1] = 0;
    }
}

template <typename T_Point>
__global__ void validateOutliers(
    const T_Point* points, const int* segments, const float* segment_line_params,
    T_Point* outliers, int* outlier_count, float margin_of_error, int num_points, int num_segments) {

    // printf("outlier count before modification is %d \n", outlier_count);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int segment = segments[idx];
    if (segment >= num_segments) return;

    float slope = segment_line_params[segment * 2 + 0];
    float intercept = segment_line_params[segment * 2 + 1];

    T_Point p = points[idx];
    float range = sqrtf(p.x * p.x + p.y * p.y);
    float predicted_z = slope * range + intercept;

    if (sqrtf((p.x * p.x) + (p.y * p.y)) > MAX_DISTANCE || (points[idx].y < 0.01f && points[idx].x < 0.01f)) return;

    if ((p.z - predicted_z) > ERROR_MARGIN && (p.z - predicted_z) < MAX_HEIGHT) {
        int out_idx = atomicAdd(outlier_count, 1);
        outliers[out_idx] = p;
    }

    // printf("outlier count is %d \n", outlier_count);
}





template <typename T_Point>
__global__ void populateLowestPoints(
    const T_Point* points, const unsigned long long* min_indices, T_Point* lowest_points, 
    int total_bins) {
    //   printf("Total num of bins is %d \n", total_bins);
    int cell_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (cell_idx >= total_bins) return;

    unsigned long long packed_val = min_indices[cell_idx];
    int index;
    float z_val;
    unpackFloatAndInt(packed_val, z_val, index);
    assert(packFloatAndInt(z_val, index) == packed_val);



    if (index >= 0) {
        // printf("index = %d, points[index].z = %f, z_val = %f, cell_idx = %d\n", index, points[index].z, z_val, cell_idx);
        assert(points[index].z == z_val);
        lowest_points[cell_idx] = points[index];
        // printf("The lowest z in bin %d is %f: \n", cell_idx, lowest_points[cell_idx].z);
    } else {
        // printf("cell index %d was never cas'd\n", cell_idx);
        T_Point default_point = {0, 0, FLT_MAX, 0, 0, 0};
        lowest_points[cell_idx] = default_point;
    }
}


__global__ __inline__ void initializeBinMinZ(unsigned long long* bin_min_z, int total_bins) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bins) return;
    bin_min_z[idx] =  packFloatAndInt(FLT_MAX, -1);
}



template <typename T_Point>
T_Point* processPointsCUDA(
    const T_Point* points_ptr, size_t num_points, int num_segments, int num_bins, int* h_outlier_count) {


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Number of points going into processPointsCUDA: " << num_points << "\n"; 

    float memAllocTime = 0.0f, assignToGridTime = 0.0f, findLowestPointInBinTime = 0.0f;

    // Allocate device memory
    T_Point* d_points;
    cudaMalloc(&d_points, num_points * sizeof(T_Point));
    cudaMemcpy(d_points, points_ptr, num_points * sizeof(T_Point), cudaMemcpyHostToDevice);

    int* d_segments, *d_bins;
    cudaMalloc(&d_segments, num_points * sizeof(int));
    cudaMalloc(&d_bins, num_points * sizeof(int));

    unsigned long long* d_bin_lowest_points;
    cudaMalloc(&d_bin_lowest_points, num_segments * num_bins * sizeof(unsigned long long));
    

    T_Point* d_lowest_points;
    cudaMalloc(&d_lowest_points, num_segments * num_bins * sizeof(T_Point));

    float* d_segment_line_params;
    cudaMalloc(&d_segment_line_params, num_segments * 2 * sizeof(float));

    T_Point* d_outliers;
    int* d_outlier_count;
    cudaMalloc(&d_outliers, num_points * sizeof(T_Point));
    cudaMalloc(&d_outlier_count, sizeof(int));
    cudaMemset(d_outlier_count, 0, sizeof(int));
    cudaDeviceSynchronize();
    cudaMemcpy(h_outlier_count, d_outlier_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "host outliers init: " << *h_outlier_count << std::endl;


    // Launch kernels
    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);


    cudaEventRecord(start);
    assignToGrid<<<grid, block>>>(d_points, d_segments, d_bins, num_points, 0.0f, M_PI / 10, 0.0f, MAX_DISTANCE, num_segments, num_bins);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&assignToGridTime, start, stop);

    cudaEventRecord(start);


    unsigned long long* d_bin_min_z;
    int* d_bin_min_indices;

    int total_bins = num_segments * num_bins;
    cudaMalloc(&d_bin_min_z, total_bins * sizeof(unsigned long long));
    cudaMalloc(&d_bin_min_indices, total_bins * sizeof(int));

   initializeBinMinZ<<<grid, block>>>(d_bin_min_z, total_bins);

   cudaDeviceSynchronize();

    std::cout << "total_bins is " << total_bins << "\n";
    shitLowestPointInBin<<<grid, block>>>(
        d_points, d_segments, d_bins, d_bin_min_z, d_bin_min_indices, 
        num_points, num_segments, num_bins);
    std::cout << "error caused by lowest point function is " << cudaGetLastError() << "\n";    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findLowestPointInBinTime, start, stop);

    int num_blocks = (total_bins + 256 - 1) / 256;
    populateLowestPoints<<<num_blocks, 256>>>(d_points, d_bin_min_z, d_lowest_points, total_bins);

    // print everything in d_lowest_points

    dim3 segmentGrid(num_segments);
    dim3 binGrid(num_bins);
    
    fitLineInSegment<<<segmentGrid, 1>>>(d_lowest_points, d_segment_line_params, num_segments, num_bins);

    // std::cout << "device outliers before call: " << *d_outlier_count << std::endl;
    validateOutliers<<<grid, block>>>(d_points, d_segments, d_segment_line_params,
                                      d_outliers, d_outlier_count, ERROR_MARGIN, num_points, num_segments);


    
    // cudaMemcpy(&h_outlier_count, d_outlier_count, sizeof(int), cudaMemcpyDeviceToHost);
    //PROBABLY WRONG
    // int* h_outlier_count;
    
    cudaMemcpy(h_outlier_count, d_outlier_count, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "host outliers: " << *h_outlier_count << std::endl;
    // Free memory and return
    T_Point* h_outliers = (T_Point*)malloc(*h_outlier_count * sizeof(T_Point));
    cudaMemcpy(h_outliers, d_outliers, *h_outlier_count * sizeof(T_Point), cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_segments);
    cudaFree(d_bins);
    cudaFree(d_bin_lowest_points);
    cudaFree(d_lowest_points);
    cudaFree(d_segment_line_params);
    cudaFree(d_outliers);
    cudaFree(d_outlier_count);
    cudaFree(d_bin_min_z);
    cudaFree(d_bin_min_indices);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return h_outliers;
}




template <typename T_Point>
__global__ void computeAnglesKernel(const T_Point* points, float* angles, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    const T_Point& p = points[idx];
    float angle = atan2f(p.y, p.x);
    if (angle < 0) angle += 2 * M_PI; // Normalize angle to [0, 2π]
    angles[idx] = angle;
}


template <typename T_Point>
__global__ void reduceMinMaxKernel(const T_Point* pinned_h_points, float* angles, float* result, int num_points, bool is_min) {
    extern __shared__ float shared_angles[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load elements into shared memory
    shared_angles[tid] = (idx < num_points) ? angles[idx] : (is_min ? FLT_MAX : -FLT_MAX);
    __syncthreads();

    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (is_min) {
                shared_angles[tid] = fminf(shared_angles[tid], shared_angles[tid + stride]);
            } else {
                shared_angles[tid] = fmaxf(shared_angles[tid], shared_angles[tid + stride]);
            }
        }
        __syncthreads();
    }

    // Write the result of this block's reduction to global memory
    if (tid == 0) {
        result[blockIdx.x] = shared_angles[0];
    }
}

// Host Function to Launch Final Reduction
// float finalReduction(float* d_result, int num_blocks, bool is_min) {
//     thrust::device_vector<float> intermediate_results(num_blocks);
//     cudaMemcpy(thrust::raw_pointer_cast(intermediate_results.data()), d_result, num_blocks * sizeof(float), cudaMemcpyDeviceToDevice);

//     if (is_min) {
//         return *thrust::min_element(intermediate_results.begin(), intermediate_results.end());
//     } else {
//         return *thrust::max_element(intermediate_results.begin(), intermediate_results.end());
//     }
// }



template <typename T_Point>
T_Point* GraceAndConrad(
    const T_Point* points_ptr,
    size_t num_points,
    float alpha,
    int num_bins,
    int* num_filtered) {

    //   int num_points = points.size();
      std::cout << "Points into GNC: " << num_points << std::endl; 
      
      // // Allocate memory for device points
      auto start = std::chrono::high_resolution_clock::now();
      
      
      T_Point* d_points;
      cudaMalloc(&d_points, num_points * sizeof(T_Point));

      auto end = std::chrono::high_resolution_clock::now();
      std::cout << "CUDA Malloc on device: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

      // // Copy host points to device
      // cudaMemcpy(d_points, std_points.data(), num_points * sizeof(Point), cudaMemcpyHostToDevice);
      

      // T_Point* pinned_h_points;
      // cudaHostAlloc(&pinned_h_points, num_points * sizeof(T_Point), cudaHostAllocDefault);
      // std::copy(points.begin(), points.end(), pinned_h_points);
      cudaMemcpy(d_points, points_ptr, num_points * sizeof(T_Point), cudaMemcpyHostToDevice);
      // cudaFreeHost(pinned_h_points);
      
      // Allocate memory for device angles
      float* d_angles;
      cudaMalloc(&d_angles, num_points * sizeof(float));

      // Kernel to compute angles
      start = std::chrono::high_resolution_clock::now();
      dim3 block(256);
      dim3 grid((num_points + block.x - 1) / block.x);
      computeAnglesKernel<<<grid, block>>>(d_points, d_angles, num_points);
      cudaDeviceSynchronize();
      end = std::chrono::high_resolution_clock::now();
      std::cout << "Compute angles: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

      // Allocate memory for block-wise results
      float* d_block_min;
      float* d_block_max;
      cudaMalloc(&d_block_min, grid.x * sizeof(float));
      cudaMalloc(&d_block_max, grid.x * sizeof(float));

      // Launch reduction kernels
      reduceMinMaxKernel<<<grid, block, block.x * sizeof(float)>>>(points_ptr, d_angles, d_block_min, num_points, true); // Min
      reduceMinMaxKernel<<<grid, block, block.x * sizeof(float)>>>(points_ptr, d_angles, d_block_max, num_points, false); // Max
      cudaDeviceSynchronize();


      // Perform final reduction on the host
      std::vector<float> h_block_min(grid.x);
      std::vector<float> h_block_max(grid.x);
      cudaMemcpy(h_block_min.data(), d_block_min, grid.x * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_block_max.data(), d_block_max, grid.x * sizeof(float), cudaMemcpyDeviceToHost);

      float angle_min = *std::min_element(h_block_min.begin(), h_block_min.end());
      float angle_max = *std::max_element(h_block_max.begin(), h_block_max.end());

      std::cout << "angle_max and angle_min respectively are " << angle_max << ", " << angle_min << "\n";

      // Free temporary memory
      cudaFree(d_block_min);
      cudaFree(d_block_max);
      cudaFree(d_points);

      // Compute number of segments
      int num_segments = static_cast<int>((angle_max - angle_min) / alpha) + 1;

      // Free temporary allocations
      cudaFree(d_angles);

      // Process points in CUDA
      cudaEvent_t start_event, stop_event;
      cudaEventCreate(&start_event);
      cudaEventCreate(&stop_event);

      cudaEventRecord(start_event);
      auto result = processPointsCUDA(points_ptr, num_points, num_segments, num_bins, num_filtered);
      cudaEventRecord(stop_event);

      cudaEventSynchronize(stop_event);

      float cuda_time = 0;
      cudaEventElapsedTime(&cuda_time, start_event, stop_event);
      std::cout << "processPointsCUDA execution time: " << cuda_time << " ms\n";

      // Clean up CUDA events
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);


      // Post-process result
      // start = std::chrono::high_resolution_clock::now();
      // thrust::host_vector<Point> result_vector(result.begin(), result.end());
      // end = std::chrono::high_resolution_clock::now();
      // std::cout << "Convert result to thrust::host_vector: "
      //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

      std::cout << "GraceAndConrad completed successfully.\n";
      std::cout << "Number of points returned: " << *num_filtered << std::endl;

      return result;
}

using namespace hesai::lidar;
template <typename T_Point>
Udp4_3ParserGpu<T_Point>::Udp4_3ParserGpu() {
  corrections_loaded_ = false;
  cudaSafeMalloc(point_data_cu_, POINT_DATA_LEN);
  cudaSafeMalloc(sensor_timestamp_cu_, SENSOR_TIMESTAMP_LEN);
}
template <typename T_Point>
Udp4_3ParserGpu<T_Point>::~Udp4_3ParserGpu() {
  cudaSafeFree(point_data_cu_);
  cudaSafeFree(sensor_timestamp_cu_);
  if (corrections_loaded_) {
    cudaSafeFree(deles_cu);
    cudaSafeFree(dazis_cu);
    cudaSafeFree(channel_elevations_cu_);
    cudaSafeFree(channel_azimuths_cu_);
    cudaSafeFree(mirror_azi_begins_cu);
    cudaSafeFree(mirror_azi_ends_cu);
    corrections_loaded_ = false;
  }
}
template <typename T_Point>
__global__ void compute_xyzs_v4_3_impl(
    T_Point *xyzs, const int32_t* channel_azimuths,
    const int32_t* channel_elevations, const int8_t* dazis, const int8_t* deles,
    const uint32_t* raw_azimuth_begin, const uint32_t* raw_azimuth_end, const uint8_t raw_correction_resolution,
    const PointDecodeData* point_data, const uint64_t* sensor_timestamp, const double raw_distance_unit, Transform transform, const uint16_t blocknum, const uint16_t lasernum, uint16_t packet_index) {
  auto iscan = blockIdx.x;
  auto ichannel = threadIdx.x;
  if (iscan >= packet_index || ichannel >= blocknum * lasernum) return;
  int point_index = iscan * blocknum * lasernum + (ichannel % (lasernum * blocknum));
  float azimuth = point_data[point_index].azimuth / kFineResolutionInt;
  int Azimuth = point_data[point_index].azimuth;
  int count = 0, field = 0;
  while (count < 3 &&
          (((Azimuth + (kFineResolutionInt * kCircle) - raw_azimuth_begin[field]) % (kFineResolutionInt * kCircle) +
          (raw_azimuth_end[field] + kFineResolutionInt * kCircle - Azimuth) % (kFineResolutionInt * kCircle)) !=
          (raw_azimuth_end[field] + kFineResolutionInt * kCircle -
          raw_azimuth_begin[field]) % (kFineResolutionInt * kCircle))) {
    field = (field + 1) % 3;
    count++;
  }
  if (count >= 3) return;
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
  gpu::setX(xyzs[point_index], x_);
  gpu::setY(xyzs[point_index],  y_);
  gpu::setZ(xyzs[point_index], z_);
  gpu::setIntensity(xyzs[point_index], point_data[point_index].reflectivities);
  gpu::setTimestamp(xyzs[point_index], double(sensor_timestamp[iscan]) / kMicrosecondToSecond);
  gpu::setRing(xyzs[point_index], ichannel % lasernum);
  gpu::setConfidence(xyzs[point_index], point_data[point_index].confidence);
}

template <typename T_Point>
int Udp4_3ParserGpu<T_Point>::ComputeXYZI(LidarDecodedFrame<T_Point> &frame) {
  if (!corrections_loaded_) return int(ReturnCode::CorrectionsUnloaded);
  std::cout << "cuda error at the start of the function? " << cudaGetLastError() << "\n";
  auto t1 = std::chrono::high_resolution_clock::now();
  assert(frame.pointData != nullptr);
  assert(point_data_cu_ != nullptr);
  std::cout << "cuda error before memcpy? " << cudaGetLastError() << "\n";
  cudaSafeCall(cudaMemcpy(point_data_cu_, frame.pointData,
                          frame.block_num * frame.laser_num * frame.packet_num * sizeof(PointDecodeData), 
                          cudaMemcpyHostToDevice), ReturnCode::CudaMemcpyHostToDeviceError);
  std::cout << "cuda error after memcpy? " << cudaGetLastError() << "\n";
  cudaSafeCall(cudaMemcpy(sensor_timestamp_cu_, frame.sensor_timestamp,
                          frame.packet_num * sizeof(uint64_t), 
                          cudaMemcpyHostToDevice), ReturnCode::CudaMemcpyHostToDeviceError);                          
  auto t2 = std::chrono::high_resolution_clock::now();
compute_xyzs_v4_3_impl<<<frame.packet_num, frame.block_num * frame.laser_num>>>(
    this->frame_.gpu()->points, (const int32_t*)channel_azimuths_cu_,
    (const int32_t*)channel_elevations_cu_, (const int8_t*)dazis_cu, deles_cu,
    mirror_azi_begins_cu, mirror_azi_ends_cu, m_PandarAT_corrections.header.resolution,
    point_data_cu_, sensor_timestamp_cu_, frame.distance_unit, this->transform_, frame.block_num, frame.laser_num, frame.packet_num);
  cudaDeviceSynchronize();
  auto t3 = std::chrono::high_resolution_clock::now();
  cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  this->frame_.DeviceToHost(0, frame.block_num * frame.laser_num * frame.packet_num * sizeof(T_Point));
  auto t4 = std::chrono::high_resolution_clock::now();
  std::memcpy(frame.points, this->frame_.cpu()->points, frame.block_num * frame.laser_num * frame.packet_num * sizeof(T_Point));
  auto t5 = std::chrono::high_resolution_clock::now();

  // std::cout << "time taken to copy data to gpu is " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n"
  //           << "time taken for kernel is " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << "\n"
  //           << "time taken to copy data from gpu is " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << "\n"
  //           << "time taken to copy cpu to cpu is " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() << "\n";

  float alpha = 1.5f;
  int num_bins = 10;
  int* num_filtered = (int*)malloc(sizeof(int));
  auto filtered_points = GraceAndConrad(frame.points, frame.block_num * frame.laser_num * frame.packet_num, alpha, num_bins, num_filtered);
  frame.filtered_points_num = static_cast<uint32_t>(*num_filtered);
  std::memcpy(frame.filtered_points, filtered_points, *num_filtered * sizeof(T_Point));
  std::cout << "num_filtered is " << *num_filtered << "\n\n\n\n";

  
  cudaDeviceSynchronize();
  int min_samples = 2;
  float eps = 0.3f;
  auto cone_clusters = runDBSCAN(filtered_points, *num_filtered, eps, min_samples); // Call your coloring function here
  free(num_filtered);
  cudaDeviceSynchronize();
  std::cout << "Number of clusters: " << cone_clusters.size() << std::endl;
  frame.cones_num = cone_clusters.size();
  std::memcpy(frame.cones, cone_clusters.data(), cone_clusters.size() * sizeof(T_Point));


  free(filtered_points);
//   cudaSafeCall(cudaGetLastError(), ReturnCode::CudaXYZComputingError);
  // Clustering

  return 0;
}

template <typename T_Point>
int Udp4_3ParserGpu<T_Point>::LoadCorrectionString(char *data) {
  try {
    char *p = data;
    PandarATCorrectionsHeader header = *(PandarATCorrectionsHeader *)p;
    if (0xee == header.delimiter[0] && 0xff == header.delimiter[1]) {
      switch (header.version[1]) {
        case 5: {
          m_PandarAT_corrections.header = header;
          auto frame_num = m_PandarAT_corrections.header.frame_number;
          auto channel_num = m_PandarAT_corrections.header.channel_number;
          p += sizeof(PandarATCorrectionsHeader);
          if (frame_num > 8 || channel_num > AT128_LASER_NUM) {
            LogError("correction error, frame_num: %u, channel_num: %u", frame_num, channel_num);
            return -1;
          }
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
          for (int i = 0; i < frame_num; ++i) {
            m_PandarAT_corrections.l.start_frame[i] =
                m_PandarAT_corrections.l.start_frame[i] *
                m_PandarAT_corrections.header.resolution;
            m_PandarAT_corrections.l.end_frame[i] =
                m_PandarAT_corrections.l.end_frame[i] *
                m_PandarAT_corrections.header.resolution;
          }
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
        } break;
        default:
          break;
      }
    }
    return -1;
  } catch (const std::exception &e) {
    LogFatal("load correction error: %s", e.what());
    return -1;
  }
  return -1;
}
template <typename T_Point>
int Udp4_3ParserGpu<T_Point>::LoadCorrectionFile(std::string lidar_correction_file) {
  LogInfo("load correction file from local correction.csv now!");
  std::ifstream fin(lidar_correction_file);
  if (fin.is_open()) {
    LogDebug("Open correction file success");
    int length = 0;
    fin.seekg(0, std::ios::end);
    length = fin.tellg();
    fin.seekg(0, std::ios::beg);
    char *buffer = new char[length];
    fin.read(buffer, length);
    fin.close();
    int ret = LoadCorrectionString(buffer);
    delete[] buffer;
    if (ret != 0) {
      LogError("Parse local Correction file Error");
    } else {
      LogInfo("Parse local Correction file Success!!!");
      return 0;
    }
  } else {
    LogError("Open correction file failed");
    return -1;
  }
  return -1;
}
