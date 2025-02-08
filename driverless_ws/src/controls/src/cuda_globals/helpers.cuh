#pragma once

#include "cuda_globals.cuh"


namespace controls {
    namespace cuda_globals {
    //TODO: move to cuda_globals?
        /**
         * @brief Sample the lookup table with a inertial pose to get the corresponding curvilinear pose
         * @param[in] world_pose [x, y, yaw] in world frame
         * @param[out] curv_pose [progress, offset, heading] in curvilinear frame
         * @param[out] out_of_bounds true if the world pose is out of bounds of the lookup table.
         */
        __device__ static void sample_curv_state(const float world_pose[3], float curv_pose[6], bool& out_of_bounds) {
            const float x = world_pose[0];
            const float y = world_pose[1];
            const float yaw = world_pose[2];

            const float xmin = curv_frame_lookup_tex_info.xcenter - curv_frame_lookup_tex_info.width / 2;
            const float ymin = curv_frame_lookup_tex_info.ycenter - curv_frame_lookup_tex_info.width / 2;

            const float u = (x - xmin) / curv_frame_lookup_tex_info.width;
            const float v = (y - ymin) / curv_frame_lookup_tex_info.width;

            const float4 left_parallel_pose = tex2D<float4>(left_curv_frame_lookup_tex, u, v);
            const float4 right_parallel_pose = tex2D<float4>(right_curv_frame_lookup_tex, u, v);
            paranoid_assert(isnan(left_parallel_pose.x));
            paranoid_assert(isnan(left_parallel_pose.y));
            paranoid_assert( isnan(left_parallel_pose.z));
            paranoid_assert(isnan(left_parallel_pose.w));
            paranoid_assert(isnan(right_parallel_pose.x) || isnan(right_parallel_pose.y) || isnan(right_parallel_pose.z) || isnan(right_parallel_pose.w));

            curv_pose[0] = left_parallel_pose.x;
            curv_pose[1] = left_parallel_pose.y;
            curv_pose[2] = yaw - left_parallel_pose.z;
            curv_pose[3] = right_parallel_pose.x;
            curv_pose[4] = right_parallel_pose.y;
            curv_pose[5] = yaw - right_parallel_pose.z;

            // if (__cudaGet_threadIdx().x == 0 && __cudaGet_blockIdx().x == 0) {
            //     printf("tex info: %f, %f, %f\n", curv_frame_lookup_tex_info.xcenter, curv_frame_lookup_tex_info.ycenter, curv_frame_lookup_tex_info.width);
            //     printf("parallel_pose: %f, %f, %f\n", parallel_pose.x, parallel_pose.y, parallel_pose.z);
            // }
            // __syncthreads();

            out_of_bounds = left_parallel_pose.w < 0.5f;
        }
    }
}