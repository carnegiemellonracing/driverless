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
        __device__ static void sample_curv_state(const float world_pose[3], float curv_pose[3], bool& out_of_bounds) {
            const float x = world_pose[0];
            const float y = world_pose[1];
            const float yaw = world_pose[2];

            const float xmin = curv_frame_lookup_tex_info.xcenter - curv_frame_lookup_tex_info.width / 2;
            const float ymin = curv_frame_lookup_tex_info.ycenter - curv_frame_lookup_tex_info.width / 2;

            const float u = (x - xmin) / curv_frame_lookup_tex_info.width;
            const float v = (y - ymin) / curv_frame_lookup_tex_info.width;

            const float4 parallel_pose = tex2D<float4>(curv_frame_lookup_tex, u, v);
            curv_pose[0] = parallel_pose.x;
            curv_pose[1] = parallel_pose.y;
            curv_pose[2] = yaw - parallel_pose.z;

            // if (__cudaGet_threadIdx().x == 0 && __cudaGet_blockIdx().x == 0) {
            //     printf("tex info: %f, %f, %f\n", curv_frame_lookup_tex_info.xcenter, curv_frame_lookup_tex_info.ycenter, curv_frame_lookup_tex_info.width);
            //     printf("parallel_pose: %f, %f, %f\n", parallel_pose.x, parallel_pose.y, parallel_pose.z);
            // }
            // __syncthreads();

            out_of_bounds = parallel_pose.w < 0;
        }
    }
}